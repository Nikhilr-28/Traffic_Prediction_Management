"""
Process PEMS-BAY data into partitioned Parquet files optimized for DuckDB
Handles temporal splits and feature engineering
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from typing import Tuple, Dict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PEMSDataProcessor:
    """
    Process PEMS-BAY data for training and inference
    Creates partitioned Parquet files with proper temporal splits
    """
    
    def __init__(self, config_path: str = "configs/data/pems_bay.yaml"):
        """Initialize with configuration"""
        # For now, hardcode config - in production, load from YAML
        self.config = {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'lookback_window': 12,  # 1 hour
            'prediction_horizons': [3, 6, 9, 12],  # 15, 30, 45, 60 minutes
            'sample_rate_minutes': 5,
            'normalize_method': 'z_score'
        }
        
        self.raw_dir = Path("data/raw/pems-bay")
        self.processed_dir = Path("data/processed/pems-bay")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sensor mapping if exists
        mapping_path = self.processed_dir / "sensor_edge_mapping.pkl"
        if mapping_path.exists():
            with open(mapping_path, 'rb') as f:
                self.mapping_data = pickle.load(f)
                logger.info("✓ Loaded sensor-to-edge mapping")
        else:
            self.mapping_data = None
            logger.warning("No sensor-to-edge mapping found")
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load raw PEMS-BAY data from H5 file
        """
        h5_path = self.raw_dir / "pems-bay.h5"
        
        if not h5_path.exists():
            raise FileNotFoundError(f"Data file not found: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            # Speed data: (num_timesteps, num_sensors)
            speeds = f['speed'][:]
            
            # Convert timestamps
            # PEMS data typically starts from 2017-01-01
            start_date = datetime(2017, 1, 1)
            timestamps = [
                start_date + timedelta(minutes=5*i) 
                for i in range(speeds.shape[0])
            ]
        
        # Create DataFrame
        df = pd.DataFrame(speeds, index=timestamps)
        df.index.name = 'timestamp'
        
        logger.info(f"✓ Loaded data shape: {df.shape}")
        logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"  Sensors: {df.shape[1]}")
        logger.info(f"  Missing values: {df.isna().sum().sum()}")
        
        # Load adjacency matrix
        adj_path = self.raw_dir / "adj_mx_bay.pkl"
        with open(adj_path, 'rb') as f:
            _, _, adj_mx = pickle.load(f, encoding='latin1')
        
        return df, adj_mx
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for better predictions
        """
        df = df.copy()
        
        # Time of day (normalized to 0-1)
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['time_of_day'] = (df['hour'] * 60 + df['minute']) / 1440
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df.index.dayofweek
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Month (for seasonal patterns)
        df['month'] = df.index.month
        
        logger.info("✓ Added temporal features")
        
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize speed data using z-score normalization
        Store normalization parameters for inference
        """
        # Only normalize speed columns (not temporal features)
        speed_cols = [col for col in df.columns if isinstance(col, int)]
        
        if self.config['normalize_method'] == 'z_score':
            mean = df[speed_cols].mean()
            std = df[speed_cols].std()
            df[speed_cols] = (df[speed_cols] - mean) / std
            
            norm_params = {'method': 'z_score', 'mean': mean.to_dict(), 'std': std.to_dict()}
        
        elif self.config['normalize_method'] == 'min_max':
            min_val = df[speed_cols].min()
            max_val = df[speed_cols].max()
            df[speed_cols] = (df[speed_cols] - min_val) / (max_val - min_val)
            
            norm_params = {'method': 'min_max', 'min': min_val.to_dict(), 'max': max_val.to_dict()}
        
        # Save normalization parameters
        with open(self.processed_dir / 'normalization_params.pkl', 'wb') as f:
            pickle.dump(norm_params, f)
        
        logger.info(f"✓ Normalized data using {self.config['normalize_method']}")
        
        return df, norm_params
    
    def create_temporal_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create temporal train/val/test splits
        Strict temporal ordering - no data leakage
        """
        n = len(df)
        train_end = int(n * self.config['train_ratio'])
        val_end = train_end + int(n * self.config['val_ratio'])
        
        splits = {
            'train': df.iloc[:train_end],
            'val': df.iloc[train_end:val_end],
            'test': df.iloc[val_end:]
        }
        
        for split_name, split_df in splits.items():
            logger.info(f"{split_name}: {len(split_df)} samples ({len(split_df)/n*100:.1f}%)")
            logger.info(f"  Date range: {split_df.index[0]} to {split_df.index[-1]}")
        
        return splits
    
    def save_as_partitioned_parquet(self, df: pd.DataFrame, name: str):
        """
        Save DataFrame as partitioned Parquet files for DuckDB
        Partition by date and hour for efficient querying
        """
        output_dir = self.processed_dir / name
        output_dir.mkdir(exist_ok=True)
        
        # Add partition columns
        df = df.copy()
        df['date'] = df.index.date
        df['hour'] = df.index.hour
        
        # Convert to PyArrow table for partitioning
        table = pa.Table.from_pandas(df.reset_index())
        
        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=str(output_dir),
            partition_cols=['date', 'hour'],
            compression='snappy',
            use_dictionary=True,
            data_page_size=2**20  # 1MB pages for better DuckDB performance
        )
        
        logger.info(f"✓ Saved {name} as partitioned Parquet to {output_dir}")
        
        # Create DuckDB view for easy querying
        self.create_duckdb_view(output_dir, name)
    
    def create_duckdb_view(self, parquet_dir: Path, view_name: str):
        """
        Create DuckDB view for efficient querying
        """
        con = duckdb.connect(str(self.processed_dir / 'traffic.duckdb'))
        
        # Create view pointing to partitioned Parquet files
        query = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT * FROM read_parquet('{parquet_dir}/*/*/*.parquet')
        """
        con.execute(query)
        
        # Create some useful aggregation views
        if view_name == 'train':
            # Hourly averages for baseline predictions
            con.execute(f"""
            CREATE OR REPLACE VIEW hourly_avg AS
            SELECT 
                hour,
                AVG(sensor_0) as avg_speed_0,
                COUNT(*) as sample_count
            FROM {view_name}
            GROUP BY hour
            """)
        
        con.close()
        logger.info(f"✓ Created DuckDB view: {view_name}")
    
    def create_graph_features(self, adj_mx: np.ndarray):
        """
        Process adjacency matrix and create graph features
        """
        # Normalize adjacency matrix (random walk normalization)
        rowsum = adj_mx.sum(axis=1, keepdims=True)
        degree_matrix = np.where(rowsum > 0, 1.0 / rowsum, 0)
        adj_normalized = degree_matrix * adj_mx
        
        # Save both original and normalized adjacency
        np.save(self.processed_dir / 'adj_mx_original.npy', adj_mx)
        np.save(self.processed_dir / 'adj_mx_normalized.npy', adj_normalized)
        
        # Calculate graph statistics
        stats = {
            'num_nodes': adj_mx.shape[0],
            'num_edges': (adj_mx > 0).sum(),
            'avg_degree': (adj_mx > 0).sum(axis=1).mean(),
            'density': (adj_mx > 0).sum() / (adj_mx.shape[0] ** 2)
        }
        
        logger.info("✓ Graph statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return adj_normalized, stats
    
    def generate_congestion_weights(self, df: pd.DataFrame):
        """
        Generate congestion weights for weighted loss calculation
        Higher weights for congested periods
        """
        # Assuming free-flow speed is around 65 mph
        FREE_FLOW_SPEED = 65.0
        
        speed_cols = [col for col in df.columns if isinstance(col, int)]
        
        # Calculate congestion factor (0 = free flow, 1 = stopped)
        congestion = 1 - (df[speed_cols] / FREE_FLOW_SPEED).clip(0, 1)
        
        # Weight = 1 + 4 * congestion (1x to 5x weight)
        weights = 1 + 4 * congestion
        
        # Save weights aligned with data
        weights_df = pd.DataFrame(weights, index=df.index)
        weights_df.to_parquet(self.processed_dir / 'congestion_weights.parquet')
        
        logger.info("✓ Generated congestion weights")
        logger.info(f"  Average weight: {weights.mean().mean():.2f}")
        logger.info(f"  Max weight: {weights.max().max():.2f}")
        
        return weights
    
    def run_pipeline(self):
        """
        Run complete data processing pipeline
        """
        logger.info("="*60)
        logger.info("Starting PEMS-BAY Data Processing Pipeline")
        logger.info("="*60)
        
        # Load raw data
        logger.info("\n1. Loading raw data...")
        df, adj_mx = self.load_raw_data()
        
        # Add temporal features
        logger.info("\n2. Creating temporal features...")
        df = self.create_temporal_features(df)
        
        # Handle missing values
        logger.info("\n3. Handling missing values...")
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize data
        logger.info("\n4. Normalizing data...")
        df, norm_params = self.normalize_data(df)
        
        # Create temporal splits
        logger.info("\n5. Creating temporal splits...")
        splits = self.create_temporal_splits(df)
        
        # Save as partitioned Parquet
        logger.info("\n6. Saving partitioned Parquet files...")
        for split_name, split_df in splits.items():
            self.save_as_partitioned_parquet(split_df, split_name)
        
        # Process graph features
        logger.info("\n7. Processing graph features...")
        adj_normalized, graph_stats = self.create_graph_features(adj_mx)
        
        # Generate congestion weights
        logger.info("\n8. Generating congestion weights...")
        self.generate_congestion_weights(df)
        
        # Save metadata
        metadata = {
            'num_sensors': df.shape[1] - 5,  # Subtract temporal feature columns
            'train_samples': len(splits['train']),
            'val_samples': len(splits['val']),
            'test_samples': len(splits['test']),
            'graph_stats': graph_stats,
            'normalization': norm_params
        }
        
        with open(self.processed_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("\n" + "="*60)
        logger.info("✅ Data processing complete!")
        logger.info("="*60)
        
        # Test DuckDB query
        logger.info("\nTesting DuckDB query...")
        con = duckdb.connect(str(self.processed_dir / 'traffic.duckdb'))
        result = con.execute("SELECT COUNT(*) FROM train").fetchone()
        logger.info(f"  Train samples in DuckDB: {result[0]}")
        con.close()
        
        return metadata


if __name__ == "__main__":
    processor = PEMSDataProcessor()
    metadata = processor.run_pipeline()
    
    print("\nNext steps:")
    print("1. Run extract_sensor_locations.py to get sensor coordinates")
    print("2. Run mapping_validator.py to validate sensor-to-edge mappings")
    print("3. Start model training with train.py")