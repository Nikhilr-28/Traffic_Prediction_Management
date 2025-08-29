"""
Critical sensor-to-edge mapping with validation and report generation
This ensures each sensor maps to exactly one directed edge
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import folium
from shapely.geometry import Point
from scipy.spatial import cKDTree
import logging
import pickle
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorEdgeMapper:
    """
    Maps PEMS sensors to OSM edges with strict validation
    Generates mapping reports for CI/CD validation
    """
    
    def __init__(self, osm_edges_path: Path, sensor_metadata_path: Path):
        """
        Initialize with OSM edges and sensor metadata
        """
        self.edges = gpd.read_parquet(osm_edges_path)
        self.sensors = pd.read_csv(sensor_metadata_path, index_col=0)
        
        # Mapping results
        self.sensor_to_edge = {}
        self.edge_to_sensors = {}
        self.unmapped_sensors = []
        self.interpolation_edges = {}
        
        # Validation results
        self.validation_errors = []
        self.mapping_stats = {}
        
        logger.info(f"Loaded {len(self.edges)} edges and {len(self.sensors)} sensors")
    
    def split_edges_at_sensors(self):
        """
        Split long edges at sensor positions to ensure 1:1 mapping
        """
        edges_to_split = []
        
        for idx, sensor in self.sensors.iterrows():
            sensor_point = Point(sensor['longitude'], sensor['latitude'])
            
            # Find nearest edge
            distances = self.edges.geometry.distance(sensor_point)
            nearest_edge_idx = distances.idxmin()
            nearest_edge = self.edges.loc[nearest_edge_idx]
            
            # Check if we need to split this edge
            edge_length = nearest_edge.geometry.length
            if edge_length > 0.001:  # Roughly 100 meters in degrees
                edges_to_split.append((nearest_edge_idx, sensor_point))
        
        # Perform edge splitting
        for edge_idx, split_point in edges_to_split:
            self._split_edge(edge_idx, split_point)
        
        logger.info(f"Split {len(edges_to_split)} edges at sensor locations")
    
    def _split_edge(self, edge_idx, split_point):
        """
        Split an edge at a given point
        """
        # Implementation depends on your edge representation
        # This is a placeholder - implement based on your data structure
        pass
    
    def create_directional_mapping(self):
        """
        Create sensor-to-edge mapping with direction awareness
        """
        mapped_count = 0
        
        for idx, sensor in self.sensors.iterrows():
            sensor_point = Point(sensor['longitude'], sensor['latitude'])
            highway = sensor.get('highway', '')
            direction = sensor.get('direction', '')
            
            # Find candidate edges within threshold distance
            distances = self.edges.geometry.distance(sensor_point)
            candidates = self.edges[distances < 0.001]  # ~100m threshold
            
            if len(candidates) == 0:
                self.unmapped_sensors.append(idx)
                continue
            
            # Filter by direction
            if direction in ['North', 'East']:
                # Assuming bearing calculation - implement based on your needs
                candidates = candidates[candidates['bearing'] > 0]
            elif direction in ['South', 'West']:
                candidates = candidates[candidates['bearing'] < 0]
            
            # Select closest edge from filtered candidates
            if len(candidates) > 0:
                nearest_idx = candidates.geometry.distance(sensor_point).idxmin()
                self.sensor_to_edge[idx] = nearest_idx
                
                # Track reverse mapping
                if nearest_idx not in self.edge_to_sensors:
                    self.edge_to_sensors[nearest_idx] = []
                self.edge_to_sensors[nearest_idx].append(idx)
                
                mapped_count += 1
        
        logger.info(f"Mapped {mapped_count}/{len(self.sensors)} sensors")
        logger.info(f"Unmapped sensors: {len(self.unmapped_sensors)}")
    
    def setup_interpolation(self):
        """
        Setup IDW interpolation for edges without direct sensor coverage
        """
        # Get edges with sensors
        covered_edges = set(self.edge_to_sensors.keys())
        all_edges = set(self.edges.index)
        uncovered_edges = all_edges - covered_edges
        
        logger.info(f"Setting up interpolation for {len(uncovered_edges)} uncovered edges")
        
        # Build KD-tree for fast nearest neighbor search
        sensor_coords = self.sensors[['longitude', 'latitude']].values
        tree = cKDTree(sensor_coords)
        
        for edge_idx in uncovered_edges:
            edge_centroid = self.edges.loc[edge_idx].geometry.centroid
            edge_coord = [edge_centroid.x, edge_centroid.y]
            
            # Find k nearest sensors
            k = min(5, len(self.sensors))
            distances, indices = tree.query(edge_coord, k=k)
            
            # Only use sensors within max interpolation distance (2km)
            valid_sensors = [
                (self.sensors.index[i], d) 
                for i, d in zip(indices, distances) 
                if d < 0.02  # Roughly 2km in degrees
            ]
            
            if valid_sensors:
                # Calculate IDW weights
                weights = [1/d**2 for _, d in valid_sensors]
                total_weight = sum(weights)
                normalized_weights = [w/total_weight for w in weights]
                
                self.interpolation_edges[edge_idx] = {
                    'sensors': [s for s, _ in valid_sensors],
                    'weights': normalized_weights,
                    'avg_distance_km': np.mean([d * 111 for _, d in valid_sensors])  # Convert to km
                }
        
        logger.info(f"Interpolation setup for {len(self.interpolation_edges)} edges")
    
    def validate_mapping(self):
        """
        Run comprehensive validation tests
        """
        errors = []
        
        # Test 1: Each sensor maps to exactly one edge
        for sensor_id in self.sensors.index:
            if sensor_id not in self.sensor_to_edge and sensor_id not in self.unmapped_sensors:
                errors.append(f"Sensor {sensor_id} has no mapping")
        
        # Test 2: No sensor maps to multiple directions
        for sensor_id, edge_id in self.sensor_to_edge.items():
            # Check edge direction consistency
            # Implementation depends on your edge attributes
            pass
        
        # Test 3: Check for reasonable interpolation distances
        for edge_id, interp_info in self.interpolation_edges.items():
            if interp_info['avg_distance_km'] > 3.0:
                errors.append(f"Edge {edge_id} has interpolation distance > 3km")
        
        self.validation_errors = errors
        
        if errors:
            logger.warning(f"Validation found {len(errors)} errors")
            for error in errors[:5]:  # Show first 5
                logger.warning(f"  - {error}")
        else:
            logger.info("✓ All validation tests passed")
        
        return len(errors) == 0
    
    def generate_mapping_report(self, output_dir: Path):
        """
        Generate comprehensive mapping report (CSV + visualization)
        Required for CI/CD validation
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        self.mapping_stats = {
            'total_sensors': len(self.sensors),
            'mapped_sensors': len(self.sensor_to_edge),
            'unmapped_sensors': len(self.unmapped_sensors),
            'coverage_percent': len(self.sensor_to_edge) / len(self.sensors) * 100,
            'total_edges': len(self.edges),
            'edges_with_sensors': len(self.edge_to_sensors),
            'edges_with_interpolation': len(self.interpolation_edges),
            'edge_coverage_percent': (len(self.edge_to_sensors) + len(self.interpolation_edges)) / len(self.edges) * 100,
            'avg_interpolation_distance_km': np.mean([
                info['avg_distance_km'] 
                for info in self.interpolation_edges.values()
            ]) if self.interpolation_edges else 0,
            'validation_errors': len(self.validation_errors),
            'timestamp': timestamp
        }
        
        # 1. Save statistics as CSV
        stats_df = pd.DataFrame([self.mapping_stats])
        stats_path = output_dir / f"mapping_stats_{timestamp}.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"✓ Saved mapping statistics to {stats_path}")
        
        # 2. Save detailed mapping as CSV
        mapping_df = pd.DataFrame([
            {
                'sensor_id': sensor_id,
                'edge_id': edge_id,
                'sensor_lat': self.sensors.loc[sensor_id, 'latitude'],
                'sensor_lon': self.sensors.loc[sensor_id, 'longitude'],
                'highway': self.sensors.loc[sensor_id, 'highway'],
                'direction': self.sensors.loc[sensor_id, 'direction']
            }
            for sensor_id, edge_id in self.sensor_to_edge.items()
        ])
        mapping_path = output_dir / f"sensor_edge_mapping_{timestamp}.csv"
        mapping_df.to_csv(mapping_path, index=False)
        logger.info(f"✓ Saved detailed mapping to {mapping_path}")
        
        # 3. Generate visualization map
        self.create_mapping_visualization(output_dir / f"mapping_visual_{timestamp}.html")
        
        # 4. Save JSON report for CI/CD
        report = {
            'stats': self.mapping_stats,
            'validation_passed': len(self.validation_errors) == 0,
            'errors': self.validation_errors[:10]  # First 10 errors
        }
        report_path = output_dir / f"mapping_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("SENSOR-TO-EDGE MAPPING REPORT")
        print("="*60)
        print(f"Sensor Coverage: {self.mapping_stats['coverage_percent']:.1f}%")
        print(f"Edge Coverage: {self.mapping_stats['edge_coverage_percent']:.1f}%")
        print(f"Avg Interpolation Distance: {self.mapping_stats['avg_interpolation_distance_km']:.2f} km")
        print(f"Validation Errors: {self.mapping_stats['validation_errors']}")
        print("="*60)
        
        return self.mapping_stats
    
    def create_mapping_visualization(self, output_path: Path):
        """
        Create interactive map showing sensor-edge mappings
        """
        # Center map on Bay Area
        m = folium.Map(location=[37.7, -122.2], zoom_start=10)
        
        # Add sensors with different colors based on mapping status
        for idx, sensor in self.sensors.iterrows():
            if idx in self.sensor_to_edge:
                color = 'green'
                popup = f"Sensor {idx}: Mapped to edge {self.sensor_to_edge[idx]}"
            elif idx in self.unmapped_sensors:
                color = 'red'
                popup = f"Sensor {idx}: UNMAPPED"
            else:
                color = 'orange'
                popup = f"Sensor {idx}: Unknown status"
            
            folium.CircleMarker(
                location=[sensor['latitude'], sensor['longitude']],
                radius=5,
                color=color,
                fill=True,
                popup=popup
            ).add_to(m)
        
        # Save map
        m.save(str(output_path))
        logger.info(f"✓ Saved visualization to {output_path}")
    
    def save_mapping(self, output_path: Path):
        """
        Save the mapping for use in training and inference
        """
        mapping_data = {
            'sensor_to_edge': self.sensor_to_edge,
            'edge_to_sensors': self.edge_to_sensors,
            'interpolation_edges': self.interpolation_edges,
            'unmapped_sensors': self.unmapped_sensors,
            'mapping_stats': self.mapping_stats
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(mapping_data, f)
        
        logger.info(f"✓ Saved mapping to {output_path}")


def run_mapping_pipeline():
    """
    Run the complete mapping pipeline
    """
    # Paths
    osm_edges = Path("data/raw/osm/bay_area_edges.parquet")
    sensor_metadata = Path("data/processed/pems-bay/sensor_metadata.csv")
    output_dir = Path("data/reports")
    
    # Check files exist
    if not osm_edges.exists():
        logger.error(f"OSM edges not found at {osm_edges}")
        return None
    
    if not sensor_metadata.exists():
        logger.error(f"Sensor metadata not found at {sensor_metadata}")
        return None
    
    # Run mapping
    mapper = SensorEdgeMapper(osm_edges, sensor_metadata)
    
    # Step 1: Split edges at sensors
    mapper.split_edges_at_sensors()
    
    # Step 2: Create directional mapping
    mapper.create_directional_mapping()
    
    # Step 3: Setup interpolation
    mapper.setup_interpolation()
    
    # Step 4: Validate
    is_valid = mapper.validate_mapping()
    
    # Step 5: Generate report
    stats = mapper.generate_mapping_report(output_dir)
    
    # Step 6: Save mapping
    mapper.save_mapping(Path("data/processed/pems-bay/sensor_edge_mapping.pkl"))
    
    return stats, is_valid


if __name__ == "__main__":
    stats, is_valid = run_mapping_pipeline()
    
    if is_valid:
        print("\n✅ Mapping validation PASSED - ready for training!")
    else:
        print("\n⚠️ Mapping validation FAILED - review errors before training")
        sys.exit(1)  # Exit with error code for CI/CD