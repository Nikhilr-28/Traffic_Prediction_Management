"""
Download and prepare PEMS-BAY dataset for traffic prediction
Also downloads OSM data for the Bay Area
"""

import os
import requests
import zipfile
import gdown
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import osmnx as ox
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_pems_bay():
    """
    Download PEMS-BAY dataset from public sources
    The dataset contains:
    - pems-bay.h5: Traffic speed data (325 sensors x time steps)
    - adj_mx_bay.pkl: Adjacency matrix based on road distances
    """
    
    data_dir = Path("data/raw/pems-bay")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # PEMS-BAY is available from multiple sources
    # Option 1: From Google Drive (most reliable)
    file_ids = {
        "pems-bay.h5": "1GQPGOK6bNKHBOJrLMJR9PVs-6pXQE9Nj",
        "adj_mx_bay.pkl": "1Wz6kGmPWMhNr9VEXRAKSEjXHsW9bEPLG"
    }
    
    for filename, file_id in file_ids.items():
        output_path = data_dir / filename
        
        if output_path.exists():
            logger.info(f"✓ {filename} already exists")
            continue
            
        logger.info(f"Downloading {filename}...")
        try:
            # Use gdown to download from Google Drive
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            logger.info(f"✓ Downloaded {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            logger.info("Alternative: Download manually from https://github.com/liyaguang/DCRNN")
            
    # Verify data integrity
    verify_pems_data(data_dir)
    
    return data_dir


def verify_pems_data(data_dir):
    """
    Verify PEMS-BAY data integrity and print statistics
    """
    import h5py
    
    # Check HDF5 file
    h5_path = data_dir / "pems-bay.h5"
    if h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            speeds = f['speed']
            logger.info(f"✓ PEMS-BAY data shape: {speeds.shape}")
            logger.info(f"  Sensors: {speeds.shape[1]}")
            logger.info(f"  Time steps: {speeds.shape[0]}")
            logger.info(f"  Date range: {pd.to_datetime(f['date'][0])} to {pd.to_datetime(f['date'][-1])}")
    
    # Check adjacency matrix
    adj_path = data_dir / "adj_mx_bay.pkl"
    if adj_path.exists():
        with open(adj_path, 'rb') as f:
            adj_mx, _ = pickle.load(f, encoding='latin1')
            logger.info(f"✓ Adjacency matrix shape: {adj_mx.shape}")
            logger.info(f"  Sparsity: {(adj_mx > 0).sum() / adj_mx.size:.2%}")


def download_osm_data():
    """
    Download OpenStreetMap data for San Francisco Bay Area
    """
    osm_dir = Path("data/raw/osm")
    osm_dir.mkdir(parents=True, exist_ok=True)
    
    # Define Bay Area bounding box (roughly covers PEMS sensors)
    # San Francisco Bay Area coordinates
    north, south = 38.0, 37.2
    east, west = -121.7, -122.5
    
    logger.info("Downloading OSM road network for Bay Area...")
    
    try:
        # Download drive network
        G = ox.graph_from_bbox(
            north, south, east, west,
            network_type='drive',
            simplify=True,
            retain_all=False
        )
        
        # Save as GraphML and shapefile
        ox.save_graphml(G, osm_dir / "bay_area_drive.graphml")
        
        # Also save as GeoDataFrame for easier processing
        nodes, edges = ox.graph_to_gdfs(G)
        edges.to_parquet(osm_dir / "bay_area_edges.parquet")
        nodes.to_parquet(osm_dir / "bay_area_nodes.parquet")
        
        logger.info(f"✓ Downloaded OSM network:")
        logger.info(f"  Nodes: {len(nodes)}")
        logger.info(f"  Edges: {len(edges)}")
        
        # Filter to major roads only (for routing)
        highway_types = ['motorway', 'trunk', 'primary', 'secondary', 'motorway_link']
        major_edges = edges[edges['highway'].isin(highway_types)]
        major_edges.to_parquet(osm_dir / "bay_area_major_roads.parquet")
        logger.info(f"  Major roads: {len(major_edges)} edges")
        
    except Exception as e:
        logger.error(f"Failed to download OSM data: {e}")
        logger.info("You may need to configure OSMnx settings or use a different method")
    
    return osm_dir


def create_sensor_metadata():
    """
    Create sensor metadata file with locations and road mappings
    This is critical for sensor-to-edge mapping
    """
    
    # PEMS sensor locations (approximate - you may need to get exact coordinates)
    # These are example coordinates for Bay Area highways
    sensor_locations = {
        # Format: sensor_id: (lat, lon, highway, direction)
        0: (37.7749, -122.4194, "US-101", "North"),
        1: (37.7849, -122.4094, "US-101", "South"),
        # Add more sensor locations as needed
    }
    
    metadata_path = Path("data/processed/pems-bay/sensor_metadata.csv")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame.from_dict(
        sensor_locations, 
        orient='index',
        columns=['latitude', 'longitude', 'highway', 'direction']
    )
    df.to_csv(metadata_path)
    logger.info(f"✓ Created sensor metadata: {metadata_path}")
    
    return metadata_path


def setup_data_directories():
    """
    Create all necessary data directories
    """
    directories = [
        "data/raw/pems-bay",
        "data/raw/osm",
        "data/processed/pems-bay",
        "data/processed/osm",
        "data/cache",
        "data/checkpoints",
        "data/reports"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Created all data directories")


if __name__ == "__main__":
    print("=" * 60)
    print("PEMS-BAY & OSM Data Download Script")
    print("=" * 60)
    
    # Setup directories
    setup_data_directories()
    
    # Download PEMS-BAY
    print("\n1. Downloading PEMS-BAY dataset...")
    pems_dir = download_pems_bay()
    
    # Download OSM data
    print("\n2. Downloading OSM road network...")
    osm_dir = download_osm_data()
    
    # Create sensor metadata
    print("\n3. Creating sensor metadata...")
    metadata_path = create_sensor_metadata()
    
    print("\n" + "=" * 60)
    print("✅ Data download complete!")
    print("\nNext steps:")
    print("1. Run data_processor.py to process and partition the data")
    print("2. Run mapping_validator.py to verify sensor-to-edge mappings")
    print("3. Start training with train.py")