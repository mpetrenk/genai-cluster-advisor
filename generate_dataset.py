import pandas as pd
import numpy as np
import random
from typing import Dict, List
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_dataset(num_samples: int = 300) -> pd.DataFrame:
    """
    Generate synthetic dataset for cluster recommendation
    """
    
    # Define workload types and their characteristics
    workload_types = [
        "Data Analytics", "Machine Learning Training", "Real-time Processing", 
        "Batch Processing", "Data Warehousing", "ETL Pipeline", 
        "Stream Processing", "Graph Analytics", "Time Series Analysis",
        "Deep Learning", "Feature Engineering", "Data Mining"
    ]
    
    # Define cluster sizes
    cluster_sizes = ["small", "medium", "large", "xlarge", "2xlarge", "4xlarge"]
    
    # Define data locations
    data_locations = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "multi-region"]
    
    # Define storage types
    storage_types = ["SSD", "HDD", "NVMe", "S3", "EFS"]
    
    dataset = []
    
    for i in range(num_samples):
        # Generate workload characteristics
        workload_type = random.choice(workload_types)
        
        # Generate resource requirements based on workload type
        if workload_type in ["Machine Learning Training", "Deep Learning"]:
            cpu_cores = np.random.randint(16, 128)
            memory_gb = np.random.randint(64, 512)
            gpu_required = random.choice([True, False])
            io_intensive = random.choice([True, False])
            network_bandwidth_gbps = np.random.uniform(10, 100)
        elif workload_type in ["Real-time Processing", "Stream Processing"]:
            cpu_cores = np.random.randint(8, 64)
            memory_gb = np.random.randint(32, 256)
            gpu_required = False
            io_intensive = True
            network_bandwidth_gbps = np.random.uniform(25, 100)
        elif workload_type in ["Data Warehousing", "Data Analytics"]:
            cpu_cores = np.random.randint(32, 256)
            memory_gb = np.random.randint(128, 1024)
            gpu_required = False
            io_intensive = True
            network_bandwidth_gbps = np.random.uniform(10, 50)
        else:  # Batch Processing, ETL, etc.
            cpu_cores = np.random.randint(4, 32)
            memory_gb = np.random.randint(16, 128)
            gpu_required = False
            io_intensive = random.choice([True, False])
            network_bandwidth_gbps = np.random.uniform(1, 25)
        
        # Generate other parameters
        data_size_tb = np.random.uniform(0.1, 100)
        concurrent_users = np.random.randint(1, 1000)
        latency_requirement_ms = np.random.uniform(1, 5000)
        availability_requirement = np.random.uniform(95, 99.99)
        
        # Determine recommended cluster size based on resource requirements
        total_compute_score = (cpu_cores * 0.3 + memory_gb * 0.002 + 
                              (50 if gpu_required else 0) + 
                              network_bandwidth_gbps * 0.5)
        
        if total_compute_score < 20:
            recommended_cluster = "small"
            node_count = np.random.randint(2, 5)
        elif total_compute_score < 50:
            recommended_cluster = "medium"
            node_count = np.random.randint(3, 8)
        elif total_compute_score < 100:
            recommended_cluster = "large"
            node_count = np.random.randint(5, 15)
        elif total_compute_score < 200:
            recommended_cluster = "xlarge"
            node_count = np.random.randint(8, 25)
        elif total_compute_score < 300:
            recommended_cluster = "2xlarge"
            node_count = np.random.randint(15, 40)
        else:
            recommended_cluster = "4xlarge"
            node_count = np.random.randint(25, 100)
        
        # Generate workload description
        workload_description = generate_workload_description(
            workload_type, cpu_cores, memory_gb, data_size_tb, 
            concurrent_users, gpu_required, io_intensive
        )
        
        dataset.append({
            'workload_id': f'workload_{i+1:03d}',
            'workload_type': workload_type,
            'workload_description': workload_description,
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb,
            'gpu_required': gpu_required,
            'data_size_tb': round(data_size_tb, 2),
            'io_intensive': io_intensive,
            'network_bandwidth_gbps': round(network_bandwidth_gbps, 2),
            'concurrent_users': concurrent_users,
            'latency_requirement_ms': round(latency_requirement_ms, 2),
            'availability_requirement': round(availability_requirement, 2),
            'data_location': random.choice(data_locations),
            'storage_type': random.choice(storage_types),
            'recommended_cluster_size': recommended_cluster,
            'recommended_node_count': node_count,
            'estimated_cost_per_hour': round(calculate_estimated_cost(recommended_cluster, node_count), 2)
        })
    
    return pd.DataFrame(dataset)

def generate_workload_description(workload_type: str, cpu_cores: int, memory_gb: int, 
                                data_size_tb: float, concurrent_users: int, 
                                gpu_required: bool, io_intensive: bool) -> str:
    """Generate realistic workload descriptions"""
    
    descriptions = {
        "Data Analytics": [
            f"Running complex analytical queries on {data_size_tb}TB of customer data with {concurrent_users} concurrent analysts",
            f"Performing statistical analysis and reporting on large datasets requiring {cpu_cores} cores and {memory_gb}GB RAM",
            f"Business intelligence workload processing {data_size_tb}TB of transactional data for real-time dashboards"
        ],
        "Machine Learning Training": [
            f"Training deep neural networks on {data_size_tb}TB of training data with {concurrent_users} data scientists",
            f"Distributed ML model training requiring {cpu_cores} cores, {memory_gb}GB RAM" + (" and GPU acceleration" if gpu_required else ""),
            f"AutoML pipeline processing large feature sets with high memory requirements"
        ],
        "Real-time Processing": [
            f"Processing streaming data from {concurrent_users} concurrent sources with sub-{int(100)}ms latency requirements",
            f"Real-time fraud detection system handling {data_size_tb}TB of transaction data daily",
            f"IoT data processing pipeline with high throughput and low latency requirements"
        ],
        "Batch Processing": [
            f"Nightly ETL jobs processing {data_size_tb}TB of data from multiple sources",
            f"Large-scale data transformation pipeline running {concurrent_users} parallel jobs",
            f"Batch analytics workload requiring {cpu_cores} cores for compute-intensive operations"
        ],
        "Data Warehousing": [
            f"Enterprise data warehouse storing {data_size_tb}TB with {concurrent_users} concurrent query users",
            f"OLAP workload requiring high memory ({memory_gb}GB) for complex aggregations",
            f"Multi-dimensional analysis on large fact tables with star schema design"
        ]
    }
    
    if workload_type in descriptions:
        return random.choice(descriptions[workload_type])
    else:
        return f"{workload_type} workload processing {data_size_tb}TB of data with {concurrent_users} users"

def calculate_estimated_cost(cluster_size: str, node_count: int) -> float:
    """Calculate estimated hourly cost based on cluster size and node count"""
    
    base_costs = {
        "small": 0.5,
        "medium": 1.2,
        "large": 2.5,
        "xlarge": 5.0,
        "2xlarge": 10.0,
        "4xlarge": 20.0
    }
    
    return base_costs.get(cluster_size, 1.0) * node_count

if __name__ == "__main__":
    # Generate dataset
    df = generate_synthetic_dataset(300)
    
    # Save to CSV
    df.to_csv('cluster_dataset.csv', index=False)
    
    # Save to JSON for easy loading
    df.to_json('cluster_dataset.json', orient='records', indent=2)
    
    print(f"Generated dataset with {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head())
    
    # Print some statistics
    print(f"\nCluster size distribution:")
    print(df['recommended_cluster_size'].value_counts())
