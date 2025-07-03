# Configuration file for Cluster Recommendation Wizard

# Application Settings
APP_TITLE = "Cluster Recommendation Wizard"
APP_DESCRIPTION = "Powered by AI and Data Analytics"
SERVER_PORT = 7860
SERVER_HOST = "0.0.0.0"

# Dataset Settings
DATASET_SIZE = 300
DATASET_FILE = "cluster_dataset.csv"

# LLM Settings
LLM_ENDPOINT = "anthropic.claude-3-7-sonnet-20250219-v1:0"
AWS_REGION = "us-east-1"
MAX_LLM_TOKENS = 1000

# Similarity Matching Settings
MAX_TFIDF_FEATURES = 1000
TOP_K_SIMILAR_WORKLOADS = 5

# Cluster Size Configurations
CLUSTER_SIZES = {
    "small": {"base_cost": 0.5, "min_nodes": 2, "max_nodes": 5},
    "medium": {"base_cost": 1.2, "min_nodes": 3, "max_nodes": 8},
    "large": {"base_cost": 2.5, "min_nodes": 5, "max_nodes": 15},
    "xlarge": {"base_cost": 5.0, "min_nodes": 8, "max_nodes": 25},
    "2xlarge": {"base_cost": 10.0, "min_nodes": 15, "max_nodes": 40},
    "4xlarge": {"base_cost": 20.0, "min_nodes": 25, "max_nodes": 100}
}

# Workload Types
WORKLOAD_TYPES = [
    "Data Analytics",
    "Machine Learning Training", 
    "Real-time Processing",
    "Batch Processing",
    "Data Warehousing",
    "ETL Pipeline",
    "Stream Processing",
    "Graph Analytics",
    "Time Series Analysis",
    "Deep Learning",
    "Feature Engineering",
    "Data Mining"
]

# Resource Thresholds for Rule-based Adjustments
CPU_THRESHOLD_HIGH = 64
CPU_THRESHOLD_VERY_HIGH = 128
MEMORY_THRESHOLD_HIGH = 256  # GB
MEMORY_THRESHOLD_VERY_HIGH = 512  # GB

# UI Settings
ENABLE_PUBLIC_SHARING = True
ENABLE_DEBUG_MODE = True
SHOW_ERROR_DETAILS = True

# Example Workloads for UI
EXAMPLE_WORKLOADS = [
    "I need to run deep learning model training on 10TB of image data. Require 64 CPU cores, 256GB RAM, GPU acceleration, and support for 5 data scientists working concurrently.",
    "Real-time fraud detection system processing 1TB of transaction data daily with sub-100ms latency requirements and 1000 concurrent users.",
    "Data warehouse for business analytics with 50TB of historical data, complex OLAP queries, and 100 concurrent business users requiring sub-second response times.",
    "ETL pipeline processing 5TB of data nightly with 32 CPU cores, 128GB RAM, and high I/O throughput requirements.",
    "Stream processing application handling IoT sensor data from 10,000 devices with real-time analytics and 16 CPU cores, 64GB RAM."
]
