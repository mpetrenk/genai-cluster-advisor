# 🧙‍♂️ Cluster Recommendation Wizard

An intelligent cluster sizing recommendation system powered by AI and data analytics. This application helps users determine optimal cluster configurations based on workload descriptions using machine learning similarity matching and LLM-powered analysis.

## 🚀 Features

- **Intelligent Workload Analysis**: Uses TF-IDF vectorization and cosine similarity to match user requirements with 300+ synthetic workload examples
- **LLM Integration**: Powered by Claude-3.7-Sonnet via AWS Bedrock for detailed workload analysis
- **Rule-based Optimization**: Combines similarity matching with rule-based adjustments for optimal recommendations
- **Cost Estimation**: Provides hourly and daily cost estimates based on AWS EC2 pricing models
- **Public Sharing**: Generates publicly shareable URLs for easy collaboration
- **Interactive UI**: Clean, modern Gradio interface with tabbed results and example workloads

## 📊 Dataset

The application uses a synthetic dataset of 300 workload examples with the following characteristics:

### Workload Types
- Data Analytics
- Machine Learning Training
- Real-time Processing
- Batch Processing
- Data Warehousing
- ETL Pipeline
- Stream Processing
- Graph Analytics
- Time Series Analysis
- Deep Learning
- Feature Engineering
- Data Mining

### Resource Parameters
- **CPU Cores**: 4-256 cores
- **Memory**: 16GB-1024GB RAM
- **Data Size**: 0.1TB-100TB
- **Concurrent Users**: 1-1000 users
- **GPU Requirements**: Boolean flag
- **I/O Intensity**: Boolean flag
- **Network Bandwidth**: 1-100 Gbps
- **Latency Requirements**: 1-5000ms
- **Availability**: 95-99.99%

### Cluster Sizes
- Small (2-5 nodes)
- Medium (3-8 nodes)
- Large (5-15 nodes)
- XLarge (8-25 nodes)
- 2XLarge (15-40 nodes)
- 4XLarge (25-100 nodes)

## 🛠️ Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "cluster recommendation wizard"
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the synthetic dataset (if not already generated):**
   ```bash
   python generate_dataset.py
   ```

## 🔧 Configuration

### AWS Bedrock Setup (Optional)
To enable full LLM analysis, configure AWS credentials with Bedrock access:

1. Install AWS CLI and configure credentials:
   ```bash
   aws configure
   ```

2. Ensure you have access to the Claude-3.7-Sonnet model in your AWS region

3. Update the region in `app.py` if needed (default: us-east-1)

**Note**: The application works in demo mode without AWS configuration, providing rule-based recommendations and simulated LLM analysis.

## 🚀 Usage

### Running the Application

```bash
python app.py
```

The application will:
1. Generate a local URL (typically http://127.0.0.1:7860)
2. Create a **publicly shareable URL** via Gradio's sharing service
3. Display both URLs in the terminal

### Using the Interface

1. **Describe Your Workload**: Enter a detailed description of your workload requirements
2. **Get Recommendations**: Click the "Get Cluster Recommendation" button
3. **Review Results**: Check the three tabs:
   - **🎯 Recommendation**: Rule-based cluster sizing with cost estimates
   - **🤖 LLM Analysis**: AI-powered detailed analysis and insights
   - **📊 Similar Workloads**: Database matches with similarity scores

### Example Workload Descriptions

```
"I need to run deep learning model training on 10TB of image data. 
Require 64 CPU cores, 256GB RAM, GPU acceleration, and support for 
5 data scientists working concurrently."
```

```
"Real-time fraud detection system processing 1TB of transaction data 
daily with sub-100ms latency requirements and 1000 concurrent users."
```

```
"Data warehouse for business analytics with 50TB of historical data, 
complex OLAP queries, and 100 concurrent business users requiring 
sub-second response times."
```

## 🏗️ Architecture

### Components

1. **Dataset Generator** (`generate_dataset.py`):
   - Creates synthetic workload examples
   - Generates realistic resource requirements
   - Calculates cluster recommendations based on compute scores

2. **Main Application** (`app.py`):
   - Gradio web interface
   - TF-IDF similarity matching
   - Rule-based recommendation engine
   - AWS Bedrock integration
   - Cost calculation

3. **Recommendation Engine**:
   - Text parsing for requirement extraction
   - Similarity-based workload matching
   - Rule-based cluster size adjustments
   - Cost estimation

### Algorithm Flow

1. **Input Processing**: Extract technical requirements from user description
2. **Similarity Matching**: Find similar workloads using TF-IDF cosine similarity
3. **Rule-based Adjustment**: Modify recommendations based on extracted requirements
4. **LLM Analysis**: Generate detailed insights using Claude-3.7-Sonnet
5. **Result Presentation**: Display recommendations, analysis, and similar workloads

## 📈 Technical Specifications

- **Framework**: Gradio 5.35.0
- **ML Libraries**: scikit-learn, pandas, numpy
- **LLM Integration**: AWS Bedrock with Claude-3.7-Sonnet
- **Similarity Algorithm**: TF-IDF with cosine similarity
- **Dataset Size**: 300 synthetic workload examples
- **Supported Cluster Sizes**: 6 tiers (small to 4xlarge)

## 🔍 Features in Detail

### Intelligent Text Parsing
- Extracts CPU, memory, data size, and user requirements from natural language
- Detects GPU and I/O intensive workload indicators
- Handles various formats and units

### Similarity Matching
- TF-IDF vectorization of workload descriptions
- Cosine similarity scoring
- Top-K similar workload retrieval

### Cost Estimation
- Based on AWS EC2 pricing models
- Hourly and daily cost projections
- Scales with cluster size and node count

### Public Sharing
- Automatic generation of shareable URLs
- No authentication required for viewers
- Temporary URLs valid for session duration

## 🚀 Deployment Options

### Local Development
```bash
python app.py
```

### Production Deployment
For production deployment, consider:
- Using a proper WSGI server (gunicorn, uvicorn)
- Setting up proper AWS IAM roles for Bedrock access
- Implementing authentication if needed
- Using environment variables for configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🔮 Future Enhancements

- **Real Workload Data**: Integration with actual cluster performance data
- **Advanced ML Models**: Deep learning models for better recommendations
- **Multi-cloud Support**: Support for Azure, GCP cluster recommendations
- **Performance Monitoring**: Integration with monitoring tools for feedback loops
- **Custom Workload Types**: User-defined workload categories
- **Batch Processing**: Support for multiple workload analysis
- **API Endpoints**: REST API for programmatic access

## 📞 Support

For issues, questions, or contributions, please refer to the project documentation or create an issue in the repository.

---

**Built with ❤️ using Gradio, scikit-learn, and AWS Bedrock**
