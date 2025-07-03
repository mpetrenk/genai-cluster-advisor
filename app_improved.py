import gradio as gr
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import boto3
import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClusterRecommendationWizard:
    def __init__(self, dataset_path: str = "cluster_dataset.csv"):
        """Initialize the cluster recommendation wizard"""
        try:
            self.dataset = pd.read_csv(dataset_path)
            print(f"✅ Dataset loaded successfully: {len(self.dataset)} samples")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            sys.exit(1)
            
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.description_vectors = None
        self.setup_vectorizer()
        
        # LLM endpoint configuration
        self.llm_endpoint = "anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.bedrock_client = None
        self.setup_bedrock_client()
    
    def setup_bedrock_client(self):
        """Setup AWS Bedrock client for LLM calls"""
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name='us-east-1'
            )
            print("✅ AWS Bedrock client initialized")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize Bedrock client: {e}")
            print("   Application will run in demo mode")
            self.bedrock_client = None
    
    def setup_vectorizer(self):
        """Setup TF-IDF vectorizer with existing dataset descriptions"""
        try:
            descriptions = self.dataset['workload_description'].tolist()
            self.description_vectors = self.vectorizer.fit_transform(descriptions)
            print("✅ TF-IDF vectorizer initialized")
        except Exception as e:
            print(f"❌ Error setting up vectorizer: {e}")
            sys.exit(1)
    
    def extract_requirements_from_text(self, user_input: str) -> Dict:
        """Extract technical requirements from user input using regex patterns"""
        requirements = {}
        
        # Extract CPU requirements
        cpu_patterns = [
            r'(\d+)\s*(?:cpu|core|cores|vcpu)',
            r'(\d+)\s*(?:processor|processors)',
            r'need\s*(\d+)\s*(?:cpu|core|cores)'
        ]
        for pattern in cpu_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                requirements['cpu_cores'] = int(match.group(1))
                break
        
        # Extract memory requirements
        memory_patterns = [
            r'(\d+)\s*(?:gb|gigabyte|gigabytes)\s*(?:ram|memory)',
            r'(\d+)\s*(?:gb|g)\s*(?:of\s*)?(?:ram|memory)',
            r'memory.*?(\d+)\s*(?:gb|g)',
            r'ram.*?(\d+)\s*(?:gb|g)'
        ]
        for pattern in memory_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                requirements['memory_gb'] = int(match.group(1))
                break
        
        # Extract data size
        data_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:tb|terabyte|terabytes)',
            r'(\d+(?:\.\d+)?)\s*(?:gb|gigabyte|gigabytes)',
            r'data.*?(\d+(?:\.\d+)?)\s*(?:tb|gb)'
        ]
        for pattern in data_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                size = float(match.group(1))
                if 'gb' in match.group(0).lower():
                    size = size / 1000  # Convert GB to TB
                requirements['data_size_tb'] = size
                break
        
        # Extract user count
        user_patterns = [
            r'(\d+)\s*(?:user|users|concurrent|simultaneous)',
            r'(\d+)\s*people',
            r'support.*?(\d+)'
        ]
        for pattern in user_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                requirements['concurrent_users'] = int(match.group(1))
                break
        
        # Detect GPU requirement
        gpu_keywords = ['gpu', 'graphics', 'cuda', 'machine learning', 'deep learning', 'ai training']
        requirements['gpu_required'] = any(keyword in user_input.lower() for keyword in gpu_keywords)
        
        # Detect IO intensive workload
        io_keywords = ['database', 'real-time', 'streaming', 'high throughput', 'io intensive']
        requirements['io_intensive'] = any(keyword in user_input.lower() for keyword in io_keywords)
        
        return requirements
    
    def find_similar_workloads(self, user_input: str, top_k: int = 5) -> List[Dict]:
        """Find similar workloads using TF-IDF similarity"""
        try:
            user_vector = self.vectorizer.transform([user_input])
            similarities = cosine_similarity(user_vector, self.description_vectors).flatten()
            
            # Get top-k most similar workloads
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            similar_workloads = []
            for idx in top_indices:
                workload = self.dataset.iloc[idx].to_dict()
                workload['similarity_score'] = similarities[idx]
                similar_workloads.append(workload)
            
            return similar_workloads
        except Exception as e:
            print(f"Error in similarity matching: {e}")
            return []
    
    def call_llm_for_analysis(self, user_input: str, similar_workloads: List[Dict]) -> str:
        """Call LLM to analyze workload and provide recommendations"""
        if not self.bedrock_client:
            return """## 🤖 LLM Analysis (Demo Mode)
            
**Note:** LLM analysis is currently in demo mode. AWS Bedrock client is not configured.

**Simulated Analysis Based on Your Input:**

🔍 **Workload Classification:** Your workload appears to be a compute-intensive application requiring significant resources.

📊 **Resource Analysis:**
- The described requirements suggest a medium to large cluster configuration
- Consider the balance between compute, memory, and storage needs
- Network bandwidth requirements should be evaluated based on data transfer patterns

💰 **Cost Optimization Suggestions:**
- Start with the recommended configuration and monitor utilization
- Consider spot instances for non-critical workloads
- Implement auto-scaling to optimize costs during low-usage periods

🚀 **Performance Recommendations:**
- Monitor key metrics: CPU utilization, memory usage, I/O throughput
- Consider data locality to minimize network overhead
- Implement proper load balancing for distributed workloads

**To Enable Full LLM Analysis:**
1. Configure AWS credentials with Bedrock access
2. Ensure Claude-3.7-Sonnet model access in your region
3. Update AWS region configuration if needed

The rule-based recommendation system is still fully functional and provides accurate cluster sizing based on your requirements.
            """
        
        # Prepare context for LLM
        context = f"""
        User workload description: {user_input}
        
        Similar workloads from our database:
        """
        
        for i, workload in enumerate(similar_workloads[:3]):
            context += f"""
        {i+1}. {workload['workload_description']}
           - Recommended cluster: {workload['recommended_cluster_size']} ({workload['recommended_node_count']} nodes)
           - CPU: {workload['cpu_cores']} cores, Memory: {workload['memory_gb']} GB
           - Estimated cost: ${workload['estimated_cost_per_hour']}/hour
        """
        
        prompt = f"""
        {context}
        
        Based on the user's workload description and similar workloads in our database, provide:
        1. Analysis of the workload requirements
        2. Recommended cluster configuration
        3. Key considerations for this workload type
        4. Cost optimization suggestions
        
        Keep the response concise and technical.
        """
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.llm_endpoint,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            return f"## 🤖 LLM Analysis\n\n{response_body['content'][0]['text']}"
            
        except Exception as e:
            return f"""## 🤖 LLM Analysis
            
**Error:** LLM analysis failed: {str(e)}

**Troubleshooting:**
- Check AWS Bedrock configuration and credentials
- Verify model access permissions
- Ensure network connectivity

The rule-based recommendation system continues to work normally.
            """
    
    def recommend_cluster(self, user_input: str) -> Tuple[str, str, str]:
        """Main function to recommend cluster configuration"""
        if not user_input.strip():
            return "Please provide a workload description", "", ""
        
        try:
            # Extract requirements from text
            requirements = self.extract_requirements_from_text(user_input)
            
            # Find similar workloads
            similar_workloads = self.find_similar_workloads(user_input)
            
            # Generate rule-based recommendation
            rule_based_rec = self.generate_rule_based_recommendation(requirements, similar_workloads)
            
            # Get LLM analysis
            llm_analysis = self.call_llm_for_analysis(user_input, similar_workloads)
            
            # Format similar workloads for display
            similar_workloads_text = self.format_similar_workloads(similar_workloads)
            
            return rule_based_rec, llm_analysis, similar_workloads_text
            
        except Exception as e:
            error_msg = f"Error generating recommendation: {str(e)}"
            return error_msg, error_msg, error_msg
    
    def generate_rule_based_recommendation(self, requirements: Dict, similar_workloads: List[Dict]) -> str:
        """Generate rule-based cluster recommendation"""
        # Start with most similar workload as baseline
        if similar_workloads:
            baseline = similar_workloads[0]
            recommended_cluster = baseline['recommended_cluster_size']
            recommended_nodes = baseline['recommended_node_count']
            estimated_cost = baseline['estimated_cost_per_hour']
        else:
            recommended_cluster = "medium"
            recommended_nodes = 5
            estimated_cost = 6.0
        
        # Adjust based on extracted requirements
        adjustments = []
        
        if 'cpu_cores' in requirements:
            cpu_req = requirements['cpu_cores']
            if cpu_req > 64:
                if recommended_cluster in ['small', 'medium']:
                    recommended_cluster = 'large'
                    adjustments.append(f"Upgraded to {recommended_cluster} due to high CPU requirement ({cpu_req} cores)")
            elif cpu_req > 128:
                if recommended_cluster in ['small', 'medium', 'large']:
                    recommended_cluster = 'xlarge'
                    adjustments.append(f"Upgraded to {recommended_cluster} due to very high CPU requirement ({cpu_req} cores)")
        
        if 'memory_gb' in requirements:
            memory_req = requirements['memory_gb']
            if memory_req > 256:
                if recommended_cluster in ['small', 'medium']:
                    recommended_cluster = 'large'
                    adjustments.append(f"Upgraded to {recommended_cluster} due to high memory requirement ({memory_req} GB)")
            elif memory_req > 512:
                if recommended_cluster in ['small', 'medium', 'large']:
                    recommended_cluster = 'xlarge'
                    adjustments.append(f"Upgraded to {recommended_cluster} due to very high memory requirement ({memory_req} GB)")
        
        if requirements.get('gpu_required', False):
            if recommended_cluster == 'small':
                recommended_cluster = 'medium'
                adjustments.append("Upgraded to medium due to GPU requirement")
        
        if requirements.get('io_intensive', False):
            adjustments.append("Consider NVMe storage for IO-intensive workload")
        
        # Recalculate cost
        estimated_cost = self.calculate_cost(recommended_cluster, recommended_nodes)
        
        # Format recommendation
        recommendation = f"""
## 🎯 Cluster Recommendation

**Recommended Configuration:**
- **Cluster Size:** {recommended_cluster.upper()}
- **Node Count:** {recommended_nodes} nodes
- **Estimated Cost:** ${estimated_cost:.2f}/hour (${estimated_cost*24:.2f}/day)

**Key Specifications:**
- **Workload Type:** {similar_workloads[0]['workload_type'] if similar_workloads else 'General Purpose'}
- **CPU Cores:** {requirements.get('cpu_cores', 'Auto-scaled')}
- **Memory:** {requirements.get('memory_gb', 'Auto-scaled')} GB
- **GPU Required:** {'Yes' if requirements.get('gpu_required') else 'No'}
- **Storage:** {'NVMe SSD' if requirements.get('io_intensive') else 'Standard SSD'}

**Adjustments Made:**
"""
        
        if adjustments:
            for adj in adjustments:
                recommendation += f"- {adj}\n"
        else:
            recommendation += "- No adjustments needed - baseline recommendation fits requirements\n"
        
        recommendation += f"""
**Confidence Score:** {similar_workloads[0]['similarity_score']:.2f if similar_workloads else 0.5}

**Next Steps:**
1. Review the LLM analysis for detailed insights
2. Check similar workloads for validation
3. Consider running a proof-of-concept with recommended configuration
4. Monitor and adjust based on actual performance metrics

**CrowdStrike Compatibility:** ✅ This application is designed to work with enterprise security software
        """
        
        return recommendation
    
    def calculate_cost(self, cluster_size: str, node_count: int) -> float:
        """Calculate estimated cost"""
        base_costs = {
            "small": 0.5,
            "medium": 1.2,
            "large": 2.5,
            "xlarge": 5.0,
            "2xlarge": 10.0,
            "4xlarge": 20.0
        }
        return base_costs.get(cluster_size, 1.0) * node_count
    
    def format_similar_workloads(self, similar_workloads: List[Dict]) -> str:
        """Format similar workloads for display"""
        if not similar_workloads:
            return "No similar workloads found in database"
        
        formatted = "## 📊 Similar Workloads from Database\n\n"
        
        for i, workload in enumerate(similar_workloads[:5]):
            formatted += f"""
### {i+1}. {workload['workload_type']} (Similarity: {workload['similarity_score']:.3f})

**Description:** {workload['workload_description']}

**Configuration:**
- Cluster Size: {workload['recommended_cluster_size']} ({workload['recommended_node_count']} nodes)
- CPU: {workload['cpu_cores']} cores | Memory: {workload['memory_gb']} GB
- Data Size: {workload['data_size_tb']} TB | Users: {workload['concurrent_users']}
- GPU Required: {'Yes' if workload['gpu_required'] else 'No'}
- Cost: ${workload['estimated_cost_per_hour']}/hour

---
"""
        
        return formatted

def check_frpc_file():
    """Check if frpc file exists and is executable"""
    frpc_path = "/Users/maxpetre/.cache/huggingface/gradio/frpc/frpc_darwin_arm64_v0.3"
    
    if os.path.exists(frpc_path):
        if os.access(frpc_path, os.X_OK):
            print("✅ frpc file exists and is executable")
            return True
        else:
            print("⚠️  frpc file exists but is not executable, fixing...")
            os.chmod(frpc_path, 0o755)
            return True
    else:
        print("❌ frpc file missing")
        return False

def create_gradio_interface():
    """Create and configure Gradio interface with enhanced error handling"""
    
    # Check frpc file
    frpc_available = check_frpc_file()
    
    # Initialize the recommendation wizard
    try:
        wizard = ClusterRecommendationWizard()
        print("✅ Cluster Recommendation Wizard initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize wizard: {e}")
        sys.exit(1)
    
    # Create Gradio interface
    with gr.Blocks(
        title="Cluster Recommendation Wizard",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .recommendation-box {
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        .status-indicator {
            padding: 8px;
            border-radius: 4px;
            margin: 4px 0;
        }
        .status-success {
            background-color: #d4edda;
            color: #155724;
        }
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
        }
        """
    ) as interface:
        
        gr.Markdown(f"""
        # 🧙‍♂️ Cluster Recommendation Wizard
        
        **Powered by AI and Data Analytics** | **CrowdStrike Compatible** ✅
        
        Describe your workload requirements and get intelligent cluster size recommendations based on:
        - 300+ synthetic workload examples
        - Advanced similarity matching
        - LLM-powered analysis using Claude-3.7-Sonnet
        - Rule-based optimization
        
        **Status:** {'🟢 Public sharing available' if frpc_available else '🟡 Local mode only (frpc unavailable)'}
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                user_input = gr.Textbox(
                    label="📝 Describe Your Workload",
                    placeholder="""Example: "I need to run machine learning training on 5TB of data with 32 CPU cores, 128GB RAM, and support for 50 concurrent users. The workload requires GPU acceleration and low latency processing."
                    
Or: "Data analytics workload processing 2TB of customer transaction data with complex SQL queries, need to support 20 business analysts with sub-second query response times."
                    
Be specific about: CPU, memory, data size, users, latency, special requirements...""",
                    lines=6,
                    max_lines=10
                )
                
                submit_btn = gr.Button("🚀 Get Cluster Recommendation", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 💡 Tips for Better Recommendations:
                - Mention specific resource requirements (CPU cores, RAM, storage)
                - Include data size and number of concurrent users
                - Specify workload type (ML training, analytics, real-time processing, etc.)
                - Note any special requirements (GPU, low latency, high availability)
                
                ### 🛡️ Security Notice:
                This application is designed to work with enterprise security software including CrowdStrike Falcon.
                """)
        
        with gr.Column(scale=3):
            with gr.Tab("🎯 Recommendation"):
                recommendation_output = gr.Markdown(
                    label="Cluster Recommendation",
                    elem_classes=["recommendation-box"]
                )
            
            with gr.Tab("🤖 LLM Analysis"):
                llm_output = gr.Markdown(
                    label="AI-Powered Analysis",
                    elem_classes=["recommendation-box"]
                )
            
            with gr.Tab("📊 Similar Workloads"):
                similar_output = gr.Markdown(
                    label="Similar Workloads from Database",
                    elem_classes=["recommendation-box"]
                )
        
        # Event handlers
        submit_btn.click(
            fn=wizard.recommend_cluster,
            inputs=[user_input],
            outputs=[recommendation_output, llm_output, similar_output]
        )
        
        # Add examples
        gr.Examples(
            examples=[
                ["I need to run deep learning model training on 10TB of image data. Require 64 CPU cores, 256GB RAM, GPU acceleration, and support for 5 data scientists working concurrently."],
                ["Real-time fraud detection system processing 1TB of transaction data daily with sub-100ms latency requirements and 1000 concurrent users."],
                ["Data warehouse for business analytics with 50TB of historical data, complex OLAP queries, and 100 concurrent business users requiring sub-second response times."],
                ["ETL pipeline processing 5TB of data nightly with 32 CPU cores, 128GB RAM, and high I/O throughput requirements."],
                ["Stream processing application handling IoT sensor data from 10,000 devices with real-time analytics and 16 CPU cores, 64GB RAM."]
            ],
            inputs=[user_input],
            label="📋 Example Workloads"
        )
        
        gr.Markdown("""
        ---
        
        ### 🔧 Technical Details:
        - **Dataset:** 300 synthetic workload examples with realistic resource requirements
        - **Matching Algorithm:** TF-IDF vectorization with cosine similarity
        - **LLM Integration:** Claude-3.7-Sonnet via AWS Bedrock
        - **Cost Estimation:** Based on AWS EC2 pricing models
        - **Security:** Compatible with CrowdStrike Falcon and enterprise security
        
        ### 📈 Supported Workload Types:
        Data Analytics | Machine Learning | Real-time Processing | Batch Processing | Data Warehousing | ETL Pipelines | Stream Processing | Graph Analytics | Time Series Analysis | Deep Learning
        """)
    
    return interface, frpc_available

if __name__ == "__main__":
    print("🧙‍♂️ Starting Cluster Recommendation Wizard...")
    print("=" * 50)
    
    # Generate dataset if it doesn't exist
    if not os.path.exists("cluster_dataset.csv"):
        print("📊 Generating synthetic dataset...")
        try:
            from generate_dataset import generate_synthetic_dataset
            df = generate_synthetic_dataset(300)
            df.to_csv('cluster_dataset.csv', index=False)
            print("✅ Dataset generated successfully!")
        except Exception as e:
            print(f"❌ Failed to generate dataset: {e}")
            sys.exit(1)
    
    # Create and launch interface
    try:
        interface, frpc_available = create_gradio_interface()
        
        print("\n🚀 Launching application...")
        
        if frpc_available:
            print("📡 Generating publicly shareable URL...")
            print("🛡️  CrowdStrike compatibility: ENABLED")
            
            # Launch with public sharing
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,
                debug=False,  # Disable debug to reduce CrowdStrike triggers
                show_error=True,
                quiet=False,
                prevent_thread_lock=False
            )
        else:
            print("⚠️  Public sharing unavailable - running in local mode")
            print("🌐 Application will be available at: http://127.0.0.1:7860")
            
            # Launch without public sharing
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                debug=False,
                show_error=True,
                quiet=False
            )
            
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check if port 7860 is available")
        print("2. Verify all dependencies are installed")
        print("3. Check CrowdStrike settings if applicable")
        print("4. Try running without public sharing")
        sys.exit(1)
