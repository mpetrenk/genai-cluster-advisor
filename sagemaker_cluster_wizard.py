# Cluster Recommendation Wizard - SageMaker Notebook Version
# Identical behavior to desktop version, optimized for SageMaker environment

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
import subprocess
import warnings
warnings.filterwarnings('ignore')

def install_requirements():
    """Install required packages in SageMaker environment"""
    packages = ['gradio>=4.0.0', 'pandas>=1.5.0', 'numpy>=1.21.0', 'scikit-learn>=1.0.0']
    
    print("📦 Installing packages for SageMaker...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✅ {package.split('>=')[0]}")
        except:
            print(f"⚠️  Warning: Could not install {package}")
    print("✅ Installation complete")

def generate_sagemaker_dataset(num_samples: int = 300) -> pd.DataFrame:
    """Generate synthetic dataset for SageMaker"""
    np.random.seed(42)
    
    workload_types = [
        "Data Analytics", "Machine Learning Training", "Real-time Processing", 
        "Batch Processing", "Data Warehousing", "ETL Pipeline", 
        "Stream Processing", "Graph Analytics", "Time Series Analysis",
        "Deep Learning", "Feature Engineering", "Data Mining"
    ]
    
    dataset = []
    
    for i in range(num_samples):
        workload_type = np.random.choice(workload_types)
        
        # Generate resource requirements based on workload type
        if workload_type in ["Machine Learning Training", "Deep Learning"]:
            cpu_cores = np.random.randint(16, 128)
            memory_gb = np.random.randint(64, 512)
            gpu_required = np.random.choice([True, False])
        elif workload_type in ["Real-time Processing", "Stream Processing"]:
            cpu_cores = np.random.randint(8, 64)
            memory_gb = np.random.randint(32, 256)
            gpu_required = False
        elif workload_type in ["Data Warehousing", "Data Analytics"]:
            cpu_cores = np.random.randint(32, 256)
            memory_gb = np.random.randint(128, 1024)
            gpu_required = False
        else:
            cpu_cores = np.random.randint(4, 32)
            memory_gb = np.random.randint(16, 128)
            gpu_required = False
        
        data_size_tb = np.random.uniform(0.1, 100)
        concurrent_users = np.random.randint(1, 1000)
        io_intensive = np.random.choice([True, False])
        
        # Calculate cluster recommendation
        compute_score = cpu_cores * 0.3 + memory_gb * 0.002 + (50 if gpu_required else 0)
        
        if compute_score < 20:
            cluster_size, node_count = "small", np.random.randint(2, 5)
        elif compute_score < 50:
            cluster_size, node_count = "medium", np.random.randint(3, 8)
        elif compute_score < 100:
            cluster_size, node_count = "large", np.random.randint(5, 15)
        elif compute_score < 200:
            cluster_size, node_count = "xlarge", np.random.randint(8, 25)
        else:
            cluster_size, node_count = "2xlarge", np.random.randint(15, 40)
        
        # Generate description
        descriptions = {
            "Machine Learning Training": f"ML training on {data_size_tb:.1f}TB data, {cpu_cores} cores, {memory_gb}GB RAM, {concurrent_users} users",
            "Data Analytics": f"Analytics on {data_size_tb:.1f}TB data with {concurrent_users} analysts, {cpu_cores} cores needed",
            "Real-time Processing": f"Real-time processing {data_size_tb:.1f}TB daily, {concurrent_users} concurrent sources",
        }
        
        description = descriptions.get(workload_type, 
            f"{workload_type} processing {data_size_tb:.1f}TB with {concurrent_users} users")
        
        cost = {"small": 0.5, "medium": 1.2, "large": 2.5, "xlarge": 5.0, "2xlarge": 10.0}[cluster_size] * node_count
        
        dataset.append({
            'workload_id': f'workload_{i+1:03d}',
            'workload_type': workload_type,
            'workload_description': description,
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb,
            'gpu_required': gpu_required,
            'data_size_tb': round(data_size_tb, 2),
            'io_intensive': io_intensive,
            'concurrent_users': concurrent_users,
            'recommended_cluster_size': cluster_size,
            'recommended_node_count': node_count,
            'estimated_cost_per_hour': round(cost, 2)
        })
    
    return pd.DataFrame(dataset)

class SageMakerClusterWizard:
    def __init__(self):
        """Initialize the SageMaker cluster recommendation wizard"""
        print("🧙‍♂️ Initializing SageMaker Cluster Wizard...")
        
        # Generate or load dataset
        self.dataset = generate_sagemaker_dataset(300)
        print(f"✅ Dataset loaded: {len(self.dataset)} samples")
        
        # Setup vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        descriptions = self.dataset['workload_description'].tolist()
        self.description_vectors = self.vectorizer.fit_transform(descriptions)
        print("✅ TF-IDF vectorizer initialized")
        
        # Setup Bedrock client (SageMaker has built-in AWS credentials)
        self.setup_bedrock_client()
        
    def setup_bedrock_client(self):
        """Setup AWS Bedrock client - SageMaker has built-in credentials"""
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
            print("✅ AWS Bedrock client initialized (SageMaker credentials)")
        except Exception as e:
            print(f"⚠️  Bedrock client unavailable: {e}")
            self.bedrock_client = None
    
    def extract_requirements(self, user_input: str) -> Dict:
        """Extract technical requirements from user input"""
        requirements = {}
        
        # CPU extraction
        cpu_match = re.search(r'(\d+)\s*(?:cpu|core|cores|vcpu)', user_input.lower())
        if cpu_match:
            requirements['cpu_cores'] = int(cpu_match.group(1))
        
        # Memory extraction
        memory_match = re.search(r'(\d+)\s*(?:gb|gigabyte|gigabytes)\s*(?:ram|memory)', user_input.lower())
        if memory_match:
            requirements['memory_gb'] = int(memory_match.group(1))
        
        # Data size extraction
        data_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:tb|terabyte|terabytes)', user_input.lower())
        if data_match:
            requirements['data_size_tb'] = float(data_match.group(1))
        
        # User count extraction
        user_match = re.search(r'(\d+)\s*(?:user|users|concurrent)', user_input.lower())
        if user_match:
            requirements['concurrent_users'] = int(user_match.group(1))
        
        # GPU and IO detection
        gpu_keywords = ['gpu', 'graphics', 'cuda', 'machine learning', 'deep learning']
        requirements['gpu_required'] = any(kw in user_input.lower() for kw in gpu_keywords)
        
        io_keywords = ['database', 'real-time', 'streaming', 'high throughput']
        requirements['io_intensive'] = any(kw in user_input.lower() for kw in io_keywords)
        
        return requirements
    
    def find_similar_workloads(self, user_input: str, top_k: int = 5) -> List[Dict]:
        """Find similar workloads using TF-IDF similarity"""
        try:
            user_vector = self.vectorizer.transform([user_input])
            similarities = cosine_similarity(user_vector, self.description_vectors).flatten()
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
    
    def call_llm_analysis(self, user_input: str, similar_workloads: List[Dict]) -> str:
        """Call LLM for analysis - optimized for SageMaker"""
        if not self.bedrock_client:
            return """## 🤖 LLM Analysis (SageMaker Demo Mode)
            
**Note:** Running in demo mode. To enable full LLM analysis, ensure SageMaker execution role has Bedrock permissions.

**Simulated Analysis:**
Based on your workload description and similar patterns in our database:

🔍 **Workload Classification:** Your requirements suggest a compute-intensive application
📊 **Resource Analysis:** The workload appears to need balanced compute and memory resources
💰 **Cost Optimization:** Consider starting with recommended size and scaling based on actual usage
🚀 **SageMaker Integration:** This workload could benefit from SageMaker's managed infrastructure

**To Enable Full LLM Analysis:**
1. Ensure SageMaker execution role has `bedrock:InvokeModel` permissions
2. Verify Claude-3.7-Sonnet model access in your region
3. The analysis will automatically activate once permissions are configured
            """
        
        # Prepare LLM prompt
        context = f"User workload: {user_input}\n\nSimilar workloads:\n"
        for i, wl in enumerate(similar_workloads[:3]):
            context += f"{i+1}. {wl['workload_description']} -> {wl['recommended_cluster_size']} ({wl['recommended_node_count']} nodes)\n"
        
        prompt = f"{context}\n\nProvide: 1) Workload analysis 2) Cluster recommendation 3) SageMaker considerations 4) Cost optimization"
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            return f"## 🤖 LLM Analysis\n\n{result['content'][0]['text']}"
            
        except Exception as e:
            return f"## 🤖 LLM Analysis\n\nAnalysis failed: {str(e)}\n\nCheck SageMaker execution role permissions for Bedrock access."
    
    def recommend_cluster(self, user_input: str) -> Tuple[str, str, str]:
        """Main recommendation function - identical behavior to desktop version"""
        if not user_input.strip():
            return "Please provide a workload description", "", ""
        
        try:
            # Extract requirements
            requirements = self.extract_requirements(user_input)
            
            # Find similar workloads
            similar_workloads = self.find_similar_workloads(user_input)
            
            # Generate recommendation
            recommendation = self.generate_recommendation(requirements, similar_workloads)
            
            # Get LLM analysis
            llm_analysis = self.call_llm_analysis(user_input, similar_workloads)
            
            # Format similar workloads
            similar_text = self.format_similar_workloads(similar_workloads)
            
            return recommendation, llm_analysis, similar_text
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg, error_msg, error_msg
    
    def generate_recommendation(self, requirements: Dict, similar_workloads: List[Dict]) -> str:
        """Generate cluster recommendation with identical logic"""
        # Baseline from most similar workload
        if similar_workloads:
            baseline = similar_workloads[0]
            cluster_size = baseline['recommended_cluster_size']
            node_count = baseline['recommended_node_count']
        else:
            cluster_size, node_count = "medium", 5
        
        # Apply rule-based adjustments
        adjustments = []
        
        if 'cpu_cores' in requirements:
            cpu_req = requirements['cpu_cores']
            if cpu_req > 64 and cluster_size in ['small', 'medium']:
                cluster_size = 'large'
                adjustments.append(f"Upgraded to {cluster_size} for {cpu_req} CPU cores")
            elif cpu_req > 128 and cluster_size != 'xlarge':
                cluster_size = 'xlarge'
                adjustments.append(f"Upgraded to {cluster_size} for {cpu_req} CPU cores")
        
        if 'memory_gb' in requirements:
            mem_req = requirements['memory_gb']
            if mem_req > 256 and cluster_size in ['small', 'medium']:
                cluster_size = 'large'
                adjustments.append(f"Upgraded to {cluster_size} for {mem_req}GB memory")
            elif mem_req > 512 and cluster_size != 'xlarge':
                cluster_size = 'xlarge'
                adjustments.append(f"Upgraded to {cluster_size} for {mem_req}GB memory")
        
        if requirements.get('gpu_required') and cluster_size == 'small':
            cluster_size = 'medium'
            adjustments.append("Upgraded to medium for GPU requirements")
        
        # Calculate cost
        costs = {"small": 0.5, "medium": 1.2, "large": 2.5, "xlarge": 5.0, "2xlarge": 10.0}
        cost = costs.get(cluster_size, 1.0) * node_count
        
        # Format recommendation
        rec = f"""## 🎯 SageMaker Cluster Recommendation

**Recommended Configuration:**
- **Cluster Size:** {cluster_size.upper()}
- **Node Count:** {node_count} nodes
- **Estimated Cost:** ${cost:.2f}/hour (${cost*24:.2f}/day)

**Key Specifications:**
- **Workload Type:** {similar_workloads[0]['workload_type'] if similar_workloads else 'General Purpose'}
- **CPU Cores:** {requirements.get('cpu_cores', 'Auto-scaled')}
- **Memory:** {requirements.get('memory_gb', 'Auto-scaled')} GB
- **GPU Required:** {'Yes' if requirements.get('gpu_required') else 'No'}
- **Storage:** {'High-performance' if requirements.get('io_intensive') else 'Standard'}

**SageMaker Integration:**
- Compatible with SageMaker Processing Jobs
- Supports SageMaker Training Jobs
- Can be used with SageMaker Endpoints

**Adjustments Made:**
"""
        
        if adjustments:
            for adj in adjustments:
                rec += f"- {adj}\n"
        else:
            rec += "- No adjustments needed\n"
        
        rec += f"""
**Confidence:** {similar_workloads[0]['similarity_score']:.2f if similar_workloads else 0.5}

**Next Steps:**
1. Review LLM analysis for detailed insights
2. Check similar workloads for validation
3. Test with SageMaker Processing/Training jobs
4. Monitor and optimize based on actual usage
        """
        
        return rec
    
    def format_similar_workloads(self, similar_workloads: List[Dict]) -> str:
        """Format similar workloads for display"""
        if not similar_workloads:
            return "No similar workloads found"
        
        formatted = "## 📊 Similar Workloads from Database\n\n"
        
        for i, wl in enumerate(similar_workloads[:5]):
            formatted += f"""### {i+1}. {wl['workload_type']} (Similarity: {wl['similarity_score']:.3f})

**Description:** {wl['workload_description']}

**Configuration:**
- Cluster: {wl['recommended_cluster_size']} ({wl['recommended_node_count']} nodes)
- Resources: {wl['cpu_cores']} cores, {wl['memory_gb']} GB RAM
- Data: {wl['data_size_tb']} TB, {wl['concurrent_users']} users
- GPU: {'Yes' if wl['gpu_required'] else 'No'}
- Cost: ${wl['estimated_cost_per_hour']}/hour

---
"""
        
        return formatted

def create_sagemaker_interface():
    """Create Gradio interface optimized for SageMaker"""
    
    # Initialize wizard
    wizard = SageMakerClusterWizard()
    
    # Create interface
    with gr.Blocks(
        title="SageMaker Cluster Recommendation Wizard",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("""
        # 🧙‍♂️ SageMaker Cluster Recommendation Wizard
        
        **Optimized for Amazon SageMaker** | **Identical Desktop Behavior**
        
        Get intelligent cluster recommendations for your SageMaker workloads:
        - 300+ synthetic workload examples
        - Advanced similarity matching
        - LLM analysis via Bedrock
        - SageMaker-specific optimizations
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                user_input = gr.Textbox(
                    label="📝 Describe Your SageMaker Workload",
                    placeholder="""Example: "SageMaker training job for deep learning model on 10TB image data. Need 64 CPU cores, 256GB RAM, GPU acceleration for 5 data scientists."

Or: "SageMaker processing job for real-time inference with 1TB daily data, sub-100ms latency, 1000 concurrent requests."

Include: CPU, memory, data size, users, GPU needs, latency requirements...""",
                    lines=6
                )
                
                submit_btn = gr.Button("🚀 Get SageMaker Recommendation", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 💡 SageMaker-Specific Tips:
                - Mention if this is for Training, Processing, or Inference
                - Include data location (S3, EFS, etc.)
                - Specify if you need spot instances
                - Note any compliance requirements
                """)
        
        with gr.Column(scale=3):
            with gr.Tab("🎯 Recommendation"):
                recommendation_output = gr.Markdown()
            
            with gr.Tab("🤖 LLM Analysis"):
                llm_output = gr.Markdown()
            
            with gr.Tab("📊 Similar Workloads"):
                similar_output = gr.Markdown()
        
        # Event handler
        submit_btn.click(
            fn=wizard.recommend_cluster,
            inputs=[user_input],
            outputs=[recommendation_output, llm_output, similar_output]
        )
        
        # SageMaker-specific examples
        gr.Examples(
            examples=[
                ["SageMaker training job for computer vision model on 15TB image dataset. Need 128 CPU cores, 512GB RAM, 8 GPUs, support for distributed training across 4 data scientists."],
                ["SageMaker processing job for real-time fraud detection. Processing 2TB transaction data daily, need sub-50ms inference latency, 2000 concurrent API calls."],
                ["SageMaker batch transform job for NLP model inference on 50TB text data. Need 64 CPU cores, 256GB RAM, process 1M documents per hour."],
                ["SageMaker hyperparameter tuning job testing 100 model configurations. Need 32 CPU cores per job, 128GB RAM, GPU acceleration, parallel execution."],
                ["SageMaker multi-model endpoint serving 20 models simultaneously. Need 16 CPU cores, 64GB RAM, auto-scaling for variable traffic."]
            ],
            inputs=[user_input],
            label="📋 SageMaker Example Workloads"
        )
        
        gr.Markdown("""
        ---
        
        ### 🔧 SageMaker Integration:
        - **Training Jobs:** Optimized for distributed ML training
        - **Processing Jobs:** Batch data processing and ETL
        - **Endpoints:** Real-time and batch inference
        - **Spot Instances:** Cost optimization recommendations
        - **Auto Scaling:** Dynamic resource allocation
        
        ### 📈 Supported SageMaker Workloads:
        Training | Processing | Inference | Hyperparameter Tuning | Batch Transform | Multi-Model Endpoints
        """)
    
    return interface

# Main execution functions for notebook
def launch_sagemaker_wizard(share=True, debug=False):
    """Launch the SageMaker cluster wizard"""
    print("🚀 Launching SageMaker Cluster Recommendation Wizard...")
    
    # Install requirements
    install_requirements()
    
    # Create and launch interface
    interface = create_sagemaker_interface()
    
    # Launch with SageMaker-optimized settings
    return interface.launch(
        share=share,  # Creates public URL for sharing
        debug=debug,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False
    )

# Convenience function for quick launch
def quick_launch():
    """Quick launch with default settings"""
    return launch_sagemaker_wizard(share=True, debug=False)

if __name__ == "__main__":
    # Direct execution
    quick_launch()
