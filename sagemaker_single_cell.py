# SageMaker Cluster Recommendation Wizard - Single Cell Version
# Copy this entire cell into your SageMaker notebook and run it

# Install dependencies
import subprocess
import sys

def install_packages():
    packages = ['gradio>=4.0.0', 'pandas>=1.5.0', 'numpy>=1.21.0', 'scikit-learn>=1.0.0']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
    print("✅ Dependencies installed")

try:
    import gradio as gr
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    install_packages()
    import gradio as gr
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

import json
import re
import boto3
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic dataset
def create_dataset():
    np.random.seed(42)
    workload_types = ["Data Analytics", "Machine Learning Training", "Real-time Processing", 
                     "Batch Processing", "Data Warehousing", "ETL Pipeline", "Stream Processing"]
    
    data = []
    for i in range(300):
        wl_type = np.random.choice(workload_types)
        
        if wl_type in ["Machine Learning Training"]:
            cpu, mem, gpu = np.random.randint(16, 128), np.random.randint(64, 512), True
        elif wl_type in ["Real-time Processing"]:
            cpu, mem, gpu = np.random.randint(8, 64), np.random.randint(32, 256), False
        else:
            cpu, mem, gpu = np.random.randint(4, 32), np.random.randint(16, 128), False
        
        data_size = np.random.uniform(0.1, 100)
        users = np.random.randint(1, 1000)
        
        score = cpu * 0.3 + mem * 0.002 + (50 if gpu else 0)
        if score < 20: cluster, nodes = "small", np.random.randint(2, 5)
        elif score < 50: cluster, nodes = "medium", np.random.randint(3, 8)
        elif score < 100: cluster, nodes = "large", np.random.randint(5, 15)
        else: cluster, nodes = "xlarge", np.random.randint(8, 25)
        
        desc = f"{wl_type} with {cpu} cores, {mem}GB RAM, {data_size:.1f}TB data, {users} users"
        cost = {"small": 0.5, "medium": 1.2, "large": 2.5, "xlarge": 5.0}[cluster] * nodes
        
        data.append({
            'workload_type': wl_type, 'workload_description': desc, 'cpu_cores': cpu,
            'memory_gb': mem, 'gpu_required': gpu, 'data_size_tb': data_size,
            'concurrent_users': users, 'recommended_cluster_size': cluster,
            'recommended_node_count': nodes, 'estimated_cost_per_hour': cost
        })
    
    return pd.DataFrame(data)

# Main wizard class
class SageMakerWizard:
    def __init__(self):
        self.dataset = create_dataset()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        self.vectors = self.vectorizer.fit_transform(self.dataset['workload_description'])
        
        try:
            self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        except:
            self.bedrock = None
    
    def extract_requirements(self, text):
        reqs = {}
        if m := re.search(r'(\d+)\s*(?:cpu|core|cores)', text.lower()):
            reqs['cpu_cores'] = int(m.group(1))
        if m := re.search(r'(\d+)\s*(?:gb|gigabyte).*(?:ram|memory)', text.lower()):
            reqs['memory_gb'] = int(m.group(1))
        if m := re.search(r'(\d+(?:\.\d+)?)\s*tb', text.lower()):
            reqs['data_size_tb'] = float(m.group(1))
        reqs['gpu_required'] = any(w in text.lower() for w in ['gpu', 'cuda', 'deep learning'])
        return reqs
    
    def find_similar(self, text, k=5):
        user_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(user_vec, self.vectors).flatten()
        indices = sims.argsort()[-k:][::-1]
        return [dict(self.dataset.iloc[i], similarity_score=sims[i]) for i in indices]
    
    def recommend(self, user_input):
        if not user_input.strip():
            return "Please provide a workload description", "", ""
        
        reqs = self.extract_requirements(user_input)
        similar = self.find_similar(user_input)
        
        # Base recommendation
        if similar:
            cluster, nodes = similar[0]['recommended_cluster_size'], similar[0]['recommended_node_count']
        else:
            cluster, nodes = "medium", 5
        
        # Adjustments
        adjustments = []
        if reqs.get('cpu_cores', 0) > 64 and cluster in ['small', 'medium']:
            cluster = 'large'
            adjustments.append(f"Upgraded to {cluster} for high CPU needs")
        if reqs.get('memory_gb', 0) > 256 and cluster in ['small', 'medium']:
            cluster = 'large'
            adjustments.append(f"Upgraded to {cluster} for high memory needs")
        if reqs.get('gpu_required') and cluster == 'small':
            cluster = 'medium'
            adjustments.append("Upgraded for GPU requirements")
        
        cost = {"small": 0.5, "medium": 1.2, "large": 2.5, "xlarge": 5.0}[cluster] * nodes
        
        rec = f"""## 🎯 SageMaker Recommendation

**Configuration:** {cluster.upper()} cluster with {nodes} nodes
**Cost:** ${cost:.2f}/hour (${cost*24:.2f}/day)
**Workload:** {similar[0]['workload_type'] if similar else 'General'}

**Specs:**
- CPU: {reqs.get('cpu_cores', 'Auto-scaled')} cores
- Memory: {reqs.get('memory_gb', 'Auto-scaled')} GB  
- GPU: {'Yes' if reqs.get('gpu_required') else 'No'}

**Adjustments:** {'; '.join(adjustments) if adjustments else 'None needed'}
**Confidence:** {similar[0]['similarity_score']:.2f if similar else 0.5}
"""
        
        llm_analysis = "## 🤖 LLM Analysis\n\nDemo mode - configure Bedrock for full analysis"
        if self.bedrock:
            try:
                prompt = f"Analyze this SageMaker workload and provide recommendations: {user_input}"
                response = self.bedrock.invoke_model(
                    modelId="anthropic.claude-3-7-sonnet-20250219-v1:0",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}]
                    })
                )
                result = json.loads(response['body'].read())
                llm_analysis = f"## 🤖 LLM Analysis\n\n{result['content'][0]['text']}"
            except:
                pass
        
        similar_text = "## 📊 Similar Workloads\n\n"
        for i, wl in enumerate(similar[:3]):
            similar_text += f"**{i+1}. {wl['workload_type']}** (Score: {wl['similarity_score']:.3f})\n"
            similar_text += f"- {wl['workload_description']}\n"
            similar_text += f"- Config: {wl['recommended_cluster_size']} ({wl['recommended_node_count']} nodes)\n"
            similar_text += f"- Cost: ${wl['estimated_cost_per_hour']}/hour\n\n"
        
        return rec, llm_analysis, similar_text

# Create interface
wizard = SageMakerWizard()
print(f"✅ Wizard initialized with {len(wizard.dataset)} workload examples")

with gr.Blocks(title="SageMaker Cluster Wizard", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🧙‍♂️ SageMaker Cluster Recommendation Wizard
    
    **Intelligent cluster sizing for your SageMaker workloads**
    
    Get recommendations based on 300+ workload examples with ML-powered similarity matching.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="📝 Describe Your SageMaker Workload",
                placeholder="Example: SageMaker training job for deep learning model on 10TB data. Need 64 CPU cores, 256GB RAM, GPU acceleration for 5 data scientists.",
                lines=4
            )
            submit_btn = gr.Button("🚀 Get Recommendation", variant="primary")
            
        with gr.Column(scale=3):
            with gr.Tab("🎯 Recommendation"):
                rec_output = gr.Markdown()
            with gr.Tab("🤖 LLM Analysis"):
                llm_output = gr.Markdown()
            with gr.Tab("📊 Similar Workloads"):
                similar_output = gr.Markdown()
    
    submit_btn.click(wizard.recommend, inputs=[user_input], outputs=[rec_output, llm_output, similar_output])
    
    gr.Examples([
        ["SageMaker training job for computer vision model on 15TB images. Need 128 cores, 512GB RAM, 8 GPUs."],
        ["SageMaker processing job for real-time fraud detection. 2TB daily, sub-50ms latency, 2000 concurrent calls."],
        ["SageMaker batch transform for NLP inference on 50TB text. Need 64 cores, 256GB RAM, 1M docs/hour."],
        ["SageMaker hyperparameter tuning with 100 configurations. 32 cores per job, 128GB RAM, GPU acceleration."]
    ], inputs=[user_input])

# Launch with public sharing
print("🚀 Launching SageMaker Cluster Wizard...")
print("📡 Generating public shareable URL...")

app.launch(
    share=True,  # Creates public URL
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True
)
