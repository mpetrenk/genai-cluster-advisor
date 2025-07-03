import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import boto3
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🧙‍♂️ Cluster Recommendation Wizard",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .similar-workload {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_dataset(num_samples: int = 300) -> pd.DataFrame:
    """Generate synthetic dataset - cached for performance"""
    np.random.seed(42)
    
    workload_types = [
        "Data Analytics", "Machine Learning Training", "Real-time Processing", 
        "Batch Processing", "Data Warehousing", "ETL Pipeline", 
        "Stream Processing", "Graph Analytics", "Time Series Analysis",
        "Deep Learning", "Feature Engineering", "Data Mining"
    ]
    
    data = []
    for i in range(num_samples):
        wl_type = np.random.choice(workload_types)
        
        # Generate resource requirements based on workload type
        if wl_type in ["Machine Learning Training", "Deep Learning"]:
            cpu, mem, gpu = np.random.randint(16, 128), np.random.randint(64, 512), True
        elif wl_type in ["Real-time Processing", "Stream Processing"]:
            cpu, mem, gpu = np.random.randint(8, 64), np.random.randint(32, 256), False
        elif wl_type in ["Data Warehousing", "Data Analytics"]:
            cpu, mem, gpu = np.random.randint(32, 256), np.random.randint(128, 1024), False
        else:
            cpu, mem, gpu = np.random.randint(4, 32), np.random.randint(16, 128), False
        
        data_size = np.random.uniform(0.1, 100)
        users = np.random.randint(1, 1000)
        io_intensive = np.random.choice([True, False])
        
        # Calculate cluster recommendation
        score = cpu * 0.3 + mem * 0.002 + (50 if gpu else 0)
        if score < 20:
            cluster, nodes = "small", np.random.randint(2, 5)
        elif score < 50:
            cluster, nodes = "medium", np.random.randint(3, 8)
        elif score < 100:
            cluster, nodes = "large", np.random.randint(5, 15)
        elif score < 200:
            cluster, nodes = "xlarge", np.random.randint(8, 25)
        else:
            cluster, nodes = "2xlarge", np.random.randint(15, 40)
        
        # Generate realistic description
        descriptions = {
            "Machine Learning Training": f"ML training on {data_size:.1f}TB data, {cpu} cores, {mem}GB RAM, {users} data scientists",
            "Data Analytics": f"Analytics workload on {data_size:.1f}TB with {users} analysts, requiring {cpu} cores",
            "Real-time Processing": f"Real-time processing {data_size:.1f}TB daily, {users} concurrent sources, low latency",
            "Deep Learning": f"Deep learning training {data_size:.1f}TB dataset, {cpu} cores, {mem}GB RAM, GPU acceleration",
            "Data Warehousing": f"Data warehouse {data_size:.1f}TB storage, {users} concurrent users, OLAP queries"
        }
        
        desc = descriptions.get(wl_type, f"{wl_type} processing {data_size:.1f}TB with {users} users")
        cost = {"small": 0.5, "medium": 1.2, "large": 2.5, "xlarge": 5.0, "2xlarge": 10.0}[cluster] * nodes
        
        data.append({
            'workload_id': f'workload_{i+1:03d}',
            'workload_type': wl_type,
            'workload_description': desc,
            'cpu_cores': cpu,
            'memory_gb': mem,
            'gpu_required': gpu,
            'data_size_tb': round(data_size, 2),
            'io_intensive': io_intensive,
            'concurrent_users': users,
            'recommended_cluster_size': cluster,
            'recommended_node_count': nodes,
            'estimated_cost_per_hour': round(cost, 2)
        })
    
    return pd.DataFrame(data)

class StreamlitClusterWizard:
    def __init__(self):
        """Initialize the Streamlit cluster wizard"""
        if 'wizard_initialized' not in st.session_state:
            with st.spinner("🧙‍♂️ Initializing Cluster Wizard..."):
                self.dataset = generate_synthetic_dataset(300)
                self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                descriptions = self.dataset['workload_description'].tolist()
                self.description_vectors = self.vectorizer.fit_transform(descriptions)
                
                # Setup Bedrock client
                self.setup_bedrock_client()
                
                st.session_state.wizard_initialized = True
                st.session_state.dataset = self.dataset
                st.session_state.vectorizer = self.vectorizer
                st.session_state.description_vectors = self.description_vectors
                st.session_state.bedrock_client = getattr(self, 'bedrock_client', None)
        else:
            self.dataset = st.session_state.dataset
            self.vectorizer = st.session_state.vectorizer
            self.description_vectors = st.session_state.description_vectors
            self.bedrock_client = st.session_state.bedrock_client
    
    def setup_bedrock_client(self):
        """Setup AWS Bedrock client"""
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        except Exception:
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
        user_match = re.search(r'(\d+)\s*(?:user|users|concurrent|simultaneous)', user_input.lower())
        if user_match:
            requirements['concurrent_users'] = int(user_match.group(1))
        
        # GPU and IO detection
        gpu_keywords = ['gpu', 'graphics', 'cuda', 'machine learning', 'deep learning', 'ai training']
        requirements['gpu_required'] = any(kw in user_input.lower() for kw in gpu_keywords)
        
        io_keywords = ['database', 'real-time', 'streaming', 'high throughput', 'io intensive']
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
            st.error(f"Error in similarity matching: {e}")
            return []
    
    def call_llm_analysis(self, user_input: str, similar_workloads: List[Dict]) -> str:
        """Call LLM for analysis"""
        if not self.bedrock_client:
            return """**🤖 LLM Analysis (Demo Mode)**

**Note:** LLM analysis is in demo mode. Configure AWS Bedrock credentials for full functionality.

**Simulated Analysis:**
- **Workload Classification:** Based on your description, this appears to be a compute-intensive workload
- **Resource Optimization:** The recommended configuration balances compute, memory, and cost efficiency
- **Performance Predictions:** Expected good performance with the suggested cluster size
- **Cost Analysis:** The pricing estimate includes both compute and storage costs
- **Best Practices:** Consider starting with the recommended size and scaling based on actual usage

**To Enable Full LLM Analysis:**
- Configure AWS credentials with Bedrock access
- Ensure Claude-3.7-Sonnet model availability in your region
            """
        
        # Prepare LLM prompt
        context = f"User workload: {user_input}\n\nSimilar workloads:\n"
        for i, wl in enumerate(similar_workloads[:3]):
            context += f"{i+1}. {wl['workload_description']} -> {wl['recommended_cluster_size']} ({wl['recommended_node_count']} nodes)\n"
        
        prompt = f"{context}\n\nProvide: 1) Workload analysis 2) Cluster recommendation 3) Key considerations 4) Cost optimization"
        
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
            return f"**🤖 LLM Analysis**\n\n{result['content'][0]['text']}"
            
        except Exception as e:
            return f"**🤖 LLM Analysis**\n\nAnalysis failed: {str(e)}\n\nCheck AWS Bedrock configuration and credentials."
    
    def generate_recommendation(self, requirements: Dict, similar_workloads: List[Dict]) -> Dict:
        """Generate cluster recommendation with identical logic to original"""
        # Baseline from most similar workload
        if similar_workloads:
            baseline = similar_workloads[0]
            cluster_size = baseline['recommended_cluster_size']
            node_count = baseline['recommended_node_count']
        else:
            cluster_size, node_count = "medium", 5
        
        # Apply rule-based adjustments (identical logic)
        adjustments = []
        
        if 'cpu_cores' in requirements:
            cpu_req = requirements['cpu_cores']
            if cpu_req > 64 and cluster_size in ['small', 'medium']:
                cluster_size = 'large'
                adjustments.append(f"Upgraded to {cluster_size} for {cpu_req} CPU cores")
            elif cpu_req > 128 and cluster_size in ['small', 'medium', 'large']:
                cluster_size = 'xlarge'
                adjustments.append(f"Upgraded to {cluster_size} for {cpu_req} CPU cores")
        
        if 'memory_gb' in requirements:
            mem_req = requirements['memory_gb']
            if mem_req > 256 and cluster_size in ['small', 'medium']:
                cluster_size = 'large'
                adjustments.append(f"Upgraded to {cluster_size} for {mem_req}GB memory")
            elif mem_req > 512 and cluster_size in ['small', 'medium', 'large']:
                cluster_size = 'xlarge'
                adjustments.append(f"Upgraded to {cluster_size} for {mem_req}GB memory")
        
        if requirements.get('gpu_required') and cluster_size == 'small':
            cluster_size = 'medium'
            adjustments.append("Upgraded to medium for GPU requirements")
        
        if requirements.get('io_intensive'):
            adjustments.append("Consider NVMe storage for IO-intensive workload")
        
        # Calculate cost (identical logic)
        costs = {"small": 0.5, "medium": 1.2, "large": 2.5, "xlarge": 5.0, "2xlarge": 10.0}
        cost = costs.get(cluster_size, 1.0) * node_count
        
        return {
            'cluster_size': cluster_size,
            'node_count': node_count,
            'cost_per_hour': cost,
            'cost_per_day': cost * 24,
            'adjustments': adjustments,
            'workload_type': similar_workloads[0]['workload_type'] if similar_workloads else 'General Purpose',
            'confidence': similar_workloads[0]['similarity_score'] if similar_workloads else 0.5,
            'requirements': requirements
        }

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧙‍♂️ Cluster Recommendation Wizard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3>🚀 Powered by AI and Data Analytics | 🌐 Public Streamlit Sharing</h3>
        <p>Get intelligent cluster size recommendations based on 300+ synthetic workload examples</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize wizard
    wizard = StreamlitClusterWizard()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📊 Dataset Information")
        st.info(f"**Workload Examples:** {len(wizard.dataset)}")
        st.info(f"**Workload Types:** {wizard.dataset['workload_type'].nunique()}")
        st.info(f"**Cluster Sizes:** {wizard.dataset['recommended_cluster_size'].nunique()}")
        
        st.markdown("### 🔧 Features")
        st.success("✅ TF-IDF Similarity Matching")
        st.success("✅ Rule-based Optimization")
        st.success("✅ Cost Estimation")
        st.success("✅ LLM Analysis (Claude-3.7)")
        st.success("✅ Public Sharing")
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Mention specific CPU cores and RAM
        - Include data size and user count
        - Specify GPU requirements
        - Note latency needs
        """)
    
    # Main input area
    st.markdown('<h2 class="sub-header">📝 Describe Your Workload</h2>', unsafe_allow_html=True)
    
    # Example buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🤖 ML Training Example"):
            st.session_state.example_input = "Deep learning model training on 10TB image data. Need 64 CPU cores, 256GB RAM, GPU acceleration for 5 data scientists."
    with col2:
        if st.button("⚡ Real-time Example"):
            st.session_state.example_input = "Real-time fraud detection processing 1TB daily with sub-100ms latency for 1000 concurrent users."
    with col3:
        if st.button("📊 Analytics Example"):
            st.session_state.example_input = "Data warehouse with 50TB historical data, complex OLAP queries, 100 concurrent business users."
    
    # Text input
    user_input = st.text_area(
        "Workload Description",
        value=st.session_state.get('example_input', ''),
        height=120,
        placeholder="Example: I need to run machine learning training on 5TB of data with 32 CPU cores, 128GB RAM, and support for 50 concurrent users. The workload requires GPU acceleration and low latency processing.",
        help="Be specific about CPU, memory, data size, users, and special requirements"
    )
    
    # Generate recommendation button
    if st.button("🚀 Get Cluster Recommendation", type="primary"):
        if user_input.strip():
            with st.spinner("🔍 Analyzing workload and generating recommendations..."):
                # Extract requirements
                requirements = wizard.extract_requirements(user_input)
                
                # Find similar workloads
                similar_workloads = wizard.find_similar_workloads(user_input)
                
                # Generate recommendation
                recommendation = wizard.generate_recommendation(requirements, similar_workloads)
                
                # Get LLM analysis
                llm_analysis = wizard.call_llm_analysis(user_input, similar_workloads)
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["🎯 Recommendation", "🤖 LLM Analysis", "📊 Similar Workloads"])
                
                with tab1:
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                    
                    # Main recommendation
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cluster Size", recommendation['cluster_size'].upper())
                    with col2:
                        st.metric("Node Count", recommendation['node_count'])
                    with col3:
                        st.metric("Cost/Hour", f"${recommendation['cost_per_hour']:.2f}")
                    with col4:
                        st.metric("Cost/Day", f"${recommendation['cost_per_day']:.2f}")
                    
                    # Specifications
                    st.markdown("#### 🔧 Key Specifications")
                    spec_col1, spec_col2 = st.columns(2)
                    with spec_col1:
                        st.write(f"**Workload Type:** {recommendation['workload_type']}")
                        st.write(f"**CPU Cores:** {requirements.get('cpu_cores', 'Auto-scaled')}")
                        st.write(f"**Memory:** {requirements.get('memory_gb', 'Auto-scaled')} GB")
                    with spec_col2:
                        st.write(f"**GPU Required:** {'Yes' if requirements.get('gpu_required') else 'No'}")
                        st.write(f"**Storage:** {'NVMe SSD' if requirements.get('io_intensive') else 'Standard SSD'}")
                        st.write(f"**Confidence:** {recommendation['confidence']:.2f}")
                    
                    # Adjustments
                    if recommendation['adjustments']:
                        st.markdown("#### ⚙️ Adjustments Made")
                        for adj in recommendation['adjustments']:
                            st.success(f"✅ {adj}")
                    else:
                        st.info("ℹ️ No adjustments needed - baseline recommendation fits requirements")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown(llm_analysis)
                
                with tab3:
                    st.markdown("### 📊 Similar Workloads from Database")
                    
                    for i, wl in enumerate(similar_workloads[:5]):
                        with st.expander(f"{i+1}. {wl['workload_type']} (Similarity: {wl['similarity_score']:.3f})"):
                            st.write(f"**Description:** {wl['workload_description']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Cluster:** {wl['recommended_cluster_size']} ({wl['recommended_node_count']} nodes)")
                                st.write(f"**CPU:** {wl['cpu_cores']} cores")
                                st.write(f"**Memory:** {wl['memory_gb']} GB")
                            with col2:
                                st.write(f"**Data Size:** {wl['data_size_tb']} TB")
                                st.write(f"**Users:** {wl['concurrent_users']}")
                                st.write(f"**Cost:** ${wl['estimated_cost_per_hour']}/hour")
                            
                            if wl['gpu_required']:
                                st.success("🎮 GPU Required")
                            if wl['io_intensive']:
                                st.info("💾 IO Intensive")
        else:
            st.warning("Please provide a workload description to get recommendations.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <h4>🔧 Technical Details</h4>
        <p><strong>Dataset:</strong> 300 synthetic workload examples | <strong>Algorithm:</strong> TF-IDF + Cosine Similarity | <strong>LLM:</strong> Claude-3.7-Sonnet</p>
        <p><strong>Supported Workloads:</strong> Data Analytics | Machine Learning | Real-time Processing | Batch Processing | Data Warehousing | ETL Pipelines</p>
        <p><strong>🌐 Streamlit Public Sharing:</strong> This app can be shared publicly via Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
