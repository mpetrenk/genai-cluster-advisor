import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List
import json
import warnings
warnings.filterwarnings('ignore')

# LangChain imports (with error handling for missing packages)
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Fallback to original implementation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page configuration
st.set_page_config(
    page_title="🧙‍♂️ RAG-Enhanced Cluster Wizard",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .rag-indicator {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .vector-db-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class RAGClusterWizard:
    def __init__(self, use_langchain: bool = True):
        """Initialize RAG-enhanced cluster wizard"""
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        if 'rag_wizard_initialized' not in st.session_state:
            with st.spinner("🔧 Initializing RAG-Enhanced Cluster Wizard..."):
                self.dataset = self.generate_synthetic_dataset()
                
                if self.use_langchain:
                    self.setup_langchain_rag()
                else:
                    self.setup_traditional_similarity()
                
                st.session_state.rag_wizard_initialized = True
                st.session_state.dataset = self.dataset
                st.session_state.use_langchain = self.use_langchain
                
                if self.use_langchain:
                    st.session_state.vector_store = self.vector_store
                    st.session_state.retriever = self.retriever
                else:
                    st.session_state.vectorizer = self.vectorizer
                    st.session_state.description_vectors = self.description_vectors
        else:
            self.dataset = st.session_state.dataset
            self.use_langchain = st.session_state.use_langchain
            
            if self.use_langchain:
                self.vector_store = st.session_state.vector_store
                self.retriever = st.session_state.retriever
            else:
                self.vectorizer = st.session_state.vectorizer
                self.description_vectors = st.session_state.description_vectors
    
    def generate_synthetic_dataset(self) -> pd.DataFrame:
        """Generate enhanced synthetic dataset for RAG"""
        np.random.seed(42)
        
        workload_types = [
            "Data Analytics", "Machine Learning Training", "Real-time Processing", 
            "Batch Processing", "Data Warehousing", "ETL Pipeline", 
            "Stream Processing", "Graph Analytics", "Time Series Analysis",
            "Deep Learning", "Feature Engineering", "Data Mining"
        ]
        
        data = []
        for i in range(300):
            wl_type = np.random.choice(workload_types)
            
            if wl_type in ["Machine Learning Training", "Deep Learning"]:
                cpu, mem, gpu = np.random.randint(16, 128), np.random.randint(64, 512), True
            elif wl_type in ["Real-time Processing", "Stream Processing"]:
                cpu, mem, gpu = np.random.randint(8, 64), np.random.randint(32, 256), False
            else:
                cpu, mem, gpu = np.random.randint(4, 32), np.random.randint(16, 128), False
            
            data_size = np.random.uniform(0.1, 100)
            users = np.random.randint(1, 1000)
            
            score = cpu * 0.3 + mem * 0.002 + (50 if gpu else 0)
            if score < 20:
                cluster, nodes = "small", np.random.randint(2, 5)
            elif score < 50:
                cluster, nodes = "medium", np.random.randint(3, 8)
            elif score < 100:
                cluster, nodes = "large", np.random.randint(5, 15)
            else:
                cluster, nodes = "xlarge", np.random.randint(8, 25)
            
            # Enhanced description for better RAG performance
            desc = f"""
            Workload: {wl_type} requiring {cpu} CPU cores and {mem}GB RAM
            Data Processing: {data_size:.1f}TB of data with {users} concurrent users
            Performance: {'GPU-accelerated computing' if gpu else 'Standard CPU processing'}
            Scale: {'High-performance' if cpu > 64 else 'Standard'} computing requirements
            Use Case: {wl_type.lower()} workload optimized for {'machine learning' if 'ML' in wl_type or 'Learning' in wl_type else 'data processing'}
            """
            
            cost = {"small": 0.5, "medium": 1.2, "large": 2.5, "xlarge": 5.0}[cluster] * nodes
            
            data.append({
                'workload_id': f'workload_{i+1:03d}',
                'workload_type': wl_type,
                'workload_description': desc.strip(),
                'cpu_cores': cpu,
                'memory_gb': mem,
                'gpu_required': gpu,
                'data_size_tb': round(data_size, 2),
                'concurrent_users': users,
                'recommended_cluster_size': cluster,
                'recommended_node_count': nodes,
                'estimated_cost_per_hour': round(cost, 2)
            })
        
        return pd.DataFrame(data)
    
    def setup_langchain_rag(self):
        """Setup LangChain RAG system"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Convert dataset to documents
            documents = []
            for _, row in self.dataset.iterrows():
                doc = Document(
                    page_content=row['workload_description'],
                    metadata={
                        'workload_type': row['workload_type'],
                        'cpu_cores': row['cpu_cores'],
                        'memory_gb': row['memory_gb'],
                        'gpu_required': row['gpu_required'],
                        'recommended_cluster_size': row['recommended_cluster_size'],
                        'recommended_node_count': row['recommended_node_count'],
                        'estimated_cost_per_hour': row['estimated_cost_per_hour']
                    }
                )
                documents.append(doc)
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            st.success("✅ LangChain RAG system initialized with FAISS vector database")
            
        except Exception as e:
            st.error(f"❌ LangChain setup failed: {e}")
            self.use_langchain = False
            self.setup_traditional_similarity()
    
    def setup_traditional_similarity(self):
        """Fallback to traditional TF-IDF similarity"""
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        descriptions = self.dataset['workload_description'].tolist()
        self.description_vectors = self.vectorizer.fit_transform(descriptions)
        st.info("ℹ️ Using traditional TF-IDF similarity matching")
    
    def get_rag_recommendations(self, user_query: str) -> Dict:
        """Get recommendations using RAG approach"""
        if self.use_langchain:
            return self.langchain_rag_search(user_query)
        else:
            return self.traditional_similarity_search(user_query)
    
    def langchain_rag_search(self, user_query: str) -> Dict:
        """LangChain RAG-based search"""
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search_with_score(user_query, k=5)
            
            similar_workloads = []
            for doc, score in docs:
                similar_workloads.append({
                    'workload_description': doc.page_content,  # Fixed: use correct key
                    'workload_type': doc.metadata.get('workload_type'),
                    'cpu_cores': doc.metadata.get('cpu_cores'),
                    'memory_gb': doc.metadata.get('memory_gb'),
                    'gpu_required': doc.metadata.get('gpu_required'),
                    'recommended_cluster_size': doc.metadata.get('recommended_cluster_size'),
                    'recommended_node_count': doc.metadata.get('recommended_node_count'),
                    'estimated_cost_per_hour': doc.metadata.get('estimated_cost_per_hour'),
                    'similarity_score': 1 - (score / 2),  # Convert distance to similarity
                    'retrieval_method': 'LangChain Vector Similarity'
                })
            
            return {
                'similar_workloads': similar_workloads,
                'retrieval_method': 'LangChain FAISS Vector Database',
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
            }
            
        except Exception as e:
            return {
                'error': f"RAG search failed: {str(e)}",
                'similar_workloads': [],
                'retrieval_method': 'Error in LangChain RAG'
            }
    
    def traditional_similarity_search(self, user_query: str) -> Dict:
        """Traditional TF-IDF similarity search"""
        try:
            user_vector = self.vectorizer.transform([user_query])
            similarities = cosine_similarity(user_vector, self.description_vectors).flatten()
            top_indices = similarities.argsort()[-5:][::-1]
            
            similar_workloads = []
            for idx in top_indices:
                workload = self.dataset.iloc[idx].to_dict()
                workload['similarity_score'] = similarities[idx]
                workload['retrieval_method'] = 'TF-IDF Cosine Similarity'
                similar_workloads.append(workload)
            
            return {
                'similar_workloads': similar_workloads,
                'retrieval_method': 'Traditional TF-IDF + Cosine Similarity',
                'embedding_model': 'TF-IDF Sparse Vectors'
            }
            
        except Exception as e:
            return {
                'error': f"Traditional search failed: {str(e)}",
                'similar_workloads': [],
                'retrieval_method': 'Error in Traditional Search'
            }

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧙‍♂️ RAG-Enhanced Cluster Wizard</h1>', unsafe_allow_html=True)
    
    # RAG Status Indicator
    if LANGCHAIN_AVAILABLE:
        st.markdown("""
        <div class="rag-indicator">
            🚀 RAG-POWERED: Using LangChain + Vector Database for Enhanced Recommendations
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="rag-indicator" style="background: linear-gradient(90deg, #ff7f0e, #d62728);">
            ⚠️ FALLBACK MODE: Install LangChain for full RAG capabilities
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize wizard
    wizard = RAGClusterWizard()
    
    # Sidebar with RAG information
    with st.sidebar:
        st.markdown("### 🤖 RAG System Status")
        
        if wizard.use_langchain:
            st.success("✅ LangChain RAG Active")
            st.info("🔍 Vector Database: FAISS")
            st.info("🧠 Embeddings: HuggingFace")
            st.info("📊 Documents: 300 workloads")
        else:
            st.warning("⚠️ Traditional Mode")
            st.info("🔍 Similarity: TF-IDF")
            st.info("📊 Vectors: Sparse")
        
        st.markdown("### 📊 Dataset Information")
        st.info(f"**Workload Examples:** {len(wizard.dataset)}")
        st.info(f"**Workload Types:** {wizard.dataset['workload_type'].nunique()}")
        
        st.markdown("### 🔧 RAG vs Traditional")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RAG Benefits:**")
            st.markdown("- Dense embeddings")
            st.markdown("- Semantic search")
            st.markdown("- Better context")
        
        with col2:
            st.markdown("**Traditional:**")
            st.markdown("- Sparse vectors")
            st.markdown("- Keyword matching")
            st.markdown("- Faster setup")
    
    # Main input area
    st.markdown("### 📝 Describe Your Workload")
    
    # Example buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🤖 ML Training"):
            st.session_state.example_input = "Deep learning model training on 15TB image dataset requiring 128 CPU cores, 512GB RAM, and GPU acceleration for distributed training across multiple data scientists."
    with col2:
        if st.button("⚡ Real-time Processing"):
            st.session_state.example_input = "Real-time fraud detection system processing 2TB of transaction data daily with sub-50ms latency requirements and 2000 concurrent API calls."
    with col3:
        if st.button("📊 Data Analytics"):
            st.session_state.example_input = "Large-scale data analytics workload processing 50TB of customer data with complex OLAP queries for 100 concurrent business users."
    
    # Text input
    user_input = st.text_area(
        "Workload Description",
        value=st.session_state.get('example_input', ''),
        height=120,
        placeholder="Describe your workload requirements including CPU, memory, data size, users, and special requirements...",
        help="Be specific about technical requirements for better RAG-based recommendations"
    )
    
    # Generate recommendation button
    if st.button("🚀 Get RAG-Enhanced Recommendations", type="primary"):
        if user_input.strip():
            with st.spinner("🔍 Performing RAG-based analysis..."):
                # Get RAG recommendations
                rag_results = wizard.get_rag_recommendations(user_input)
                
                # Display results
                if 'error' not in rag_results:
                    similar_workloads = rag_results['similar_workloads']
                    
                    # RAG System Information
                    st.markdown(f"""
                    <div class="vector-db-info">
                        <strong>🤖 RAG System Used:</strong> {rag_results['retrieval_method']}<br>
                        <strong>🧠 Embedding Model:</strong> {rag_results['embedding_model']}<br>
                        <strong>📊 Retrieved:</strong> {len(similar_workloads)} similar workloads
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tabs for results
                    tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "🔍 RAG Analysis", "📊 Vector Similarity"])
                    
                    with tab1:
                        if similar_workloads:
                            baseline = similar_workloads[0]
                            
                            # Main metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Cluster Size", baseline['recommended_cluster_size'].upper())
                            with col2:
                                st.metric("Node Count", baseline['recommended_node_count'])
                            with col3:
                                st.metric("Cost/Hour", f"${baseline['estimated_cost_per_hour']:.2f}")
                            with col4:
                                st.metric("Confidence", f"{baseline['similarity_score']:.2f}")
                            
                            # Specifications
                            st.markdown("#### 🔧 Recommended Configuration")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Workload Type:** {baseline['workload_type']}")
                                st.write(f"**CPU Cores:** {baseline['cpu_cores']}")
                                st.write(f"**Memory:** {baseline['memory_gb']} GB")
                            with col2:
                                st.write(f"**GPU Required:** {'Yes' if baseline['gpu_required'] else 'No'}")
                                st.write(f"**Data Size:** {baseline.get('data_size_tb', 'N/A')} TB")
                                st.write(f"**Users:** {baseline.get('concurrent_users', 'N/A')}")
                    
                    with tab2:
                        st.markdown("### 🤖 RAG-Enhanced Analysis")
                        
                        if wizard.use_langchain:
                            st.success("✅ **Vector Database Analysis**")
                            st.markdown("""
                            **RAG Process:**
                            1. **Query Embedding:** Your description was converted to dense vectors
                            2. **Semantic Search:** Vector database found semantically similar workloads
                            3. **Context Retrieval:** Retrieved top-5 most relevant examples
                            4. **Similarity Scoring:** Calculated semantic similarity scores
                            
                            **Advantages over Traditional Search:**
                            - **Semantic Understanding:** Captures meaning beyond keywords
                            - **Dense Representations:** Better context understanding
                            - **Scalable:** Efficient for large knowledge bases
                            """)
                        else:
                            st.info("ℹ️ **Traditional TF-IDF Analysis**")
                            st.markdown("""
                            **Process:**
                            1. **Text Vectorization:** Description converted to TF-IDF vectors
                            2. **Cosine Similarity:** Mathematical similarity calculation
                            3. **Ranking:** Top matches based on term frequency
                            
                            **To Enable RAG:** Install LangChain dependencies
                            """)
                    
                    with tab3:
                        st.markdown("### 📊 Similar Workloads (Vector Similarity)")
                        
                        for i, wl in enumerate(similar_workloads[:5]):
                            with st.expander(f"{i+1}. {wl['workload_type']} (Similarity: {wl['similarity_score']:.3f})"):
                                # Fixed: use correct key name
                                description = wl.get('workload_description', 'No description available')
                                st.write(f"**Description:** {description[:200]}...")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Cluster:** {wl['recommended_cluster_size']} ({wl['recommended_node_count']} nodes)")
                                    st.write(f"**CPU:** {wl['cpu_cores']} cores")
                                    st.write(f"**Memory:** {wl['memory_gb']} GB")
                                with col2:
                                    st.write(f"**Cost:** ${wl['estimated_cost_per_hour']}/hour")
                                    st.write(f"**GPU:** {'Yes' if wl['gpu_required'] else 'No'}")
                                    st.write(f"**Method:** {wl.get('retrieval_method', 'N/A')}")
                else:
                    st.error(f"❌ {rag_results['error']}")
        else:
            st.warning("Please provide a workload description.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <h4>🔧 RAG-Enhanced Technical Details</h4>
        <p><strong>Vector Database:</strong> FAISS with HuggingFace embeddings | <strong>Fallback:</strong> TF-IDF + Cosine Similarity</p>
        <p><strong>RAG Benefits:</strong> Semantic search, dense embeddings, better context understanding</p>
        <p><strong>🌐 Deploy:</strong> This RAG-enhanced version can be deployed to Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
