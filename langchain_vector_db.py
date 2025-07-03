# LangChain Vector Database Implementation for Cluster Recommendation Wizard
# This transforms the current TF-IDF approach into a true RAG system

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json

# LangChain imports for vector database
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate

class LangChainClusterWizard:
    def __init__(self, vector_store_type: str = "faiss"):
        """
        Initialize LangChain-based Cluster Wizard with vector database
        
        Args:
            vector_store_type: "faiss", "chroma", or "pinecone"
        """
        self.vector_store_type = vector_store_type
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Generate synthetic dataset
        self.dataset = self.generate_synthetic_dataset()
        
        # Initialize embeddings and vector store
        self.setup_embeddings()
        self.create_vector_database()
        self.setup_retrieval_chain()
    
    def setup_embeddings(self):
        """Setup embedding model for vector database"""
        try:
            # Try OpenAI embeddings first (best quality)
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002"
            )
            print("✅ Using OpenAI embeddings")
        except:
            # Fallback to HuggingFace embeddings (free, local)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✅ Using HuggingFace embeddings (local)")
    
    def generate_synthetic_dataset(self) -> pd.DataFrame:
        """Generate synthetic workload dataset"""
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
            
            # Calculate cluster recommendation
            score = cpu * 0.3 + mem * 0.002 + (50 if gpu else 0)
            if score < 20:
                cluster, nodes = "small", np.random.randint(2, 5)
            elif score < 50:
                cluster, nodes = "medium", np.random.randint(3, 8)
            elif score < 100:
                cluster, nodes = "large", np.random.randint(5, 15)
            else:
                cluster, nodes = "xlarge", np.random.randint(8, 25)
            
            # Enhanced description for better embeddings
            desc = f"""
            Workload Type: {wl_type}
            Resource Requirements: {cpu} CPU cores, {mem}GB RAM, {data_size:.1f}TB data storage
            Concurrency: {users} concurrent users
            GPU Required: {'Yes' if gpu else 'No'}
            Recommended Configuration: {cluster} cluster with {nodes} nodes
            Use Case: Processing {data_size:.1f}TB of data with {users} concurrent users
            Performance Requirements: {'High-performance computing' if gpu else 'Standard processing'}
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
    
    def create_vector_database(self):
        """Create vector database from workload dataset"""
        print(f"🔧 Creating {self.vector_store_type.upper()} vector database...")
        
        # Convert dataset to LangChain documents
        documents = []
        for _, row in self.dataset.iterrows():
            doc = Document(
                page_content=row['workload_description'],
                metadata={
                    'workload_id': row['workload_id'],
                    'workload_type': row['workload_type'],
                    'cpu_cores': row['cpu_cores'],
                    'memory_gb': row['memory_gb'],
                    'gpu_required': row['gpu_required'],
                    'data_size_tb': row['data_size_tb'],
                    'concurrent_users': row['concurrent_users'],
                    'recommended_cluster_size': row['recommended_cluster_size'],
                    'recommended_node_count': row['recommended_node_count'],
                    'estimated_cost_per_hour': row['estimated_cost_per_hour']
                }
            )
            documents.append(doc)
        
        # Create vector store based on type
        if self.vector_store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Save FAISS index for persistence
            self.vector_store.save_local("faiss_cluster_index")
            
        elif self.vector_store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="chroma_cluster_db"
            )
            self.vector_store.persist()
            
        elif self.vector_store_type == "pinecone":
            # Note: Requires Pinecone API key
            import pinecone
            pinecone.init(api_key="your-pinecone-api-key", environment="your-env")
            
            self.vector_store = Pinecone.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name="cluster-recommendations"
            )
        
        print(f"✅ Vector database created with {len(documents)} workload examples")
    
    def setup_retrieval_chain(self):
        """Setup LangChain retrieval chain for RAG"""
        # Create retriever from vector store
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 similar workloads
        )
        
        # Setup Bedrock LLM
        try:
            llm = Bedrock(
                model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
                region_name="us-east-1"
            )
        except:
            # Fallback to a mock LLM for testing
            from langchain.llms.fake import FakeListLLM
            llm = FakeListLLM(responses=["Mock LLM response for testing"])
        
        # Create custom prompt template
        prompt_template = """
        You are an expert cluster sizing consultant. Based on the following similar workloads and user requirements, provide intelligent cluster recommendations.

        User Query: {question}

        Similar Workloads from Database:
        {context}

        Please provide:
        1. Recommended cluster size and configuration
        2. Cost analysis and optimization suggestions
        3. Performance considerations
        4. Key factors that influenced your recommendation

        Recommendation:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ RAG retrieval chain configured")
    
    def get_recommendations(self, user_query: str) -> Dict:
        """Get cluster recommendations using RAG approach"""
        try:
            # Use LangChain RAG chain
            result = self.qa_chain({"query": user_query})
            
            # Extract similar workloads from source documents
            similar_workloads = []
            for doc in result.get("source_documents", []):
                similar_workloads.append({
                    'description': doc.page_content,
                    'metadata': doc.metadata,
                    'workload_type': doc.metadata.get('workload_type'),
                    'recommended_cluster_size': doc.metadata.get('recommended_cluster_size'),
                    'recommended_node_count': doc.metadata.get('recommended_node_count'),
                    'estimated_cost_per_hour': doc.metadata.get('estimated_cost_per_hour')
                })
            
            return {
                'llm_recommendation': result.get("result", ""),
                'similar_workloads': similar_workloads,
                'retrieval_method': 'LangChain RAG with vector similarity',
                'vector_store_type': self.vector_store_type
            }
            
        except Exception as e:
            return {
                'error': f"RAG retrieval failed: {str(e)}",
                'similar_workloads': [],
                'retrieval_method': 'Error in LangChain RAG'
            }
    
    def search_similar_workloads(self, query: str, k: int = 5) -> List[Dict]:
        """Direct vector similarity search"""
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score),
                    'workload_type': doc.metadata.get('workload_type'),
                    'cluster_recommendation': doc.metadata.get('recommended_cluster_size')
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return []
    
    def save_vector_database(self, path: str = "cluster_vector_db"):
        """Save vector database for persistence"""
        if self.vector_store_type == "faiss":
            self.vector_store.save_local(path)
        elif self.vector_store_type == "chroma":
            # Chroma auto-persists if persist_directory is set
            pass
        print(f"✅ Vector database saved to {path}")
    
    def load_vector_database(self, path: str = "cluster_vector_db"):
        """Load existing vector database"""
        try:
            if self.vector_store_type == "faiss":
                self.vector_store = FAISS.load_local(path, self.embeddings)
            elif self.vector_store_type == "chroma":
                self.vector_store = Chroma(
                    persist_directory=path,
                    embedding_function=self.embeddings
                )
            print(f"✅ Vector database loaded from {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load vector database: {e}")
            return False

# Example usage and testing
def test_langchain_vector_db():
    """Test the LangChain vector database implementation"""
    print("🧪 Testing LangChain Vector Database Implementation")
    print("=" * 60)
    
    # Test with FAISS (local, fast)
    wizard = LangChainClusterWizard(vector_store_type="faiss")
    
    # Test queries
    test_queries = [
        "Machine learning training with 64 CPU cores and 256GB RAM",
        "Real-time processing system with low latency requirements",
        "Data analytics workload for 100 concurrent users"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        # Test direct similarity search
        similar = wizard.search_similar_workloads(query, k=3)
        print(f"📊 Found {len(similar)} similar workloads")
        
        for i, result in enumerate(similar[:2]):
            print(f"  {i+1}. {result['workload_type']} (Score: {result['similarity_score']:.3f})")
            print(f"     Recommendation: {result['cluster_recommendation']}")
        
        # Test RAG chain
        recommendations = wizard.get_recommendations(query)
        if 'error' not in recommendations:
            print(f"🤖 RAG Recommendation: {recommendations['llm_recommendation'][:100]}...")
        else:
            print(f"⚠️  RAG Error: {recommendations['error']}")
    
    print("\n✅ LangChain vector database testing complete!")

if __name__ == "__main__":
    test_langchain_vector_db()
