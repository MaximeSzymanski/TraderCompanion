import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore  # <--- The Magic Fix
from langchain_ollama import OllamaEmbeddings
import streamlit as st
# Global variable to hold the database in RAM
# We don't save to a file, so no "Read-Only" errors!

def get_embeddings():
    """
    Uses your local Ollama 'nomic-embed-text' model.
    """
    return OllamaEmbeddings(
        model="nomic-embed-text",        
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434") 
    )
@st.cache_resource
def get_vector_store():
    # This ensures the DB persists across reruns but is unique to the session logic
    # Note: For a true multi-user app, you'd handle this differently, 
    # but for a portfolio local demo, this prevents "variable not defined" errors.
    return InMemoryVectorStore(embedding=get_embeddings())

def ingest_pdf(file_path: str):
    print(f"--- [RAG] Indexing {file_path} into RAM ...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    if not chunks: return 0

    # Get the store (Cached)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks) # Add to existing or new store
    
    yield 1.0
    print(f"--- [RAG] Finished indexing in RAM.")

def query_rag(query_text: str, k=3):
    vector_store = get_vector_store()
    
    # Check if store is empty (hacky check for InMemory)
    # Ideally, track if ingestion happened in session state
    if not vector_store.store: 
        return None
        
    results = vector_store.similarity_search(query_text, k=k)
    if results:
        return "\n\n".join([f"[Chunk {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])
    return None
    
from langchain_core.tools import tool
# Add this import based on your project structure
from my_agent.utils.rag import query_rag 

# --- Define a tool interface for the Router to recognize RAG intent ---
@tool("lookup_knowledge_base")
def lookup_knowledge_base(query: str):
    """
    Useful for answering questions about specific trading strategies, 
    internal documentation, pdfs, agent capabilities, or how the backtester works.
    Use this when the user asks 'how to', 'explain strategy', or about internal concepts.
    """
    return "trigger_rag"