import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore  # <--- The Magic Fix
from langchain_ollama import OllamaEmbeddings

# Global variable to hold the database in RAM
# We don't save to a file, so no "Read-Only" errors!
RAM_DB = None

def get_embeddings():
    """
    Uses your local Ollama 'nomic-embed-text' model.
    """
    return OllamaEmbeddings(
        model="nomic-embed-text",        
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434") 
    )

def ingest_pdf(file_path: str):
    """
    Ingests PDF into RAM.
    """
    global RAM_DB
    print(f"--- [RAG] Indexing {file_path} into RAM ...")
    
    # 1. Load
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        return 0

    # 3. Create In-Memory DB (Fresh start every time)
    # We initialize it with the first batch of chunks
    print(f"--- [RAG] Creating In-Memory Store for {len(chunks)} chunks...")
    
    RAM_DB = InMemoryVectorStore.from_documents(
        chunks,
        get_embeddings()
    )
    
    # (Optional) Yield progress just to satisfy the UI loop
    # Since it's in-memory, it's instant, so we just yield 100%
    yield 1.0
    
    print(f"--- [RAG] Finished indexing in RAM.")

def query_rag(query_text: str, k=3):
    """
    Searches the RAM database.
    """
    global RAM_DB
    if RAM_DB is None:
        return None
        
    print(f"--- [RAG] Searching RAM for: '{query_text}'")
    
    # Similarity Search
    results = RAM_DB.similarity_search(query_text, k=k)
    
    if results:
        context_str = "\n\n".join([f"[Chunk {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])
        return context_str
    else:
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