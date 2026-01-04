import os
import re
import uuid
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage

# --- CUSTOM AGENT IMPORTS ---
# Ensure these match your project structure
from my_agent.agent import app
from my_agent.utils.rag import ingest_pdf

# =============================================================================
# 1. CONFIG & UTILS
# =============================================================================

st.set_page_config(
    page_title="Trader Companion AI",
    page_icon="📈",
    layout="wide" 
)

def initialize_session():
    """Initializes Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    if "rag_capable" not in st.session_state:
        st.session_state.rag_capable = False

def extract_chart_path(content: str):
    """
    Regex to find chart file paths in bot response.
    Supports dots/hyphens in filenames (e.g., 'charts/UBI.PA_123.html').
    """
    match = re.search(r"(charts/[\w\.-]+\.html)", content)
    if match:
        return match.group(1)
    return None

def save_uploaded_file(uploaded_file):
    """Saves uploaded PDF to a temporary file."""
    temp_path = "temp_doc.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

# =============================================================================
# 2. UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Renders the Knowledge Base sidebar and handles PDF ingestion."""
    st.sidebar.title("📚 Knowledge Base")
    
    # 1. File Uploader
    uploaded_file = st.sidebar.file_uploader("Upload Company Report (PDF)", type="pdf")
    
    if uploaded_file:
        temp_path = save_uploaded_file(uploaded_file)
        
        # Progress Bar Logic
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            for progress in ingest_pdf(temp_path):
                progress_bar.progress(progress)
                status_text.text(f"Indexing... {int(progress * 100)}%")
                
            status_text.empty()
            st.sidebar.success("✅ Indexing Complete!")
            
            # Auto-enable RAG
            st.session_state.rag_capable = True
            st.session_state["rag_toggle"] = True
            
        except Exception as e:
            st.sidebar.error(f"Indexing failed: {e}")

    # 2. Master Toggle Switch
    use_rag = st.sidebar.toggle(
        "🟢 Enable Document Search", 
        key="rag_toggle", 
        disabled=not st.session_state.rag_capable,
        help="ON: Reads PDF. OFF: Searches Web/Yahoo."
    )
    
    # 3. Status Indicator
    if use_rag:
        st.sidebar.info("🟢 RAG Mode Active: Analyzing PDF")
    else:
        st.sidebar.info("⚪ Web Mode Active: Searching Yahoo Finance")
        
    return use_rag

def render_chat_history():
    """Displays existing chat messages and interactive charts."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("is_html"):
                # Render HTML Chart
                if os.path.exists(message["content"]):
                    with open(message["content"], 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
                else:
                    st.error(f"Chart file missing: {message['content']}")
            else:
                # Render Markdown Text
                st.markdown(message["content"])

# =============================================================================
# 3. CORE LOGIC
# =============================================================================

def handle_user_input(prompt, use_rag):
    """Processing loop for user input -> LangGraph -> UI Update."""
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Agent Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🕵️ *Thinking...*")
        
        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            inputs = {
                "messages": [HumanMessage(content=prompt)],
                "is_rag_active": use_rag
            }
            
            # Invoke Graph
            final_state = app.invoke(inputs, config=config)
            
            # Extract response details
            raw_response = final_state["messages"][-1].content
            chart_path = None
            
            # Scan all recent messages for chart paths (handling multi-message outputs)
            for msg in final_state["messages"][::-1]: # Look backwards
                content = msg.content if hasattr(msg, "content") else msg.get("content")
                path = extract_chart_path(content)
                if path:
                    chart_path = path
                    break

            # 3. Render Output
            if chart_path:
                placeholder.empty() # Remove loading text
                
                # Render Chart
                if os.path.exists(chart_path):
                    with open(chart_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
                    
                    # Save Chart to History
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": chart_path,
                        "is_html": True
                    })
                else:
                    st.error(f"File not found: {chart_path}")
                
                # Render Text Analysis (Cleaning path from text)
                clean_text = raw_response.replace(chart_path, "").strip()
                if clean_text:
                    placeholder.markdown(clean_text) # Re-use placeholder for text
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": clean_text,
                        "is_html": False
                    })
            else:
                # Text Only Response
                placeholder.markdown(raw_response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": raw_response,
                    "is_html": False
                })

        except Exception as e:
            placeholder.error(f"An error occurred: {str(e)}")

# =============================================================================
# 4. MAIN APP LOOP
# =============================================================================

def main():
    initialize_session()
    
    st.title("📈 Trader Companion AI")
    st.caption("Interactive Technical Analysis & Document Search powered by LangGraph.")
    
    # Render Layout
    use_rag = render_sidebar()
    render_chat_history()
    
    # Chat Input
    if prompt := st.chat_input("Ask about a stock (e.g., 'Analyze Nvidia') or your PDF..."):
        handle_user_input(prompt, use_rag)

if __name__ == "__main__":
    main()