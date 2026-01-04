import streamlit as st
import streamlit.components.v1 as components
import uuid
import os
import re
from langchain_core.messages import HumanMessage

# --- IMPORTS FOR YOUR CUSTOM AGENT ---
from my_agent.agent import app
from my_agent.utils.rag import ingest_pdf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Trader Companion AI",
    page_icon="📈",
    layout="wide" 
)

st.title("📈 Trader Companion AI")
st.caption("Interactive Technical Analysis & Document Search powered by LangGraph.")

# --- SESSION STATE SETUP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Tracks if a file has EVER been uploaded in this session
if "rag_capable" not in st.session_state:
    st.session_state.rag_capable = False

# --- HELPER: DETECT CHART PATH ---
def extract_chart_path(content: str):
    """
    Checks if the bot response contains a file path to an HTML chart.
    """
    match = re.search(r"(charts/[a-zA-Z0-9_]+\.html)", content)
    if match:
        return match.group(1)
    return None

# --- SIDEBAR: KNOWLEDGE BASE (RAG) ---
st.sidebar.title("📚 Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload Company Report (PDF)", type="pdf")

if uploaded_file:
    # Save to disk temporarily
    temp_path = "temp_doc.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # --- PROGRESS BAR LOGIC ---
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        # Loop through the generator to update the bar
        for progress in ingest_pdf(temp_path):
            progress_bar.progress(progress)
            status_text.text(f"Indexing... {int(progress * 100)}%")
            
        status_text.empty() # Clear text
        st.sidebar.success("✅ Indexing Complete!")
        
        # 1. Mark system as "Capable" of RAG
        st.session_state.rag_capable = True
        
        # 2. Force the Toggle Switch to ON (if it exists in state)
        # This makes the UI intuitive: Upload -> Auto-Enable
        st.session_state["rag_toggle"] = True
        
    except Exception as e:
        st.sidebar.error(f"Indexing failed: {e}")

# --- MANUAL TOGGLE (THE MASTER SWITCH) ---
# We link this widget to 'rag_toggle' key so we can control it from the upload block above
use_rag = st.sidebar.toggle(
    "🟢 Enable Document Search", 
    key="rag_toggle", 
    disabled=not st.session_state.rag_capable, # Grey out if no file
    help="If On, bot reads your PDF. If Off, bot searches Yahoo Finance."
)

# --- VISUAL STATUS INDICATOR ---
# This now looks at the SWITCH (use_rag), not the file upload status.
if use_rag:
    st.sidebar.info("🟢 RAG Mode Active: Analyzing PDF")
else:
    st.sidebar.info("⚪ Web Mode Active: Searching Yahoo Finance")


# --- UI: DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 1. Handle HTML Charts (Interactive)
        if message.get("is_html"):
            if os.path.exists(message["content"]):
                with open(message["content"], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                components.html(html_content, height=600, scrolling=True)
            else:
                st.error(f"Chart file missing: {message['content']}")
        
        # 2. Handle Text
        else:
            st.markdown(message["content"])

# --- UI: CHAT INPUT ---
if prompt := st.chat_input("Ask about a stock (e.g., 'Analyze Nvidia') or your PDF..."):
    
    # User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Agent Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🕵️ *Thinking...*")
        
        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # --- CRITICAL: PASS THE TOGGLE VALUE ---
            inputs = {
                "messages": [HumanMessage(content=prompt)],
                "is_rag_active": use_rag  # <--- True if Toggle is On, False if Off
            }
            
            # Run the Graph
            final_state = app.invoke(inputs, config=config)
            
            bot_response_content = final_state["messages"][-1].content
            
            # Check for Chart
            chart_path = extract_chart_path(bot_response_content)
            
            if chart_path:
                # === RENDER CHART ===
                message_placeholder.empty() # Clear loading text
                
                if os.path.exists(chart_path):
                    with open(chart_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Render the interactive Plotly HTML
                    components.html(html_content, height=600, scrolling=True)
                    
                    # Save to history with 'is_html' flag
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": chart_path,
                        "is_html": True
                    })
                else:
                    st.error(f"File not found: {chart_path}")
            
            else:
                # === RENDER TEXT ===
                message_placeholder.markdown(bot_response_content)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": bot_response_content,
                    "is_html": False
                })

        except Exception as e:
            message_placeholder.error(f"An error occurred: {str(e)}")