import uuid
import os
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- IMPORTS FOR NODES & STATE ---
# Ensure these match your project structure
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import (
    node_extract_entities, 
    node_validate_ticker, 
    node_validate_dates, 
    node_fetch_data, 
    node_technical_analysis, 
    node_clarify_preference, 
    node_generate_viz,
    node_ask_user, 
    node_sentiment_analysis,
    node_forecast,
    node_rag_search,
    node_risk_disclaimer
)

# =============================================================================
# 1. ROUTER DEFINITIONS
# =============================================================================

def route_rag(state):
    """Diamond 0: RAG or Web Search?"""
    if state.get("is_rag_active"):
        return "rag_search"
    return "check_ticker"

def route_ticker_check(state):
    """Diamond 1: Ticker Exists?"""
    if state.get('ticker') and not state.get('error_message'):
        return "check_dates"   # Success path
    return "ask_human"        # Failure path

def route_date_check(state):
    """Diamond 2: Dates Valid?"""
    if state.get('error_message'):
        return "ask_human"     
    return "fetcher"       

def route_fetcher(state):
    """Diamond 3: News, Data, or Tool Answer?"""
    if state.get("news_summary"):
        return "sentiment_analyst" 
    if state.get("stock_data"):
        return "analyst"
    if not state.get("error_message"):
        return END 
    return "ask_human"

def route_style_check(state):
    """Diamond 4: Plot or Table?"""
    if state.get('error_message'):
        return "ask_human"
    if state.get('output_preference'):
        return "visualizer"
    return "ask_human"

def route_rag_result(state):
    """
    RAG Fallback Logic:
    If RAG fails (fallback=True), route to Web Search (check_ticker).
    Otherwise, End.
    """
    if state.get("rag_fallback"):
        print("--- [ROUTER] RAG failed, falling back to Web Search ---")
        return "check_ticker"
    return END

# =============================================================================
# 2. GRAPH CONSTRUCTION
# =============================================================================

def build_financial_agent():
    """Builds and compiles the StateGraph workflow."""
    workflow = StateGraph(AgentState)

    # --- Add Nodes ---
    workflow.add_node("extractor", node_extract_entities)
    workflow.add_node("check_ticker", node_validate_ticker)
    workflow.add_node("check_dates", node_validate_dates)
    workflow.add_node("fetcher", node_fetch_data)
    workflow.add_node("analyst", node_technical_analysis)
    workflow.add_node("check_style", node_clarify_preference)
    workflow.add_node("visualizer", node_generate_viz)
    workflow.add_node("ask_human", node_ask_user)
    workflow.add_node("sentiment_analyst", node_sentiment_analysis)
    workflow.add_node("forecaster", node_forecast)
    workflow.add_node("rag_search", node_rag_search)
    workflow.add_node("risk_disclaimer", node_risk_disclaimer)

    # --- Set Entry Point ---
    workflow.set_entry_point("extractor")

    # --- Standard Edges ---
    # Flow: Analyst -> Forecaster -> Style Check
    workflow.add_edge("analyst", "forecaster")
    workflow.add_edge("forecaster", "check_style")
    
    # Flow: Visualizer -> Risk Disclaimer -> End
    workflow.add_edge("visualizer", "risk_disclaimer")
    workflow.add_edge("risk_disclaimer", END)
    
    # Terminal Edges
    workflow.add_edge("ask_human", END)
    workflow.add_edge("sentiment_analyst", END)

    # --- Conditional Edges ---

    # 1. Extractor -> RAG OR Ticker Check
    workflow.add_conditional_edges(
        "extractor",
        route_rag,
        {"rag_search": "rag_search", "check_ticker": "check_ticker"}
    )

    # 2. RAG Result -> Fallback (Ticker Check) OR End
    workflow.add_conditional_edges(
        "rag_search",
        route_rag_result,
        {"check_ticker": "check_ticker", END: END}
    )

    # 3. Ticker Check -> Dates OR Human
    workflow.add_conditional_edges(
        "check_ticker",
        route_ticker_check,
        {"check_dates": "check_dates", "ask_human": "ask_human"}
    )

    # 4. Dates Check -> Fetcher OR Human
    workflow.add_conditional_edges(
        "check_dates",
        route_date_check,
        {"fetcher": "fetcher", "ask_human": "ask_human"}
    )

    # 5. Fetcher -> Sentiment | Analyst | Human | End
    workflow.add_conditional_edges(
        "fetcher", 
        route_fetcher,
        {
            'sentiment_analyst': "sentiment_analyst",
            "analyst": "analyst",
            "ask_human": "ask_human",
            END: END
        }
    )

    # 6. Style Check -> Visualizer OR Human
    workflow.add_conditional_edges(
        "check_style",
        route_style_check,
        {"visualizer": "visualizer", "ask_human": "ask_human"}
    )

    # --- Compile ---
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

# Initialize the App (Global instance for Streamlit import)
app = build_financial_agent()

# =============================================================================
# 3. MAIN EXECUTION (CLI TEST)
# =============================================================================
if __name__ == "__main__":
    from IPython.display import Image, display
    
    # Generate Diagram
    try:
        diagram = app.get_graph().draw_mermaid_png()
        with open("financial_assistant_workflow.png", "wb") as f:
            f.write(diagram)
        print("--- [INFO] Workflow diagram saved to 'financial_assistant_workflow.png' ---")
        # display(Image(diagram)) # Only works in Jupyter
    except Exception as e:
        print(f"--- [WARN] Could not generate diagram: {e}")

    # CLI Loop
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("--- Financial Assistant Started (Type 'quit' to exit) ---")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            # Stream events
            for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="values"):
                pass
            
            # Get Final Response
            snapshot = app.get_state(config)
            if snapshot.values.get('messages'):
                last_msg = snapshot.values['messages'][-1]
                print(f"Assistant: {last_msg.content}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"--- [ERROR] {e}")