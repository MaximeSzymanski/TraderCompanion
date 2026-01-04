import os
import re
import json
import uuid
import difflib
from datetime import datetime, timedelta
from typing import Optional

# --- DATA SCIENCE IMPORTS ---
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# --- LANGCHAIN / AI IMPORTS ---
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# --- PROJECT IMPORTS ---
from my_agent.utils.tools import (
    lookup_knowledge_base, 
    search_financial_news, 
    get_company_fundamentals
)
from my_agent.utils.rag import query_rag

# =============================================================================
# 0. GLOBAL CONFIGURATION & MODELS
# =============================================================================

# Load S&P 500 Data for local lookup
try:
    if os.path.exists("my_agent/sp500_tickers.csv"):
        SP500_DF = pd.read_csv("my_agent/sp500_tickers.csv")
    else:
        FileNotFoundError("sp500_tickers.csv not found.")
        SP500_DF = pd.DataFrame(columns=['Symbol', 'Security'])
except Exception as e:
    SP500_DF = pd.DataFrame(columns=['Symbol', 'Security'])

# Schema for Entity Extraction
class SearchIntent(BaseModel):
    company_name: Optional[str] = Field(
        None, 
        description="The name of the company or stock symbol mentioned (e.g., 'Apple', 'MSFT')."
    )
    start_date: Optional[str] = Field(
        None, 
        description="The start date for the data in YYYY-MM-DD format. Calculate relative dates."
    )
    end_date: Optional[str] = Field(
        None, 
        description="The end date for the data in YYYY-MM-DD format."
    )
    visualization_type: Optional[str] = Field(
        None, 
        description="The preferred output format. Must be either 'plot' or 'table'."
    )

# =============================================================================
# 1. INPUT PROCESSING & VALIDATION NODES
# =============================================================================

def node_extract_entities(state):
    """
    Analyzes the conversation history to extract search parameters (Ticker, Date, Style).
    """
    print("\n--- [NODE] EXTRACT ENTITIES ---")
    
    messages = state.get("messages", [])
    if messages:
        print(f"--- [LOG] Last User Message: '{messages[-1].content}'")
    
    parser = JsonOutputParser(pydantic_object=SearchIntent)
    llm = ChatOllama(model="qwen2.5:7b", temperature=0, format="json")

    current_date = datetime.now().strftime("%Y-%m-%d")
    
    system_prompt = """You are a financial assistant data extractor.
    Today is {current_date}.
    
    Extract the company name, date range, and visualization preference from the conversation.
    If a value is not mentioned, return null.
    
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    
    prompt_with_instructions = prompt.partial(
        format_instructions=parser.get_format_instructions(),
        current_date=current_date
    )
    
    chain = prompt_with_instructions | llm | parser

    try:
        print("--- [LOG] Invoking LLM for Extraction...")
        result = chain.invoke({"messages": state["messages"]})
        print(f"--- [LOG] Extracted Raw JSON: {result}")
        
        updates = {}
        if result.get("company_name"):
            updates["ticker"] = result["company_name"]
            updates["error_message"] = None

        if result.get("start_date"):
            updates["start_date"] = result["start_date"]
            
        if result.get("end_date"):
            updates["end_date"] = result["end_date"]
            
        if result.get("visualization_type"):
            updates["output_preference"] = result["visualization_type"]

        print(f"--- [LOG] Updates to State: {updates}")
        return updates

    except Exception as e:
        print(f"--- [ERROR] Extraction Error: {e}")
        return {}

def node_validate_ticker(state):
    """
    Robust Ticker Validation Node.
    1. Checks S&P 500 CSV (Fast Path).
    2. Searches Web for multiple candidates (Slow Path).
    3. Validates against Yahoo Finance data AND Company Name to avoid collisions.
    """
    print("\n--- [NODE] VALIDATE TICKER (FINAL ROBUST) ---")
    ticker_input = state.get("ticker")
    
    if not ticker_input:
        return {"error_message": "No company name provided."}

    # --- STEP 1: CHECK LOCALLY (S&P 500 CSV) ---
    if not SP500_DF.empty:
        mask = (SP500_DF['Security'].str.contains(ticker_input, case=False, na=False) | 
                SP500_DF['Symbol'].str.contains(ticker_input, case=False, na=False))
        match = SP500_DF[mask]
        if not match.empty:
            found_ticker = match.iloc[0]['Symbol']
            return {"ticker": found_ticker, "error_message": None}

    # --- STEP 2: SEARCH WEB & RETRIEVE CANDIDATES ---
    candidates = []
    try:
        search = DuckDuckGoSearchRun()
        query = f"Yahoo finance ticker symbol for {ticker_input}"
        search_results = search.invoke(query)
        
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        extraction_prompt = f"""
        Search Results: "{search_results}"
        Task: Identify valid Yahoo Finance tickers for '{ticker_input}'.
        Rules: Return up to 3 likely tickers, separated by commas.
        """
        response = llm.invoke(extraction_prompt)
        raw_list = [c.strip() for c in response.content.split(',')]
        for c in raw_list:
            clean = re.sub(r'[^A-Z0-9\.-]', '', c.upper())
            if clean and "NOTFOUND" not in clean:
                candidates.append(clean)
    except Exception:
        pass

    if not candidates:
        candidates = [ticker_input.upper()]

    # --- STEP 3: EXPAND CANDIDATES (Suffixes AND Roots) ---
    expanded_candidates = []
    for c in candidates:
        if c not in expanded_candidates:
            expanded_candidates.append(c)
        
        if "." not in c:
            # If Root, try adding International Suffixes
            for suffix in [".TO", ".PA", ".L", ".SW"]:
                if f"{c}{suffix}" not in expanded_candidates:
                    expanded_candidates.append(f"{c}{suffix}")
        else:
            # If Suffix exists, try the Root (e.g. STLA.PA -> STLA)
            root = c.split('.')[0]
            if root not in expanded_candidates:
                expanded_candidates.append(root)

    # --- STEP 4: VALIDATION LOOP ---
    suffix_map = {".TSX": ".TO", ".TV": ".TO", ".PAR": ".PA", ".TYO": ".T", ".LSE": ".L", ".HKG": ".HK", ".ASX": ".AX"}
    
    for cand in expanded_candidates:
        # Auto-correct suffix
        for bad, good in suffix_map.items():
            if cand.endswith(bad):
                cand = cand.replace(bad, good)
        
        try:
            print(f"--- [LOG] Testing Candidate: '{cand}'...")
            stock = yf.Ticker(cand)
            hist = stock.history(period="1d")
            
            if hist.empty:
                continue

            # Check Name Similarity
            info = stock.info
            yahoo_name = info.get("longName", "").lower()
            user_query = ticker_input.lower()
            
            is_match = False
            if user_query in yahoo_name:
                is_match = True
            else:
                ratio = difflib.SequenceMatcher(None, user_query, yahoo_name).ratio()
                if cand.split('.')[0] == ticker_input.upper(): ratio += 0.3 
                if ratio > 0.4: is_match = True
            
            if is_match:
                print(f"--- [LOG] Match Confirmed! '{cand}' is '{yahoo_name}'")
                return {"ticker": cand, "error_message": None}

        except Exception:
            continue

    return {"ticker": None, "error_message": f"Could not verify ticker for '{ticker_input}'."}

def node_validate_dates(state):
    """
    Validates start/end dates. 
    """
    print("\n--- [NODE] VALIDATE DATES ---")
    start_date_str = state.get("start_date")
    end_date_str = state.get("end_date")
    preference = state.get("output_preference")
    
    updates = {}
    error_msgs = []
    
    if not start_date_str and not end_date_str:
        if preference:
            return {"error_message": "Please provide start and end dates."}
        else:
            return {"error_message": None}
            
    dt_start = None
    dt_end = None
    today = datetime.now()

    if start_date_str:
        try:
            dt_start = datetime.strptime(start_date_str, "%Y-%m-%d")
            updates["start_date"] = dt_start.strftime("%Y-%m-%d")
            if dt_start > today:
                error_msgs.append(f"Start date cannot be in the future.")
        except ValueError:
            error_msgs.append(f"Start date '{start_date_str}' is invalid. Use YYYY-MM-DD.")
    else:
        error_msgs.append("Please provide a start date.")

    if end_date_str:
        try:
            dt_end = datetime.strptime(end_date_str, "%Y-%m-%d")
            updates["end_date"] = dt_end.strftime("%Y-%m-%d")
            if dt_end > today:
                 error_msgs.append(f"End date cannot be in the future.")
        except ValueError:
            error_msgs.append(f"End date '{end_date_str}' is invalid. Use YYYY-MM-DD.")
    else:
        error_msgs.append("Please provide an end date.")

    if dt_start and dt_end:
        if dt_start > dt_end:
            error_msgs.append("Start date cannot be after end date.")

    if error_msgs:
        return {"error_message": " ".join(error_msgs)}
    
    updates["error_message"] = None
    return updates

def node_clarify_preference(state):
    """
    Checks if the user has specified a visualization format (Plot vs Table).
    """
    print("\n--- [NODE] CLARIFY PREFERENCE ---")
    preference = state.get("output_preference")
    
    if preference:
        pref_clean = preference.lower().strip()
        if pref_clean in ["plot", "chart", "graph", "drawing"]:
            return {"output_preference": "plot", "error_message": None}
        elif pref_clean in ["table", "list", "data", "rows", "csv"]:
            return {"output_preference": "table", "error_message": None}

    return {
        "output_preference": None,
        "error_message": "Would you like to see a plot or a table?"
    }

# =============================================================================
# 2. DATA RETRIEVAL (ROUTER)
# =============================================================================

def node_fetch_data(state):
    """
    Smart Router: Decides between News, Fundamentals, RAG, or Stock Data.
    Prioritizes Stock Data if keywords like 'price', 'values', or 'chart' are detected.
    """
    print("\n--- [NODE] FETCH DATA (ROUTER) ---")
    
    ticker = state.get("ticker")
    start_date = state.get("start_date")
    end_date = state.get("end_date")
    existing_preference = state.get("output_preference")
    
    messages = state.get("messages", [])
    last_msg = messages[-1].content.lower() if messages else ""

    # --- KEYWORD GUARD ---
    data_keywords = ["price", "value", "values", "chart", "plot", "graph", "history", "trend", "data"]
    
    if existing_preference in ["plot", "table"] or any(k in last_msg for k in data_keywords):
        print(f"--- [LOG] Data intent detected. Forcing Data Fetch.")
        should_fetch_data = True
    else:
        print("--- [LOG] Intent unclear. Asking Router LLM...")
        should_fetch_data = False

    # --- LLM ROUTER ---
    if not should_fetch_data:
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        llm_with_tools = llm.bind_tools([
            search_financial_news, 
            get_company_fundamentals, 
            lookup_knowledge_base
        ])      
        
        msg = (
            f"User Input: '{last_msg}'\n"
            f"Context: Ticker={ticker}, DateRange={start_date} to {end_date}.\n"
            "Decide the best tool to use:\n"
            "- Use 'lookup_knowledge_base' for strategy, PDFs, or internal docs.\n"
            "- Use 'search_financial_news' for 'news', 'why', 'headlines'.\n"
            "- Use 'get_company_fundamentals' for 'sector', 'industry', 'market cap'.\n"
            "- IF the user asks for prices, values, history, or charts, DO NOT use a tool. Return empty."
        )
        
        response = llm_with_tools.invoke([HumanMessage(content=msg)])
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"--- [LOG] LLM Decided: USE TOOL '{tool_name}'")
            
            if tool_name == "lookup_knowledge_base":
                return {
                    "is_rag_query": True,
                    "messages": [AIMessage(content="Checking internal knowledge base...")]
                }

            elif tool_name == "search_financial_news":
                # Force ticker into the search (Safe Query)
                tool_result = search_financial_news.invoke(ticker)
                return {
                    "news_summary": tool_result, 
                    "messages": [AIMessage(content=f"Fetching latest headlines for **{ticker}**...")]
                }

            elif tool_name == "get_company_fundamentals":
                tool_result = get_company_fundamentals.invoke(tool_args)
                content = (
                    f"### 🏢 {tool_result.get('name', 'Company')}\n"
                    f"**Sector:** {tool_result.get('sector', 'N/A')} | **Industry:** {tool_result.get('industry', 'N/A')}\n"
                    f"**Market Cap:** {tool_result.get('market_cap', 'N/A')} | **P/E Ratio:** {tool_result.get('pe_ratio', 'N/A')}\n\n"
                    f"_{tool_result.get('summary', 'No summary available.')}_"
                )
                return {
                    "messages": [AIMessage(content=content)],
                    "error_message": None
                }
        else:
            should_fetch_data = True

    # --- STOCK DATA FETCHING ---
    if should_fetch_data:
        if not ticker or not start_date or not end_date:
            return {"error_message": "Missing ticker or date range for data fetch."}
        
        try:
            print(f"--- [LOG] Fetching yfinance history for {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return {"error_message": f"No data found for {ticker}."}
            
            hist.index = hist.index.strftime('%Y-%m-%d')
            data_serializable = hist.to_dict(orient="index")
            
            updates = {
                "stock_data": data_serializable,
                "error_message": None
            }
            if not existing_preference:
                updates["output_preference"] = "plot"
            return updates
            
        except Exception as e:
            return {"error_message": f"Error fetching data: {str(e)}"}
            
    return {}

# =============================================================================
# 3. ANALYSIS & LOGIC NODES
# =============================================================================

def node_technical_analysis(state):
    """
    Calculates RSI and SMA indicators.
    """
    print("\n--- [NODE] TECHNICAL ANALYSIS ---")
    stock_data = state.get("stock_data")
    ticker = state.get("ticker", "Stock")
    
    if not stock_data:
        return {"messages": [AIMessage(content="No stock data available for technical analysis.")]}
    
    df = pd.DataFrame.from_dict(stock_data, orient="index")
    df.index = pd.to_datetime(df.index)
    
    # Calculate Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df = df.fillna(0)
    df.index = df.index.strftime('%Y-%m-%d')
    
    return {
        "stock_data": df.to_dict(orient="index"),
        "messages": [AIMessage(content=f"Technical analysis for **{ticker}** completed.")]
    }
    
def node_sentiment_analysis(state):
    """
    Analyzes news summary using LLM.
    """
    print("\n--- [NODE] SENTIMENT ANALYSIS ---")
    news_text = state.get("news_summary")
    if not news_text:
        return {}

    llm = ChatOllama(model="qwen2.5:7b", temperature=0, format="json")
    
    system = """You are a veteran financial analyst. 
    Analyze the following news summary and determine the market sentiment.
    Return a JSON with: "score" (-1.0 to 1.0), "verdict", "explanation".
    """
    msg = f"News Summary: {news_text}"
    
    try:
        response = llm.invoke([("system", system), ("human", msg)])
        analysis = json.loads(response.content)
        
        formatted_msg = (
            f"**Sentiment Verdict:** {analysis['verdict']} (Score: {analysis['score']})\n\n"
            f"_{analysis['explanation']}_\n\n"
            f"---\n**News Source:**\n{news_text[:500]}..."
        )
        return {
            "sentiment": analysis,
            "messages": [AIMessage(content=formatted_msg)],
            "news_summary": None 
        }
    except Exception as e:
        return {"error_message": f"Sentiment analysis failed: {e}"}
    
def node_forecast(state):
    """
    Linear Regression Forecast for next 30 days.
    """
    print("\n--- [NODE] FORECASTING ---")
    stock_data = state.get("stock_data")
    if not stock_data:
        return {}
    
    df = pd.DataFrame.from_dict(stock_data, orient="index")
    df.index = pd.to_datetime(df.index)
    
    # Feature Engineering
    df['Date_Ordinal'] = df.index.map(pd.Timestamp.toordinal)
    recent_df = df.tail(30)
    X = recent_df[['Date_Ordinal']].values
    y = recent_df['Close'].values
    
    # Model Fit
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 30)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    predictions = model.predict(future_ordinals)
    r2 = model.score(X, y)
    trend = "upward" if model.coef_[0] > 0 else "downward"
    
    forecast_dict = {
        date.strftime('%Y-%m-%d'): float(pred)
        for date, pred in zip(future_dates, predictions)
    }
    
    print(f"--- [LOG] R^2: {r2:.4f}, Trend: {trend}")
    return {"forecast_data": forecast_dict, "forecast_meta" : {"r2_score": round(r2,4), "trend": trend}}

def node_company_profile(state):
    """
    Extracts profile info from stock data.
    """
    print("--- [NODE] COMPANY PROFILE ---")
    stock_data = state.get("stock_data")
    ticker = state.get("ticker")
    
    if not stock_data or "info" not in stock_data:
        return {} 
        
    info = stock_data["info"]
    name = info.get("longName", ticker)
    mkt_cap = info.get("marketCap", 0)
    mkt_cap_str = f"${mkt_cap / 1e9:.2f}B" if mkt_cap else "N/A"
    
    profile_text = (
        f"### 🏢 {name}\n"
        f"**Sector:** {info.get('sector', 'Unknown')} | **Market Cap:** {mkt_cap_str}\n\n"
        f"_{info.get('longBusinessSummary', 'No summary')[:400]}..._"
    )
    return {"messages": [AIMessage(content=profile_text)]}

# =============================================================================
# 4. RAG SEARCH NODE
# =============================================================================

def node_rag_search(state):
    """
    Queries vector DB. Sets fallback flag if not found.
    """
    print("\n--- [NODE] RAG SEARCH ---")
    messages = state.get("messages")
    user_query = messages[-1].content
    
    context = query_rag(user_query)
    
    if not context:
        print("--- [RAG] No context found. Fallback. ---")
        return {"rag_fallback": True}

    llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    system_prompt = """You are a documentation expert. Answer STRICTLY based on Context.
    If answer is not in context, output: "NOT_FOUND"
    Context: {context}
    """
    
    msg = f"Question: {user_query}"
    response = llm.invoke([("system", system_prompt.format(context=context)), ("human", msg)])
    
    if "NOT_FOUND" in response.content:
        print("--- [RAG] LLM could not answer. Fallback. ---")
        return {"rag_fallback": True}

    return {
        "messages": [AIMessage(content=response.content)],
        "rag_fallback": False,
        "error_message": None
    }

# =============================================================================
# 5. VISUALIZATION & UI NODES
# =============================================================================

def node_generate_viz(state):
    """
    Generates Plotly Charts or Markdown Tables.
    """
    print("\n--- [NODE] GENERATE VIZ ---")
    stock_data = state.get("stock_data")
    preference = state.get("output_preference")
    ticker = state.get("ticker", "Stock")

    if not stock_data:
        return {"messages": [AIMessage(content="No data to visualize.")]}

    df = pd.DataFrame.from_dict(stock_data, orient="index")
    df.index = pd.to_datetime(df.index)

    # --- TABLE MODE ---
    if preference == "table":
        cols = ['Close', 'Volume']
        if 'RSI' in df.columns: cols.append('RSI')
        table_md = df[cols].tail(10).to_markdown()
        msg = f"Here is the recent data for **{ticker}**:\n\n{table_md}"
        return {"messages": [AIMessage(content=msg)], "output_preference": None, "stock_data": None}

    # --- PLOT MODE ---
    else:
        output_dir = "charts"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{ticker}_{uuid.uuid4().hex[:8]}.html"
        filepath = os.path.join(output_dir, filename)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
            subplot_titles=(f"{ticker} Price Trend", "RSI Momentum"), row_heights=[0.7, 0.3]
        )

        # Price Trace
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(color="blue")), row=1, col=1)

        # Forecast Trace
        forecast_data = state.get("forecast_data")
        if forecast_data:
            f_dates = list(forecast_data.keys())
            f_prices = list(forecast_data.values())
            # Connector
            f_dates.insert(0, df.index[-1].strftime('%Y-%m-%d'))
            f_prices.insert(0, df['Close'].iloc[-1])
            
            fig.add_trace(go.Scatter(
                x=f_dates, y=f_prices, mode="lines", name="Forecast", 
                line=dict(dash="dot", color="orange"), visible="legendonly"
            ), row=1, col=1)

        # Indicators
        if "SMA_20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(dash="dash", color="green")), row=1, col=1)
        if "RSI" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple"), visible="legendonly"), row=2, col=1)

        fig.update_layout(height=800, title_text=f"{ticker} Technical Analysis", xaxis_rangeslider_visible=False)
        fig.write_html(filepath)
        
        meta = state.get("forecast_meta", {})
        msg = f"I have generated the chart for **{ticker}**. Saved at: `{filepath}`\n\n**Forecast Info:** R²={meta.get('r2_score')}, Trend={meta.get('trend')}"
        
        return {"messages": [AIMessage(content=msg)], "output_preference": None, "stock_data": None}

def node_ask_user(state):
    """Returns error message to user."""
    print("\n--- [NODE] ASK USER ---")
    msg = state.get("error_message") or "I'm not sure how to proceed."
    return {"messages": [AIMessage(content=msg)]}

def node_risk_disclaimer(state):
    """Appends disclaimer if analysis performed."""
    if state.get("stock_data") or state.get("forecast_data"):
        return {"messages": state.get("messages", []) + [AIMessage(content="⚠️ This analysis is for educational purposes only.")]}
    return {}