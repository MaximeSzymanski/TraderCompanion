import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.linear_model import LinearRegression
import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
import pandas_ta as ta
# Ensure this import matches your project structure
from my_agent.utils.tools import lookup_knowledge_base, search_financial_news, get_company_fundamentals



try:
    if os.path.exists("my_agent/sp500_tickers.csv"):
        SP500_DF = pd.read_csv("my_agent/sp500_tickers.csv")
    else:
        # raise FileNotFoundError
        FileNotFoundError("sp500_tickers.csv not found.")
except Exception as e:
    SP500_DF = pd.DataFrame(columns=['Symbol', 'Security'])
    
# 1. Define the Schema for the LLM to fill
class SearchIntent(BaseModel):
    company_name: Optional[str] = Field(
        None, 
        description="The name of the company or stock symbol mentioned (e.g., 'Apple', 'MSFT')."
    )
    start_date: Optional[str] = Field(
        None, 
        description="The start date for the data in YYYY-MM-DD format. Calculate relative dates (e.g., 'last month') based on today."
    )
    end_date: Optional[str] = Field(
        None, 
        description="The end date for the data in YYYY-MM-DD format."
    )
    visualization_type: Optional[str] = Field(
        None, 
        description="The preferred output format. Must be either 'plot' or 'table'."
    )


def node_extract_entities(state):
    """
    Analyzes the conversation history to extract search parameters.
    """
    print("\n--- [NODE] EXTRACT ENTITIES ---")
    
    # Log the input messages
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
    Verifies the ticker actually exists on Yahoo Finance.
    """
    print("\n--- [NODE] VALIDATE TICKER ---")
    ticker_input = state.get("ticker")
    print(f"--- [LOG] Input Ticker: {ticker_input}")

    if not ticker_input:
        print("--- [LOG] No ticker provided. Returning error.")
        return {"error_message": "No company name provided."}

    resolved_ticker = ticker_input.strip().upper()
    print(f"--- [LOG] Normalized Ticker: {resolved_ticker}")
    
    if not SP500_DF.empty:
        mask = (SP500_DF['Security'].str.contains(ticker_input, case=False, na=False) | 
                SP500_DF['Symbol'].str.contains(ticker_input, case=False, na=False))
        print(f"--- [LOG] Searching CSV for matches...")
        print(SP500_DF[mask])
        match = SP500_DF[mask]
        if not match.empty:
            resolved_ticker = match.iloc[0]['Symbol']
            print(f"--- [LOG] Found in CSV Match: {resolved_ticker}")

    try:
        print(f"--- [LOG] verifying {resolved_ticker} with yfinance...")
        stock = yf.Ticker(resolved_ticker)
        history = stock.history(period="1d")
        
        if history.empty:
            error_msg = f"I couldn't find a valid stock for '{ticker_input}'. Did you mean '{resolved_ticker}'?"
            print(f"--- [LOG] Verification Failed. {error_msg}")
            return {"ticker": None, "error_message": error_msg}
            
        print("--- [LOG] Ticker Validated Successfully.")
    except Exception as e:
        print(f"--- [ERROR] yfinance Verification Error: {e}")
        return {"ticker": None, "error_message": f"Error validating ticker: {str(e)}"}

    return {
        "ticker": resolved_ticker, 
        "error_message": None 
    }
    
def node_validate_dates(state):
    """
    Validates start/end dates. 
    """
    print("\n--- [NODE] VALIDATE DATES ---")
    start_date_str = state.get("start_date")
    end_date_str = state.get("end_date")
    preference = state.get("output_preference")
    print(f"--- [LOG] Checking Dates: Start={start_date_str}, End={end_date_str}")
    
    updates = {}
    error_msgs = []
    
    if not start_date_str and not end_date_str:
        if preference :
            return {"error_message": "Pleasae provide start and end dates."}
        else:
            print("--- [LOG] No dates provided. But chart no reqeusted. Passing trhough.")
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
        print(f"--- [LOG] Date Validation Errors: {error_msgs}")
        return {"error_message": " ".join(error_msgs)}
    
    print("--- [LOG] Dates Validated Successfully.")
    updates["error_message"] = None
    return updates
def node_fetch_data(state):
    """
    Smart Router: Decides between News, Fundamentals, RAG (Internal Docs), or Stock Data.
    """
    print("\n--- [NODE] FETCH DATA (ROUTER) ---")
    ticker = state.get("ticker")
    start_date = state.get("start_date")
    end_date = state.get("end_date")
    existing_preference = state.get("output_preference")
    
    # If the user explicitly asked for a plot/table, we skip the router and go straight to data
    if existing_preference in ["plot", "table"]:
        print(f"--- [LOG] Preference '{existing_preference}' found. Defaulting to Data Fetch.")
        should_fetch_data = True
    else:
        print("--- [LOG] Preference unknown. Asking Router LLM...")
        
        # 1. Bind all available tools including the new RAG tool
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        llm_with_tools = llm.bind_tools([
            search_financial_news, 
            get_company_fundamentals, 
            lookup_knowledge_base
        ])    
        
        last_msg = state['messages'][-1].content
        
        # Context string to help the LLM decide
        msg = (
            f"User Input: '{last_msg}'\n"
            f"Context: Ticker={ticker}, DateRange={start_date} to {end_date}.\n"
            "Decide the best tool to use. If they ask about strategy/docs, use lookup_knowledge_base. "
            "If they ask for price/chart/table, do not use a tool."
        )
        
        response = llm_with_tools.invoke([HumanMessage(content=msg)])
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"--- [LOG] LLM Decided: USE TOOL '{tool_name}'")
            
            # --- CASE A: RAG / KNOWLEDGE BASE ---
            if tool_name == "lookup_knowledge_base":
                # We return a flag/key that the Graph Edges will use to route to 'node_rag_search'
                # Note: You must update your conditional edges in the main graph file to look for 'is_rag_query'
                return {
                    "is_rag_query": True,
                    "messages": [AIMessage(content="Checking internal knowledge base...")]
                }

            # --- CASE B: NEWS ---
            elif tool_name == "search_financial_news":
                print(f"--- [LOG] Invoking News Tool with: {tool_args}")
                tool_result = search_financial_news.invoke(tool_args)
                return {
                    "news_summary": tool_result, 
                    "messages": [AIMessage(content=f"Searching news for {ticker}...")]
                }

            # --- CASE C: FUNDAMENTALS ---
            elif tool_name == "get_company_fundamentals":
                print(f"--- [LOG] Invoking Fundamentals Tool with: {tool_args}")
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
            # No tool selected -> Default to Data Fetching (Prices/Volume)
            print(f"--- [LOG] LLM Decided: FETCH STOCK DATA.")
            should_fetch_data = True

    # --- DEFAULT PATH: STOCK DATA FETCHING ---
    if should_fetch_data:
        if not ticker or not start_date or not end_date:
            print("--- [LOG] Missing parameters for data fetch.")
            return {"error_message": "Missing ticker or date range for data fetch."}
        
        try:
            print(f"--- [LOG] Fetching yfinance history for {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return {"error_message": f"No data found for {ticker}."}
            
            hist.index = hist.index.strftime('%Y-%m-%d')
            data_serializable = hist.to_dict(orient="index")
            
            return {
                "stock_data": data_serializable,
                "error_message": None
            }
            
        except Exception as e:
            return {"error_message": f"Error fetching data: {str(e)}"}

def node_clarify_preference(state):
    """
    Checks if the user has specified a visualization format.
    """
    print("\n--- [NODE] CLARIFY PREFERENCE ---")
    preference = state.get("output_preference")
    print(f"--- [LOG] Current Preference: {preference}")
    
    if preference:
        pref_clean = preference.lower().strip()
        if pref_clean in ["plot", "chart", "graph", "drawing"]:
            print("--- [LOG] Standardized to 'plot'")
            return {"output_preference": "plot", "error_message": None}
        elif pref_clean in ["table", "list", "data", "rows", "csv"]:
            print("--- [LOG] Standardized to 'table'")
            return {"output_preference": "table", "error_message": None}

    print("--- [LOG] Preference unclear. Setting error message to ask user.")
    return {
        "output_preference": None,
        "error_message": "Would you like to see a plot or a table?"
    }

def node_ask_user(state):
    """
    Formats the error/question as an AI Message.
    """
    print("\n--- [NODE] ASK USER ---")
    message_text = state.get("error_message")
    print(f"--- [LOG] Asking User: '{message_text}'")
    
    if not message_text:
        message_text = "I'm not sure how to proceed. Could you provide more details?"

    return {
        "messages": [AIMessage(content=message_text)],
    }

def node_generate_viz(state):
    print("\n--- [NODE] GENERATE VIZ (INTERACTIVE) ---")
    stock_data = state.get("stock_data")
    preference = state.get("output_preference")
    ticker = state.get("ticker", "Stock")

    if not stock_data:
        return {"messages": [AIMessage(content="No data to visualize.")]}

    df = pd.DataFrame.from_dict(stock_data, orient="index")
    df.index = pd.to_datetime(df.index)

    # --- TABLE MODE ---
    if preference == "table":
        cols_to_show = ['Close', 'Volume']
        if 'RSI' in df.columns: cols_to_show.append('RSI')
        if 'SMA_20' in df.columns: cols_to_show.append('SMA_20')
        
        table_md = df[cols_to_show].tail(10).to_markdown()
        msg = f"Here is the recent market data for **{ticker}**:\n\n{table_md}"
        return {"messages": [AIMessage(content=msg)], "output_preference": None, "stock_data": None}

    # --- PLOT MODE (INTERACTIVE) ---
    else:
        output_dir = "charts"
        os.makedirs(output_dir, exist_ok=True)
        
        # NOTE: We save as .html now, not .png
        filename = f"{ticker}_{uuid.uuid4().hex[:8]}.html"
        filepath = os.path.join(output_dir, filename)

        # Create Subplots (Row 1 = Price, Row 2 = RSI)
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1, 
            subplot_titles=(f"{ticker} Price Trend", "RSI Momentum"),
            row_heights=[0.7, 0.3]
        )

        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')), row=1, col=1)

        # --- NEW: ADD FORECAST TRACE ---
        forecast_data = state.get("forecast_data")
        if forecast_data:
            # Convert dict to lists
            f_dates = list(forecast_data.keys())
            f_prices = list(forecast_data.values())
            
            # Add a "connector" line from the last real price to the first predicted price
            # This makes the line look continuous
            f_dates.insert(0, df.index[-1].strftime('%Y-%m-%d'))
            f_prices.insert(0, df['Close'].iloc[-1])
            
            fig.add_trace(go.Scatter(
                x=f_dates, 
                y=f_prices, 
                mode='lines', 
                name='7-Day Forecast', 
                line=dict(color='green', dash='dot', width=2)
            ))
        # 2. SMA (if exists)
        if 'SMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA (20)', line=dict(color='orange', dash='dash')), row=1, col=1)

        # 3. RSI (if exists)
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
            # Add Overbought/Oversold lines
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1, annotation_text="Oversold")

        # Layout Polish
        fig.update_layout(
            height=800, 
            title_text=f"{ticker} Technical Analysis", 
            showlegend=True,
            xaxis_rangeslider_visible=False # Hide the bottom slider to save space
        )
        
        # Save to HTML
        fig.write_html(filepath)
        print(f"--- [LOG] Interactive chart saved to {filepath}")

        msg = f"I have generated the interactive chart for **{ticker}**. Saved at: `{filepath}`"
        return {"messages": [AIMessage(content=msg)], "output_preference": None, "stock_data": None}
    
    
def node_technical_analysis(state):
    """
    Analyzes stock data for technical indicators (Placeholder), like RSI and SMA.
    """
    print("\n--- [NODE] TECHNICAL ANALYSIS ---")
    stock_data = state.get("stock_data")
    ticker = state.get("ticker", "Stock")
    
    if not stock_data:
        print("--- [LOG] No stock data found in state for analysis.")
        return {"messages": [AIMessage(content="No stock data available for technical analysis.")]}
    
    df = pd.DataFrame.from_dict(stock_data, orient="index")
    df.index = pd.to_datetime(df.index)
    
    # Placeholder for actual technical analysis logic
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    
    df = df.fillna(0)
    
    df.index = df.index.strftime('%Y-%m-%d')
    
    print("--- [LOG] Technical analysis completed.")
    return {
        "stock_data": df.to_dict(orient="index"),
        "messages": [AIMessage(content=f"Technical analysis for **{ticker}** completed.")]
    }
    
def node_sentiment_analysis(state):
    print("\n--- [NODE] SENTIMENT ANALYSIS ---")
    news_text = state.get("news_summary")
    if not news_text:
        print("--- [LOG] No news summary found in state for sentiment analysis.")
        return {}

    # 1. Initialize LLM (Force JSON output)
    llm = ChatOllama(model="qwen2.5:7b", temperature=0, format="json")
    
    # 2. Prompt
    system = """You are a veteran financial analyst. 
    Analyze the following news summary and determine the market sentiment.
    Return a JSON with:
    - "score": A float from -1.0 (Very Bearish) to 1.0 (Very Bullish).
    - "verdict": A short string (e.g., "Bullish", "Neutral", "Bearish").
    - "explanation": A 1-sentence explanation of why.
    """
    
    msg = f"News Summary: {news_text}"
    
    # 3. Invoke
    # (We use a simple invoke here since we just want a raw JSON string to parse)
    # Ideally, reuse your JsonOutputParser pattern, but for brevity:
    try:
        response = llm.invoke([
            ("system", system),
            ("human", msg)
        ])
        import json
        analysis = json.loads(response.content)
        
        # Format a nice message for the user
        formatted_msg = (
            f"**Sentiment Verdict:** {analysis['verdict']} (Score: {analysis['score']})\n\n"
            f"_{analysis['explanation']}_\n\n"
            f"---\n**News Source:**\n{news_text[:500]}..." # Truncate raw news
        )
        print("--- [LOG] Sentiment analysis completed.")
        return {
            "sentiment": analysis,
            "messages": [AIMessage(content=formatted_msg)],
            "news_summary": None # Clear raw news to save memory
        }
    except Exception as e:
        return {"error_message": f"Sentiment analysis failed: {e}"}
    
def  node_forecast(state):
    print("\n--- [NODE] FORECASTING ---")
    stock_data = state.get("stock_data")
    if not stock_data:
        return {}
    
    # 1. Prepare Data
    df = pd.DataFrame.from_dict(stock_data, orient="index")
    df.index = pd.to_datetime(df.index)
    
    # Use ordinal dates for regression (math needs numbers, not date objects)
    df['Date_Ordinal'] = df.index.map(pd.Timestamp.toordinal)
    
    # Train on the last 30 days only (for short-term trend relevance)
    recent_df = df.tail(30)
    X = recent_df[['Date_Ordinal']].values
    y = recent_df['Close'].values
    
    # 2. Fit Model (Simple Linear Regression)
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Predict Future (Next 7 days)
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    predictions = model.predict(future_ordinals)
    
    # 4. Format for State
    # We store forecast as a separate dict to keep 'stock_data' clean
    forecast_dict = {
        date.strftime('%Y-%m-%d'): float(pred)
        for date, pred in zip(future_dates, predictions)
    }
    
    print(f"--- [LOG] Generated forecast for next {len(forecast_dict)} days.")
    return {"forecast_data": forecast_dict}


from my_agent.utils.rag import query_rag

def node_rag_search(state):
    """
    Queries the vector database for internal context.
    If context is found, answers the user.
    If not, sets a flag to fall back to web search.
    """
    print("\n--- [NODE] RAG SEARCH ---")
    messages = state.get("messages")
    # We use the last message as the query
    user_query = messages[-1].content
    
    # 1. Retrieve Context
    print(f"--- [LOG] Querying RAG with: {user_query}")
    context = query_rag(user_query)
    
    # CASE A: No relevant text chunks found in DB
    if not context:
        print("--- [RAG] No context found. Falling back to Web. ---")
        return {"rag_fallback": True} # The graph should catch this and route to Web Search

    # 2. Synthesize Answer
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    
    # We tell the LLM to output a specific token if it doesn't know
    system_prompt = """You are a documentation expert for this trading agent. 
    Answer the user's question STRICTLY based on the provided Context chunks below.
    
    If the answer is NOT in the context, output exactly: "NOT_FOUND"
    
    Context:
    {context}
    """
    
    msg = f"Question: {user_query}"
    
    print("--- [LOG] Synthesizing RAG Answer...")
    response = llm.invoke([("system", system_prompt.format(context=context)), ("human", msg)])
    
    # CASE B: Context found, but LLM says it's irrelevant
    if "NOT_FOUND" in response.content:
        print("--- [RAG] LLM could not answer based on context. Falling back to Web. ---")
        return {"rag_fallback": True}
    print("--- [RAG] LLM provided an answer based on context. ---")
    # CASE C: Success
    return {
        "messages": [AIMessage(content=response.content)],
        "rag_fallback": False,
        "error_message": None
    }
    

def node_company_profile(state):
    print("--- [NODE] COMPANY PROFILE ---")
    
    # 1. Get the data from state
    stock_data = state.get("stock_data")
    ticker = state.get("ticker")
    
    # Safety Check: If no info was found, skip silently
    if not stock_data or "info" not in stock_data:
        return {} 
        
    info = stock_data["info"]
    
    # 2. Extract & Format Fields
    name = info.get("longName", ticker)
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    
    # Handle Summary (Truncate if too long)
    summary = info.get("longBusinessSummary", "No summary available.")
    if len(summary) > 400:
        summary = summary[:400] + "..."
        
    # Handle Financials (Convert to Billions)
    mkt_cap = info.get("marketCap", 0)
    if mkt_cap:
        mkt_cap_str = f"${mkt_cap / 1e9:.2f}B"
    else:
        mkt_cap_str = "N/A"
        
    pe_ratio = info.get("trailingPE", "N/A")
    if isinstance(pe_ratio, float):
        pe_ratio = f"{pe_ratio:.2f}"
    
    # 3. Create the Markdown Message
    profile_text = (
        f"### 🏢 {name}\n"
        f"**Sector:** {sector} | **Industry:** {industry} | **Market Cap:** {mkt_cap_str} | **P/E:** {pe_ratio}\n\n"
        f"_{summary}_"
    )
    
    # 4. Return as a separate message
    return {"messages": [AIMessage(content=profile_text)]}


