import pytest
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage

# --- IMPORT YOUR MODULES ---
# Adjust paths if your folder structure is different
from my_agent.utils.nodes import (
    node_extract_entities,
    node_validate_ticker,
    node_validate_dates,
    node_fetch_data,
    node_technical_analysis,
    node_sentiment_analysis,
    node_forecast,
    node_rag_search,
    node_company_profile,
    node_clarify_preference,
    node_generate_viz,
    node_ask_user,
    node_risk_disclaimer
)
# Import tools explicitly to test them
from my_agent.utils.tools import search_financial_news, get_company_fundamentals

# =============================================================================
# 0. FIXTURES
# =============================================================================

@pytest.fixture
def base_state():
    """Returns a fresh, empty state for every test."""
    return {
        "messages": [HumanMessage(content="Test message")],
        "ticker": None,
        "start_date": None,
        "end_date": None,
        "stock_data": None,
        "forecast_data": None,
        "output_preference": None,
        "error_message": None,
        "news_summary": None,
        "rag_fallback": False,
        "is_rag_active": False
    }

@pytest.fixture
def mock_stock_data():
    """Creates a dummy pandas DataFrame resembling yfinance data."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    data = {
        "Open": np.random.rand(50) * 100,
        "High": np.random.rand(50) * 100,
        "Low": np.random.rand(50) * 100,
        "Close": np.linspace(100, 150, 50), # Linear upward trend for forecast testing
        "Volume": np.random.randint(1000, 5000, 50)
    }
    df = pd.DataFrame(data, index=dates)
    return df.to_dict(orient="index")

# =============================================================================
# 1. TEST ENTITY EXTRACTION
# =============================================================================

@patch("my_agent.utils.nodes.ChatOllama")
def test_extract_entities_success(mock_chat, base_state):
    """Test successful JSON extraction from LLM."""
    base_state["messages"] = [HumanMessage(content="Analyze Apple from 2023-01-01 to 2023-02-01")]
    
    # Mock chain execution: chain.invoke(...) -> returns dict
    # Since the node constructs the chain dynamically, we mock the final parser output
    # But since it's hard to mock the pipe '|', we mock the invoke call on the result of the prompt.
    # ALTERNATIVE: Mock ChatOllama to return a JSON string, and the real parser handles it?
    # Simpler: The node code calls `chain.invoke`. Let's assume we can patch the chain construction.
    # Given the complexity of patching inside the function, let's mock the `chain.invoke` return value 
    # by patching the class method if possible, or just the LLM response if the chain is robust.
    
    # Let's try mocking the object returned by the chain construction:
    with patch("my_agent.utils.nodes.ChatPromptTemplate") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "company_name": "Apple",
            "start_date": "2023-01-01",
            "end_date": "2023-02-01",
            "visualization_type": "plot"
        }
        # We make the pipe operator return our mock chain
        mock_prompt.from_messages.return_value.partial.return_value.__or__.return_value.__or__.return_value = mock_chain
        
        result = node_extract_entities(base_state)
        
        assert result["ticker"] == "Apple"
        assert result["start_date"] == "2023-01-01"
        assert result["output_preference"] == "plot"

@patch("my_agent.utils.nodes.ChatOllama")
def test_extract_entities_failure_handling(mock_chat, base_state):
    """Test that it doesn't crash if LLM fails."""
    with patch("my_agent.utils.nodes.ChatPromptTemplate") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM Timeout")
        mock_prompt.from_messages.return_value.partial.return_value.__or__.return_value.__or__.return_value = mock_chain
        
        result = node_extract_entities(base_state)
        assert result == {} # Graceful exit

# =============================================================================
# 2. TEST TICKER VALIDATION (The Complex Node)
# =============================================================================

@patch("my_agent.utils.nodes.SP500_DF", pd.DataFrame({'Symbol': ['AAPL'], 'Security': ['Apple Inc.']}))
def test_validate_ticker_sp500_fast_path(base_state):
    """Test fast lookup in S&P 500 CSV."""
    base_state["ticker"] = "Apple"
    result = node_validate_ticker(base_state)
    assert result["ticker"] == "AAPL"
    assert result["error_message"] is None

@patch("my_agent.utils.nodes.SP500_DF", pd.DataFrame(columns=['Symbol', 'Security']))
@patch("my_agent.utils.nodes.ChatOllama")
@patch("my_agent.utils.nodes.DuckDuckGoSearchRun")
@patch("my_agent.utils.nodes.yf.Ticker")
def test_validate_ticker_collision_boralex(mock_ticker, mock_search, mock_chat, base_state):
    """
    CRITICAL TEST: The Boralex vs Banco Latinoamericano collision.
    1. Web Search finds "BLX".
    2. Loop 1: Checks BLX. Info says "Banco". Mismatch.
    3. Loop 2: Checks BLX.TO. Info says "Boralex". Match.
    """
    base_state["ticker"] = "Boralex"
    
    # 1. Web Search finds the generic ticker
    mock_search.return_value.invoke.return_value = "Boralex Inc (BLX) Stock..."
    mock_chat.return_value.invoke.return_value = MagicMock(content="BLX")

    # 2. Mock YFinance for different calls
    # Mock Object for BLX (Banco)
    mock_banco = MagicMock()
    mock_banco.history.return_value = pd.DataFrame({'Close': [10]})
    mock_banco.info = {"longName": "Banco Latinoamericano de Comercio Exterior, S. A."}

    # Mock Object for BLX.TO (Boralex)
    mock_boralex = MagicMock()
    mock_boralex.history.return_value = pd.DataFrame({'Close': [20]})
    mock_boralex.info = {"longName": "Boralex Inc."}

    # Side Effect: Return Banco first, then Boralex when suffix added
    def side_effect(ticker):
        if ticker == "BLX": return mock_banco
        if ticker == "BLX.TO": return mock_boralex
        return MagicMock(history=lambda period: pd.DataFrame()) # Return empty for others

    mock_ticker.side_effect = side_effect

    # Run
    result = node_validate_ticker(base_state)

    # Assert
    assert result["ticker"] == "BLX.TO"
    assert result["error_message"] is None

@patch("my_agent.utils.nodes.SP500_DF", pd.DataFrame(columns=['Symbol', 'Security']))
@patch("my_agent.utils.nodes.ChatOllama")
@patch("my_agent.utils.nodes.DuckDuckGoSearchRun")
@patch("my_agent.utils.nodes.yf.Ticker")
def test_validate_ticker_retry_root(mock_ticker, mock_search, mock_chat, base_state):
    """
    Test logic: If STLA.PA fails, try STLA (Root).
    """
    base_state["ticker"] = "Stellantis"
    
    # Web search says STLA.PA
    mock_search.return_value.invoke.return_value = "..."
    mock_chat.return_value.invoke.return_value = MagicMock(content="STLA.PA")

    # STLA.PA fails (empty history), STLA succeeds
    mock_fail = MagicMock()
    mock_fail.history.return_value = pd.DataFrame()
    
    mock_success = MagicMock()
    mock_success.history.return_value = pd.DataFrame({'Close': [10]})
    mock_success.info = {"longName": "Stellantis N.V."}

    mock_ticker.side_effect = lambda t: mock_success if t == "STLA" else mock_fail

    result = node_validate_ticker(base_state)
    # The robust node should strip .PA and try STLA
    assert result["ticker"] == "STLA"

# =============================================================================
# 3. TEST DATE VALIDATION
# =============================================================================

def test_validate_dates_missing_inputs(base_state):
    """If dates are missing and no preference is set, pass."""
    base_state["ticker"] = "AAPL"
    result = node_validate_dates(base_state)
    assert result["error_message"] is None

def test_validate_dates_missing_inputs_with_plot_pref(base_state):
    """If preference is PLOT, missing dates is an error."""
    base_state["output_preference"] = "plot"
    result = node_validate_dates(base_state)
    assert "provide start and end dates" in result["error_message"]

def test_validate_dates_future_error(base_state):
    future = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
    base_state["start_date"] = "2023-01-01"
    base_state["end_date"] = future
    result = node_validate_dates(base_state)
    assert "cannot be in the future" in result["error_message"]

def test_validate_dates_start_after_end(base_state):
    base_state["start_date"] = "2023-02-01"
    base_state["end_date"] = "2023-01-01"
    result = node_validate_dates(base_state)
    assert "Start date cannot be after end date" in result["error_message"]

# =============================================================================
# 4. TEST DATA FETCHING & ROUTING (Keyword Guards & RSS)
# =============================================================================

def test_fetch_data_keyword_guard(base_state):
    """Test that 'price' keyword skips LLM router."""
    base_state["ticker"] = "AAPL"
    base_state["start_date"] = "2023-01-01"
    base_state["end_date"] = "2023-01-02"
    base_state["messages"] = [HumanMessage(content="What is the price history?")] # Contains 'price'

    with patch("my_agent.utils.nodes.yf.Ticker") as mock_ticker:
        with patch("my_agent.utils.nodes.ChatOllama") as mock_chat:
            mock_stock = MagicMock()
            mock_stock.history.return_value = pd.DataFrame({'Close': [100]}, index=pd.to_datetime(["2023-01-01"]))
            mock_ticker.return_value = mock_stock

            result = node_fetch_data(base_state)
            
            # ChatOllama should NOT be called because of keyword guard
            mock_chat.assert_not_called()
            assert result["stock_data"] is not None

@patch("my_agent.utils.nodes.search_financial_news")
@patch("my_agent.utils.nodes.ChatOllama")
def test_fetch_data_routes_to_news(mock_chat, mock_tool, base_state):
    """Test routing to news tool."""
    base_state["messages"] = [HumanMessage(content="Why is it down?")]
    base_state["ticker"] = "AAPL"

    # Mock Router Response
    mock_msg = MagicMock()
    mock_msg.tool_calls = [{"name": "search_financial_news", "args": {"ticker": "AAPL"}}]
    mock_chat.return_value.bind_tools.return_value.invoke.return_value = mock_msg

    # Mock Tool execution
    mock_tool.invoke.return_value = "Fake RSS Data"

    result = node_fetch_data(base_state)
    
    assert result["news_summary"] == "Fake RSS Data"
    mock_tool.invoke.assert_called_with("AAPL") # Ensure ticker was passed, not search string

@patch("my_agent.utils.nodes.get_company_fundamentals")
@patch("my_agent.utils.nodes.ChatOllama")
def test_fetch_data_routes_to_fundamentals(mock_chat, mock_tool, base_state):
    """Test routing to fundamentals."""
    base_state["messages"] = [HumanMessage(content="What does this company do?")]
    base_state["ticker"] = "AAPL"

    mock_msg = MagicMock()
    mock_msg.tool_calls = [{"name": "get_company_fundamentals", "args": {"ticker": "AAPL"}}]
    mock_chat.return_value.bind_tools.return_value.invoke.return_value = mock_msg

    mock_tool.invoke.return_value = {"name": "Apple", "summary": "Tech stuff", "pe_ratio": 30}

    result = node_fetch_data(base_state)
    assert isinstance(result["messages"][0], AIMessage)
    assert "Tech stuff" in result["messages"][0].content

@patch("my_agent.utils.nodes.yf.Ticker")
@patch("my_agent.utils.nodes.ChatOllama")
def test_fetch_data_missing_params(mock_chat, mock_ticker, base_state):
    """Test error when routing to data but dates are missing."""
    base_state["messages"] = [HumanMessage(content="Show me price")] # triggers keyword guard
    base_state["ticker"] = "AAPL"
    base_state["start_date"] = None # Missing

    result = node_fetch_data(base_state)
    assert "Missing ticker or date range" in result["error_message"]

# =============================================================================
# 5. TEST RSS & FUNDAMENTALS TOOLS (Unit Tests)
# =============================================================================

@patch("requests.get")
def test_tool_rss_success(mock_get):
    """Test the RSS parser tool."""
    xml = """<rss><channel><item>
             <title>Test News</title><link>http://link</link><pubDate>Mon</pubDate>
             </item></channel></rss>"""
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = xml.encode()

    result = search_financial_news.invoke("AAPL")
    assert "Test News" in result

@patch("requests.get")
def test_tool_rss_failure(mock_get):
    """Test RSS network failure."""
    mock_get.return_value.status_code = 404
    result = search_financial_news.invoke("AAPL")
    assert "Failed to retrieve" in result

@patch("my_agent.utils.tools.yf.Ticker")
def test_tool_fundamentals(mock_ticker):
    """Test fundamentals extraction."""
    mock_info = {"longName": "Test Co", "marketCap": 1000000000, "sector": "Tech"}
    mock_ticker.return_value.info = mock_info
    
    result = get_company_fundamentals.invoke("TEST")
    assert result["name"] == "Test Co"
    assert "Billion" in result["market_cap"]

# =============================================================================
# 6. TEST MATH NODES (Forecast & Tech Analysis)
# =============================================================================

def test_forecast_linear_regression(base_state, mock_stock_data):
    """Test that forecasting produces valid numbers and R2 score."""
    base_state["stock_data"] = mock_stock_data
    
    result = node_forecast(base_state)
    
    assert "forecast_data" in result
    assert "forecast_meta" in result
    meta = result["forecast_meta"]
    # Since mock data is perfectly linear (linspace), R2 should be close to 1.0
    assert meta["r2_score"] > 0.95 
    assert meta["trend"] == "upward"
    assert len(result["forecast_data"]) == 29

def test_technical_analysis_calcs(base_state, mock_stock_data):
    """Test RSI and SMA addition."""
    base_state["stock_data"] = mock_stock_data
    result = node_technical_analysis(base_state)
    first_row = list(result["stock_data"].values())[-1]
    assert "RSI" in first_row
    assert "SMA_20" in first_row

@patch("my_agent.utils.nodes.ChatOllama")
def test_sentiment_analysis(mock_chat, base_state):
    """Test sentiment JSON parsing."""
    base_state["news_summary"] = "Good news"
    
    # Mock LLM returning JSON string
    mock_chat.return_value.invoke.return_value = AIMessage(content='{"score": 0.8, "verdict": "Bullish", "explanation": "Good stuff"}')
    
    result = node_sentiment_analysis(base_state)
    assert result["sentiment"]["score"] == 0.8
    assert result["news_summary"] is None # Should be cleared

# =============================================================================
# 7. TEST RAG & OTHER NODES
# =============================================================================

@patch("my_agent.utils.nodes.query_rag")
@patch("my_agent.utils.nodes.ChatOllama")
def test_rag_success(mock_chat, mock_query, base_state):
    """Test successful retrieval."""
    base_state["messages"] = [HumanMessage(content="What is in the doc?")]
    mock_query.return_value = "Context Chunk 1..."
    mock_chat.return_value.invoke.return_value = AIMessage(content="The doc says X.")
    
    result = node_rag_search(base_state)
    assert result["rag_fallback"] is False

@patch("my_agent.utils.nodes.query_rag")
def test_rag_no_context(mock_query, base_state):
    """Test fallback when no context found."""
    base_state["messages"] = [HumanMessage(content="Unknown")]
    mock_query.return_value = None
    result = node_rag_search(base_state)
    assert result["rag_fallback"] is True

def test_company_profile(base_state):
    """Test company profile formatting."""
    base_state["ticker"] = "AAPL"
    base_state["stock_data"] = {"info": {"longName": "Apple", "marketCap": 2000000000}}
    result = node_company_profile(base_state)
    assert "Apple" in result["messages"][0].content
    assert "$2.00B" in result["messages"][0].content

def test_risk_disclaimer(base_state):
    """Test disclaimer triggers only if signals exist."""
    # 1. Test Negative Case (No data -> No disclaimer)
    base_state["signals"] = None
    base_state["forecast_data"] = None
    assert node_risk_disclaimer(base_state) == {}

    # 2. Test Positive Case (Forecast exists -> Add disclaimer)
    base_state["forecast_data"] = {"2024-01-01": 100}
    
    # Run Node
    result = node_risk_disclaimer(base_state)
    
    # Assert
    # FIX: Check the LAST message [-1], not the first one [0]
    last_message = result["messages"][-1]
    
    assert isinstance(last_message, AIMessage)
    assert "educational purposes" in last_message.content

def test_clarify_preference(base_state):
    """Test preference standardization."""
    base_state["output_preference"] = "graph"
    assert node_clarify_preference(base_state)["output_preference"] == "plot"
    
    base_state["output_preference"] = "unknown"
    result = node_clarify_preference(base_state)
    assert result["output_preference"] is None
    assert result["error_message"] is not None

# =============================================================================
# 8. TEST VISUALIZATION (Plotly & Table)
# =============================================================================

@patch("my_agent.utils.nodes.os.makedirs")
@patch("my_agent.utils.nodes.go.Figure.write_html")
def test_generate_viz_plot(mock_write, mock_dirs, base_state, mock_stock_data):
    """Test interactive plot generation."""
    base_state["stock_data"] = mock_stock_data
    base_state["output_preference"] = "plot"
    base_state["ticker"] = "TEST"
    
    result = node_generate_viz(base_state)
    assert "charts/" in result["messages"][0].content
    mock_write.assert_called_once()

def test_generate_viz_table(base_state, mock_stock_data):
    """Test markdown table generation."""
    base_state["stock_data"] = mock_stock_data
    base_state["output_preference"] = "table"
    
    result = node_generate_viz(base_state)
    assert "Close" in result["messages"][0].content
    assert "|" in result["messages"][0].content