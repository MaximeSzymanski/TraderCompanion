from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import yfinance as yf
@tool
def search_financial_news(query: str):
    """
    Searches for financial news and reasons behind stock movements.
    Use this when the user asks 'Why', 'News', or 'What happened'.
    """
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

@tool
def get_company_fundamentals(ticker: str):
    """
    Retrieves fundamental data for a company (P/E ratio, Market Cap, Sector, Business Summary).
    Useful when the user asks "what does this company do?", "is it overvalued?", or about its valuation/sector.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Format the numbers nicely
        market_cap = info.get("marketCap", "N/A")
        if isinstance(market_cap, int):
            market_cap = f"${market_cap / 1e9:.2f} Billion"
            
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": market_cap,
            "pe_ratio": info.get("trailingPE", "N/A"),
            "summary": info.get("longBusinessSummary", "No summary available.")[:400] + "..." # Truncate long text
        }
    except Exception as e:
        return f"Error getting fundamentals for {ticker}: {e}"
    
    
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