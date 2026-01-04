import requests
import xml.etree.ElementTree as ET
import yfinance as yf
from langchain_core.tools import tool

# Import RAG query function (Adjust path if necessary)
# We use a try/except block or direct import depending on your project structure
try:
    from my_agent.utils.rag import query_rag
except ImportError:
    # Fallback or placeholder if running isolated tests without the full package
    query_rag = None

# =============================================================================
# 1. NEWS RETRIEVAL (Yahoo Finance RSS)
# =============================================================================

@tool
def search_financial_news(ticker: str):
    """
    Fetches the latest official news from Yahoo Finance RSS feed.
    Input should be a ticker (e.g. "STLA", "CVO.TO").
    """
    try:
        # We enforce region=US and lang=en-US to filter out foreign spam.
        # This matches the content you see on the standard Yahoo Quote page.
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        
        # Use a browser-like header to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"Failed to retrieve news. Status: {response.status_code}"

        # Parse the XML data directly
        root = ET.fromstring(response.content)
        
        news_list = []
        # Grab the top 5 stories
        for item in root.findall('.//item')[:5]:
            title = item.find('title').text if item.find('title') is not None else "No Title"
            link = item.find('link').text if item.find('link') is not None else "#"
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            
            # Create a clean string for the LLM
            news_list.append(f"Headline: {title}\nDate: {pub_date}\nLink: {link}\n")

        if not news_list:
            return f"No news found for {ticker} in the RSS feed."

        return "\n---\n".join(news_list)

    except Exception as e:
        return f"Error fetching Yahoo news: {e}"

# =============================================================================
# 2. FUNDAMENTAL DATA (YFinance)
# =============================================================================

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

# =============================================================================
# 3. KNOWLEDGE BASE TRIGGER (RAG)
# =============================================================================

@tool("lookup_knowledge_base")
def lookup_knowledge_base(query: str):
    """
    Useful for answering questions about specific trading strategies, 
    internal documentation, pdfs, agent capabilities, or how the backtester works.
    Use this when the user asks 'how to', 'explain strategy', or about internal concepts.
    """
    # This tool acts as a "Flag" for the router.
    # The actual RAG search happens in the 'node_rag_search' node.
    return "trigger_rag"