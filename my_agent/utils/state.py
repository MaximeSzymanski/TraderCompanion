from typing import TypedDict, List, Union
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]

    ticker: Union[str, None]
    start_date: Union[str, None]
    end_date: Union[str, None]

    stock_data: Union[dict, None]
    forecast_data: Union[dict, None]

    output_preference: Union[str, None]
    error_message: Union[str, None]

    # NEWS / FUNDAMENTALS
    want_news: Union[bool, None]
    want_informations: Union[bool, None]
    news_summary: Union[str, None]
    sentiment: Union[dict, None]

    # POST-PLOT FOLLOWUP
    ask_news_after_plot: Union[bool, None]

    # RAG (UNCHANGED)
    rag_context: Union[str, None]
    is_rag_active: Union[bool, None]
    rag_fallback: bool
