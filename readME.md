---
title: Trader Companion AI
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.11"
app_file: app.py
app_port: 7860
fullWidth: true
header: default
short_description: Autonomous financial agent.
tags:
  - finance
  - stocks
  - technical-analysis
  - RAG
  - forecast
  - agent
  - langgraph
  - ollama
---


# 📈 Trader Companion AI

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)
![Ollama](https://img.shields.io/badge/AI-Ollama%20\(Local\)-black?logo=ollama)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue?logo=docker)

**Trader Companion AI** is an autonomous financial agent that combines real-time stock analysis with private document retrieval (RAG).

Unlike simple chatbots, this agent uses a **cyclic graph architecture** to self-correct errors. If it can't find a ticker (e.g., "Boralex"), it searches the web, verifies candidates against official listings, checks for name collisions, and validates data availability before answering.

---

## 🧭 High-Level Flow (Agent Graph)

The diagram below illustrates how the agent routes intent, validates tickers, self-corrects failures, and produces analysis or RAG-based answers.

```mermaid
graph TD
    Start([User Input]) --> Extractor[Extract Entities]
    Extractor --> Router{RAG or Web?}
    
    Router -- RAG Active --> RagSearch[Query Vector DB]
    RagSearch -->|Success| End([Response])
    RagSearch -->|Fallback| TickerCheck
    
    Router -- Web Mode --> TickerCheck[Validate Ticker]
    
    TickerCheck -->|Invalid| WebSearch[DuckDuckGo Search]
    WebSearch --> TickerCheck
    
    TickerCheck -->|Valid| DateCheck[Validate Dates]
    DateCheck --> Fetcher[Smart Data Fetcher]
    
    Fetcher -->|News Intent| Sentiment[Sentiment Analysis]
    Fetcher -->|Data Intent| Analyst[Tech Analysis]
    
    Analyst --> Forecast[Linear Regression Forecast]
    Forecast --> Viz[Generate Plotly Chart]
    
    Sentiment --> End
    Viz --> End

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px
    classDef decision fill:#ffefdb,stroke:#f6b26b,stroke-width:2px
    classDef process fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef endNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    
    class Router decision
    class Extractor,RagSearch,TickerCheck,WebSearch,DateCheck,Fetcher,Sentiment,Analyst,Forecast,Viz process
    class Start,End endNode
```

---

## 🧠 System Architecture

The agent is powered by **LangGraph**, enabling non-linear workflows and state persistence. The system operates through a specialized node architecture:

* **Router Logic:** Intelligently decides between pulling real-time market data, searching the web for news, or querying the internal Knowledge Base (PDFs) based on user intent.
* **Self-Correction Loop:** If a stock ticker is invalid or ambiguous, the agent enters a fallback loop—searching the web, verifying exchange suffixes (e.g., converting `.TSX` to `.TO`), and validating against official company names before proceeding.
* **Hybrid Search:** Combines DuckDuckGo for general queries and a local Vector Store (Ollama embeddings) for private document analysis.

---

## ✨ Key Capabilities

### 1. 🛡️ Robust Ticker Resolution

* **Self-Correction:** Automatically maps informal names (e.g., "Ubisoft") to accurate tickers (`UBI.PA`) using a multi-step web search and validation loop.
* **Collision Detection:** Uses fuzzy matching to distinguish between similar tickers (e.g., `BLX` for Boralex vs. Banco Latinoamericano).
* **Suffix Handling:** Automatically converts exchange suffixes (e.g., `.TSX` → `.TO`) for API compatibility.

### 2. 🧠 Smart Routing

* Detects intent to route between **Fundamental Analysis**, **Technical Charts**, **News Sentiment**, or **Internal RAG** queries.
* **Keyword Guards:** Bypasses LLM latency for direct data requests (e.g., "Show me the price of Apple").

### 3. 📊 Interactive Visualization

* Generates dynamic **Plotly** charts with zoom/pan.
* Overlays **SMA (Simple Moving Average)** and **RSI (Relative Strength Index)**.
* Projects a 7-day trend forecast using Linear Regression.

### 4. 📚 Local RAG (Retrieval-Augmented Generation)

* Ingests PDF reports into an in-memory Vector Store.
* Uses **Ollama (nomic-embed-text)** for fully local, private document analysis.

---

## 🛠️ Installation & Setup

### Option 1: Docker (Recommended)

```bash
# Build the image
docker build -t trader-ai .

# Run the container (Exposes port 7860)
docker run -p 7860:7860 trader-ai
```

### Option 2: Local Development

Requires Python 3.11+ and [Ollama](https://ollama.com/) running locally.

1. **Clone and Install**

   ```bash
   git clone https://github.com/yourusername/trader-companion.git
   cd trader-companion
   pip install -r requirements.txt
   ```

2. **Start Ollama**

   ```bash
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ollama serve
   ```

3. **Run Streamlit**

   ```bash
   streamlit run app.py
   ```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_agent.py

# Run specific ticker validation tests
pytest -k "validate_ticker"
```

---

## ⚠️ Disclaimer

*This project is for educational purposes only. The financial forecasts and analysis provided by the AI are based on simple statistical models and should not be used as financial advice.*
