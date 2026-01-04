# 📈 Trader Companion AI

A toy but professional **AI-powered trading assistant** built with **LangGraph, LangChain, Streamlit, and Plotly**.

Trader Companion AI lets you explore stocks, generate interactive technical charts, run basic forecasts, analyze sentiment, and query your own documents — all through a conversational interface.

> ⚠️ Educational project only. This tool does **not** provide financial advice.

---

## ✨ Features Overview

### 1. Conversational Stock Analysis

Ask natural language questions such as:

* "Analyze Amazon from 2024-01-01 to 2024-03-01"
* "Show me Nvidia technicals"
* "Plot Apple with RSI and forecast"

The agent automatically:

* Extracts tickers and dates
* Validates symbols via Yahoo Finance
* Fetches historical market data

---

### 2. Interactive Technical Charts (Plotly)

The agent generates **interactive HTML charts** with:

* **Close Price**
* **Simple Moving Average (SMA 20)**
* **Relative Strength Index (RSI)**
* **Linear Regression Forecast** (short-term)

Chart characteristics:

* One color per indicator
* RSI and Forecast hidden by default (toggle via legend)
* Shared time axis
* Saved as standalone `.html` files

#### Example Plot Layout

* Top panel: Price, SMA, Forecast
* Bottom panel: RSI

```
| Price + SMA + Forecast |
|------------------------|
| RSI Momentum           |
```

---

### 3. Technical Indicators

The agent computes technical indicators automatically:

* **RSI (14)** using `pandas-ta`
* **SMA (20)**

Indicators are added to both:

* Interactive charts
* Tabular output (when requested)

---

### 4. Short-Term Forecasting

A simple but transparent forecasting module:

* Uses **Linear Regression** on the last 30 trading days
* Predicts up to **30 future days**
* Displays:

  * Forecast curve
  * R² score
  * Trend direction (upward / downward)

The forecast is:

* Clearly separated from historical prices
* Hidden by default
* Labeled as educational

---

### 5. Market Sentiment Analysis (News)

When news is requested, the agent:

1. Fetches recent financial news
2. Runs LLM-based sentiment analysis
3. Outputs:

   * Sentiment score (-1 to +1)
   * Verdict (Bullish / Neutral / Bearish)
   * Short explanation

---

### 6. Company Fundamentals

For company-level questions, the agent can display:

* Sector and industry
* Market capitalization
* P/E ratio
* Business summary

Data is retrieved live from Yahoo Finance.

---

### 7. Document Search (RAG)

Upload your own PDFs (earnings reports, strategy notes, research):

* Automatic indexing
* Vector-based retrieval
* LLM answers strictly grounded in your documents

The UI includes:

* Sidebar upload
* Progress indicator
* Toggle to enable / disable RAG mode

If RAG is off, the agent falls back to web-based financial data.

---

### 8. Table Output Mode

Instead of charts, you can ask for tabular data:

* Recent closing prices
* Volume
* RSI and SMA (if available)

Example:

```
| Date       | Close | Volume | RSI | SMA_20 |
```

---

## 🧠 Architecture

* **Frontend**: Streamlit
* **Agent Orchestration**: LangGraph
* **LLMs**: Ollama (Qwen 2.5)
* **Market Data**: yFinance
* **Charts**: Plotly
* **Technical Analysis**: pandas-ta
* **Forecasting**: scikit-learn
* **RAG**: Vector database + embeddings

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure Ollama is running locally with the required model installed.

---

## 🎯 What This Project Is

* A learning-oriented trading assistant
* A clean LangGraph reference architecture
* A sandbox for combining LLMs, charts, and financial data

## ❌ What This Project Is Not

* A production trading system
* A reliable forecasting engine
* Financial advice

---

## 📌 Next Ideas to Explore

* Trend-based coloring (bullish / bearish)
* Support & resistance detection
* Multi-indicator strategies
* Backtesting engine
* Portfolio-level analysis

---

## 📄 License

MIT — use, modify, and experiment freely.

---

Happy experimenting 🚀
