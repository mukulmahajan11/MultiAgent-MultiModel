# agent.py
from typing import Dict, Any, List
from config import settings
from rag_store import retrieve, format_context
from tools.market_data import get_prices, latest_quote, fetch_news_stub
from tools.risk import compute_returns, portfolio_stats, max_drawdown, historical_var_es
from tools.optimizer import mean_variance_opt, risk_parity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import pandas as pd
import numpy as np

@tool
def tool_latest_quote(symbol: str) -> Dict[str, Any]:
    return latest_quote(symbol)

@tool
def tool_prices(symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
    df = get_prices(symbols, period=period, interval=interval)
    return {"index": [str(x) for x in df.index], "data": df.to_dict(orient="list")}

@tool
def tool_risk_metrics(symbols: List[str]) -> Dict[str, Any]:
    df = get_prices(symbols)
    rets = compute_returns(df)
    w = np.ones(len(symbols)) / len(symbols)
    stats = portfolio_stats(rets, w)
    mdd = max_drawdown(df, w)
    daily_port = (rets @ w).rename("port")
    tail = historical_var_es(daily_port, alpha=0.95)
    return {"stats": stats, "max_drawdown": mdd, "tail": tail}

@tool
def tool_optimize(symbols: List[str], method: str = "max_sharpe", target_risk: float = 0.0) -> Dict[str, Any]:
    df = get_prices(symbols)
    rets = compute_returns(df)
    if method == "risk_parity":
        res = risk_parity(rets)
    else:
        res = mean_variance_opt(rets, target_risk=target_risk if target_risk > 0 else None)
    w = res["weights"]
    stats = portfolio_stats(rets, np.array(w))
    mdd = max_drawdown(df, np.array(w))
    return {"weights": w, "stats": stats, "max_drawdown": mdd}

@tool
def tool_news(query: str, k: int = 5) -> Dict[str, Any]:
    return {"items": fetch_news_stub(query, k)}

@tool
def tool_rag(query: str, k: int = 5) -> Dict[str, Any]:
    ctx = retrieve(query, k=k)
    return {"snippets": ctx, "context_text": format_context(ctx)}

SYSTEM_PROMPT = """You are a cautious financial analyst agent. 
Prefer tools and RAG context for numbers and facts.
Provide transparent reasoning and disclaimers.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{input}")]
)

def build_agent():
    llm = ChatOpenAI(model=settings.LLM_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0.1)
    tools = [tool_latest_quote, tool_prices, tool_risk_metrics, tool_optimize, tool_news, tool_rag]
    bound = llm.bind_functions(tools)
    return bound

def run_agent(user_input: str) -> Dict[str, Any]:
    agent = build_agent()
    final = agent.invoke(prompt.format(input=user_input))
    return {"text": final.content}
