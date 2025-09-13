# streamlit_app.py
import json
import streamlit as st
from agent import run_agent

st.set_page_config(page_title="Financial Advisor", layout="wide")
st.title("💬 Financial Advisor (Langflow • RAG • AstraDB • LLMs • LiveKit)")

q = st.text_area("Ask a finance question:", value="", height=120)
if st.button("Run"):
    with st.spinner("Thinking…"):
        res = run_agent(q)
    st.markdown(res["text"])
    try:
        start = res["text"].rfind("{")
        end = res["text"].rfind("}")
        if start != -1 and end != -1 and end > start:
            metrics = json.loads(res["text"][start:end+1])
            st.subheader("Metrics")
            st.json(metrics)
    except Exception:
        pass

st.info("Voice mode runs via LiveKit worker (separate process).")
