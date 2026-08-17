import streamlit as st

from app import graph

st.set_page_config(
    page_title="Multi-Step Research Agent",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Multi-Step Research Agent")

st.write(
    "Ask a research question and let the LangGraph agent "
    "decompose, research, evaluate, and synthesize the answer."
)

query = st.text_area(
    "Research Question",
    placeholder="Example: How do RAG and fine-tuning compare for enterprise customer support?"
)

if st.button("🚀 Start Research"):

    if not query.strip():
        st.warning("Please enter a research question.")
        st.stop()

    initial_state = {
        "query": query,
        "sub_questions": [],
        "findings": [],
        "gaps": [],
        "sufficient": False,
        "research_round": 0,
        "final_report": ""
    }

    with st.spinner("Researching..."):

        result = graph.invoke(initial_state)

    st.success("Research completed!")

    if result.get("final_report"):

        st.markdown("## 📄 Final Research Report")

        st.markdown(
            result["final_report"]
        )

    else:

        st.warning("The agent did not produce a final report.")