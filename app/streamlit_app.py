"""
CitizenSense - Streamlit demo.

Usage:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
from inference import CitizenSenseAnalyzer

st.set_page_config(page_title="CitizenSense", page_icon="\U0001F4EC", layout="centered")


@st.cache_resource
def get_analyzer():
    return CitizenSenseAnalyzer()


st.title("CitizenSense")
st.caption("NLP-Based Public Feedback Classification and Insight System")
st.caption("Learning prototype - AIRI Team PITB AI Internship Task 4")

st.markdown(
    "Enter a citizen complaint or feedback message below. CitizenSense will "
    "classify its category, sentiment and priority, extract keywords, and "
    "generate a short summary and suggested routing department."
)

example = (
    "The road in our area has been broken for months and rainwater collects "
    "every day. Please fix it urgently."
)

if "feedback_text" not in st.session_state:
    st.session_state["feedback_text"] = ""

col1, col2 = st.columns([1, 1])
with col1:
    use_example = st.button("Use example")
with col2:
    predict_clicked = st.button("Predict", type="primary")

if use_example:
    st.session_state["feedback_text"] = example

text = st.text_area("Feedback text", key="feedback_text", height=120, placeholder=example)

if predict_clicked or use_example:
    if not text.strip():
        st.warning("Please enter some feedback text first.")
    else:
        with st.spinner("Analyzing feedback (loading models on first run)..."):
            analyzer = get_analyzer()
            result = analyzer.analyze(text)

        st.subheader("Insight Card")
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", result["category"])
        c2.metric("Sentiment", result["sentiment"])
        c3.metric("Priority", result["priority"])

        st.progress(result["category_confidence"], text=f"Category confidence: {result['category_confidence']:.2f}")
        st.progress(result["sentiment_confidence"], text=f"Sentiment confidence: {result['sentiment_confidence']:.2f}")

        st.markdown("**Keywords:** " + ", ".join(result["keywords"]) if result["keywords"] else "**Keywords:** -")
        st.markdown("**Summary:** " + result["summary"])
        st.markdown("**Suggested Department:** " + result["suggested_department"])

        with st.expander("Raw JSON output"):
            st.json(result)

st.divider()
st.caption(
    "This is a learning prototype for public-service feedback analysis, "
    "not a production government decision system. Predictions should be "
    "treated as suggestions."
)
