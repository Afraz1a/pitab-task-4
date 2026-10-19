"""
CitizenSense - FastAPI demo (optional alternative to the CLI/Streamlit demos).

Usage:
    uvicorn app.api:app --reload --app-dir app
Then POST to http://127.0.0.1:8000/analyze-feedback
    {"text": "There is no water supply in our area for three days."}
"""

from fastapi import FastAPI
from pydantic import BaseModel

from inference import CitizenSenseAnalyzer

app = FastAPI(title="CitizenSense API", description="NLP-based public feedback classification and insight API")

_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = CitizenSenseAnalyzer()
    return _analyzer


class FeedbackRequest(BaseModel):
    text: str


@app.post("/analyze-feedback")
def analyze_feedback(payload: FeedbackRequest):
    analyzer = get_analyzer()
    result = analyzer.analyze(payload.text)
    return {
        "category": result["category"],
        "sentiment": result["sentiment"],
        "priority": result["priority"],
        "keywords": result["keywords"],
        "summary": result["summary"],
        "suggested_department": result["suggested_department"],
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "CitizenSense API. POST /analyze-feedback with {'text': '...'}"}
