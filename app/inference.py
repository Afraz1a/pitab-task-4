"""
CitizenSense - Combined inference: category + sentiment + priority +
keywords + summary + suggested department, used by both the CLI and
Streamlit demos so behavior is identical.
"""

import os
import pickle

from utils import clean_text, extract_keywords, generate_summary, suggested_department
from sentiment import predict_sentiment

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "models")

PRIORITY_BY_CATEGORY_AND_SENTIMENT = {
    # simple, explainable priority rule combining predicted category and
    # sentiment; used for the demo's Priority output (the dataset itself
    # has ground-truth priority labels used during evaluation/training data
    # generation, but a live prediction needs a priority rule at inference
    # time since the baseline/HF models here target category + sentiment).
}


def load_baseline_model():
    with open(os.path.join(MODELS_DIR, "baseline_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


URGENT_WORDS = {"urgent", "urgently", "immediately", "emergency", "dangerous", "unsafe", "sparking"}
HIGH_WORDS = {"no", "not", "broken", "overflowing", "leaking", "burst", "shortage", "damaged"}


def estimate_priority(cleaned_text: str, sentiment_label: str) -> str:
    words = set(cleaned_text.split())
    if words & URGENT_WORDS:
        return "Urgent"
    if sentiment_label == "Negative" and (words & HIGH_WORDS):
        return "High"
    if sentiment_label == "Negative":
        return "Medium"
    return "Low"


class CitizenSenseAnalyzer:
    def __init__(self):
        self.model, self.encoder = load_baseline_model()

    def analyze(self, text: str) -> dict:
        cleaned = clean_text(text)
        category_id = self.model.predict([cleaned])[0]
        category = self.encoder.inverse_transform([category_id])[0]
        proba = self.model.predict_proba([cleaned])[0]
        confidence = float(proba.max())

        sentiment_label, sentiment_score = predict_sentiment(text)
        priority = estimate_priority(cleaned, sentiment_label)
        keywords = extract_keywords(text, top_n=5)
        summary = generate_summary(text, category)
        department = suggested_department(category)

        return {
            "category": category,
            "category_confidence": round(confidence, 3),
            "sentiment": sentiment_label,
            "sentiment_confidence": round(sentiment_score, 3),
            "priority": priority,
            "keywords": keywords,
            "summary": summary,
            "suggested_department": department,
        }


if __name__ == "__main__":
    analyzer = CitizenSenseAnalyzer()
    out = analyzer.analyze(
        "The road in our area has been broken for months and rainwater collects every day. Please fix it urgently."
    )
    for k, v in out.items():
        print(f"{k}: {v}")
