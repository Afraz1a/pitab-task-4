"""
Shared utilities for CitizenSense: text cleaning, keyword extraction,
extractive summarization, and category -> department mapping.

Used by the baseline training script, the CLI demo, and the Streamlit demo
so that preprocessing stays identical between training and inference.
"""

import os
import re
import pickle
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(__file__)
KEYWORD_VECTORIZER_PATH = os.path.join(HERE, "..", "models", "keyword_vectorizer.pkl")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
PHONE_RE = re.compile(r"\b(\+?\d[\d\-\s]{7,}\d)\b")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_KEEP_RE = re.compile(r"[^a-zA-Z0-9'?.,!\s]")

DEPARTMENT_MAP = {
    "Roads / Infrastructure": "Municipal / Infrastructure Department",
    "Water Supply": "Water & Sanitation Agency (WASA)",
    "Electricity": "Electricity Supply Company (DISCO)",
    "Sanitation": "Solid Waste Management Department",
    "Health": "Health Department",
    "Education": "Education Department",
    "Traffic": "Traffic Police Department",
    "General": "General Administration / Citizen Services",
}

# Minimal keyword list used to nudge the rule-based priority heuristic
# used only as a fallback when no priority model/label is available
# (the dataset itself carries ground-truth priority labels for evaluation).
URGENT_WORDS = {"urgent", "urgently", "immediately", "emergency", "dangerous", "unsafe"}
HIGH_WORDS = {"no", "not", "broken", "overflowing", "leaking", "burst", "shortage"}


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = URL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = PUNCT_KEEP_RE.sub(" ", text)
    text = text.lower()
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=1)
def _get_corpus_vectorizer():
    """Load the TF-IDF vectorizer fit on the full cleaned dataset
    (see train_keyword_vectorizer.py). Returns None if it hasn't been
    trained yet, so callers can fall back to a single-document method."""
    if not os.path.exists(KEYWORD_VECTORIZER_PATH):
        return None
    with open(KEYWORD_VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def extract_keywords(text: str, top_n: int = 5):
    """TF-IDF-based keyword extraction.

    Prefers scoring the document's terms against a TF-IDF vectorizer fit on
    the full CitizenSense corpus (real document-frequency statistics, so
    common words like "area" or "please" are correctly down-weighted versus
    topic-specific words like "rainwater" or "transformer"). Falls back to a
    single-document TF-IDF (or plain frequency) if the corpus vectorizer
    hasn't been trained yet, e.g. a first run before train_keyword_vectorizer.py.
    This is a lightweight, dependency-free alternative to RAKE/KeyBERT
    suitable for short complaint-style text.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    corpus_vectorizer = _get_corpus_vectorizer()
    if corpus_vectorizer is not None:
        try:
            tfidf = corpus_vectorizer.transform([cleaned])
            scores = tfidf.toarray()[0]
            terms = corpus_vectorizer.get_feature_names_out()
            ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
            keywords = [t for t, s in ranked if s > 0][:top_n]
            if keywords:
                return keywords
        except ValueError:
            pass

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=20,
        )
        tfidf = vectorizer.fit_transform([cleaned])
        scores = tfidf.toarray()[0]
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        keywords = [t for t, s in ranked if s > 0][:top_n]
        if keywords:
            return keywords
    except ValueError:
        pass

    # fallback: simple frequency over remaining words for very short/odd text
    words = [w for w in cleaned.split() if len(w) > 2]
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return seen[:top_n]


def generate_summary(text: str, category: str) -> str:
    """Lightweight extractive/template summary for short feedback text.

    Citizen feedback in this dataset is usually one or two sentences, so a
    heavy abstractive transformer is unnecessary overhead for a one-week
    prototype. Instead we build a structured one-line summary from the
    cleaned text and predicted category, which mirrors the "Complaint
    Insight Card" format requested in the task brief.
    """
    cleaned = clean_text(text)
    snippet = cleaned[:140].rstrip()
    if len(cleaned) > 140:
        snippet += "..."
    return f"Citizen reports a {category.lower()} issue: {snippet}"


def suggested_department(category: str) -> str:
    return DEPARTMENT_MAP.get(category, "General Administration / Citizen Services")


def rule_based_priority(text: str) -> str:
    """Fallback heuristic priority (used only if no priority label/model
    is supplied). Not used in the main pipeline since the dataset carries
    ground-truth priority labels, but kept for robustness in the demo when
    a user types free-form text outside the training distribution."""
    cleaned = clean_text(text)
    words = set(cleaned.split())
    if words & URGENT_WORDS:
        return "Urgent"
    if words & HIGH_WORDS:
        return "High"
    return "Medium"
