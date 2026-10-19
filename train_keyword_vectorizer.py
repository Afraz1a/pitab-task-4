"""
CitizenSense - Fit a corpus-level TF-IDF vectorizer for keyword extraction.

A single-document TF-IDF (fit only on the text being analyzed) cannot tell
common words ("area", "please", "fix") apart from topic-specific ones
("rainwater", "transformer", "garbage"), because every term in a one-document
corpus gets the same IDF weight. Fitting the vectorizer once on the full
cleaned dataset gives real document-frequency statistics, so at inference
time we can score a new document's words by real TF-IDF weight and get much
more meaningful top keywords. Saved to models/keyword_vectorizer.pkl and
used by app/utils.py::extract_keywords.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(HERE, "data", "processed")
MODELS_DIR = os.path.join(HERE, "models")


def main():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "full_cleaned.csv"))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=3000,
        min_df=1,
    )
    vectorizer.fit(df["clean_text"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "keyword_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Fitted keyword TF-IDF vectorizer on {len(df)} documents, "
          f"{len(vectorizer.vocabulary_)} vocabulary terms.")
    print("Saved -> models/keyword_vectorizer.pkl")


if __name__ == "__main__":
    main()
