"""
CitizenSense - Evaluate the Hugging Face sentiment pipeline against the
dataset's ground-truth Sentiment labels (on the test split).

This is a zero-shot evaluation: the HF `sentiment-analysis` pipeline is
pretrained (not fine-tuned on our data), so this measures how well an
off-the-shelf transformer pipeline transfers to civic-complaint text.
"""

import os
import sys
import json
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
from sentiment import predict_sentiment  # noqa: E402

HERE = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(HERE, "data", "processed")
OUTPUTS_DIR = os.path.join(HERE, "outputs")


def main():
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    predictions = []
    for text in test_df["text"]:
        label, score = predict_sentiment(text)
        predictions.append(label)

    test_df = test_df.copy()
    test_df["predicted_sentiment"] = predictions

    acc = accuracy_score(test_df["sentiment"], test_df["predicted_sentiment"])
    report = classification_report(
        test_df["sentiment"], test_df["predicted_sentiment"], zero_division=0
    )

    print(f"HF sentiment pipeline accuracy on test set: {acc:.4f}")
    print(report)

    out_path = os.path.join(OUTPUTS_DIR, "sentiment_evaluation.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("CitizenSense - Hugging Face sentiment pipeline evaluation (zero-shot)\n")
        f.write(f"Model: distilbert-base-uncased-finetuned-sst-2-english (+ neutral threshold 0.70)\n")
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report)

    test_df[["text", "sentiment", "predicted_sentiment"]].to_csv(
        os.path.join(OUTPUTS_DIR, "sentiment_predictions.csv"), index=False
    )

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
