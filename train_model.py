"""
CitizenSense - Baseline model training (TF-IDF + Logistic Regression).

Trains a category classifier on the processed train split, evaluates it on
validation + test splits, and saves:
  - models/baseline_model.pkl   (sklearn Pipeline: TfidfVectorizer + LogisticRegression)
  - models/label_encoder.pkl    (category LabelEncoder)
  - outputs/classification_report.txt
  - outputs/confusion_matrix.png
  - outputs/sample_predictions.csv  (>=20 test predictions)
"""

import os
import json
import pickle

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

HERE = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(HERE, "data", "processed")
MODELS_DIR = os.path.join(HERE, "models")
OUTPUTS_DIR = os.path.join(HERE, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_split(name):
    return pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))


def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    # re-fit a clean label encoder on the full category set so class order
    # is deterministic and consistent across train/val/test
    all_categories = sorted(
        set(train_df["category"]) | set(val_df["category"]) | set(test_df["category"])
    )
    encoder = LabelEncoder()
    encoder.fit(all_categories)

    X_train, y_train = train_df["clean_text"], encoder.transform(train_df["category"])
    X_val, y_val = val_df["clean_text"], encoder.transform(val_df["category"])
    X_test, y_test = test_df["clean_text"], encoder.transform(test_df["category"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    model.fit(X_train, y_train)

    # ---- Validation metrics ----
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_acc:.4f}")

    # ---- Test metrics ----
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_preds, average="weighted", zero_division=0
    )
    print(f"Test accuracy: {test_acc:.4f}  precision: {precision:.4f}  "
          f"recall: {recall:.4f}  f1: {f1:.4f}")

    report_text = classification_report(
        y_test, test_preds, target_names=encoder.classes_, zero_division=0
    )
    print(report_text)

    with open(os.path.join(OUTPUTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("CitizenSense Baseline Model (TF-IDF + Logistic Regression)\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test weighted precision: {precision:.4f}\n")
        f.write(f"Test weighted recall: {recall:.4f}\n")
        f.write(f"Test weighted F1-score: {f1:.4f}\n\n")
        f.write("Per-class report:\n")
        f.write(report_text)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(encoder.classes_)))
    ax.set_yticks(range(len(encoder.classes_)))
    ax.set_xticklabels(encoder.classes_, rotation=45, ha="right")
    ax.set_yticklabels(encoder.classes_)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title("CitizenSense - Confusion Matrix (Test Set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ---- Sample predictions (>=20) ----
    sample_df = test_df.copy()
    sample_df["predicted_category"] = encoder.inverse_transform(test_preds)
    sample_df["correct"] = sample_df["category"] == sample_df["predicted_category"]
    sample_out = sample_df[["text", "category", "predicted_category", "correct",
                             "sentiment", "priority", "language"]].head(30)
    sample_out.to_csv(os.path.join(OUTPUTS_DIR, "sample_predictions.csv"), index=False)

    # save full test predictions for error analysis script
    sample_df[["text", "category", "predicted_category", "correct",
               "sentiment", "priority", "language"]].to_csv(
        os.path.join(OUTPUTS_DIR, "test_predictions_full.csv"), index=False
    )

    # ---- Save model + encoder ----
    with open(os.path.join(MODELS_DIR, "baseline_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    metrics_summary = {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_precision_weighted": precision,
        "test_recall_weighted": recall,
        "test_f1_weighted": f1,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "classes": list(encoder.classes_),
    }
    with open(os.path.join(OUTPUTS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("Saved model, label encoder, and evaluation outputs.")


if __name__ == "__main__":
    main()
