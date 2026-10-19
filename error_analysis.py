"""
CitizenSense - Error analysis.

Loads the trained baseline model, re-runs predictions on the validation and
test splits, and builds a manual-review table of at least 15 incorrect or
low-confidence ("weak") predictions with an inferred error type and
possible reason for each, plus a short list of improvement suggestions.
"""

import os
import pickle
import pandas as pd

HERE = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(HERE, "data", "processed")
MODELS_DIR = os.path.join(HERE, "models")
OUTPUTS_DIR = os.path.join(HERE, "outputs")

MIN_CASES = 15


def load_model():
    with open(os.path.join(MODELS_DIR, "baseline_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def classify_error(row):
    """Infer a human-readable error type + likely reason from the row."""
    text = row["text"]
    actual = row["actual"]
    predicted = row["predicted"]
    n_words = len(str(text).split())

    if row["is_weak"]:
        return "Low-confidence (weak) prediction", (
            "Model was not confident even though the prediction was correct; "
            "text lacks a strong category-specific keyword."
        )

    if n_words <= 6:
        return "Wrong category (short text)", (
            "Very few words / weak keyword signal, hard to disambiguate from context alone."
        )

    if row["language"] == "roman_ur":
        return "Wrong category (language mismatch)", (
            "Roman Urdu / mixed-language text; TF-IDF vocabulary was built mostly on English tokens."
        )

    actual_l, predicted_l = actual.lower(), predicted.lower()
    overlap_pairs = {
        frozenset(["roads / infrastructure", "sanitation"]): "Contains both road and sanitation/water cues",
        frozenset(["roads / infrastructure", "water supply"]): "Contains both road and water issue",
        frozenset(["electricity", "general"]): "Short, generic phrasing with few electricity-specific keywords",
        frozenset(["health", "education"]): "Shared 'staff' vocabulary between health and education templates",
        frozenset(["general", "water supply"]): "Vague wording also plausible as a general complaint",
        frozenset(["general", "health"]): "Vague wording also plausible as a general complaint",
        frozenset(["traffic", "roads / infrastructure"]): "Traffic and road-issue vocabulary overlap",
    }
    key = frozenset([actual_l, predicted_l])
    reason = overlap_pairs.get(key, "Ambiguous/generic wording without a strong category-specific keyword")
    return "Wrong category", reason


def main():
    model, encoder = load_model()

    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    combined = pd.concat([val_df, test_df], ignore_index=True)
    X = combined["clean_text"]
    y_true_ids = encoder.transform(combined["category"])

    pred_ids = model.predict(X)
    probs = model.predict_proba(X)
    confidences = probs.max(axis=1)

    combined["actual"] = combined["category"]
    combined["predicted"] = encoder.inverse_transform(pred_ids)
    combined["confidence"] = confidences
    combined["is_correct"] = combined["actual"] == combined["predicted"]

    errors = combined[~combined["is_correct"]].copy()
    errors["is_weak"] = False

    n_needed = MIN_CASES - len(errors)
    if n_needed > 0:
        weak_pool = combined[combined["is_correct"]].copy()
        weak_pool = weak_pool.sort_values("confidence").head(n_needed)
        weak_pool["is_weak"] = True
        review_set = pd.concat([errors, weak_pool], ignore_index=True)
    else:
        review_set = errors.sample(n=max(MIN_CASES, len(errors)), random_state=42) if len(errors) > MIN_CASES else errors

    rows = []
    for _, r in review_set.iterrows():
        error_type, reason = classify_error(r)
        rows.append({
            "Text Sample": r["text"],
            "Actual Label": r["actual"],
            "Predicted Label": r["predicted"],
            "Confidence": round(float(r["confidence"]), 3),
            "Error Type": error_type,
            "Possible Reason": reason,
        })

    error_df = pd.DataFrame(rows).sort_values("Confidence")
    out_path = os.path.join(OUTPUTS_DIR, "error_analysis.csv")
    error_df.to_csv(out_path, index=False)

    print(f"Reviewed {len(error_df)} cases ({len(errors)} true errors, "
          f"{max(0, n_needed)} weak/low-confidence correct predictions added).")
    print(error_df[["Text Sample", "Actual Label", "Predicted Label", "Error Type"]].to_string(index=False))
    print(f"\nSaved error analysis table -> {out_path}")

    # improvement suggestions (also surfaced in README / final report)
    suggestions = [
        "Add more labeled examples for classes with lower F1 (General, Water Supply, Health).",
        "Balance weak/ambiguous classes with additional, more distinctive training examples.",
        "Improve label guidelines so short/vague complaints get a consistent category (e.g. default to General).",
        "Add more Roman Urdu / multilingual training data rather than a small bonus slice.",
        "Fine-tune a transformer (e.g. DistilBERT) which can use context better than bag-of-words TF-IDF.",
        "Add a human-review queue for predictions below a confidence threshold (e.g. < 0.55).",
    ]
    with open(os.path.join(OUTPUTS_DIR, "improvement_suggestions.txt"), "w", encoding="utf-8") as f:
        f.write("CitizenSense - Suggested improvements based on error analysis\n\n")
        for s in suggestions:
            f.write(f"- {s}\n")


if __name__ == "__main__":
    main()
