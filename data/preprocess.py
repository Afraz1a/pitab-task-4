"""
CitizenSense - Text preprocessing and train/val/test split.

Steps applied (see README for rationale of each):
1. Lowercasing (English text only; Roman Urdu left as-is since it is
   already lowercase in our generator and case carries little meaning there).
2. Whitespace normalization (collapse multiple spaces).
3. URL removal.
4. Phone-number / CNIC-like pattern removal (defensive; dataset has none,
   kept as a safety net per the task's privacy rules).
5. Punctuation handling: light touch - keep sentence content, strip stray
   symbols that are not useful for TF-IDF, keep apostrophes.
6. Duplicate / near-duplicate removal (exact match on cleaned text).
7. Missing value handling (drop empty rows).
8. Tokenization (simple whitespace/regex tokenizer, used for keyword
   extraction later; TF-IDF vectorizer tokenizes internally for modeling).
9. Label encoding for category (and sentiment/priority for reference).
10. Train (70%) / validation (15%) / test (15%) split, stratified by category.

Stopword removal / stemming were intentionally NOT applied for the TF-IDF
baseline: complaint texts are short, and function words like "no", "not"
carry sentiment-bearing signal that stopword lists often strip. This is
noted explicitly in the README as a documented preprocessing decision.
"""

import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(__file__)
RAW_PATH = os.path.join(HERE, "raw", "citizen_feedback_raw.csv")
PROCESSED_DIR = os.path.join(HERE, "processed")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
PHONE_RE = re.compile(r"\b(\+?\d[\d\-\s]{7,}\d)\b")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_KEEP_RE = re.compile(r"[^a-zA-Z0-9'?.,!\s]")


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = URL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = PUNCT_KEEP_RE.sub(" ", text)
    text = text.lower()
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def simple_tokenize(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())


def main():
    df = pd.read_csv(RAW_PATH)
    before = len(df)

    df = df.dropna(subset=["text", "category"])
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    after = len(df)
    print(f"Rows before cleaning/dedup: {before}, after: {after}")

    df["tokens"] = df["clean_text"].apply(lambda t: " ".join(simple_tokenize(t)))

    cat_encoder = LabelEncoder()
    df["category_id"] = cat_encoder.fit_transform(df["category"])

    sent_encoder = LabelEncoder()
    df["sentiment_id"] = sent_encoder.fit_transform(df["sentiment"])

    prio_encoder = LabelEncoder()
    df["priority_id"] = prio_encoder.fit_transform(df["priority"])

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # stratified 70/15/15 split on category
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["category"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["category"]
    )

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    df.to_csv(os.path.join(PROCESSED_DIR, "full_cleaned.csv"), index=False)

    # labels.csv required by project structure: category label map
    labels_df = pd.DataFrame({
        "class_id": range(len(cat_encoder.classes_)),
        "category": cat_encoder.classes_,
    })
    labels_df.to_csv(os.path.join(HERE, "labels.csv"), index=False)

    label_maps = {
        "category": {int(i): c for i, c in enumerate(cat_encoder.classes_)},
        "sentiment": {int(i): c for i, c in enumerate(sent_encoder.classes_)},
        "priority": {int(i): c for i, c in enumerate(prio_encoder.classes_)},
    }
    with open(os.path.join(PROCESSED_DIR, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("Category classes:", list(cat_encoder.classes_))
    print("Saved processed splits to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
