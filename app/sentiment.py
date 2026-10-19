"""
CitizenSense - Sentiment via a Hugging Face pretrained pipeline.

Satisfies the task's "Transformer / Hugging Face Requirement" using
Option A (pretrained pipeline). We use the default sentiment-analysis
pipeline (distilbert-base-uncased-finetuned-sst-2-english), which is
binary (POSITIVE/NEGATIVE). Since CitizenSense needs a 3-way label
(Negative/Neutral/Positive) to match the category labels used for
training/evaluation, we add a confidence threshold: when the model is
not confident (score below NEUTRAL_THRESHOLD), the message is treated
as Neutral rather than forcing it into Positive/Negative.
"""

from functools import lru_cache

NEUTRAL_THRESHOLD = 0.70


@lru_cache(maxsize=1)
def _get_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis")


def predict_sentiment(text: str):
    """Returns (label, confidence) where label is Negative/Neutral/Positive."""
    clf = _get_pipeline()
    result = clf(text[:512])[0]
    raw_label = result["label"]
    score = float(result["score"])

    if score < NEUTRAL_THRESHOLD:
        return "Neutral", score

    label = "Positive" if raw_label == "POSITIVE" else "Negative"
    return label, score


if __name__ == "__main__":
    samples = [
        "There is no water supply in our area for three days.",
        "The newly repaired road is smooth and well done, thank you.",
        "When will the road repair work be completed?",
    ]
    for s in samples:
        print(s, "->", predict_sentiment(s))
