"""
CitizenSense - CLI demo.

Usage:
    python app/demo.py
"""

from inference import CitizenSenseAnalyzer


def main():
    print("=" * 60)
    print("CitizenSense - Public Feedback Classification & Insight System")
    print("=" * 60)
    print("Loading models (baseline classifier + Hugging Face pipeline)...")
    analyzer = CitizenSenseAnalyzer()
    print("Ready. Type 'quit' to exit.\n")

    while True:
        text = input("Enter feedback text: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if not text:
            continue

        result = analyzer.analyze(text)
        print("\n--- Insight Card ---")
        print(f"Category:            {result['category']} (confidence: {result['category_confidence']})")
        print(f"Sentiment:           {result['sentiment']} (confidence: {result['sentiment_confidence']})")
        print(f"Priority:            {result['priority']}")
        print(f"Keywords:            {', '.join(result['keywords'])}")
        print(f"Summary:             {result['summary']}")
        print(f"Suggested Action:    Route to {result['suggested_department']}")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
