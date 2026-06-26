"""
Train the financial sentiment classifier.

Loads financial_phrasebank from HuggingFace (sentences_allagree split),
trains a TF-IDF + LogisticRegression pipeline, evaluates on test set,
and saves the pipeline to models/sentiment/sentiment_pipeline.joblib.

Usage:
    python scripts/train_sentiment.py

Target accuracy: > 0.80 on financial_phrasebank test split.
"""

import os
import sys
import time

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join("models", "sentiment")
MODEL_PATH = os.path.join(OUTPUT_DIR, "sentiment_pipeline.joblib")


def main() -> None:
    """Train and save the sklearn sentiment pipeline."""
    print("=" * 60)
    print("Financial Sentiment Model Training")
    print("=" * 60)

    # --- Load dataset ---
    print("\n[1/5] Loading financial_phrasebank from HuggingFace...")
    start = time.time()
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "financial_phrasebank",
            "sentences_allagree",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"ERROR loading dataset: {exc}")
        print("Ensure you have internet access and 'datasets' installed.")
        sys.exit(1)

    train_data = dataset["train"]
    texts = [item["sentence"] for item in train_data]
    labels = [item["label"] for item in train_data]

    print(f"    Loaded {len(texts)} samples in {time.time()-start:.1f}s")
    label_counts = {0: 0, 1: 0, 2: 0}
    for lb in labels:
        label_counts[lb] += 1
    print(
        f"    Distribution: negative={label_counts[0]}, "
        f"neutral={label_counts[1]}, positive={label_counts[2]}"
    )

    # --- Split ---
    print("\n[2/5] Splitting dataset (80/20, random_state=42)...")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"    Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # --- Build pipeline ---
    print("\n[3/5] Building TF-IDF + LogisticRegression pipeline...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                    analyzer="word",
                    token_pattern=r"\w{1,}",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    random_state=42,
                    solver="lbfgs",
                    multi_class="multinomial",
                ),
            ),
        ]
    )

    # --- Train ---
    print("\n[4/5] Training...")
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"    Training complete in {time.time()-train_start:.1f}s")

    # --- Evaluate ---
    print("\n[5/5] Evaluating on test set...")
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    label_names = ["negative", "neutral", "positive"]
    report = classification_report(y_test, y_pred, target_names=label_names)

    print(f"\n    Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    if accuracy < 0.80:
        print(f"⚠️  WARNING: Accuracy {accuracy:.4f} is below target of 0.80")
        print("   Consider adjusting hyperparameters or using more features.")
    else:
        print(f"✅ Accuracy target met: {accuracy:.4f} >= 0.80")

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import joblib

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"   File size: {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
