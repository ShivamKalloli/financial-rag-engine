"""
Financial sentiment classifier using scikit-learn TF-IDF + LogisticRegression.

Trained on the financial_phrasebank dataset.
Model is loaded lazily from disk on first use and cached as a singleton.
"""

import os
import threading
from typing import List, Optional

from app.monitoring.logger import get_logger
from app.monitoring.metrics import rag_sentiment_total

logger = get_logger(__name__)

_MODEL_PATH = os.path.join("models", "sentiment", "sentiment_pipeline.joblib")
_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentClassifier:
    """
    Singleton sklearn TF-IDF + LogisticRegression sentiment classifier.

    Trained on financial_phrasebank. Labels: positive, negative, neutral.
    Model is loaded from disk on first use (lazy initialization).
    """

    _instance: Optional["SentimentClassifier"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Load the saved sklearn pipeline from disk."""
        import joblib

        if not os.path.isfile(_MODEL_PATH):
            raise FileNotFoundError(
                f"Sentiment model not found at {_MODEL_PATH}. "
                "Run: python scripts/train_sentiment.py"
            )
        self._pipeline = joblib.load(_MODEL_PATH)
        logger.info("sentiment_model_loaded", extra={"path": _MODEL_PATH})

    @classmethod
    def get_instance(cls) -> "SentimentClassifier":
        """
        Return the singleton SentimentClassifier.

        Thread-safe via double-checked locking.

        Returns:
            The singleton SentimentClassifier instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Clear the singleton (for testing only).

        Allows tests to inject a mock classifier.
        """
        with cls._lock:
            cls._instance = None

    def classify(self, text: str) -> dict:
        """
        Classify the sentiment of a single text string.

        Args:
            text: Input financial text (e.g., a chunk or sentence).

        Returns:
            dict with:
                - label (str): "positive", "negative", or "neutral"
                - confidence (float): Probability of the predicted class [0, 1]
        """
        if not text or not text.strip():
            return {"label": "neutral", "confidence": 0.0}

        proba = self._pipeline.predict_proba([text])[0]
        pred_class = int(proba.argmax())
        label = _LABEL_MAP.get(pred_class, "neutral")
        confidence = float(proba[pred_class])

        rag_sentiment_total.labels(label=label).inc()
        return {"label": label, "confidence": round(confidence, 4)}

    def classify_bulk(self, texts: List[str]) -> List[dict]:
        """
        Classify the sentiment of multiple texts in one batch call.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of dicts, same format as classify(), one per input text.
        """
        if not texts:
            return []

        probas = self._pipeline.predict_proba(texts)
        results = []
        for proba in probas:
            pred_class = int(proba.argmax())
            label = _LABEL_MAP.get(pred_class, "neutral")
            confidence = float(proba[pred_class])
            rag_sentiment_total.labels(label=label).inc()
            results.append({"label": label, "confidence": round(confidence, 4)})
        return results


def is_model_available() -> bool:
    """
    Check whether the sentiment model file exists on disk.

    Returns:
        True if the model joblib file is present.
    """
    return os.path.isfile(_MODEL_PATH)
