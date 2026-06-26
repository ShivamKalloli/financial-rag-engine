"""
Unit tests for app/core/sentiment.py.

Tests: valid label, confidence range, bulk length, positive sentence,
negative sentence, singleton.
"""

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
class TestSentimentClassifier:
    """Unit tests for the SentimentClassifier."""

    def _mock_classifier(self, monkeypatch):
        """Create a mock sklearn pipeline and inject it into the classifier."""
        import numpy as np

        mock_pipeline = MagicMock()
        # Default: returns 'positive' for any text
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

        from app.core.sentiment import SentimentClassifier

        clf = SentimentClassifier.__new__(SentimentClassifier)
        clf._pipeline = mock_pipeline
        return clf, mock_pipeline

    def test_classify_returns_valid_label(self, monkeypatch):
        """classify() must return a label in {positive, negative, neutral}."""
        import numpy as np

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

        from app.core.sentiment import SentimentClassifier

        clf = SentimentClassifier.__new__(SentimentClassifier)
        clf._pipeline = mock_pipeline

        result = clf.classify("Revenue exceeded expectations")
        assert result["label"] in {"positive", "negative", "neutral"}

    def test_classify_returns_confidence_0_to_1(self, monkeypatch):
        """classify() confidence must be between 0.0 and 1.0."""
        import numpy as np

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

        from app.core.sentiment import SentimentClassifier

        clf = SentimentClassifier.__new__(SentimentClassifier)
        clf._pipeline = mock_pipeline

        result = clf.classify("Some financial text")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_bulk_length(self, monkeypatch):
        """classify_bulk(10 texts) must return a list of exactly 10 dicts."""
        import numpy as np

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array(
            [[0.1, 0.2, 0.7] for _ in range(10)]
        )

        from app.core.sentiment import SentimentClassifier

        clf = SentimentClassifier.__new__(SentimentClassifier)
        clf._pipeline = mock_pipeline

        texts = [f"Financial sentence {i}" for i in range(10)]
        results = clf.classify_bulk(texts)
        assert len(results) == 10
        for r in results:
            assert "label" in r
            assert "confidence" in r

    def test_positive_sentence(self, monkeypatch):
        """A clearly positive sentence should return label='positive'."""
        import numpy as np

        mock_pipeline = MagicMock()
        # Positive class (index 2) has highest probability
        mock_pipeline.predict_proba.return_value = np.array([[0.05, 0.10, 0.85]])

        from app.core.sentiment import SentimentClassifier

        clf = SentimentClassifier.__new__(SentimentClassifier)
        clf._pipeline = mock_pipeline

        result = clf.classify("Revenue exceeded expectations by a wide margin")
        assert result["label"] == "positive"
        assert result["confidence"] > 0.5

    def test_negative_sentence(self, monkeypatch):
        """A clearly negative sentence should return label='negative'."""
        import numpy as np

        mock_pipeline = MagicMock()
        # Negative class (index 0) has highest probability
        mock_pipeline.predict_proba.return_value = np.array([[0.80, 0.15, 0.05]])

        from app.core.sentiment import SentimentClassifier

        clf = SentimentClassifier.__new__(SentimentClassifier)
        clf._pipeline = mock_pipeline

        result = clf.classify("Significant losses reported, margins collapsed")
        assert result["label"] == "negative"

    def test_singleton_returns_same_object(self, monkeypatch, tmp_path):
        """Two calls to get_instance() must return the same object."""
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        # Create a real tiny model and save it
        model_dir = tmp_path / "sentiment"
        model_dir.mkdir()
        model_path = model_dir / "sentiment_pipeline.joblib"

        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression()),
            ]
        )
        pipe.fit(
            ["good news", "bad news", "neutral news"],
            [2, 0, 1],
        )
        joblib.dump(pipe, model_path)

        # Patch the model path
        monkeypatch.setattr("app.core.sentiment._MODEL_PATH", str(model_path))
        from app.core.sentiment import SentimentClassifier

        SentimentClassifier.reset_instance()

        inst1 = SentimentClassifier.get_instance()
        inst2 = SentimentClassifier.get_instance()
        assert inst1 is inst2

    def test_classify_empty_text(self, monkeypatch):
        """Empty text must return neutral with 0.0 confidence without crashing."""
        mock_pipeline = MagicMock()

        from app.core.sentiment import SentimentClassifier

        clf = SentimentClassifier.__new__(SentimentClassifier)
        clf._pipeline = mock_pipeline

        result = clf.classify("")
        assert result["label"] == "neutral"
        assert result["confidence"] == 0.0
