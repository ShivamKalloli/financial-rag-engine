"""
Unit tests for app/core/rag_pipeline.py.
"""

from unittest.mock import MagicMock

import pytest

from app.core.rag_pipeline import RAGPipeline


@pytest.mark.unit
class TestRAGPipelineUnit:
    """Unit tests for the RAGPipeline orchestrator."""

    def test_pipeline_initialization(self):
        """Should accept injected retriever, chain, and sentiment classifier."""
        mock_retriever = MagicMock()
        mock_chain = MagicMock()
        mock_sentiment = MagicMock()

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_chain=mock_chain,
            sentiment_classifier=mock_sentiment,
        )

        assert pipeline._get_retriever() is mock_retriever
        assert pipeline._get_chain() is mock_chain
        assert pipeline._get_sentiment() is mock_sentiment

    def test_pipeline_query_flow(self):
        """Should run validation, retrieve chunks, generate answer,
        classify sentiment, and format response.
        """
        mock_retriever = MagicMock()
        mock_chain = MagicMock()
        mock_sentiment = MagicMock()

        # Setup mock behavior
        chunks = [
            {
                "text": "Apple Q4 revenue was $89.5B",
                "source": "apple.txt",
                "page": 1,
                "score": 0.95,
            }
        ]
        mock_retriever.search.return_value = chunks
        mock_chain.answer.return_value = "Mocked LLM answer citing apple.txt"
        mock_chain.model_used = "mock-groq-model"

        mock_sentiment.classify_bulk.return_value = [
            {"label": "positive", "confidence": 0.9}
        ]

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_chain=mock_chain,
            sentiment_classifier=mock_sentiment,
        )

        resp = pipeline.query(
            "What was Apple's Q4 revenue?", top_k=3, retrieval_mode="hybrid"
        )

        # Assertions
        assert resp["answer"] == "Mocked LLM answer citing apple.txt"
        assert len(resp["sources"]) == 1
        assert resp["sources"][0]["source"] == "apple.txt"
        assert resp["sources"][0]["score"] == 0.95
        assert resp["sentiment"]["label"] == "positive"
        assert resp["sentiment"]["confidence"] == 0.9
        assert resp["retrieval_mode"] == "hybrid"
        assert resp["model_used"] == "mock-groq-model"

        # Verify correct methods were invoked
        mock_retriever.search.assert_called_once_with(
            query="What was Apple's Q4 revenue?", top_k=3, mode="hybrid"
        )
        mock_chain.answer.assert_called_once_with(
            "What was Apple's Q4 revenue?", chunks, stream=False
        )
        mock_sentiment.classify_bulk.assert_called_once_with(
            ["Apple Q4 revenue was $89.5B"]
        )

    def test_query_flow_empty_chunks(self):
        """Should handle empty chunk list gracefully, returning
        neutral sentiment with 0 confidence.
        """
        mock_retriever = MagicMock()
        mock_chain = MagicMock()
        mock_sentiment = MagicMock()

        mock_retriever.search.return_value = []
        mock_chain.answer.return_value = (
            "I cannot answer this from the available documents."
        )
        mock_chain.model_used = "mock-groq-model"

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_chain=mock_chain,
            sentiment_classifier=mock_sentiment,
        )

        resp = pipeline.query("How many cars did Tesla deliver?", top_k=3)

        assert resp["answer"] == "I cannot answer this from the available documents."
        assert len(resp["sources"]) == 0
        assert resp["sentiment"]["label"] == "neutral"
        assert resp["sentiment"]["confidence"] == 0.0
        assert not mock_sentiment.classify_bulk.called

    def test_query_flow_no_classifier(self):
        """Should return neutral sentiment if the classifier is None."""
        from unittest.mock import patch

        mock_retriever = MagicMock()
        mock_chain = MagicMock()

        chunks = [{"text": "Some text", "source": "doc.txt", "page": 1, "score": 0.8}]
        mock_retriever.search.return_value = chunks
        mock_chain.answer.return_value = "Answer string"

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_chain=mock_chain,
            sentiment_classifier=None,
        )

        with patch.object(pipeline, "_get_sentiment", return_value=None):
            resp = pipeline.query("Any question?", top_k=3)
            assert resp["sentiment"]["label"] == "neutral"
            assert resp["sentiment"]["confidence"] == 0.0
