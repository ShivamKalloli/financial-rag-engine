"""
LangChain LCEL generation chain with Groq/Ollama backend.

Auto-detects backend: GROQ_API_KEY set → Groq (llama3-8b-8192),
else → Ollama (llama3.2:3b).

Handles multi-step queries (containing 'and', 'vs', 'compare', multiple '?').
Truncates context to 3000 tokens before sending.
"""

import re
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.monitoring.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt — used verbatim as specified
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a precise financial analysis assistant.

RULES:
1. Answer ONLY using information from the provided context below.
2. Always cite your source: mention the document name and relevant details.
3. Be specific with numbers, percentages, and dates when available in context.
4. If the context does not contain enough information to answer, respond with \
exactly: "I cannot answer this from the available documents."
5. Do not make up any numbers or facts not present in the context.
6. Keep answers concise and factual.

Context:
{context}"""

_MAX_CONTEXT_TOKENS = 3000
_APPROX_CHARS_PER_TOKEN = 4  # conservative approximation


def get_llm():
    """
    Return the appropriate LLM based on environment configuration.

    If GROQ_API_KEY is set in settings, returns ChatGroq (free tier).
    Otherwise returns Ollama with llama3.2:3b for local inference.

    Returns:
        LangChain chat model instance.
    """
    if settings.llm_backend == "groq":
        from langchain_groq import ChatGroq

        logger.info("llm_backend_selected", extra={"backend": "groq"})
        return ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            groq_api_key=settings.groq_api_key,
            max_tokens=1024,
        )
    else:
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            from langchain_community.llms.ollama import Ollama

        logger.info("llm_backend_selected", extra={"backend": "ollama"})
        return Ollama(model="llama3.2:3b", temperature=settings.llm_temperature)


def _format_context(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    Each chunk is formatted as:
        [Source: {source}, Page: {page}]
        {text}

    Chunks are separated by "\\n\\n---\\n\\n".
    The result is truncated to MAX_CONTEXT_TOKENS.

    Args:
        chunks: List of chunk dicts with 'text', 'source', 'page' keys.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "No relevant context found."

    parts = []
    total_chars = 0
    max_chars = _MAX_CONTEXT_TOKENS * _APPROX_CHARS_PER_TOKEN

    for chunk in chunks:
        src = chunk.get("source", "unknown")
        pg = chunk.get("page", 1)
        part = f"[Source: {src}, Page: {pg}]\n{chunk['text']}"
        if total_chars + len(part) > max_chars:
            # Truncate this chunk to fit
            remaining = max_chars - total_chars
            if remaining > 100:  # Only add if meaningful content remains
                parts.append(part[:remaining] + "... [truncated]")
            break
        parts.append(part)
        total_chars += len(part)

    return "\n\n---\n\n".join(parts)


def _is_multi_step_query(question: str) -> bool:
    """
    Detect whether a question requires multi-step reasoning.

    Returns True if the question contains ' and ', ' vs ', ' compare ',
    or has more than one question mark.

    Args:
        question: The raw question string.

    Returns:
        True if multi-step handling should be applied.
    """
    triggers = [" and ", " vs ", " vs. ", " compare ", " versus "]
    q_lower = question.lower()
    if question.count("?") > 1:
        return True
    return any(t in q_lower for t in triggers)


def _split_sub_questions(question: str) -> List[str]:
    """
    Split a compound question into sub-questions.

    Splits on ' and ', ' vs ', or '?' boundaries.

    Args:
        question: The compound question string.

    Returns:
        List of individual sub-questions. Falls back to [question] if
        splitting does not yield multiple parts.
    """
    # Try splitting on '?' first
    parts = [p.strip() + "?" for p in question.split("?") if p.strip()]
    if len(parts) > 1:
        return parts

    # Try splitting on ' and '
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+|\s+compare\s+", question, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [question]


class RAGChain:
    """
    LangChain LCEL chain wrapping Groq or Ollama with financial RAG prompt.

    Handles:
    - Context formatting and token truncation
    - Multi-step query decomposition and synthesis
    - Streaming (passthrough — handled upstream)
    """

    def __init__(self, llm=None) -> None:
        """
        Initialise the chain.

        Args:
            llm: Optional LangChain LLM instance. If None, get_llm() is called.
        """
        self._llm = llm or get_llm()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", "Question: {question}"),
            ]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    @property
    def model_used(self) -> str:
        """Return a descriptive string of the active LLM."""
        if settings.llm_backend == "groq":
            return f"groq/{settings.llm_model}"
        return "ollama/llama3.2:3b"

    def answer(
        self,
        question: str,
        context_chunks: List[dict],
        stream: bool = False,
    ) -> str:
        """
        Generate an answer grounded in the provided context chunks.

        If the question is detected as multi-step, it is decomposed into
        sub-questions that are each answered individually, then synthesized.

        Args:
            question: The user's question string.
            context_chunks: Retrieved chunks from the retrieval pipeline.
            stream: Ignored in this implementation (synchronous by default).

        Returns:
            Generated answer string.
        """
        if _is_multi_step_query(question):
            return self._multi_step_answer(question, context_chunks)
        return self._single_answer(question, context_chunks)

    def _single_answer(self, question: str, chunks: List[dict]) -> str:
        """
        Run one LLM call with formatted context.

        Args:
            question: Question string.
            chunks: Context chunks.

        Returns:
            Answer string from LLM.
        """
        context = _format_context(chunks)
        logger.info(
            "llm_call",
            extra={
                "backend": settings.llm_backend,
                "context_chunks": len(chunks),
                "context_chars": len(context),
            },
        )
        return self._chain.invoke({"context": context, "question": question})

    def _multi_step_answer(self, question: str, chunks: List[dict]) -> str:
        """
        Decompose a multi-part question and synthesize individual answers.

        Args:
            question: Compound question string.
            chunks: Context chunks (used for all sub-questions).

        Returns:
            Synthesized answer covering all sub-questions.
        """
        sub_questions = _split_sub_questions(question)
        if len(sub_questions) == 1:
            return self._single_answer(question, chunks)

        logger.info(
            "multi_step_query",
            extra={"sub_question_count": len(sub_questions)},
        )

        sub_answers = []
        for sq in sub_questions:
            ans = self._single_answer(sq, chunks)
            sub_answers.append(f"**{sq}**\n{ans}")

        synthesis_prompt = (
            "Based on the following individual answers, provide a concise "
            "combined summary:\n\n" + "\n\n".join(sub_answers)
        )
        context = _format_context(chunks)
        return self._chain.invoke({"context": context, "question": synthesis_prompt})
