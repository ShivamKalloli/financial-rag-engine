"""
Generation evaluation: ROUGE-L and keyword overlap scoring.

For each QA pair, runs the full RAG pipeline and scores the answer
against the reference using ROUGE-L and keyword overlap.

Marked @pytest.mark.slow — skipped in CI unless RUN_EVAL=true.

Results saved to results/generation_eval.json.
"""

import os
import re
import sys
from typing import Dict, List

import pytest

# Ensure project root on path when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _extract_key_terms(text: str) -> List[str]:
    """
    Extract nouns and numbers from text using simple regex.

    Args:
        text: Input text string.

    Returns:
        List of lowercase key term strings.
    """
    # Numbers (including decimal, percentages, $ amounts)
    numbers = re.findall(r"\$?[\d,]+\.?\d*%?", text)
    # Capitalized words (proper nouns / company names)
    proper = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    # All terms lowercased and deduplicated
    terms = list({t.lower() for t in numbers + proper if len(t) > 1})
    return terms


def _rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between hypothesis and reference.

    Args:
        hypothesis: Generated answer.
        reference: Reference answer.

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure
    except Exception:
        # Fallback: simple token overlap
        hyp_tokens = set(hypothesis.lower().split())
        ref_tokens = set(reference.lower().split())
        if not ref_tokens:
            return 0.0
        overlap = len(hyp_tokens & ref_tokens)
        precision = overlap / len(hyp_tokens) if hyp_tokens else 0
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def _keyword_overlap(answer: str, question: str) -> float:
    """
    Compute keyword overlap: fraction of question key terms in answer.

    Args:
        answer: Generated answer text.
        question: Original question text.

    Returns:
        Float in [0, 1]: fraction of key terms from question found in answer.
    """
    key_terms = _extract_key_terms(question)
    if not key_terms:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for term in key_terms if term in answer_lower)
    return found / len(key_terms)


@pytest.mark.slow
def run_generation_eval(qa_pairs: List[dict]) -> Dict:
    """
    Evaluate generation quality with ROUGE-L and keyword overlap.

    Args:
        qa_pairs: List of QA pairs (subset recommended for speed).

    Returns:
        Dict with per-pair results and aggregate averages.
    """
    from app.core.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    per_pair = []
    total_rouge = 0.0
    total_keyword = 0.0

    for pair in qa_pairs:
        question = pair["question"]
        reference = pair["answer"]

        try:
            result = pipeline.query(question, top_k=5, retrieval_mode="hybrid")
            generated = result.get("answer", "")
        except Exception as exc:
            generated = f"ERROR: {exc}"

        rouge = _rouge_l(generated, reference)
        kw_overlap = _keyword_overlap(generated, question)

        total_rouge += rouge
        total_keyword += kw_overlap

        per_pair.append(
            {
                "question": question,
                "reference": reference,
                "generated": generated[:500],
                "rouge_l": round(rouge, 4),
                "keyword_overlap": round(kw_overlap, 4),
            }
        )

    n = len(qa_pairs)
    return {
        "avg_rouge_l": round(total_rouge / n, 4) if n > 0 else 0.0,
        "avg_keyword_overlap": round(total_keyword / n, 4) if n > 0 else 0.0,
        "total_evaluated": n,
        "per_pair": per_pair,
    }


def main() -> None:
    """Run generation evaluation as a standalone script."""
    import json

    from tests.evaluation.eval_dataset import generate_qa_dataset

    qa_pairs = generate_qa_dataset()[:10]  # Use 10 pairs for speed
    print(f"Evaluating generation on {len(qa_pairs)} QA pairs...")

    results = run_generation_eval(qa_pairs)

    print(f"\n  Average ROUGE-L:         {results['avg_rouge_l']:.4f}")
    print(f"  Average Keyword Overlap: {results['avg_keyword_overlap']:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "generation_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
