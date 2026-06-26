"""
Retrieval evaluation: Precision@k and MRR.

For each QA pair, checks whether the correct source document appears
in the top-k retrieved chunks for each retrieval mode.

Metrics computed per mode (semantic, keyword, hybrid):
- Precision@1: correct doc is top-1 result
- Precision@3: correct doc appears in top-3 results
- Precision@5: correct doc appears in top-5 results
- MRR: Mean Reciprocal Rank (1/rank of first correct result)

Results saved to results/retrieval_eval.json.
"""

import os
import sys
from typing import Dict, List

# Ensure project root on path when run standalone
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def run_retrieval_eval(qa_pairs: List[dict]) -> Dict[str, dict]:
    """
    Evaluate retrieval performance across all three modes.

    Args:
        qa_pairs: List of QA pair dicts with 'question', 'answer', 'source_doc'.

    Returns:
        Dict mapping mode → metric_dict with precision_at_1/3/5 and mrr.
    """
    from app.core.retriever import get_global_retriever

    retriever = get_global_retriever()
    modes = ["semantic", "keyword", "hybrid"]
    results = {}

    for mode in modes:
        p_at_1 = p_at_3 = p_at_5 = 0
        mrr_sum = 0.0
        total = len(qa_pairs)

        for pair in qa_pairs:
            question = pair["question"]
            expected_source = pair["source_doc"]

            try:
                chunks = retriever.search(question, top_k=5, mode=mode)
            except Exception:
                chunks = []

            sources = [c.get("source", "") for c in chunks]

            # Precision@1
            if sources and sources[0] == expected_source:
                p_at_1 += 1

            # Precision@3
            if expected_source in sources[:3]:
                p_at_3 += 1

            # Precision@5
            if expected_source in sources[:5]:
                p_at_5 += 1

            # MRR: 1/rank of first correct result
            for rank, source in enumerate(sources, start=1):
                if source == expected_source:
                    mrr_sum += 1.0 / rank
                    break

        results[mode] = {
            "precision_at_1": round(p_at_1 / total, 4) if total > 0 else 0.0,
            "precision_at_3": round(p_at_3 / total, 4) if total > 0 else 0.0,
            "precision_at_5": round(p_at_5 / total, 4) if total > 0 else 0.0,
            "mrr": round(mrr_sum / total, 4) if total > 0 else 0.0,
            "total_questions": total,
        }

    return results


def main() -> None:
    """Run retrieval evaluation as a standalone script."""
    import json

    from tests.evaluation.eval_dataset import generate_qa_dataset

    qa_pairs = generate_qa_dataset()
    print(f"Evaluating retrieval on {len(qa_pairs)} QA pairs...")

    results = run_retrieval_eval(qa_pairs)

    # Print table
    print(f"\n{'Metric':<20} {'Semantic':>10} {'Keyword':>10} {'Hybrid':>10}")
    print("-" * 52)
    for metric in ["precision_at_1", "precision_at_3", "precision_at_5", "mrr"]:
        row = f"{metric:<20}"
        for mode in ["semantic", "keyword", "hybrid"]:
            row += f" {results[mode][metric]:>10.4f}"
        print(row)

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "retrieval_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
