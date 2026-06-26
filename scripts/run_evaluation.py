"""
Full evaluation suite runner.

Generates QA dataset → runs retrieval evaluation → runs generation evaluation
→ prints comparison table → saves JSON results to results/.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --skip-generation  # fast mode, retrieval only
"""

import argparse
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"


def _print_table(title: str, data: dict) -> None:
    """Print a formatted evaluation results table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'Semantic':>10} {'Keyword':>10} {'Hybrid':>10}")
    print(f"{'-' * 55}")
    for metric, values in data.items():
        row = f"{metric:<25}"
        for mode in ("semantic", "keyword", "hybrid"):
            val = values.get(mode, 0.0)
            row += f" {val:>10.4f}"
        print(row)
    print(f"{'=' * 60}")


def main() -> None:
    """Run the complete evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite.")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip slow generation evaluation (ROUGE-L).",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Financial RAG Engine — Evaluation Suite")
    print("=" * 60)

    # --- Step 1: Generate QA dataset ---
    print("\n[1/3] Generating QA dataset...")
    from tests.evaluation.eval_dataset import generate_qa_dataset

    qa_pairs = generate_qa_dataset()
    qa_path = os.path.join("tests", "evaluation", "qa_dataset.json")
    with open(qa_path, "w", encoding="utf-8") as fh:
        json.dump(qa_pairs, fh, indent=2)
    print(f"    ✓ {len(qa_pairs)} QA pairs saved to {qa_path}")

    # --- Step 2: Retrieval evaluation ---
    print("\n[2/3] Running retrieval evaluation (Precision@k, MRR)...")
    from tests.evaluation.eval_retrieval import run_retrieval_eval

    retrieval_results = run_retrieval_eval(qa_pairs)

    retrieval_path = os.path.join(RESULTS_DIR, "retrieval_eval.json")
    with open(retrieval_path, "w", encoding="utf-8") as fh:
        json.dump(retrieval_results, fh, indent=2)
    print(f"    ✓ Retrieval results saved to {retrieval_path}")

    # Print table
    _print_table(
        "Retrieval Evaluation Results",
        {
            "Precision@1": {
                m: retrieval_results[m]["precision_at_1"] for m in retrieval_results
            },
            "Precision@3": {
                m: retrieval_results[m]["precision_at_3"] for m in retrieval_results
            },
            "Precision@5": {
                m: retrieval_results[m]["precision_at_5"] for m in retrieval_results
            },
            "MRR": {m: retrieval_results[m]["mrr"] for m in retrieval_results},
        },
    )

    # Check targets
    hybrid_p5 = retrieval_results.get("hybrid", {}).get("precision_at_5", 0.0)
    keyword_p5 = retrieval_results.get("keyword", {}).get("precision_at_5", 0.0)
    hybrid_mrr = retrieval_results.get("hybrid", {}).get("mrr", 0.0)

    print(f"\n{'TARGET CHECKS':}")
    p_status = "✅ PASS" if hybrid_p5 > 0.88 else "❌ FAIL"
    print(f"  Precision@5 (hybrid) > 0.88: {p_status} ({hybrid_p5:.4f})")

    m_status = "✅ PASS" if hybrid_mrr > 0.75 else "❌ FAIL"
    print(f"  MRR (hybrid) > 0.75:          {m_status} ({hybrid_mrr:.4f})")

    comp_status = "✅ PASS" if hybrid_p5 > keyword_p5 else "❌ FAIL"
    print(
        f"  Hybrid P@5 > Keyword P@5:     {comp_status} "
        f"({hybrid_p5:.4f} vs {keyword_p5:.4f})"
    )

    # --- Step 3: Generation evaluation ---
    if not args.skip_generation:
        print("\n[3/3] Running generation evaluation (ROUGE-L)...")
        from tests.evaluation.eval_generation import run_generation_eval

        gen_results = run_generation_eval(qa_pairs[:10])  # subset for speed

        gen_path = os.path.join(RESULTS_DIR, "generation_eval.json")
        with open(gen_path, "w", encoding="utf-8") as fh:
            json.dump(gen_results, fh, indent=2)
        print(f"    ✓ Generation results saved to {gen_path}")

        avg_rouge = gen_results.get("avg_rouge_l", 0.0)
        print(
            f"  ROUGE-L > 0.25: {'✅ PASS' if avg_rouge > 0.25 else '❌ FAIL'} ({avg_rouge:.4f})"
        )
    else:
        print("\n[3/3] Generation evaluation skipped (--skip-generation flag set).")

    print("\n✅ Evaluation suite complete. Results in results/")


if __name__ == "__main__":
    main()
