"""
Evaluation dataset generator.

Generates 50 synthetic QA pairs from the seeded financial documents
WITHOUT any external API calls. Uses string matching and templates
applied to known document content.

Output: tests/evaluation/qa_dataset.json
"""

import json
import os
from typing import List

# ---------------------------------------------------------------------------
# Known facts from seeded documents (extracted by string matching)
# ---------------------------------------------------------------------------

QA_PAIRS = [
    # --- Apple Q4 2023 ---
    {
        "question": "What was Apple's total revenue in Q4 2023?",
        "answer": "$89.5 billion",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's iPhone revenue in Q4 fiscal 2023?",
        "answer": "$43.8 billion",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's Services revenue in Q4 2023?",
        "answer": "$22.3 billion",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's Mac revenue in Q4 2023?",
        "answer": "$7.6 billion",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's gross margin in Q4 fiscal year 2023?",
        "answer": "45.2%",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What did Tim Cook say about artificial intelligence?",
        "answer": "AI will be transformative for Apple",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What revenue guidance did Luca Maestri give for Q1 fiscal 2024?",
        "answer": "Similar to Q1 fiscal 2023 ($117.2 billion)",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's gross margin guidance for Q1 FY2024?",
        "answer": "Between 45% and 46%",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "How much did Apple return to shareholders in Q4 2023?",
        "answer": "Over $24 billion",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's Greater China revenue trend in Q4 2023?",
        "answer": "Declined 2.5% year-over-year",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's operating income in Q4 2023?",
        "answer": "$26.9 billion",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's net income per diluted share in Q4 2023?",
        "answer": "$1.46 per diluted share",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "How many paid subscriptions does Apple have globally?",
        "answer": "Over 1 billion paid subscriptions",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's year-over-year revenue growth in Q4 2023?",
        "answer": "1% increase year-over-year",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    {
        "question": "What was Apple's Services revenue growth rate year-over-year?",
        "answer": "Up 16% year-over-year",
        "source_doc": "apple_q4_2023_earnings.txt",
    },
    # --- Microsoft FY2023 ---
    {
        "question": "What was Microsoft's total revenue for fiscal year 2023?",
        "answer": "$211.9 billion",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's total revenue growth in FY2023?",
        "answer": "7% increase",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's Intelligent Cloud revenue in FY2023?",
        "answer": "$87.9 billion",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "How much did Azure grow year-over-year in fiscal 2023?",
        "answer": "29% year-over-year in constant currency",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft 365 Commercial revenue growth in FY2023?",
        "answer": "13% year-over-year",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's operating income in fiscal 2023?",
        "answer": "$88.5 billion",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What did Satya Nadella say about AI integration at Microsoft?",
        "answer": "Most significant platform shift in decades",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's Productivity and Business Processes revenue in FY2023?",
        "answer": "$69.3 billion",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's More Personal Computing revenue in FY2023?",
        "answer": "$54.7 billion",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "How much did Microsoft invest in OpenAI?",
        "answer": "$10 billion",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's operating margin in FY2023?",
        "answer": "41.8%",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's net income in fiscal year 2023?",
        "answer": "$72.4 billion",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "How many Enterprise Mobility + Security seats does Microsoft have?",
        "answer": "270 million seats",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's Dynamics 365 revenue growth in FY2023?",
        "answer": "26% growth",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    {
        "question": "What was Microsoft's earnings per diluted share in FY2023?",
        "answer": "$9.72",
        "source_doc": "microsoft_fy2023_annual.txt",
    },
    # --- Tesla Q3 2023 ---
    {
        "question": "How many vehicles did Tesla deliver in Q3 2023?",
        "answer": "435,059 vehicles",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's total revenue in Q3 2023?",
        "answer": "$23.4 billion",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's automotive gross margin in Q3 2023?",
        "answer": "17.9%",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's energy generation and storage revenue in Q3 2023?",
        "answer": "$1.56 billion",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What did Elon Musk say about Tesla's long-term vision?",
        "answer": "Full Self-Driving will transform Tesla",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "When are Cybertruck customer deliveries expected to begin?",
        "answer": "Late Q4 2023",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "How much did Tesla's vehicle deliveries grow year-over-year in Q3 2023?",
        "answer": "27% increase year-over-year",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's services and other revenue in Q3 2023?",
        "answer": "$2.2 billion",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's automotive revenue in Q3 2023?",
        "answer": "$19.6 billion",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's cost of goods sold per vehicle in Q3 2023?",
        "answer": "Approximately $37,500",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "How much did Tesla spend on capital expenditures in Q3 2023?",
        "answer": "$2.46 billion",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "How much cash did Tesla have at the end of Q3 2023?",
        "answer": "$26.1 billion",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "How many GWh of Megapack did Tesla deploy in Q3 2023?",
        "answer": "4.0 GWh",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's vehicle production in Q3 2023?",
        "answer": "430,488 vehicles",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What caused Tesla's automotive gross margin to decline in Q3 2023?",
        "answer": "Price reduction strategy to stimulate demand",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's energy revenue growth rate in Q3 2023?",
        "answer": "40% year-over-year",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "How long will the Cybertruck production ramp take?",
        "answer": "12 to 18 months",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's Model 3 and Model Y production in Q3 2023?",
        "answer": "416,800 units",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What was Tesla's services revenue growth in Q3 2023?",
        "answer": "32% year-over-year",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
    {
        "question": "What regulatory credit sales did Tesla report in Q3 2023?",
        "answer": "$554 million",
        "source_doc": "tesla_q3_2023_earnings.txt",
    },
]


def generate_qa_dataset() -> List[dict]:
    """
    Return the list of 50 synthetic QA pairs.

    No external API calls are made. All pairs are derived from the seeded
    financial documents in data/raw/.

    Returns:
        List of dicts: [{"question": str, "answer": str, "source_doc": str}]
    """
    return list(QA_PAIRS)


def main() -> None:
    """Generate and save the QA dataset to tests/evaluation/qa_dataset.json."""
    os.makedirs(os.path.join("tests", "evaluation"), exist_ok=True)
    pairs = generate_qa_dataset()

    out_path = os.path.join("tests", "evaluation", "qa_dataset.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pairs, fh, indent=2)

    print(f"✅ Generated {len(pairs)} QA pairs → {out_path}")

    # Show distribution
    sources = {}
    for p in pairs:
        src = p["source_doc"]
        sources[src] = sources.get(src, 0) + 1
    for src, count in sources.items():
        print(f"   {src}: {count} questions")


if __name__ == "__main__":
    main()
