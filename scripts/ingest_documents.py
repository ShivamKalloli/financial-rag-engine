"""
CLI ingestion runner.

Scans a directory for PDF and TXT files and ingests each into the
FAISS index using the IngestionPipeline.

Usage:
    python scripts/ingest_documents.py --path data/raw
    python scripts/ingest_documents.py --path data/raw --verbose
"""

import argparse
import os
import sys
import time

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    """Scan a directory and ingest all supported documents."""
    parser = argparse.ArgumentParser(description="Ingest financial documents into the FAISS index.")
    parser.add_argument(
        "--path",
        required=True,
        help="Directory containing .pdf or .txt files to ingest.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress per file.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"ERROR: Path does not exist or is not a directory: {args.path}")
        sys.exit(1)

    # Collect supported files
    supported_extensions = {".pdf", ".txt"}
    files = [
        os.path.join(args.path, f)
        for f in sorted(os.listdir(args.path))
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if not files:
        print(f"No supported documents (.pdf, .txt) found in: {args.path}")
        sys.exit(0)

    print(f"Found {len(files)} file(s) to process in: {args.path}")
    print("-" * 60)

    # Import pipeline after sys.path is set
    from app.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()

    total_start = time.time()
    total_chunks = 0
    processed = 0
    skipped = 0
    errors = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            result = pipeline.ingest(file_path)
            if result["skipped"]:
                print(f"  ⟳ {filename} — already ingested (skipped)")
                skipped += 1
            else:
                print(
                    f"  ✓ {filename} — "
                    f"{result['chunks_added']} chunks added "
                    f"(doc_id: {result['doc_id'][:8]}...)"
                )
                total_chunks += result["chunks_added"]
                processed += 1
        except Exception as exc:
            print(f"  ✗ {filename} — ERROR: {exc}")
            errors += 1

    elapsed = time.time() - total_start
    print("-" * 60)
    print(f"✅ Ingestion complete in {elapsed:.1f}s")
    print(f"   Documents processed:  {processed}")
    print(f"   Chunks added:         {total_chunks}")
    print(f"   Already ingested:     {skipped}")
    print(f"   Errors:               {errors}")
    print()
    print("   Next: python app/main.py  OR  make run")


if __name__ == "__main__":
    main()
