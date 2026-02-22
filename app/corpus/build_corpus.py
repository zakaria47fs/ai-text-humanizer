"""
Standalone script to build the ChromaDB corpus from human-written text samples.

Usage:
    python -m app.corpus.build_corpus
    python -m app.corpus.build_corpus --samples-dir app/corpus/samples --db-path ./chroma_db
"""

import argparse
import sys
import os

# Ensure the project root is in path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.rag import build_corpus


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB corpus from human writing samples.")
    parser.add_argument(
        "--samples-dir",
        default="app/corpus/samples",
        help="Directory containing .txt sample files (default: app/corpus/samples)"
    )
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path for ChromaDB persistence (default: ./chroma_db)"
    )
    args = parser.parse_args()

    print(f"Building corpus from: {args.samples_dir}")
    print(f"ChromaDB path: {args.db_path}")
    build_corpus(args.samples_dir, db_path=args.db_path)
    print("Done.")


if __name__ == "__main__":
    main()
