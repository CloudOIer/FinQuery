from __future__ import annotations

import argparse

from finquery_agent.config import load_rag_settings
from finquery_agent.rag.index import build_rag_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FinQuery research report RAG index.")
    parser.add_argument("--no-vector", action="store_true", help="Build only BM25-readable chunk files, skip FAISS vectors.")
    parser.add_argument("--strict-vector", action="store_true", help="Fail instead of falling back when FAISS vector build fails.")
    args = parser.parse_args()

    settings = load_rag_settings()
    use_vector = False if args.no_vector else settings.use_vector
    try:
        rag_index = build_rag_index(settings, use_vector=use_vector)
    except Exception as exc:
        if not use_vector or args.strict_vector:
            raise
        print(f"[warning] vector index build failed, falling back to BM25-only index: {exc}")
        rag_index = build_rag_index(settings, use_vector=False)
    print(
        f"RAG index built: documents={len(rag_index.documents)} "
        f"chunks={len(rag_index.chunks)} index_dir={rag_index.index_dir} "
        f"vector={rag_index.vector_index is not None}"
    )


if __name__ == "__main__":
    main()