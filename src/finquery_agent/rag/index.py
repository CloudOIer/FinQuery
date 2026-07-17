from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finquery_agent.config import RAGSettings
from finquery_agent.rag.loader import chunk_documents, load_research_documents
from finquery_agent.rag.models import ResearchChunk, ResearchDocument


@dataclass
class RAGIndex:
    documents: list[ResearchDocument]
    chunks: list[ResearchChunk]
    index_dir: Path
    vector_index: Any | None = None
    vector_chunk_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.index_dir / "documents.jsonl", [document.to_dict() for document in self.documents])
        _write_jsonl(self.index_dir / "chunks.jsonl", [chunk.to_dict() for chunk in self.chunks])
        meta = self.metadata or {}
        meta.update({"document_count": len(self.documents), "chunk_count": len(self.chunks)})
        (self.index_dir / "index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.vector_index is not None:
            import faiss

            faiss.write_index(self.vector_index, str(self.index_dir / "faiss.index"))
            (self.index_dir / "vector_chunk_ids.json").write_text(json.dumps(self.vector_chunk_ids or [], ensure_ascii=False), encoding="utf-8")


def build_rag_index(settings: RAGSettings, use_vector: bool | None = None) -> RAGIndex:
    use_vector = settings.use_vector if use_vector is None else use_vector
    documents = load_research_documents(settings)
    chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    vector_index = None
    vector_chunk_ids: list[str] | None = None
    metadata = {
        "embedding_model": settings.embedding_model if use_vector else None,
        "use_vector": use_vector,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    if use_vector and chunks:
        vector_index, vector_chunk_ids = _build_faiss_index(chunks, settings.embedding_model, settings.embedding_batch_size)
    rag_index = RAGIndex(
        documents=documents,
        chunks=chunks,
        index_dir=settings.index_dir or Path("data/rag"),
        vector_index=vector_index,
        vector_chunk_ids=vector_chunk_ids,
        metadata=metadata,
    )
    rag_index.save()
    return rag_index


def load_rag_index(index_dir: Path) -> RAGIndex:
    documents = [ResearchDocument.from_dict(item) for item in _read_jsonl(index_dir / "documents.jsonl")]
    chunks = [ResearchChunk.from_dict(item) for item in _read_jsonl(index_dir / "chunks.jsonl")]
    metadata_path = index_dir / "index_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    vector_index = None
    vector_chunk_ids = None
    faiss_path = index_dir / "faiss.index"
    ids_path = index_dir / "vector_chunk_ids.json"
    if faiss_path.exists() and ids_path.exists():
        import faiss

        vector_index = faiss.read_index(str(faiss_path))
        vector_chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    return RAGIndex(
        documents=documents,
        chunks=chunks,
        index_dir=index_dir,
        vector_index=vector_index,
        vector_chunk_ids=vector_chunk_ids,
        metadata=metadata,
    )


def _build_faiss_index(chunks: list[ResearchChunk], model_name: str, batch_size: int):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [_embedding_text(chunk) for chunk in chunks]
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    matrix = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, [chunk.chunk_id for chunk in chunks]


def _embedding_text(chunk: ResearchChunk) -> str:
    parts = [chunk.title, chunk.section_title, chunk.stock_name, chunk.industry_name, chunk.text]
    return "\n".join(part for part in parts if part)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]