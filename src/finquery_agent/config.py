from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    postgres_data_dir: Path
    postgres_log_file: Path
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    llm_config_file: Path
    rag_config_file: Path

    @property
    def database_url(self) -> str:
        explicit_url = os.getenv("DATABASE_URL")
        if explicit_url:
            return explicit_url
        return (
            f"postgresql+psycopg://{self.postgres_user}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.getenv("FINQUERY_DATA_ROOT", project_root / "第一批数据"))
    postgres_data_dir = Path(os.getenv("FINQUERY_PGDATA", project_root / ".local" / "postgres"))
    return Settings(
        project_root=project_root,
        data_root=data_root,
        postgres_data_dir=postgres_data_dir,
        postgres_log_file=Path(os.getenv("FINQUERY_PGLOG", project_root / ".local" / "postgres.log")),
        postgres_host=os.getenv("FINQUERY_PGHOST", "127.0.0.1"),
        postgres_port=int(os.getenv("FINQUERY_PGPORT", "55432")),
        postgres_db=os.getenv("FINQUERY_PGDATABASE", "finquery"),
        postgres_user=os.getenv("FINQUERY_PGUSER", os.getenv("USER", "postgres")),
        llm_config_file=Path(os.getenv("FINQUERY_LLM_CONFIG", project_root / "config" / "llm.json")),
        rag_config_file=Path(os.getenv("FINQUERY_RAG_CONFIG", project_root / "config" / "rag.json")),
    )


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool = False
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str | None = None
    timeout_seconds: int = 60
    # 意图识别是否走 LLM(HybridIntentEngine)。与 enabled 分开是因为:
    # 答案润色失败只影响表达,意图解析失败会改变查询结果,风险等级不同,
    # 需要能单独开关(例如评测时只开答案 LLM、关意图 LLM 做消融)。
    intent_enabled: bool = False


def load_llm_settings(config_file: Path | None = None) -> LLMSettings:
    settings = get_settings()
    path = config_file or settings.llm_config_file
    if not path.exists():
        return LLMSettings()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LLMSettings(
        enabled=bool(payload.get("enabled", False)),
        provider=str(payload.get("provider", "")),
        model=str(payload.get("model", "")),
        api_key=str(payload.get("api_key", "")),
        base_url=payload.get("base_url"),
        timeout_seconds=int(payload.get("timeout_seconds", 60)),
        intent_enabled=bool(payload.get("intent_enabled", False)),
    )


@dataclass(frozen=True)
class RAGSettings:
    enabled: bool = True
    data_roots: tuple[Path, ...] = ()
    stock_metadata_file: Path | None = None
    industry_metadata_file: Path | None = None
    index_dir: Path | None = None
    chunk_size: int = 800
    chunk_overlap: int = 150
    use_vector: bool = True
    embedding_model: str = "bge-base-zh-v1.5"
    bm25_top_k: int = 20
    vector_top_k: int = 20
    final_top_k: int = 8
    embedding_batch_size: int = 32
    # 两阶段检索:粗排(BM25+向量融合)取 rerank_candidate_k 个候选,再用
    # cross-encoder 精排。精排模型加载失败时自动降级为粗排结果。
    use_reranker: bool = True
    reranker_model: str = "bge-reranker-base"
    rerank_candidate_k: int = 30
    rerank_batch_size: int = 16
    # 同一文档最多返回的 chunk 数(0=不限),避免 top_k 被单篇文档占满。
    max_chunks_per_doc: int = 2


def default_rag_settings() -> RAGSettings:
    settings = get_settings()
    data_root = settings.data_root
    return RAGSettings(
        data_roots=(data_root / "研报数据" / "研报markdown",),
        stock_metadata_file=data_root / "研报信息" / "个股_研报信息.CSV",
        industry_metadata_file=data_root / "研报信息" / "行业_研报信息.CSV",
        index_dir=settings.project_root / "data" / "rag",
    )


def load_rag_settings(config_file: Path | None = None) -> RAGSettings:
    defaults = default_rag_settings()
    settings = get_settings()
    path = config_file or settings.rag_config_file
    if not path.exists():
        return defaults
    payload = json.loads(path.read_text(encoding="utf-8"))
    data_roots = tuple(Path(item) for item in payload.get("data_roots", ())) or defaults.data_roots
    metadata_files = payload.get("metadata_files", {}) or {}
    retrieval = payload.get("retrieval", {}) or {}
    return RAGSettings(
        enabled=bool(payload.get("enabled", defaults.enabled)),
        data_roots=tuple(_resolve_project_path(path_item) for path_item in data_roots),
        stock_metadata_file=_resolve_project_path(metadata_files.get("stock_reports")) if metadata_files.get("stock_reports") else defaults.stock_metadata_file,
        industry_metadata_file=_resolve_project_path(metadata_files.get("industry_reports")) if metadata_files.get("industry_reports") else defaults.industry_metadata_file,
        index_dir=_resolve_project_path(payload.get("index_dir")) if payload.get("index_dir") else defaults.index_dir,
        chunk_size=int(payload.get("chunk_size", defaults.chunk_size)),
        chunk_overlap=int(payload.get("chunk_overlap", defaults.chunk_overlap)),
        use_vector=bool(retrieval.get("use_vector", defaults.use_vector)),
        embedding_model=str(retrieval.get("embedding_model", defaults.embedding_model)),
        bm25_top_k=int(retrieval.get("bm25_top_k", defaults.bm25_top_k)),
        vector_top_k=int(retrieval.get("vector_top_k", defaults.vector_top_k)),
        final_top_k=int(retrieval.get("final_top_k", defaults.final_top_k)),
        embedding_batch_size=int(retrieval.get("embedding_batch_size", defaults.embedding_batch_size)),
        use_reranker=bool(retrieval.get("use_reranker", defaults.use_reranker)),
        reranker_model=str(retrieval.get("reranker_model", defaults.reranker_model)),
        rerank_candidate_k=int(retrieval.get("rerank_candidate_k", defaults.rerank_candidate_k)),
        rerank_batch_size=int(retrieval.get("rerank_batch_size", defaults.rerank_batch_size)),
        max_chunks_per_doc=int(retrieval.get("max_chunks_per_doc", defaults.max_chunks_per_doc)),
    )


def _resolve_project_path(value: str | Path | None) -> Path:
    if value is None:
        return Path()
    path = Path(value)
    if path.is_absolute():
        return path
    return get_settings().project_root / path
