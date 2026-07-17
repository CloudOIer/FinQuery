from finquery_agent.config import LLMSettings, load_llm_settings, load_rag_settings


def test_load_llm_settings_returns_disabled_when_file_missing(tmp_path):
    assert load_llm_settings(tmp_path / "missing.json") == LLMSettings()


def test_load_llm_settings_reads_json_config(tmp_path):
    config_path = tmp_path / "llm.json"
    config_path.write_text(
        '{"enabled": true, "provider": "openai-compatible", "model": "demo", "api_key": "secret", "base_url": "https://example.test/v1", "timeout_seconds": 12}',
        encoding="utf-8",
    )

    settings = load_llm_settings(config_path)

    assert settings.enabled is True
    assert settings.provider == "openai-compatible"
    assert settings.model == "demo"
    assert settings.api_key == "secret"
    assert settings.base_url == "https://example.test/v1"
    assert settings.timeout_seconds == 12


def test_load_rag_settings_reads_json_config(tmp_path):
        markdown_dir = tmp_path / "md"
        stock_meta = tmp_path / "stock.csv"
        industry_meta = tmp_path / "industry.csv"
        index_dir = tmp_path / "index"
        config_path = tmp_path / "rag.json"
        config_path.write_text(
                f"""{{
                    "enabled": true,
                    "data_roots": ["{markdown_dir}"],
                    "metadata_files": {{
                        "stock_reports": "{stock_meta}",
                        "industry_reports": "{industry_meta}"
                    }},
                    "index_dir": "{index_dir}",
                    "chunk_size": 500,
                    "chunk_overlap": 80,
                    "retrieval": {{
                        "use_vector": false,
                        "embedding_model": "demo-model",
                        "bm25_top_k": 11,
                        "vector_top_k": 12,
                        "final_top_k": 5,
                        "embedding_batch_size": 4
                    }}
                }}""",
                encoding="utf-8",
        )

        settings = load_rag_settings(config_path)

        assert settings.enabled is True
        assert settings.data_roots == (markdown_dir,)
        assert settings.stock_metadata_file == stock_meta
        assert settings.industry_metadata_file == industry_meta
        assert settings.index_dir == index_dir
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 80
        assert settings.use_vector is False
        assert settings.embedding_model == "demo-model"
        assert settings.bm25_top_k == 11
        assert settings.vector_top_k == 12
        assert settings.final_top_k == 5
        assert settings.embedding_batch_size == 4
