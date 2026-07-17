from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from finquery_agent.ingestion.financial_staging import FinancialStagingRecord, extract_financial_staging_records
from finquery_agent.ingestion.metadata import infer_report_metadata, refine_report_metadata_from_text
from finquery_agent.ingestion.models import ExtractedPage, ExtractedTable, ParsedPdfDocument, ReportMetadata
from finquery_agent.ingestion.period_derivations import derive_period_growth_records
from finquery_agent.schema import load_default_registry


def store_parsed_document(engine: Engine, parsed: ParsedPdfDocument, tool_name: str = "pdfplumber") -> int:
    with engine.begin() as connection:
        document_id = _upsert_source_document(connection, parsed.metadata)
        run_id = connection.execute(
            text(
                """
                INSERT INTO extraction_runs (document_id, tool_name, tool_version, status, finished_at, metadata)
                VALUES (:document_id, :tool_name, :tool_version, 'completed', now(), :metadata)
                RETURNING run_id
                """
            ),
            {
                "document_id": document_id,
                "tool_name": tool_name,
                "tool_version": "local",
                "metadata": json.dumps(
                    {
                        "markdown_path": str(parsed.markdown_path),
                        "clean_markdown_path": str(parsed.clean_markdown_path),
                        "page_count": len(parsed.pages),
                    },
                    ensure_ascii=False,
                ),
            },
        ).scalar_one()

        for page in parsed.pages:
            connection.execute(
                text(
                    """
                    INSERT INTO extracted_pages (run_id, page_number, text_content, markdown_content, metadata)
                    VALUES (:run_id, :page_number, :text_content, :markdown_content, '{}'::jsonb)
                    ON CONFLICT (run_id, page_number) DO UPDATE SET
                        text_content = EXCLUDED.text_content,
                        markdown_content = EXCLUDED.markdown_content
                    """
                ),
                {
                    "run_id": run_id,
                    "page_number": page.page_number,
                    "text_content": page.text_content,
                    "markdown_content": page.markdown_content,
                },
            )
            for table in page.tables:
                connection.execute(
                    text(
                        """
                        INSERT INTO extracted_tables
                            (run_id, page_number, table_index, section_title, raw_markdown, raw_json, confidence)
                        VALUES
                            (:run_id, :page_number, :table_index, :section_title, :raw_markdown, :raw_json, :confidence)
                        ON CONFLICT (run_id, table_index) DO UPDATE SET
                            page_number = EXCLUDED.page_number,
                            section_title = EXCLUDED.section_title,
                            raw_markdown = EXCLUDED.raw_markdown,
                            raw_json = EXCLUDED.raw_json,
                            confidence = EXCLUDED.confidence
                        """
                    ),
                    {
                        "run_id": run_id,
                        "page_number": table.page_number,
                        "table_index": table.table_index,
                        "section_title": table.section_title,
                        "raw_markdown": table.markdown,
                        "raw_json": json.dumps(table.raw_rows, ensure_ascii=False),
                        "confidence": None,
                    },
                )
        registry = load_default_registry()
        records = extract_financial_staging_records(parsed, registry)
        _insert_staging_records(connection, document_id, run_id, parsed.metadata, records)
        period_growth_records = derive_period_growth_records(connection, run_id, parsed.metadata, registry)
        _insert_staging_records(connection, document_id, run_id, parsed.metadata, period_growth_records)
        return int(run_id)


def rebuild_staging_for_run(engine: Engine, run_id: int) -> int:
    with engine.begin() as connection:
        parsed = _load_parsed_document_for_run(connection, run_id)
        metadata = parsed.metadata
        connection.execute(
            text(
                """
                UPDATE source_documents
                SET stock_code = :stock_code,
                    report_year = :report_year,
                    report_period = :report_period,
                    file_date = :file_date
                WHERE document_id = :document_id
                """
            ),
            {
                "document_id": _document_id_for_run(connection, run_id),
                "stock_code": metadata.stock_code,
                "report_year": metadata.report_year,
                "report_period": metadata.report_period,
                "file_date": metadata.file_date,
            },
        )
        connection.execute(text("DELETE FROM validation_results WHERE run_id = :run_id"), {"run_id": run_id})
        connection.execute(text("DELETE FROM financial_staging WHERE run_id = :run_id"), {"run_id": run_id})
        connection.execute(text("DELETE FROM field_mapping_logs WHERE run_id = :run_id"), {"run_id": run_id})

        registry = load_default_registry()
        records = extract_financial_staging_records(parsed, registry)
        document_id = _document_id_for_run(connection, run_id)
        _insert_staging_records(connection, document_id, run_id, metadata, records)
        period_growth_records = derive_period_growth_records(connection, run_id, metadata, registry)
        _insert_staging_records(connection, document_id, run_id, metadata, period_growth_records)
        return len(records) + len(period_growth_records)


def _insert_staging_records(
    connection,
    document_id: int,
    run_id: int,
    metadata: ReportMetadata,
    records: tuple[FinancialStagingRecord, ...],
) -> None:
    if not records or not metadata.stock_code or metadata.report_year is None or not metadata.report_period:
        return
    for record in records:
        connection.execute(
            text(
                """
                INSERT INTO financial_staging
                    (run_id, document_id, stock_code, report_year, report_period, target_table, target_field,
                     source_label, raw_value, raw_unit, standard_value, standard_unit, period_scope,
                     source_period_label, is_derived, derivation_formula, page_number, table_id,
                     confidence, validation_status)
                VALUES
                    (:run_id, :document_id, :stock_code, :report_year, :report_period, :target_table, :target_field,
                     :source_label, :raw_value, :raw_unit, :standard_value, :standard_unit, :period_scope,
                     :source_period_label, :is_derived, :derivation_formula, :page_number,
                     (SELECT table_id FROM extracted_tables WHERE run_id = :run_id AND table_index = :table_index LIMIT 1),
                     :confidence, 'pending')
                ON CONFLICT (run_id, target_table, target_field, stock_code, report_year, report_period, period_scope, is_derived)
                DO UPDATE SET
                    source_label = EXCLUDED.source_label,
                    raw_value = EXCLUDED.raw_value,
                    raw_unit = EXCLUDED.raw_unit,
                    standard_value = EXCLUDED.standard_value,
                    standard_unit = EXCLUDED.standard_unit,
                    period_scope = EXCLUDED.period_scope,
                    source_period_label = EXCLUDED.source_period_label,
                    is_derived = EXCLUDED.is_derived,
                    derivation_formula = EXCLUDED.derivation_formula,
                    page_number = EXCLUDED.page_number,
                    table_id = EXCLUDED.table_id,
                    confidence = EXCLUDED.confidence,
                    validation_status = EXCLUDED.validation_status
                """
            ),
            {
                "run_id": run_id,
                "document_id": document_id,
                "stock_code": metadata.stock_code,
                "report_year": metadata.report_year,
                "report_period": metadata.report_period,
                "target_table": record.target_table,
                "target_field": record.target_field,
                "source_label": record.source_label,
                "raw_value": record.raw_value,
                "raw_unit": record.raw_unit,
                "standard_value": record.standard_value,
                "standard_unit": record.standard_unit,
                "period_scope": record.period_scope,
                "source_period_label": record.source_period_label,
                "is_derived": record.is_derived,
                "derivation_formula": record.derivation_formula,
                "page_number": record.page_number,
                "table_index": record.table_index,
                "confidence": record.confidence,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO field_mapping_logs
                    (run_id, source_label, target_table, target_field, mapping_method, confidence, metadata)
                VALUES
                    (:run_id, :source_label, :target_table, :target_field, :mapping_method, :confidence,
                     :metadata)
                """
            ),
            {
                "run_id": run_id,
                "source_label": record.source_label,
                "target_table": record.target_table,
                "target_field": record.target_field,
                "mapping_method": "derived_formula" if record.is_derived else "deterministic_alias",
                "confidence": record.confidence,
                "metadata": json.dumps(
                    {
                        "page_number": record.page_number,
                        "table_index": record.table_index,
                        "raw_unit": record.raw_unit,
                        "source_period_label": record.source_period_label,
                        "is_derived": record.is_derived,
                        "derivation_formula": record.derivation_formula,
                    },
                    ensure_ascii=False,
                ),
            },
        )


def _document_id_for_run(connection, run_id: int) -> int:
    return int(
        connection.execute(text("SELECT document_id FROM extraction_runs WHERE run_id = :run_id"), {"run_id": run_id}).scalar_one()
    )


def _load_parsed_document_for_run(connection, run_id: int) -> ParsedPdfDocument:
    run = connection.execute(
        text(
            """
            SELECT er.run_id, er.metadata, sd.source_path
            FROM extraction_runs er
            JOIN source_documents sd ON sd.document_id = er.document_id
            WHERE er.run_id = :run_id
            """
        ),
        {"run_id": run_id},
    ).mappings().one()
    source_path = Path(run["source_path"])
    pages = _load_pages_for_run(connection, run_id)
    markdown = "\n\n".join(page.markdown_content for page in pages if page.markdown_content)
    page_text = "\n\n".join(page.text_content for page in pages if page.text_content)
    metadata = refine_report_metadata_from_text(infer_report_metadata(source_path), markdown or page_text)
    run_metadata = run["metadata"] or {}
    return ParsedPdfDocument(
        metadata=metadata,
        markdown=markdown,
        clean_markdown="",
        pages=pages,
        markdown_path=Path(run_metadata.get("markdown_path") or ""),
        clean_markdown_path=Path(run_metadata.get("clean_markdown_path") or ""),
    )


def _load_pages_for_run(connection, run_id: int) -> tuple[ExtractedPage, ...]:
    page_rows = connection.execute(
        text(
            """
            SELECT page_number, text_content, markdown_content
            FROM extracted_pages
            WHERE run_id = :run_id
            ORDER BY page_number
            """
        ),
        {"run_id": run_id},
    ).mappings().all()
    table_rows = connection.execute(
        text(
            """
            SELECT page_number, table_index, section_title, raw_markdown, raw_json
            FROM extracted_tables
            WHERE run_id = :run_id
            ORDER BY page_number, table_index
            """
        ),
        {"run_id": run_id},
    ).mappings().all()
    tables_by_page: dict[int, list[ExtractedTable]] = {}
    for row in table_rows:
        raw_rows = row["raw_json"] or []
        if isinstance(raw_rows, str):
            raw_rows = json.loads(raw_rows)
        tables_by_page.setdefault(row["page_number"], []).append(
            ExtractedTable(
                page_number=row["page_number"],
                table_index=row["table_index"],
                markdown=row["raw_markdown"] or "",
                raw_rows=raw_rows,
                section_title=row["section_title"],
            )
        )
    return tuple(
        ExtractedPage(
            page_number=row["page_number"],
            text_content=row["text_content"] or "",
            markdown_content=row["markdown_content"] or "",
            tables=tuple(tables_by_page.get(row["page_number"], ())),
        )
        for row in page_rows
    )


def _upsert_source_document(connection, metadata: ReportMetadata) -> int:
    path = metadata.source_path.resolve()
    digest = _sha256(path)
    return connection.execute(
        text(
            """
            INSERT INTO source_documents
                (source_path, document_type, stock_code, report_year, report_period, file_date, sha256)
            VALUES
                (:source_path, 'financial_report_pdf', :stock_code, :report_year, :report_period, :file_date, :sha256)
            ON CONFLICT (source_path) DO UPDATE SET
                stock_code = EXCLUDED.stock_code,
                report_year = EXCLUDED.report_year,
                report_period = EXCLUDED.report_period,
                file_date = EXCLUDED.file_date,
                sha256 = EXCLUDED.sha256
            RETURNING document_id
            """
        ),
        {
            "source_path": str(path),
            "stock_code": metadata.stock_code,
            "report_year": metadata.report_year,
            "report_period": metadata.report_period,
            "file_date": metadata.file_date,
            "sha256": digest,
        },
    ).scalar_one()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
