from __future__ import annotations

import re

from finquery_agent.schema.models import FieldDefinition, TableDefinition
from finquery_agent.schema.registry import SchemaRegistry


class DDLGenerator:
    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

    def generate_all(self) -> str:
        statements = [self.generate_company_table()]
        statements.extend(self.generate_financial_table(table) for table in self.registry.tables.values())
        statements.extend(self.generate_ingestion_tables())
        return "\n\n".join(statements)

    def generate_company_table(self) -> str:
        return "\n".join(
            [
                "CREATE TABLE IF NOT EXISTS company_info (",
                "    stock_code VARCHAR(20) PRIMARY KEY,",
                "    stock_abbr VARCHAR(50) NOT NULL,",
                "    company_name VARCHAR(255) NOT NULL,",
                "    exchange VARCHAR(100),",
                "    industry VARCHAR(255)",
                ");",
            ]
        )

    def generate_financial_table(self, table: TableDefinition) -> str:
        column_lines = [f"    {field.name} {_to_sql_type(field)}" for field in table.fields]
        column_lines.extend(
            [
                "    PRIMARY KEY (stock_code, report_year, report_period)",
                "    FOREIGN KEY (stock_code) REFERENCES company_info(stock_code)",
            ]
        )
        return "\n".join(
            [
                f"CREATE TABLE IF NOT EXISTS {table.name} (",
                ",\n".join(column_lines),
                ");",
            ]
        )

    def generate_ingestion_tables(self) -> list[str]:
        return [
            _statement(
                "source_documents",
                [
                    "document_id BIGSERIAL PRIMARY KEY",
                    "source_path TEXT NOT NULL UNIQUE",
                    "document_type VARCHAR(50) NOT NULL",
                    "stock_code VARCHAR(20)",
                    "report_year INTEGER",
                    "report_period VARCHAR(20)",
                    "file_date DATE",
                    "sha256 VARCHAR(64)",
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
                ],
            ),
            _statement(
                "extraction_runs",
                [
                    "run_id BIGSERIAL PRIMARY KEY",
                    "document_id BIGINT NOT NULL REFERENCES source_documents(document_id)",
                    "tool_name VARCHAR(100) NOT NULL",
                    "tool_version VARCHAR(100)",
                    "status VARCHAR(50) NOT NULL DEFAULT 'pending'",
                    "started_at TIMESTAMPTZ NOT NULL DEFAULT now()",
                    "finished_at TIMESTAMPTZ",
                    "error_message TEXT",
                    "metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
                ],
            ),
            _statement(
                "extracted_pages",
                [
                    "page_id BIGSERIAL PRIMARY KEY",
                    "run_id BIGINT NOT NULL REFERENCES extraction_runs(run_id)",
                    "page_number INTEGER NOT NULL",
                    "text_content TEXT",
                    "markdown_content TEXT",
                    "metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
                    "UNIQUE (run_id, page_number)",
                ],
            ),
            _statement(
                "extracted_tables",
                [
                    "table_id BIGSERIAL PRIMARY KEY",
                    "run_id BIGINT NOT NULL REFERENCES extraction_runs(run_id)",
                    "page_number INTEGER",
                    "table_index INTEGER NOT NULL",
                    "section_title TEXT",
                    "raw_markdown TEXT",
                    "raw_json JSONB NOT NULL DEFAULT '{}'::jsonb",
                    "confidence NUMERIC(8, 6)",
                    "UNIQUE (run_id, table_index)",
                ],
            ),
            _statement(
                "financial_staging",
                [
                    "staging_id BIGSERIAL PRIMARY KEY",
                    "run_id BIGINT NOT NULL REFERENCES extraction_runs(run_id)",
                    "document_id BIGINT NOT NULL REFERENCES source_documents(document_id)",
                    "stock_code VARCHAR(20) NOT NULL REFERENCES company_info(stock_code)",
                    "report_year INTEGER NOT NULL",
                    "report_period VARCHAR(20) NOT NULL",
                    "target_table VARCHAR(100) NOT NULL",
                    "target_field VARCHAR(100) NOT NULL",
                    "source_label TEXT",
                    "raw_value TEXT",
                    "raw_unit VARCHAR(50)",
                    "standard_value NUMERIC(24, 6)",
                    "standard_unit VARCHAR(50)",
                    "period_scope VARCHAR(50) NOT NULL DEFAULT 'unknown'",
                    "source_period_label TEXT",
                    "is_derived BOOLEAN NOT NULL DEFAULT false",
                    "derivation_formula TEXT",
                    "page_number INTEGER",
                    "table_id BIGINT REFERENCES extracted_tables(table_id)",
                    "confidence NUMERIC(8, 6)",
                    "validation_status VARCHAR(50) NOT NULL DEFAULT 'pending'",
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
                    "CONSTRAINT financial_staging_unique_metric_period_scope UNIQUE (run_id, target_table, target_field, stock_code, report_year, report_period, period_scope, is_derived)",
                ],
            ),
            _statement(
                "validation_results",
                [
                    "validation_id BIGSERIAL PRIMARY KEY",
                    "run_id BIGINT REFERENCES extraction_runs(run_id)",
                    "document_id BIGINT REFERENCES source_documents(document_id)",
                    "staging_id BIGINT REFERENCES financial_staging(staging_id)",
                    "rule_name VARCHAR(100) NOT NULL",
                    "status VARCHAR(50) NOT NULL",
                    "message TEXT",
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
                ],
            ),
            _statement(
                "field_mapping_logs",
                [
                    "mapping_id BIGSERIAL PRIMARY KEY",
                    "run_id BIGINT NOT NULL REFERENCES extraction_runs(run_id)",
                    "source_label TEXT NOT NULL",
                    "target_table VARCHAR(100)",
                    "target_field VARCHAR(100)",
                    "mapping_method VARCHAR(100) NOT NULL",
                    "confidence NUMERIC(8, 6)",
                    "metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
                ],
            ),
        ]


def _to_sql_type(field: FieldDefinition) -> str:
    data_type = field.data_type.strip().lower()
    if data_type == "int":
        return "INTEGER"
    if data_type == "varchar(20)" or data_type == "varchar(50)":
        return data_type.upper()
    decimal_match = re.match(r"decimal\((\d+),(\d+)\)", data_type)
    if decimal_match:
        return f"NUMERIC({decimal_match.group(1)}, {decimal_match.group(2)})"
    if data_type.startswith("varchar"):
        return data_type.upper()
    return "TEXT"


def _statement(table_name: str, columns: list[str]) -> str:
    body = ",\n".join(f"    {column}" for column in columns)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n{body}\n);"


def generate_schema_migrations() -> list[str]:
    return [
        "ALTER TABLE source_documents DROP CONSTRAINT IF EXISTS source_documents_stock_code_fkey",
        "ALTER TABLE financial_staging ADD COLUMN IF NOT EXISTS period_scope VARCHAR(50) NOT NULL DEFAULT 'unknown'",
        "ALTER TABLE financial_staging ADD COLUMN IF NOT EXISTS source_period_label TEXT",
        "ALTER TABLE financial_staging ADD COLUMN IF NOT EXISTS is_derived BOOLEAN NOT NULL DEFAULT false",
        "ALTER TABLE financial_staging ADD COLUMN IF NOT EXISTS derivation_formula TEXT",
        "ALTER TABLE validation_results ADD COLUMN IF NOT EXISTS run_id BIGINT REFERENCES extraction_runs(run_id)",
        "ALTER TABLE validation_results ADD COLUMN IF NOT EXISTS document_id BIGINT REFERENCES source_documents(document_id)",
        """
        DO $$
        DECLARE existing_constraint text;
        BEGIN
            SELECT conname INTO existing_constraint
            FROM pg_constraint
            WHERE conrelid = 'financial_staging'::regclass
              AND contype = 'u'
              AND pg_get_constraintdef(oid) LIKE 'UNIQUE (run_id, target_table, target_field, stock_code, report_year, report_period)%'
              AND conname <> 'financial_staging_unique_metric_period_scope'
            LIMIT 1;
            IF existing_constraint IS NOT NULL THEN
                EXECUTE format('ALTER TABLE financial_staging DROP CONSTRAINT %I', existing_constraint);
            END IF;
        END $$
        """.strip(),
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'financial_staging'::regclass
                  AND conname = 'financial_staging_unique_metric_period_scope'
            ) THEN
                ALTER TABLE financial_staging
                ADD CONSTRAINT financial_staging_unique_metric_period_scope
                UNIQUE (run_id, target_table, target_field, stock_code, report_year, report_period, period_scope, is_derived);
            END IF;
        END $$
        """.strip(),
    ]
