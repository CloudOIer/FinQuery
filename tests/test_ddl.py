from finquery_agent.db import DDLGenerator
from finquery_agent.schema import load_default_registry


def test_generates_financial_table_ddl():
    registry = load_default_registry()
    ddl = DDLGenerator(registry).generate_all()

    assert "CREATE TABLE IF NOT EXISTS company_info" in ddl
    assert "CREATE TABLE IF NOT EXISTS core_performance_indicators_sheet" in ddl
    assert "CREATE TABLE IF NOT EXISTS source_documents" in ddl
    assert "CREATE TABLE IF NOT EXISTS extraction_runs" in ddl
    assert "CREATE TABLE IF NOT EXISTS financial_staging" in ddl
    assert "CREATE TABLE IF NOT EXISTS validation_results" in ddl
    assert "run_id BIGINT REFERENCES extraction_runs(run_id)" in ddl
    assert "document_id BIGINT REFERENCES source_documents(document_id)" in ddl
    assert "total_operating_revenue NUMERIC(20, 2)" in ddl
    assert "PRIMARY KEY (stock_code, report_year, report_period)" in ddl
    assert "PRIMARY KEY (stock_code, report_year, report_period),," not in ddl
    assert "FOREIGN KEY (stock_code) REFERENCES company_info(stock_code)" in ddl
    source_documents_ddl = ddl.split("CREATE TABLE IF NOT EXISTS source_documents", 1)[1].split(");", 1)[0]
    assert "stock_code VARCHAR(20)" in source_documents_ddl
    assert "FOREIGN KEY (stock_code)" not in source_documents_ddl


def test_ingestion_staging_tracks_source_and_validation():
    registry = load_default_registry()
    ddl = DDLGenerator(registry).generate_all()

    assert "target_table VARCHAR(100) NOT NULL" in ddl
    assert "target_field VARCHAR(100) NOT NULL" in ddl
    assert "standard_value NUMERIC(24, 6)" in ddl
    assert "validation_status VARCHAR(50) NOT NULL DEFAULT 'pending'" in ddl
    assert "period_scope VARCHAR(50) NOT NULL DEFAULT 'unknown'" in ddl
    assert "is_derived BOOLEAN NOT NULL DEFAULT false" in ddl
    assert "financial_staging_unique_metric_period_scope" in ddl
    assert "report_period, period_scope, is_derived" in ddl
