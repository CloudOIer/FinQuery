from decimal import Decimal

from sqlalchemy import create_engine, text

from finquery_agent.nl2sql.executor import QueryExecutor
from finquery_agent.nl2sql.sql_builder import SQLQuery
from finquery_agent.schema import load_default_registry


def test_query_executor_returns_rows_columns_and_units():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE core_performance_indicators_sheet (
                    stock_code TEXT,
                    stock_abbr TEXT,
                    report_year INTEGER,
                    report_period TEXT,
                    total_operating_revenue NUMERIC
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO core_performance_indicators_sheet
                    (stock_code, stock_abbr, report_year, report_period, total_operating_revenue)
                VALUES ('600332', '白云山', 2024, 'FY', 12345.67)
                """
            )
        )

    query = SQLQuery(
        sql="SELECT stock_code, stock_abbr, report_year, report_period, total_operating_revenue FROM core_performance_indicators_sheet LIMIT :limit",
        params={"limit": 10},
        table_name="core_performance_indicators_sheet",
        metric_columns=("total_operating_revenue",),
    )

    result = QueryExecutor(engine, load_default_registry()).execute(query)

    assert result.columns == ("stock_code", "stock_abbr", "report_year", "report_period", "total_operating_revenue")
    assert result.row_count == 1
    assert result.rows[0]["stock_code"] == "600332"
    assert result.rows[0]["total_operating_revenue"] == 12345.67
    assert result.units["total_operating_revenue"] == "万元"


def test_query_executor_converts_decimal_values_to_json_safe_numbers():
    row = {"value": Decimal("10.50")}

    from finquery_agent.nl2sql.executor import _json_safe_row

    assert _json_safe_row(row) == {"value": 10.5}
