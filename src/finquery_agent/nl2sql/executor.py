from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from finquery_agent.nl2sql.sql_builder import SQLQuery
from finquery_agent.schema.models import FieldDefinition
from finquery_agent.schema.registry import SchemaRegistry


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    units: dict[str, str | None]
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": list(self.rows),
            "units": self.units,
            "row_count": self.row_count,
        }


class QueryExecutor:
    """Execute SQLQuery objects produced by SQLBuilder.

    The executor accepts only SQLQuery, not raw user SQL. That keeps execution behind
    the DSL/SQLBuilder/validator chain and prevents bypassing our safety layer.
    """

    def __init__(self, engine: Engine, registry: SchemaRegistry):
        self.engine = engine
        self.registry = registry

    def execute(self, query: SQLQuery) -> QueryResult:
        with self.engine.connect() as connection:
            rows = connection.execute(text(query.sql), query.params).mappings().all()
        normalized_rows = tuple(_json_safe_row(dict(row)) for row in rows)
        columns = tuple(rows[0].keys()) if rows else _columns_from_query(query)
        units = self._units_for_columns(query.table_name, columns)
        return QueryResult(columns=columns, rows=normalized_rows, units=units, row_count=len(normalized_rows))

    def _units_for_columns(self, table_name: str, columns: tuple[str, ...]) -> dict[str, str | None]:
        table = self.registry.tables.get(table_name)
        fields: dict[str, FieldDefinition] = {field.name: field for field in table.fields} if table else {}
        return {column: fields[column].unit if column in fields else None for column in columns}


def _columns_from_query(query: SQLQuery) -> tuple[str, ...]:
    return ("stock_code", "stock_abbr", "report_year", "report_period", *query.metric_columns)


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in row.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value
