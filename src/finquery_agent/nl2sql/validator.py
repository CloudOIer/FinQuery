from __future__ import annotations

import re


class SQLValidationError(ValueError):
    pass


def validate_select_sql(sql: str, allowed_tables: set[str]) -> None:
    normalized = sql.strip().lower()
    if not normalized.startswith("select "):
        raise SQLValidationError("Only SELECT statements are allowed")
    forbidden = (" insert ", " update ", " delete ", " drop ", " alter ", " create ", " truncate ")
    padded = f" {normalized} "
    if any(token in padded for token in forbidden):
        raise SQLValidationError("Mutating SQL statements are not allowed")
    if ";" in normalized.rstrip(";"):
        raise SQLValidationError("Multiple SQL statements are not allowed")

    referenced_tables = set(re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)", normalized))
    referenced_tables.update(re.findall(r"\bjoin\s+([a-zA-Z_][\w]*)", normalized))
    unknown_tables = referenced_tables - allowed_tables
    if unknown_tables:
        raise SQLValidationError(f"Unknown tables: {', '.join(sorted(unknown_tables))}")
