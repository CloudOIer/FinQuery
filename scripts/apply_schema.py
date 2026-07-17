from __future__ import annotations

from sqlalchemy import text

from finquery_agent.db import DDLGenerator, create_database_engine
from finquery_agent.db.ddl import generate_schema_migrations
from finquery_agent.schema import load_default_registry


def split_statements(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


def main() -> None:
    registry = load_default_registry()
    ddl = DDLGenerator(registry).generate_all()
    engine = create_database_engine()
    with engine.begin() as connection:
        for statement in split_statements(ddl):
            connection.execute(text(statement))
        for statement in generate_schema_migrations():
            connection.execute(text(statement))
    print("Applied FinQuery database schema")


if __name__ == "__main__":
    main()