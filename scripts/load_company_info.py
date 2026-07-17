from __future__ import annotations

from sqlalchemy import text

from finquery_agent.db import create_database_engine
from finquery_agent.schema import load_default_registry


UPSERT_COMPANY_SQL = text(
    """
    INSERT INTO company_info (stock_code, stock_abbr, company_name, exchange, industry)
    VALUES (:stock_code, :stock_abbr, :company_name, :exchange, :industry)
    ON CONFLICT (stock_code) DO UPDATE SET
        stock_abbr = EXCLUDED.stock_abbr,
        company_name = EXCLUDED.company_name,
        exchange = EXCLUDED.exchange,
        industry = EXCLUDED.industry
    """
)


def main() -> None:
    registry = load_default_registry()
    engine = create_database_engine()
    with engine.begin() as connection:
        for company in registry.companies.values():
            connection.execute(
                UPSERT_COMPANY_SQL,
                {
                    "stock_code": company.stock_code,
                    "stock_abbr": company.stock_abbr,
                    "company_name": company.company_name,
                    "exchange": company.exchange,
                    "industry": company.industry,
                },
            )
    print(f"Loaded {len(registry.companies)} companies into company_info")


if __name__ == "__main__":
    main()
