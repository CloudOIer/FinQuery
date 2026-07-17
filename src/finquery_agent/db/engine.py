from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from finquery_agent.config import get_settings


def create_database_engine(database_url: str | None = None) -> Engine:
    settings = get_settings()
    return create_engine(database_url or settings.database_url, pool_pre_ping=True)
