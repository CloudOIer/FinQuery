from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    chinese_name: str
    data_type: str
    description: str
    table_name: str
    unit: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    is_dimension: bool = False


@dataclass(frozen=True)
class TableDefinition:
    name: str
    chinese_name: str
    fields: tuple[FieldDefinition, ...]


@dataclass(frozen=True)
class Company:
    stock_code: str
    stock_abbr: str
    company_name: str
    exchange: str | None = None
    industry: str | None = None
