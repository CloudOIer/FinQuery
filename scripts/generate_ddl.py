from __future__ import annotations

from finquery_agent.db import DDLGenerator
from finquery_agent.schema import load_default_registry


def main() -> None:
    registry = load_default_registry()
    print(DDLGenerator(registry).generate_all())


if __name__ == "__main__":
    main()
