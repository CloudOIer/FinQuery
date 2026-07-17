from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from finquery_agent.config import get_settings


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True)


def init_cluster() -> None:
    settings = get_settings()
    if settings.postgres_data_dir.exists():
        print(f"PostgreSQL data directory already exists: {settings.postgres_data_dir}")
        return
    settings.postgres_data_dir.parent.mkdir(parents=True, exist_ok=True)
    run([
        "initdb",
        "-D",
        str(settings.postgres_data_dir),
        "--encoding=UTF8",
        "--locale=C",
        "--auth=trust",
    ])


def start_cluster() -> None:
    settings = get_settings()
    settings.postgres_log_file.parent.mkdir(parents=True, exist_ok=True)
    run([
        "pg_ctl",
        "-D",
        str(settings.postgres_data_dir),
        "-l",
        str(settings.postgres_log_file),
        "-o",
        f"-p {settings.postgres_port} -h {settings.postgres_host}",
        "start",
    ])


def stop_cluster() -> None:
    settings = get_settings()
    run(["pg_ctl", "-D", str(settings.postgres_data_dir), "stop"], check=False)


def status_cluster() -> None:
    settings = get_settings()
    run(["pg_ctl", "-D", str(settings.postgres_data_dir), "status"], check=False)


def create_database() -> None:
    settings = get_settings()
    result = subprocess.run(
        [
            "createdb",
            "-h",
            settings.postgres_host,
            "-p",
            str(settings.postgres_port),
            "-U",
            settings.postgres_user,
            settings.postgres_db,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"Created database: {settings.postgres_db}")
        return
    if "already exists" in result.stderr:
        print(f"Database already exists: {settings.postgres_db}")
        return
    raise SystemExit(result.stderr.strip())


def print_config() -> None:
    settings = get_settings()
    print(f"PGDATA={settings.postgres_data_dir}")
    print(f"PGLOG={settings.postgres_log_file}")
    print(f"PGHOST={settings.postgres_host}")
    print(f"PGPORT={settings.postgres_port}")
    print(f"PGDATABASE={settings.postgres_db}")
    print(f"PGUSER={settings.postgres_user}")
    print(f"DATABASE_URL={settings.database_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage local PostgreSQL for FinQuery")
    parser.add_argument(
        "command",
        choices=["init", "start", "stop", "status", "createdb", "config"],
    )
    args = parser.parse_args()

    actions = {
        "init": init_cluster,
        "start": start_cluster,
        "stop": stop_cluster,
        "status": status_cluster,
        "createdb": create_database,
        "config": print_config,
    }
    actions[args.command]()


if __name__ == "__main__":
    main()
