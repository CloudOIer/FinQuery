from __future__ import annotations

import argparse
import json

from finquery_agent.db import create_database_engine
from finquery_agent.ingestion.evaluation import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PDF extraction completeness and staging quality.")
    parser.add_argument("--run-id", action="append", type=int, required=True, help="Extraction run_id to evaluate.")
    parser.add_argument("--write-validation", action="store_true", help="Persist issues into validation_results.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary.")
    args = parser.parse_args()

    engine = create_database_engine()
    reports = [evaluate_run(engine, run_id, write_validation=args.write_validation) for run_id in args.run_id]
    if args.json:
        print(json.dumps([report.to_dict() for report in reports], ensure_ascii=False, indent=2))
        return

    for report in reports:
        print(f"run_id={report.run_id} stock_code={report.stock_code} {report.report_year}/{report.report_period}")
        print(f"  status={report.overall_status}")
        print(f"  pages={report.page_count_extracted}/{report.page_count_expected} tables={report.table_count}")
        print(f"  staging_rows={report.staging_count} mapping_logs={report.mapping_log_count}")
        print(
            f"  field_coverage={report.covered_field_count}/{report.target_field_count} "
            f"({report.field_coverage_ratio:.2%})"
        )
        print(
            f"  core_coverage={report.core_present_count}/{report.core_required_count} "
            f"({report.core_coverage_ratio:.2%})"
        )
        for issue in report.issues[:20]:
            location = f" staging_id={issue.staging_id}" if issue.staging_id else ""
            print(f"  [{issue.status}] {issue.rule_name}{location}: {issue.message}")
        if len(report.issues) > 20:
            print(f"  ... {len(report.issues) - 20} more issues")


if __name__ == "__main__":
    main()