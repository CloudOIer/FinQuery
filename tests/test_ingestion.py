from datetime import date
from pathlib import Path

from finquery_agent.ingestion.clean_markdown import clean_markdown
from finquery_agent.ingestion.financial_staging import (
    extract_financial_staging_records,
    infer_table_unit,
    parse_numeric_value,
    standardize_value,
)
from finquery_agent.ingestion.metadata import infer_period_from_date
from finquery_agent.ingestion.metadata import infer_period_from_text, infer_report_metadata, refine_report_metadata_from_text
from finquery_agent.ingestion.models import ExtractedPage, ExtractedTable, ParsedPdfDocument, ReportMetadata
from finquery_agent.ingestion.pdf_markdown import table_to_markdown
from finquery_agent.schema import load_default_registry


def test_infer_period_from_disclosure_date():
    assert infer_period_from_date(date(2023, 3, 31)) == (2022, "FY")
    assert infer_period_from_date(date(2024, 4, 25)) == (2023, "FY")
    assert infer_period_from_date(date(2025, 4, 28)) == (2024, "FY")
    assert infer_period_from_date(date(2025, 8, 30)) == (2025, "HY")
    assert infer_period_from_date(date(2025, 10, 28)) == (2025, "Q3")


def test_infer_period_from_report_title_text():
    assert infer_period_from_text("昭衍新药 2022 年年度报告") == (2022, "FY")
    assert infer_period_from_text("成都先导 2024 年第一季度报告") == (2024, "Q1")

    metadata = ReportMetadata(source_path=Path("603127_20230331.pdf"), stock_code="603127", report_year=2023, report_period="Q1")
    refined = refine_report_metadata_from_text(metadata, "北京昭衍新药研究中心股份有限公司 2022 年年度报告")

    assert refined.report_year == 2022
    assert refined.report_period == "FY"


def test_infer_metadata_from_chinese_filename_and_text_stock_code():
    filename_metadata = infer_report_metadata(Path("凯莱英：2023年年度报告.pdf"))

    assert filename_metadata.stock_code == "002821"

    base = infer_report_metadata(Path("unknown.pdf"))
    refined = refine_report_metadata_from_text(
        base,
        "证券代码：301033 证券简称：迈普医学 广州迈普再生医学科技股份有限公司 2025 年半年度报告摘要",
    )

    assert refined.stock_code == "301033"
    assert refined.report_year == 2025
    assert refined.report_period == "HY"


def test_table_to_markdown_escapes_cells():
    markdown = table_to_markdown([["项目", "本期"], ["营业收入", "1|2"]])

    assert "| 项目 | 本期 |" in markdown
    assert "1\\|2" in markdown


def test_clean_markdown_keeps_financial_lines():
    source = """
本公司及董事会全体成员保证信息真实
# 公司治理
董事会成员情况
# 合并利润表
| 项目 | 本期发生额 |
| 利润总额 | 100 |
"""

    cleaned = clean_markdown(source)

    assert "保证信息真实" not in cleaned
    assert "董事会成员情况" not in cleaned
    assert "合并利润表" in cleaned
    assert "利润总额" in cleaned


def test_clean_markdown_preserves_statement_sections_and_drops_notes():
    source = """
# 第三节 管理层讨论与分析
公司业务概要和风险分析
# 主要会计数据和财务指标
单位：元
| 主要会计数据 | 2024年 | 2023年 |
| 营业收入 | 200 | 100 |
# 财务报表附注
| 项目 | 说明 |
| 长期股权投资 | 附注明细 |
# 合并现金流量表
| 项目 | 2024年度 |
| 经营活动产生的现金流量净额 | 50 |
"""

    cleaned = clean_markdown(source)

    assert "公司业务概要" not in cleaned
    assert "长期股权投资" not in cleaned
    assert "主要会计数据和财务指标" in cleaned
    assert "营业收入" in cleaned
    assert "合并现金流量表" in cleaned
    assert "经营活动产生的现金流量净额" in cleaned


def test_parse_numeric_value_handles_financial_formats():
    assert parse_numeric_value("1,234.50") == parse_numeric_value("1234.50")
    assert parse_numeric_value("(12.30)") == parse_numeric_value("-12.30")
    assert parse_numeric_value("十九、4") is None
    assert parse_numeric_value("--") is None


def test_standardize_value_converts_yuan_to_ten_thousand_yuan():
    registry = load_default_registry()
    field = registry.resolve_metric("营业总收入")

    assert field is not None
    value, unit = standardize_value(parse_numeric_value("123456789") or 0, "元币种:人民币", field)

    assert value == parse_numeric_value("12345.6789")
    assert unit == "万元"


def test_infer_table_unit_ignores_compilation_unit_company_name():
    table = ExtractedTable(page_number=1, table_index=1, markdown="", raw_rows=[])

    unit = infer_table_unit(table, "编制单位：凯莱英医药集团（天津）股份有限公司\n单位：元")

    assert unit == "元"


def test_extract_financial_staging_records_from_table():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=10,
        table_index=1,
        section_title="合并利润表 单位：元",
        markdown="单位：元",
        raw_rows=[
            ["项目", "本期发生额", "上期发生额"],
            ["营业总收入", "123456789", "100"],
            ["利润总额", "2000000", "100"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(
            source_path=Path("600332_20250425_TEST.pdf"),
            stock_code="600332",
            report_year=2024,
            report_period="FY",
        ),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=10, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)

    assert {record.target_field for record in records} == {"total_operating_revenue", "total_profit"}
    revenue = next(record for record in records if record.target_field == "total_operating_revenue")
    assert revenue.standard_value == parse_numeric_value("12345.6789")
    assert revenue.period_scope == "full_year"
    assert revenue.source_period_label == "本期发生额"


def test_extract_financial_staging_skips_note_column():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=94,
        table_index=107,
        section_title="利润表 单位：元",
        markdown="单位：元",
        raw_rows=[
            ["项目", "附注十七", "2022年度", "2021年度"],
            ["一、营业收入", "4", "551,521,552.72", "441,414,818.52"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("603127.pdf"), stock_code="603127", report_year=2022, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=94, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    revenue = next(record for record in records if record.target_table == "income_sheet" and record.target_field == "total_operating_revenue")

    assert revenue.raw_value == "551,521,552.72"
    assert revenue.standard_value == parse_numeric_value("55152.155272")
    assert revenue.source_period_label == "2022年度"


def test_extract_financial_staging_does_not_match_subtotals_as_totals():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=10,
        table_index=1,
        section_title="合并资产负债表 单位：元",
        markdown="单位：元",
        raw_rows=[
            ["项目", "2024年12月31日", "2023年12月31日"],
            ["流动资产合计", "100", "90"],
            ["资产总计", "300", "250"],
            ["流动负债合计", "40", "30"],
            ["负债合计", "120", "100"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=10, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_field = {record.target_field: record for record in records}

    assert by_field["asset_total_assets"].source_label == "资产总计"
    assert by_field["liability_total_liabilities"].source_label == "负债合计"


def test_extract_financial_staging_skips_non_period_value_columns():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=20,
        table_index=1,
        section_title="分部信息 单位：元",
        markdown="单位：元",
        raw_rows=[
            ["项目", "中国区实验室服务", "美国区实验室服务"],
            ["营业利润", "100", "200"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=20, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    assert extract_financial_staging_records(parsed, registry) == ()


def test_extract_financial_staging_skips_note_detail_tables():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=169,
        table_index=427,
        section_title=None,
        markdown="合营企业或联营企业 持股比例 投资的账面价值",
        raw_rows=[
            ["", "2024年", "2023年"],
            ["资产合计", "914260131.51", "767628760.56"],
            ["负债合计", "127940.41", "85440.41"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("002821.pdf"), stock_code="002821", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=169, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    assert extract_financial_staging_records(parsed, registry) == ()


def test_extract_financial_staging_handles_mixed_statement_table():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=1,
        table_index=93,
        section_title=None,
        markdown="合并利润表 合并现金流量表 单位：元",
        raw_rows=[
            ["项目", "附注", "2024年度", "2023年度"],
            ["一、营业总收入", "", "300000000", "200000000"],
            ["其中:营业成本", "七、1", "120000000", "100000000"],
            ["二、营业利润", "", "80000000", "60000000"],
            ["经营活动产生的现金流量净额", "", "50000000", "40000000"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=1, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    assert by_key[("income_sheet", "total_operating_revenue")].standard_value == parse_numeric_value("30000")
    assert by_key[("income_sheet", "operating_expense_cost_of_sales")].standard_value == parse_numeric_value("12000")
    assert by_key[("income_sheet", "operating_profit")].standard_value == parse_numeric_value("8000")
    assert by_key[("cash_flow_sheet", "operating_cf_net_amount")].standard_value == parse_numeric_value("5000")


def test_extract_financial_staging_defaults_formal_statement_unit_to_yuan():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=96,
        table_index=192,
        section_title="1、合并资产负债表",
        markdown="合并资产负债表",
        raw_rows=[
            ["", "项目", "", "", "2024年12月31日", "", "", "2023年12月31日", ""],
            ["", "货币资金", "", "5,289,594,427.89", "", "", "6,234,457,167.58", "", ""],
            ["", "结算备付金", "", "", "", "", "", "", ""],
            ["", "资产总计", "", "12,610,011,324.42", "", "", "10,310,396,863.27", "", ""],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("002821.pdf"), stock_code="002821", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=96, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    assert by_key[("balance_sheet", "asset_cash_and_cash_equivalents")].source_label == "货币资金"
    assert by_key[("balance_sheet", "asset_cash_and_cash_equivalents")].standard_value == parse_numeric_value("528959.442789")
    assert by_key[("balance_sheet", "asset_total_assets")].standard_value == parse_numeric_value("1261001.132442")


def test_extract_financial_staging_strips_parenthetical_label_notes():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=1,
        table_index=1,
        section_title="主要会计数据和财务指标 单位：元",
        markdown="单位：元",
        raw_rows=[
            ["主要财务指标", "2024年", "2023年"],
            ["加权平均净资产收益率(%)", "3.09", "1.91"],
            ["二、营业利润(亏损以“-”号填列)", "80000000", "60000000"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=1, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    assert by_key[("core_performance_indicators_sheet", "roe")].standard_value == parse_numeric_value("3.09")


def test_extract_financial_staging_strips_unclosed_parenthetical_label_notes():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=99,
        table_index=195,
        section_title="合并利润表 单位：元",
        markdown="合并利润表 单位：元",
        raw_rows=[
            ["", "项目", "", "", "2024年度", "", "", "2023年度", ""],
            ["", "三、营业利润（亏损以“－”号填", "", "80,000,000", "", "", "60,000,000", "", ""],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("002821.pdf"), stock_code="002821", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=99, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    assert by_key[("income_sheet", "operating_profit")].standard_value == parse_numeric_value("8000")


def test_extract_financial_staging_skips_unknown_company_dimension():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=1,
        table_index=1,
        markdown="单位：元",
        raw_rows=[["项目", "本期"], ["营业总收入", "100",]],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("999999.pdf"), stock_code="999999", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=1, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    assert extract_financial_staging_records(parsed, registry) == ()


def test_extract_financial_staging_uses_table_type_and_current_period():
    registry = load_default_registry()
    core_table = ExtractedTable(
        page_number=5,
        table_index=1,
        section_title="主要会计数据和财务指标 单位：万元",
        markdown="主要会计数据和财务指标 单位：万元",
        raw_rows=[
            ["主要会计数据", "2024年", "2023年", "本期比上年同期增减(%)"],
            ["营业收入", "2000", "1000", "100.00"],
            ["归属于上市公司股东的净利润", "300", "200", "50.00"],
        ],
    )
    income_table = ExtractedTable(
        page_number=20,
        table_index=2,
        section_title="合并利润表 单位：元",
        markdown="合并利润表 单位：元",
        raw_rows=[
            ["项目", "附注", "2024年度", "2023年度"],
            ["一、营业收入", "1", "20000000", "10000000"],
            ["减：营业成本", "2", "12000000", "9000000"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("600332_20250425_TEST.pdf"), stock_code="600332", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(
            ExtractedPage(page_number=5, text_content="", markdown_content="", tables=(core_table,)),
            ExtractedPage(page_number=20, text_content="", markdown_content="", tables=(income_table,)),
        ),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    assert by_key[("core_performance_indicators_sheet", "total_operating_revenue")].standard_value == parse_numeric_value("2000")
    assert by_key[("core_performance_indicators_sheet", "operating_revenue_yoy_growth")].standard_value == parse_numeric_value("100.00")
    assert by_key[("income_sheet", "total_operating_revenue")].standard_value == parse_numeric_value("2000")
    assert by_key[("income_sheet", "operating_expense_cost_of_sales")].standard_value == parse_numeric_value("1200")
    assert by_key[("income_sheet", "total_operating_revenue")].source_period_label == "2024年度"


def test_extract_financial_staging_handles_sparse_szse_core_table():
    registry = load_default_registry()
    table = ExtractedTable(
        page_number=9,
        table_index=10,
        section_title="六、主要会计数据和财务指标",
        markdown="主要会计数据和财务指标",
        raw_rows=[
            ["", "", "", "", "2024年", "", "", "2023年", "", "", "本年比上年增减", "", "", "2022年", ""],
            ["", "营业收入（元）", "", "10,255,325,392.82", "", "", "4,638,834,177.53", "", "", "121.08%", "", "", "3,149,689,675.80", "", ""],
            ["", "归属于上市公司股东", "", "3,301,635,019.64", "", "", "1,069,273,577.50", "", "", "208.77%", "", "", "722,091,360.68", "", ""],
            ["", "的净利润（元）", "", "", "", "", "", "", "", "", "", "", "", "", ""],
            ["", "经营活动产生的现金", "", "3,286,910,705.82", "", "", "113,150,121.36", "", "", "2,804.91%", "", "", "569,291,589.49", "", ""],
            ["", "流量净额", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("002821.pdf"), stock_code="002821", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=9, text_content="", markdown_content="", tables=(table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    assert by_key[("core_performance_indicators_sheet", "total_operating_revenue")].standard_value == parse_numeric_value("1025532.539282")
    assert by_key[("core_performance_indicators_sheet", "operating_revenue_yoy_growth")].standard_value == parse_numeric_value("121.08")
    assert by_key[("core_performance_indicators_sheet", "net_profit_10k_yuan")].standard_value == parse_numeric_value("330163.501964")
    assert by_key[("core_performance_indicators_sheet", "net_profit_yoy_growth")].standard_value == parse_numeric_value("208.77")
    assert by_key[("cash_flow_sheet", "operating_cf_net_amount")].standard_value == parse_numeric_value("328691.070582")


def test_extract_financial_staging_reconciles_equivalent_profit_fields():
    registry = load_default_registry()
    core_table = ExtractedTable(
        page_number=1,
        table_index=1,
        section_title="主要会计数据和财务指标 单位：元",
        markdown="主要会计数据和财务指标 单位：元",
        raw_rows=[
            ["主要会计数据", "2024年", "2023年", "本期比上年同期增减(%)"],
            ["归属于上市公司股东的净利润", "40718459.76", "25266025.36", "61.16"],
        ],
    )
    income_table = ExtractedTable(
        page_number=20,
        table_index=2,
        section_title="财务报表附注 单位：元",
        markdown="财务报表附注 单位：元",
        raw_rows=[
            ["项目", "期末余额/本期发生额", "期初余额/上期发生额"],
            ["净利润", "-3103474.35", "-6109094.80"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=1, text_content="", markdown_content="", tables=(core_table, income_table)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    income_profit = by_key[("income_sheet", "net_profit")]
    income_profit_growth = by_key[("income_sheet", "net_profit_yoy_growth")]
    assert income_profit.standard_value == parse_numeric_value("4071.845976")
    assert income_profit.is_derived is True
    assert income_profit.source_label == "equivalent:core_performance_indicators_sheet.net_profit_10k_yuan"
    assert income_profit_growth.standard_value == parse_numeric_value("61.16")
    assert income_profit_growth.is_derived is True


def test_extract_financial_staging_backfills_core_revenue_from_income_when_missing():
    registry = load_default_registry()
    income_table = ExtractedTable(
        page_number=20,
        table_index=2,
        section_title="合并利润表 单位：元",
        markdown="合并利润表 单位：元",
        raw_rows=[
            ["项目", "2024年度", "2023年度"],
            ["一、营业总收入", "20000000", "10000000"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=20, text_content="", markdown_content="", tables=(income_table,)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    core_revenue = by_key[("core_performance_indicators_sheet", "total_operating_revenue")]
    assert core_revenue.standard_value == parse_numeric_value("2000")
    assert core_revenue.is_derived is True
    assert core_revenue.source_label == "equivalent:income_sheet.total_operating_revenue"


def test_extract_financial_staging_derives_missing_fields():
    registry = load_default_registry()
    balance_table = ExtractedTable(
        page_number=30,
        table_index=1,
        section_title="合并资产负债表 单位：元",
        markdown="合并资产负债表 单位：元",
        raw_rows=[
            ["项目", "2024年12月31日", "2023年12月31日"],
            ["资产总计", "3000000000", "2500000000"],
            ["负债合计", "1200000000", "1000000000"],
            ["归属于母公司所有者权益合计", "1500000000", "1300000000"],
            ["股本", "500000000", "500000000"],
        ],
    )
    income_table = ExtractedTable(
        page_number=40,
        table_index=2,
        section_title="合并利润表 单位：元",
        markdown="合并利润表 单位：元",
        raw_rows=[
            ["项目", "2024年度", "2023年度"],
            ["营业总收入", "200000000", "100000000"],
            ["营业成本", "120000000", "80000000"],
            ["销售费用", "10000000", "9000000"],
            ["管理费用", "8000000", "7000000"],
            ["财务费用", "2000000", "1000000"],
            ["研发费用", "30000000", "20000000"],
            ["税金及附加", "1000000", "900000"],
        ],
    )
    cash_flow_table = ExtractedTable(
        page_number=50,
        table_index=3,
        section_title="合并现金流量表 单位：元",
        markdown="合并现金流量表 单位：元",
        raw_rows=[
            ["项目", "2024年度", "2023年度"],
            ["经营活动产生的现金流量净额", "100000000", "50000000"],
            ["投资活动产生的现金流量净额", "-40000000", "-30000000"],
            ["筹资活动产生的现金流量净额", "20000000", "10000000"],
            ["现金及现金等价物净增加额", "80000000", "30000000"],
        ],
    )
    parsed = ParsedPdfDocument(
        metadata=ReportMetadata(source_path=Path("600332_20250425_TEST.pdf"), stock_code="600332", report_year=2024, report_period="FY"),
        markdown="",
        clean_markdown="",
        pages=(ExtractedPage(page_number=1, text_content="", markdown_content="", tables=(balance_table, income_table, cash_flow_table)),),
        markdown_path=Path("raw.md"),
        clean_markdown_path=Path("clean.md"),
    )

    records = extract_financial_staging_records(parsed, registry)
    by_key = {(record.target_table, record.target_field): record for record in records}

    gross_margin = by_key[("core_performance_indicators_sheet", "gross_profit_margin")]
    assert gross_margin.is_derived is True
    assert gross_margin.standard_value == parse_numeric_value("40.0")
    assert by_key[("balance_sheet", "asset_liability_ratio")].standard_value == parse_numeric_value("40.0")
    assert by_key[("core_performance_indicators_sheet", "net_asset_per_share")].standard_value == parse_numeric_value("3")
    assert by_key[("income_sheet", "total_operating_expenses")].standard_value == parse_numeric_value("17100")
    assert by_key[("cash_flow_sheet", "operating_cf_ratio_of_net_cf")].standard_value == parse_numeric_value("125.00")
