from __future__ import annotations

import re


IMAGE_RE = re.compile(r"^!\[.*?\]\(.*?\)\s*$")
HEADING_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$")
BOILERPLATE_RE = re.compile(
    r"本公司及董事会全体成员保证|没有虚假记载|误导性陈述|重大遗漏|"
    r"所有董事均已出席|前瞻性陈述|^公告编号|^\s*目录\s*$"
)
DROP_SECTION_RE = re.compile(
    r"股东信息|普通股股份变动|公司治理|管理层讨论|重要事项|股份变动|优先股|债券|审计报告|"
    r"风险提示|释义|备查文件|社会责任|环境和社会责任|投资者关系|财务报表附注|董事、监事|员工情况"
)
KEEP_SECTION_RE = re.compile(
    r"主要会计数据|主要财务指标|财务指标|合并资产负债表|母公司资产负债表|资产负债表|"
    r"合并利润表|母公司利润表|利润表|合并现金流量表|母公司现金流量表|现金流量表|财务报表"
)
KEEP_LINE_RE = re.compile(
    r"证券代码|证券简称|公司名称|主要会计数据|主要财务指标|资产负债表|利润表|现金流量表|"
    r"营业收入|营业总收入|营业成本|利润总额|净利润|扣除非经常性损益|每股收益|净资产收益率|"
    r"经营活动产生的现金流量净额|投资活动产生的现金流量净额|筹资活动产生的现金流量净额|"
    r"现金及现金等价物净增加额|总资产|资产总计|负债合计|总负债|所有者权益|股东权益|货币资金|"
    r"应收账款|存货|合同负债|研发费用|销售费用|管理费用|财务费用"
)
STATEMENT_TABLE_RE = re.compile(
    r"\|.*(项目|主要会计数据|主要财务指标|营业收入|资产总计|负债合计|净利润|现金流量).*\|"
)


def clean_markdown(markdown: str) -> str:
    lines = []
    section_mode = "neutral"
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if IMAGE_RE.match(line) or BOILERPLATE_RE.search(line):
            continue
        heading_match = HEADING_RE.match(line)
        if heading_match:
            heading = heading_match.group(1)
            if DROP_SECTION_RE.search(heading):
                section_mode = "drop"
                continue
            if KEEP_SECTION_RE.search(heading):
                section_mode = "keep"
                lines.append(line)
                continue
            section_mode = "neutral"
            continue

        if KEEP_LINE_RE.search(line):
            lines.append(line)
            continue
        if line.startswith("|") and (section_mode == "keep" or STATEMENT_TABLE_RE.search(line)):
            lines.append(line)
            continue
        if section_mode == "drop":
            continue
        if section_mode == "keep" and _looks_like_statement_context(line):
            lines.append(line)
            continue
    return "\n".join(lines)


def _looks_like_statement_context(line: str) -> bool:
    return bool(
        line.startswith("|")
        or re.search(r"单位\s*[:：]", line)
        or re.search(r"20\d{2}年", line)
        or re.search(r"本期|上期|期末|期初", line)
    )
