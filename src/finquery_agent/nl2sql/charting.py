from __future__ import annotations

import base64
import html
import re
from dataclasses import dataclass
from numbers import Number
from typing import Any

from finquery_agent.nl2sql.intent import StructuredIntent
from finquery_agent.schema.models import FieldDefinition
from finquery_agent.schema.registry import SchemaRegistry


PERIOD_LABELS = {"FY": "年报", "Q1": "第一季度", "HY": "半年度", "Q3": "第三季度"}


@dataclass(frozen=True)
class ChartImage:
    chart_type: str
    title: str
    x_axis_label: str
    y_axis_label: str
    image_data_url: str
    alt_text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "image_data_url": self.image_data_url,
            "alt_text": self.alt_text,
        }


class ChartRenderer:
    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

    def render(self, intent: StructuredIntent, queries: list[dict[str, Any]]) -> ChartImage | None:
        images = self.render_all(intent, queries)
        return images[0] if images else None

    def render_all(self, intent: StructuredIntent, queries: list[dict[str, Any]]) -> list[ChartImage]:
        if intent.chart is None:
            return []

        images: list[ChartImage] = []
        for query, metric, rows in self._iter_chart_data(intent, queries):
            image = self._render_metric_image(intent, query, metric, rows)
            if image:
                images.append(image)
        return images

    def _render_metric_image(self, intent: StructuredIntent, query: dict[str, Any], metric: str, rows: list[dict[str, Any]]) -> ChartImage | None:
        if intent.chart is None:
            return None
        chart_type = "line" if intent.chart.chart_type == "line" else "bar"
        table_name = str(query.get("table_name") or "")
        metric_label = self._field_label(table_name, metric)
        unit = self._metric_unit(query, table_name, metric)
        value_axis_label = _label_with_unit(metric_label, unit)
        category_axis_label = "报告期间" if chart_type == "line" else "公司/报告期"
        title = intent.chart.title or self._chart_title(rows, metric_label, chart_type)

        if chart_type == "line":
            chart_rows = sorted(rows[:18], key=_row_sort_key)
            svg = self._render_line_svg(chart_rows, metric, title, category_axis_label, value_axis_label, unit)
            x_axis_label = category_axis_label
            y_axis_label = value_axis_label
        else:
            chart_rows = rows[:14]
            svg = self._render_bar_svg(chart_rows, metric, title, category_axis_label, value_axis_label, unit)
            x_axis_label = value_axis_label
            y_axis_label = category_axis_label

        return ChartImage(
            chart_type=chart_type,
            title=title,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            image_data_url=_svg_data_url(svg),
            alt_text=f"{title}，横轴为{x_axis_label}，纵轴为{y_axis_label}。",
        )

    def _iter_chart_data(self, intent: StructuredIntent, queries: list[dict[str, Any]]):
        for query in queries:
            result = query.get("result") or {}
            rows = list(result.get("rows") or [])
            metric_columns = tuple(query.get("metric_columns") or ())
            if not rows or not metric_columns:
                continue
            for metric in self._chart_metrics(intent, str(query.get("table_name") or ""), metric_columns):
                numeric_rows = [row for row in rows if _is_number(row.get(metric))]
                if numeric_rows:
                    yield query, metric, numeric_rows

    def _chart_metrics(self, intent: StructuredIntent, table_name: str, metric_columns: tuple[str, ...]) -> tuple[str, ...]:
        if len(metric_columns) > 1:
            return metric_columns
        chart_metric = intent.chart.y if intent.chart else None
        if not chart_metric:
            return metric_columns
        field = self.registry.resolve_metric(chart_metric)
        if field and field.table_name == table_name and field.name in metric_columns:
            return (field.name,)
        if chart_metric in metric_columns:
            return (chart_metric,)
        return metric_columns

    def _metric_unit(self, query: dict[str, Any], table_name: str, metric: str) -> str | None:
        result = query.get("result") or {}
        units = result.get("units") or {}
        if units.get(metric):
            return str(units[metric])
        field = self._field(table_name, metric)
        return field.unit if field else None

    def _field_label(self, table_name: str, metric: str) -> str:
        field = self._field(table_name, metric)
        if field is None:
            return metric
        return _strip_unit(field.chinese_name) or field.chinese_name or metric

    def _field(self, table_name: str, metric: str) -> FieldDefinition | None:
        table = self.registry.tables.get(table_name)
        if table is None:
            return None
        return next((field for field in table.fields if field.name == metric), None)

    def _row_label(self, row: dict[str, Any], *, include_company: bool = True) -> str:
        stock_code = str(row.get("stock_code") or "")
        company = self._company_label(row)
        year = row.get("report_year")
        period = _period_label(row.get("report_period"))
        time_label = f"{year}年{period}" if year and period else str(year or period or "")
        if include_company and company:
            return f"{company} {time_label}".strip()
        return time_label or str(company or stock_code or "")

    def _company_label(self, row: dict[str, Any]) -> str:
        stock_code = str(row.get("stock_code") or "")
        return str(row.get("stock_abbr") or (self.registry.companies.get(stock_code).stock_abbr if stock_code in self.registry.companies else "") or stock_code or "")

    def _chart_title(self, rows: list[dict[str, Any]], metric_label: str, chart_type: str) -> str:
        companies = {self._company_label(row) for row in rows}
        companies.discard("")
        subject = next(iter(companies)) if len(companies) == 1 else ""
        suffix = "趋势图" if chart_type == "line" else "对比图"
        return f"{subject}{metric_label}{suffix}"

    def _render_bar_svg(
        self,
        rows: list[dict[str, Any]],
        metric: str,
        title: str,
        category_axis_label: str,
        value_axis_label: str,
        unit: str | None,
    ) -> str:
        width = 920
        top = 78
        left = 190
        right = 140
        bottom = 72
        row_height = 36
        height = max(340, top + bottom + len(rows) * row_height)
        plot_width = width - left - right
        values = [float(row[metric]) for row in rows]
        min_value = min(0.0, min(values))
        max_value = max(0.0, max(values))
        if min_value == max_value:
            max_value = min_value + 1.0
        span = max_value - min_value

        def x_for(value: float) -> float:
            return left + ((value - min_value) / span) * plot_width

        zero_x = x_for(0.0)
        tick_parts = []
        for index in range(5):
            value = min_value + (span * index / 4)
            x = x_for(value)
            tick_parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" />')
            tick_parts.append(f'<text class="tick" x="{x:.1f}" y="{height - bottom + 22}" text-anchor="middle">{_svg(_format_number(value))}</text>')

        bar_parts = []
        for index, row in enumerate(rows):
            value = float(row[metric])
            y = top + index * row_height + 8
            x = x_for(min(value, 0.0))
            bar_width = max(3, abs(x_for(value) - zero_x))
            value_x = min(width - 18, x + bar_width + 8) if value >= 0 else max(left, x - 8)
            anchor = "start" if value >= 0 else "end"
            fill_class = "bar negative" if value < 0 else "bar"
            label = _truncate(self._row_label(row), 18)
            bar_parts.append(f'<text class="category" x="{left - 14}" y="{y + 13}" text-anchor="end">{_svg(label)}</text>')
            bar_parts.append(f'<rect class="{fill_class}" x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="16" rx="4" />')
            bar_parts.append(f'<text class="value" x="{value_x:.1f}" y="{y + 13}" text-anchor="{anchor}">{_svg(_format_value(value, unit))}</text>')

        return _svg_document(
            width,
            height,
            f"""
            <text class="title" x="{width / 2}" y="34" text-anchor="middle">{_svg(title)}</text>
            <text class="axis-name" x="{left}" y="58">纵轴：{_svg(category_axis_label)}</text>
            <text class="axis-name" x="{width / 2}" y="{height - 18}" text-anchor="middle">横轴：{_svg(value_axis_label)}</text>
            <line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" />
            <line class="axis" x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{height - bottom}" />
            {''.join(tick_parts)}
            {''.join(bar_parts)}
            """,
        )

    def _render_line_svg(
        self,
        rows: list[dict[str, Any]],
        metric: str,
        title: str,
        category_axis_label: str,
        value_axis_label: str,
        unit: str | None,
    ) -> str:
        width = 920
        height = 520
        top = 78
        left = 96
        right = 54
        bottom = 104
        plot_width = width - left - right
        plot_height = height - top - bottom
        values = [float(row[metric]) for row in rows]
        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            padding = max(abs(max_value) * 0.05, 1.0)
            min_value -= padding
            max_value += padding
        span = max_value - min_value

        def x_for(index: int) -> float:
            if len(rows) == 1:
                return left + plot_width / 2
            return left + (index * plot_width / (len(rows) - 1))

        def y_for(value: float) -> float:
            return top + ((max_value - value) / span) * plot_height

        tick_parts = []
        for index in range(5):
            value = min_value + (span * index / 4)
            y = y_for(value)
            tick_parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" />')
            tick_parts.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{_svg(_format_number(value))}</text>')

        points = [(x_for(index), y_for(float(row[metric])), row) for index, row in enumerate(rows)]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        point_parts = []
        label_parts = []
        for index, (x, y, row) in enumerate(points):
            value = float(row[metric])
            point_parts.append(f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="4.5"><title>{_svg(self._row_label(row))}：{_svg(_format_value(value, unit))}</title></circle>')
            label = _truncate(self._row_label(row, include_company=False), 10)
            if len(rows) > 6:
                label_parts.append(f'<text class="x-label" x="{x:.1f}" y="{height - bottom + 30}" text-anchor="end" transform="rotate(-32 {x:.1f} {height - bottom + 30})">{_svg(label)}</text>')
            else:
                label_parts.append(f'<text class="x-label" x="{x:.1f}" y="{height - bottom + 26}" text-anchor="middle">{_svg(label)}</text>')

        return _svg_document(
            width,
            height,
            f"""
            <text class="title" x="{width / 2}" y="34" text-anchor="middle">{_svg(title)}</text>
            <text class="axis-name" x="{width / 2}" y="{height - 18}" text-anchor="middle">横轴：{_svg(category_axis_label)}</text>
            <text class="axis-name" x="22" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2})">纵轴：{_svg(value_axis_label)}</text>
            <line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" />
            <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" />
            {''.join(tick_parts)}
            <polyline class="line" points="{polyline}" />
            {''.join(point_parts)}
            {''.join(label_parts)}
            """,
        )


def _svg_document(width: int, height: int, body: str) -> str:
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <style>
    text {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei", Arial, sans-serif; fill: #1f2937; }}
    .title {{ font-size: 24px; font-weight: 700; }}
    .axis-name {{ font-size: 14px; fill: #475467; }}
    .axis {{ stroke: #667085; stroke-width: 1.2; }}
    .grid {{ stroke: #e4e7ec; stroke-width: 1; }}
    .tick, .x-label, .category {{ font-size: 12px; fill: #667085; }}
    .value {{ font-size: 12px; fill: #1f2937; font-weight: 700; }}
    .bar {{ fill: #2563eb; }}
    .bar.negative {{ fill: #d92d20; }}
    .line {{ fill: none; stroke: #2563eb; stroke-width: 3.2; stroke-linejoin: round; stroke-linecap: round; }}
    .point {{ fill: #ffffff; stroke: #2563eb; stroke-width: 2.4; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#ffffff" />
  {body}
</svg>""".strip()


def _svg_data_url(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _svg(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _strip_unit(label: str) -> str:
    return re.sub(r"[（(][^）)]+[）)]", "", label or "").strip()


def _label_with_unit(label: str, unit: str | None) -> str:
    return f"{label}（{unit}）" if unit else label


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    period_order = {"第一季度": 1, "半年度": 2, "第三季度": 3, "年报": 4}
    year = int(row.get("report_year") or 0)
    period = period_order.get(_period_label(row.get("report_period")), 99)
    stock_code = str(row.get("stock_code") or "")
    return (year, period, stock_code)


def _period_label(period: Any) -> str:
    raw = re.sub(r"\s+", "", str(period or ""))
    upper = raw.upper()
    if upper in PERIOD_LABELS:
        return PERIOD_LABELS[upper]
    if "一季" in raw or "第1季" in raw or "1季" in raw:
        return "第一季度"
    if "半" in raw or "中期" in raw:
        return "半年度"
    if "三季" in raw or "第3季" in raw or "3季" in raw:
        return "第三季度"
    if "年报" in raw or "年度" in raw or raw == "年":
        return "年报"
    return raw


def _is_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _format_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _format_value(value: float, unit: str | None) -> str:
    return f"{_format_number(value)}{unit or ''}"


def _truncate(value: str, max_chars: int) -> str:
    text = str(value or "")
    return text if len(text) <= max_chars else f"{text[: max_chars - 1]}…"