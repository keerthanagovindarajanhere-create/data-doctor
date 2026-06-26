from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import pandas as pd


def build_report_payload(
    dataframe: pd.DataFrame,
    target_column: str | None,
    readiness_score: int,
    deductions: list[str],
    findings: list[dict[str, object]],
    recommendations: list[dict[str, object]],
    validation_metrics: pd.DataFrame | None = None,
) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "target": target_column,
            "missing_cells": int(dataframe.isna().sum().sum()),
            "duplicate_rows": int(dataframe.duplicated().sum()),
        },
        "readiness_score": readiness_score,
        "score_explanation": deductions,
        "findings": findings,
        "recommendations": recommendations,
        "validation": (
            validation_metrics.to_dict(orient="records")
            if validation_metrics is not None
            else []
        ),
    }


def report_as_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, default=str)


def report_as_html(payload: dict[str, object]) -> str:
    dataset = payload["dataset"]
    findings = payload["findings"]
    recommendations = payload["recommendations"]
    validation = payload["validation"]

    def table(rows: list[dict[str, object]]) -> str:
        if not rows:
            return "<p>No results available.</p>"
        headers = list(rows[0])
        head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(column, '')))}</td>"
                for column in headers
            )
            + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Data Doctor Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;color:#172033}}
h1,h2{{color:#173b57}} .score{{font-size:42px;font-weight:700;color:#0b8f6a}}
table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{border:1px solid #d9e2ea;padding:8px;text-align:left;vertical-align:top}}
th{{background:#eef6f8}} .meta{{color:#607080}}
</style></head><body>
<h1>Data Doctor preprocessing report</h1>
<p class="meta">Generated {html.escape(str(payload["generated_at_utc"]))}</p>
<div class="score">{payload["readiness_score"]}/100</div>
<p>{dataset["rows"]:,} rows · {dataset["columns"]} columns ·
Target: {html.escape(str(dataset["target"] or "Not selected"))}</p>
<h2>Score explanation</h2><ul>{"".join(f"<li>{html.escape(str(x))}</li>" for x in payload["score_explanation"])}</ul>
<h2>Diagnostic findings</h2>{table(findings)}
<h2>Recommendations</h2>{table(recommendations)}
<h2>Model validation</h2>{table(validation)}
</body></html>"""
