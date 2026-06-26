from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.profiler import calculate_readiness_score


@dataclass
class DatasetSummary:
    score: int
    status: str
    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    top_issues: list[str]
    next_step: str
    estimated_time: str
    markdown: str


def _status(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"


def generate_summary(df: pd.DataFrame) -> DatasetSummary:

    score, explanations = calculate_readiness_score(df)

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    issues = []

    if missing:
        issues.append("Missing values detected")

    if duplicates:
        issues.append("Duplicate rows detected")

    constant_columns = [
        c for c in df.columns if df[c].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        issues.append("Constant columns found")

    identifier_columns = []

    for c in df.columns:
        unique_ratio = df[c].nunique(dropna=True) / max(len(df), 1)

        if unique_ratio > 0.95:
            identifier_columns.append(c)

    if identifier_columns:
        issues.append("Possible identifier columns detected")

    if not issues:
        issues.append("No major quality issues detected")

    if score >= 90:
        next_step = "Dataset is ready for machine learning."
    elif score >= 75:
        next_step = "Run automatic preprocessing before training."
    elif score >= 60:
        next_step = "Address quality issues before model development."
    else:
        next_step = "Significant cleaning is recommended."

    estimated_seconds = max(
        5,
        min(
            60,
            len(df) // 10000 + len(df.columns),
        ),
    )

    markdown = f"""
### 🩺 Data Doctor Assessment

**Overall Health:** **{_status(score)}** ({score}/100)

**Dataset Overview**

- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Missing Cells: {missing:,}
- Duplicate Rows: {duplicates:,}

**Key Findings**

{chr(10).join([f"- {x}" for x in issues])}

**Recommended Next Step**

{next_step}

**Estimated preprocessing time**

~ {estimated_seconds} seconds
"""

    return DatasetSummary(
        score=score,
        status=_status(score),
        rows=len(df),
        columns=len(df.columns),
        missing_cells=missing,
        duplicate_rows=duplicates,
        top_issues=issues,
        next_step=next_step,
        estimated_time=f"{estimated_seconds} sec",
        markdown=markdown,
    )