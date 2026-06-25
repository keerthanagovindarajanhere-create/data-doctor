from __future__ import annotations

import pandas as pd


def infer_semantic_type(series: pd.Series) -> str:
    """Describe how a column is likely to be used in machine learning."""
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"

    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / max(len(series), 1)

    if unique_ratio > 0.95:
        return "Possible identifier"
    return "Categorical"


def build_column_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create one quality summary row for every dataset column."""
    rows: list[dict[str, object]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        skewness = series.skew() if pd.api.types.is_numeric_dtype(series) else None

        rows.append(
            {
                "Column": column,
                "Stored type": str(series.dtype),
                "Semantic type": infer_semantic_type(series),
                "Missing values": int(series.isna().sum()),
                "Missing %": round(float(series.isna().mean() * 100), 2),
                "Unique values": int(series.nunique(dropna=True)),
                "Unique %": round(
                    float(series.nunique(dropna=True) / max(len(non_null), 1) * 100),
                    2,
                ),
                "Skewness": (
                    round(float(skewness), 2) if pd.notna(skewness) else None
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_readiness_score(dataframe: pd.DataFrame) -> tuple[int, list[str]]:
    """Return a transparent 0–100 preliminary dataset readiness score."""
    score = 100.0
    deductions: list[str] = []
    total_cells = max(dataframe.size, 1)

    missing_rate = dataframe.isna().sum().sum() / total_cells
    if missing_rate:
        penalty = min(30.0, missing_rate * 100)
        score -= penalty
        deductions.append(f"-{penalty:.1f}: {missing_rate:.1%} of cells are missing")

    duplicate_rate = dataframe.duplicated().mean()
    if duplicate_rate:
        penalty = min(15.0, duplicate_rate * 50)
        score -= penalty
        deductions.append(
            f"-{penalty:.1f}: {duplicate_rate:.1%} of rows are exact duplicates"
        )

    constant_columns = [
        column for column in dataframe.columns if dataframe[column].nunique() <= 1
    ]
    if constant_columns:
        penalty = min(15.0, len(constant_columns) * 3.0)
        score -= penalty
        deductions.append(
            f"-{penalty:.1f}: {len(constant_columns)} constant column(s)"
        )

    identifier_columns = [
        column
        for column in dataframe.columns
        if infer_semantic_type(dataframe[column]) == "Possible identifier"
    ]
    if identifier_columns:
        penalty = min(10.0, len(identifier_columns) * 2.0)
        score -= penalty
        deductions.append(
            f"-{penalty:.1f}: {len(identifier_columns)} possible identifier column(s)"
        )

    if not deductions:
        deductions.append("No basic quality penalties were detected")

    return max(0, round(score)), deductions

