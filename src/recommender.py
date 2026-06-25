from __future__ import annotations

import pandas as pd


def build_recommendations(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Generate ranked, evidence-backed preprocessing recommendations."""
    recommendations: list[dict[str, object]] = []
    duplicate_count = int(dataframe.duplicated().sum())

    if duplicate_count:
        recommendations.append(
            {
                "Priority": 90,
                "Severity": "High",
                "Column": "Entire dataset",
                "Issue": "Duplicate rows",
                "Recommendation": "Remove exact duplicate rows",
                "Reason": (
                    f"{duplicate_count} duplicate rows may give repeated observations "
                    "too much influence during model training."
                ),
            }
        )

    for column in dataframe.columns:
        series = dataframe[column]
        missing_rate = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))

        if unique_count <= 1:
            recommendations.append(
                {
                    "Priority": 85,
                    "Severity": "High",
                    "Column": column,
                    "Issue": "Constant feature",
                    "Recommendation": "Remove this column",
                    "Reason": (
                        "It contains one or fewer distinct non-missing values, "
                        "so it cannot help distinguish predictions."
                    ),
                }
            )

        if missing_rate == 0:
            continue

        severity = "High" if missing_rate >= 0.4 else "Medium"
        priority = 80 if missing_rate >= 0.4 else 60

        if pd.api.types.is_numeric_dtype(series):
            skewness = series.skew()
            if pd.notna(skewness) and abs(skewness) > 1:
                method = "Fill missing values with the median"
                reason = (
                    f"The column is numeric and highly skewed "
                    f"(skewness {skewness:.2f}). The median is less affected "
                    "by extreme values."
                )
            else:
                method = "Fill missing values with the mean"
                reason = (
                    "The column is numeric and not highly skewed, so the mean "
                    "is a reasonable initial replacement."
                )
        else:
            method = "Fill missing values with the most frequent category"
            reason = (
                "The column is categorical, so its most frequent observed value "
                "is a simple type-safe replacement."
            )

        recommendations.append(
            {
                "Priority": priority,
                "Severity": severity,
                "Column": column,
                "Issue": f"{missing_rate:.1%} missing values",
                "Recommendation": method,
                "Reason": reason,
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "Priority": 10,
                "Severity": "Info",
                "Column": "Entire dataset",
                "Issue": "No basic issues found",
                "Recommendation": "Continue with advanced profiling",
                "Reason": "No missing values, duplicates, or constant features were found.",
            }
        )

    return sorted(
        recommendations,
        key=lambda recommendation: int(recommendation["Priority"]),
        reverse=True,
    )

