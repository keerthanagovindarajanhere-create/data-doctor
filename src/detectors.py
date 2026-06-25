from __future__ import annotations

from itertools import combinations

import pandas as pd


Finding = dict[str, object]


def detect_outliers(dataframe: pd.DataFrame) -> list[Finding]:
    """Detect numeric outliers with the explainable IQR rule."""
    findings: list[Finding] = []

    for column in dataframe.select_dtypes(include="number").columns:
        series = dataframe[column].dropna()
        if len(series) < 4 or series.nunique() < 4:
            continue

        first_quartile = series.quantile(0.25)
        third_quartile = series.quantile(0.75)
        iqr = third_quartile - first_quartile
        if iqr == 0:
            continue

        lower_bound = first_quartile - 1.5 * iqr
        upper_bound = third_quartile + 1.5 * iqr
        outlier_count = int(
            ((series < lower_bound) | (series > upper_bound)).sum()
        )

        if outlier_count:
            findings.append(
                {
                    "Severity": "Medium",
                    "Issue": "Possible outliers",
                    "Column(s)": column,
                    "Evidence": (
                        f"{outlier_count} value(s) fall outside the IQR bounds "
                        f"[{lower_bound:.2f}, {upper_bound:.2f}]."
                    ),
                }
            )

    return findings


def detect_rare_categories(
    dataframe: pd.DataFrame, threshold: float = 0.01
) -> list[Finding]:
    """Find categorical values occurring in less than the chosen row fraction."""
    findings: list[Finding] = []

    for column in dataframe.select_dtypes(exclude="number").columns:
        frequencies = dataframe[column].value_counts(normalize=True, dropna=True)
        rare_values = frequencies[frequencies < threshold]

        if not rare_values.empty:
            findings.append(
                {
                    "Severity": "Low",
                    "Issue": "Rare categories",
                    "Column(s)": column,
                    "Evidence": (
                        f"{len(rare_values)} category value(s) occur in less than "
                        f"{threshold:.0%} of non-missing rows."
                    ),
                }
            )

    return findings


def detect_near_constant_features(
    dataframe: pd.DataFrame, threshold: float = 0.95
) -> list[Finding]:
    """Find columns dominated by one value without being fully constant."""
    findings: list[Finding] = []

    for column in dataframe.columns:
        non_null = dataframe[column].dropna()
        if non_null.empty or non_null.nunique() <= 1:
            continue

        dominant_rate = float(non_null.value_counts(normalize=True).iloc[0])
        if dominant_rate >= threshold:
            findings.append(
                {
                    "Severity": "Medium",
                    "Issue": "Near-constant feature",
                    "Column(s)": column,
                    "Evidence": (
                        f"One value represents {dominant_rate:.1%} of the "
                        "non-missing observations."
                    ),
                }
            )

    return findings


def detect_high_correlations(
    dataframe: pd.DataFrame, threshold: float = 0.90
) -> list[Finding]:
    """Find strongly correlated pairs of numeric features."""
    findings: list[Finding] = []
    numeric_data = dataframe.select_dtypes(include="number")

    if numeric_data.shape[1] < 2:
        return findings

    correlations = numeric_data.corr()
    for first, second in combinations(correlations.columns, 2):
        correlation = correlations.loc[first, second]
        if pd.notna(correlation) and abs(correlation) >= threshold:
            findings.append(
                {
                    "Severity": "Medium",
                    "Issue": "High correlation",
                    "Column(s)": f"{first}, {second}",
                    "Evidence": (
                        f"Pearson correlation is {correlation:.3f}. These features "
                        "may contain overlapping information."
                    ),
                }
            )

    return findings


def detect_class_imbalance(
    dataframe: pd.DataFrame, target_column: str | None
) -> list[Finding]:
    """Detect an underrepresented class in a selected classification target."""
    if not target_column or target_column not in dataframe.columns:
        return []

    target = dataframe[target_column].dropna()
    class_count = target.nunique()
    if class_count < 2 or class_count > max(20, int(len(target) * 0.05)):
        return []

    proportions = target.value_counts(normalize=True)
    minority_class = proportions.idxmin()
    minority_rate = float(proportions.min())

    if minority_rate >= 0.20:
        return []

    severity = "High" if minority_rate < 0.10 else "Medium"
    return [
        {
            "Severity": severity,
            "Issue": "Class imbalance",
            "Column(s)": target_column,
            "Evidence": (
                f"The least frequent class ({minority_class!s}) represents only "
                f"{minority_rate:.1%} of non-missing target values."
            ),
        }
    ]


def run_all_detectors(
    dataframe: pd.DataFrame, target_column: str | None = None
) -> list[Finding]:
    """Run all current diagnostic rules and return one combined finding list."""
    findings = [
        *detect_outliers(dataframe),
        *detect_rare_categories(dataframe),
        *detect_near_constant_features(dataframe),
        *detect_high_correlations(dataframe),
        *detect_class_imbalance(dataframe, target_column),
    ]

    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(
        findings,
        key=lambda finding: severity_order.get(str(finding["Severity"]), 3),
    )

