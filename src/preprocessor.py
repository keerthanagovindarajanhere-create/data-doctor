from __future__ import annotations

import pandas as pd


def create_cleaned_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the safe transformations supported by the current prototype."""
    cleaned = dataframe.drop_duplicates().copy()

    constant_columns = [
        column for column in cleaned.columns if cleaned[column].nunique() <= 1
    ]
    cleaned = cleaned.drop(columns=constant_columns)

    for column in cleaned.columns:
        series = cleaned[column]
        if not series.isna().any():
            continue

        if pd.api.types.is_numeric_dtype(series):
            skewness = series.skew()
            replacement = (
                series.median()
                if pd.notna(skewness) and abs(skewness) > 1
                else series.mean()
            )
        else:
            modes = series.mode(dropna=True)
            replacement = modes.iloc[0] if not modes.empty else "Unknown"

        cleaned[column] = series.fillna(replacement)

    return cleaned

