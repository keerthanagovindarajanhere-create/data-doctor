import pandas as pd
import streamlit as st


st.set_page_config(page_title="Data Doctor", page_icon="🩺", layout="wide")

st.title("🩺 Data Doctor")
st.caption("Upload a CSV file to inspect its quality and receive preprocessing suggestions.")


def build_column_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create one easy-to-read quality summary row for every column."""
    rows = []

    for column in dataframe.columns:
        series = dataframe[column]
        rows.append(
            {
                "Column": column,
                "Data type": str(series.dtype),
                "Missing values": int(series.isna().sum()),
                "Missing %": round(series.isna().mean() * 100, 2),
                "Unique values": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def build_recommendations(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    """Generate simple, explainable preprocessing recommendations."""
    recommendations = []

    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count:
        recommendations.append(
            {
                "Issue": "Duplicate rows",
                "Recommendation": "Remove exact duplicate rows",
                "Reason": f"{duplicate_count} duplicate rows can bias analysis and model training.",
            }
        )

    for column in dataframe.columns:
        series = dataframe[column]
        missing_rate = series.isna().mean()

        if missing_rate == 0:
            continue

        if pd.api.types.is_numeric_dtype(series):
            skewness = series.skew()
            if pd.notna(skewness) and abs(skewness) > 1:
                method = "Fill missing values with the median"
                reason = (
                    f"{column} is numeric and highly skewed "
                    f"(skewness {skewness:.2f}); the median is resistant to extreme values."
                )
            else:
                method = "Fill missing values with the mean"
                reason = (
                    f"{column} is numeric and is not highly skewed; "
                    "the mean is a reasonable first strategy."
                )
        else:
            method = "Fill missing values with the most frequent category"
            reason = (
                f"{column} is categorical; its most frequent value provides "
                "a simple replacement without inventing a new numeric value."
            )

        recommendations.append(
            {
                "Issue": f"Missing values in {column}",
                "Recommendation": method,
                "Reason": reason,
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "Issue": "No basic issues found",
                "Recommendation": "Continue with advanced profiling",
                "Reason": "The dataset has no missing values or exact duplicate rows.",
            }
        )

    return recommendations


def create_cleaned_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply only the simple transformations recommended in version 0.1."""
    cleaned = dataframe.drop_duplicates().copy()

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


uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Start by uploading a CSV file. Your data stays in this running app.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"The CSV could not be read: {error}")
    st.stop()

if df.empty:
    st.warning("The uploaded CSV contains no data rows.")
    st.stop()

st.success("Dataset loaded successfully.")

row_count, column_count = df.shape
duplicate_count = int(df.duplicated().sum())
missing_count = int(df.isna().sum().sum())

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Rows", f"{row_count:,}")
metric_2.metric("Columns", column_count)
metric_3.metric("Missing cells", f"{missing_count:,}")
metric_4.metric("Duplicate rows", f"{duplicate_count:,}")

st.subheader("Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Choose the target column")
target_column = st.selectbox(
    "The target is the value your future ML model will try to predict.",
    options=["Not selected"] + list(df.columns),
)

if target_column != "Not selected":
    unique_targets = df[target_column].nunique(dropna=True)
    suggested_task = (
        "Classification"
        if unique_targets <= max(20, int(len(df) * 0.05))
        else "Regression"
    )
    st.info(
        f"Initial suggestion: **{suggested_task}** "
        f"because `{target_column}` has {unique_targets} unique values."
    )

st.subheader("Column quality report")
column_report = build_column_report(df)
st.dataframe(column_report, use_container_width=True, hide_index=True)

st.subheader("Explainable recommendations")
recommendations = build_recommendations(df)
st.dataframe(pd.DataFrame(recommendations), use_container_width=True, hide_index=True)

st.subheader("Create the cleaned dataset")
st.write(
    "Version 0.1 removes exact duplicates and fills missing values using the "
    "methods explained above. The original uploaded file is never changed."
)

cleaned_df = create_cleaned_dataset(df)
cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download cleaned CSV",
    data=cleaned_csv,
    file_name="data_doctor_cleaned.csv",
    mime="text/csv",
    type="primary",
)

