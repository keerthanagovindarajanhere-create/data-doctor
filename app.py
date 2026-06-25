import pandas as pd
import streamlit as st

from src.preprocessor import create_cleaned_dataset
from src.profiler import build_column_report, calculate_readiness_score
from src.recommender import build_recommendations


st.set_page_config(page_title="Data Doctor", page_icon="DD", layout="wide")

st.title("Data Doctor")
st.caption("Upload a CSV file to inspect its quality and receive preprocessing suggestions.")


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
readiness_score, score_explanation = calculate_readiness_score(df)

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Rows", f"{row_count:,}")
metric_2.metric("Columns", column_count)
metric_3.metric("Missing cells", f"{missing_count:,}")
metric_4.metric("Duplicate rows", f"{duplicate_count:,}")
metric_5.metric("Readiness score", f"{readiness_score}/100")

with st.expander("How was the readiness score calculated?"):
    for explanation in score_explanation:
        st.write(explanation)

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
    "The current prototype removes exact duplicates and constant columns, then "
    "fills missing values using the methods explained above. Your uploaded file "
    "is never changed."
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
