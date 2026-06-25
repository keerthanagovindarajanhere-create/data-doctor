import pandas as pd
import streamlit as st

from src.detectors import run_all_detectors
from src.ml_pipeline import create_ml_ready_dataset, generate_pipeline_code
from src.preprocessor import create_cleaned_dataset
from src.profiler import build_column_report, calculate_readiness_score
from src.recommender import build_recommendations
from src.validator import is_classification_target, validate_classification


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

st.subheader("Advanced diagnostic findings")
selected_target = None if target_column == "Not selected" else target_column
findings = run_all_detectors(df, selected_target)
if findings:
    st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
    st.caption(
        "These are warnings supported by statistical evidence, not automatic "
        "instructions to delete data."
    )
else:
    st.success("No outlier, rarity, correlation, or imbalance warnings were detected.")

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

st.subheader("Build the ML-ready dataset")
if selected_target is None:
    st.info("Select a target column above to build the Scikit-learn pipeline.")
else:
    st.write(
        "Data Doctor will remove identifier-like columns, limit categorical "
        "expansion, and check estimated memory before creating the file."
    )
    if st.button("Build ML-ready dataset"):
        try:
            pipeline_result = create_ml_ready_dataset(df, selected_target)
        except (ValueError, MemoryError) as error:
            st.warning(str(error))
        else:
            st.session_state["pipeline_result"] = pipeline_result
            st.session_state["pipeline_target"] = selected_target

    pipeline_result = st.session_state.get("pipeline_result")
    pipeline_target = st.session_state.get("pipeline_target")
    if pipeline_result is not None and pipeline_target == selected_target:
        st.write(
            "The pipeline imputes numeric values, scales numeric features, "
            "one-hot encodes categories, and preserves the target column."
        )

        summary_1, summary_2, summary_3 = st.columns(3)
        summary_1.metric("Numeric features", len(pipeline_result.numeric_columns))
        summary_2.metric(
            "Categorical features", len(pipeline_result.categorical_columns)
        )
        summary_3.metric(
            "ML-ready columns", pipeline_result.transformed_data.shape[1]
        )

        if pipeline_result.removed_columns:
            st.caption(
                "Removed constant features: "
                + ", ".join(pipeline_result.removed_columns)
            )
        if pipeline_result.high_cardinality_columns:
            st.caption(
                "Excluded identifier-like or extremely high-cardinality columns: "
                + ", ".join(pipeline_result.high_cardinality_columns)
            )
        if pipeline_result.dropped_target_rows:
            st.caption(
                f"Dropped {pipeline_result.dropped_target_rows} row(s) with a "
                "missing target because a model cannot learn their correct label."
            )

        with st.expander("Preview transformed data"):
            st.dataframe(
                pipeline_result.transformed_data.head(20),
                use_container_width=True,
            )

        ml_ready_csv = pipeline_result.transformed_data.to_csv(index=False).encode(
            "utf-8"
        )
        pipeline_code = generate_pipeline_code(selected_target)

        download_1, download_2 = st.columns(2)
        download_1.download_button(
            "Download ML-ready CSV",
            data=ml_ready_csv,
            file_name="data_doctor_ml_ready.csv",
            mime="text/csv",
        )
        download_2.download_button(
            "Download reusable pipeline code",
            data=pipeline_code,
            file_name="data_doctor_pipeline.py",
            mime="text/x-python",
        )

        st.subheader("Validate preprocessing with ML models")
        if not is_classification_target(df[selected_target]):
            st.info(
                "This validation milestone currently supports classification targets. "
                "Regression validation will be added next."
            )
        elif st.button("Run classification validation", type="primary"):
            with st.spinner(
                "Training baseline models across multiple folds. This can take a minute..."
            ):
                try:
                    validation = validate_classification(df, selected_target)
                except ValueError as error:
                    st.warning(str(error))
                except Exception as error:
                    st.error(f"Validation could not finish: {error}")
                else:
                    st.success(
                        f"Compared both approaches using {validation.folds}-fold "
                        f"cross-validation on {validation.rows_used:,} rows."
                    )
                    st.dataframe(
                        validation.metrics,
                        use_container_width=True,
                        hide_index=True,
                    )

                    chart_data = validation.metrics.pivot(
                        index="Model",
                        columns="Approach",
                        values="F1",
                    )
                    st.write("F1-score comparison")
                    st.bar_chart(chart_data)
                    st.caption(
                        "A higher score is better, but small differences may be normal "
                        "cross-validation variation rather than a real improvement."
                    )
