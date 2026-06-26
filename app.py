from __future__ import annotations

import pandas as pd
import streamlit as st
from src.certification import generate_certificate
from src.detectors import run_all_detectors
from src.ml_pipeline import create_ml_ready_dataset, generate_pipeline_code
from src.preprocessor import create_cleaned_dataset
from src.profiler import build_column_report, calculate_readiness_score
from src.recommender import build_recommendations
from src.reporting import build_report_payload, report_as_html, report_as_json
from src.validator import (
    infer_task_type,
    recommend_models,
    validate_classification,
    validate_regression,
)


st.set_page_config(page_title="Data Doctor", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1450px}
    [data-testid="stMetric"] {background:#f4f8fa;border:1px solid #dce8ed;
      padding:14px;border-radius:14px}
    .hero {padding:24px 28px;border-radius:20px;
      background:linear-gradient(120deg,#12364a,#087f6b);color:white;margin-bottom:20px}
    .hero h1 {margin:0;color:white}.hero p{margin:8px 0 0;color:#d9f4ef}
    </style>
    <div class="hero"><h1>Data Doctor</h1>
    <p>Explainable dataset diagnosis, preprocessing, and ML-readiness validation.</p></div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
    help="For the hosted version, use a reasonably sized tabular CSV.",
)
if uploaded_file is None:
    st.info("Upload a CSV to begin. The dashboard will guide you through each stage.")
    st.stop()

try:
    dataframe = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"The CSV could not be read: {error}")
    st.stop()
if dataframe.empty:
    st.warning("The uploaded CSV has no data rows.")
    st.stop()

target_column = st.selectbox(
    "Target column",
    ["Not selected"] + list(dataframe.columns),
    help="The value a future model should predict.",
)
selected_target = None if target_column == "Not selected" else target_column
task_type = (
    infer_task_type(dataframe[selected_target]) if selected_target else "not selected"
)

readiness_score, score_explanation = calculate_readiness_score(dataframe)
column_report = build_column_report(dataframe)
findings = run_all_detectors(dataframe, selected_target)
recommendations = build_recommendations(dataframe)
certificate = generate_certificate(
    dataframe,
    readiness_score,
    findings,
    recommendations,
)
cleaned_data = create_cleaned_dataset(dataframe)
cleaned_score, _ = calculate_readiness_score(cleaned_data)

overview_tab, diagnostics_tab, prep_tab, validation_tab, export_tab = st.tabs(
    ["Overview", "Diagnostics", "Preprocessing", "Validation", "Export"]
)

with overview_tab:
    metrics = st.columns(6)
    metrics[0].metric("Rows", f"{len(dataframe):,}")
    metrics[1].metric("Columns", len(dataframe.columns))
    metrics[2].metric("Missing cells", f"{dataframe.isna().sum().sum():,}")
    metrics[3].metric("Duplicates", f"{dataframe.duplicated().sum():,}")
    metrics[4].metric("Readiness", f"{readiness_score}/100")
    metrics[5].metric("Task", task_type.title())

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Dataset preview")
        st.dataframe(dataframe.head(25), use_container_width=True)
    with right:
        st.subheader("Readiness explanation")
        for explanation in score_explanation:
            st.write(f"• {explanation}")
        if selected_target:
            st.subheader("Suggested models")
            st.dataframe(
                pd.DataFrame(recommend_models(dataframe, selected_target)),
                hide_index=True,
                use_container_width=True,
            )

with diagnostics_tab:
    st.subheader("Column profiles")
    st.dataframe(column_report, use_container_width=True, hide_index=True)

    st.subheader("Quality findings")
    if findings:
        st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
    else:
        st.success("No advanced quality warnings were detected.")

    numeric = dataframe.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        st.subheader("Numeric correlation matrix")
        st.dataframe(
            numeric.corr().round(3),
            use_container_width=True,
        )

with prep_tab:
    st.subheader("Ranked recommendations")
    st.caption(
        "Recommendations explain what Data Doctor proposes and why. Statistical "
        "warnings are not automatically treated as errors."
    )
    st.dataframe(
        pd.DataFrame(recommendations),
        use_container_width=True,
        hide_index=True,
    )

    # ===============================
    # Before vs After Dataset Summary
    # ===============================
    st.divider()
    st.subheader("📊 Dataset Improvement Summary")

    before_col, arrow_col, after_col = st.columns([2, 0.5, 2])

    with before_col:
        st.markdown("### Before Cleaning")
        st.metric("Rows", f"{len(dataframe):,}")
        st.metric("Columns", len(dataframe.columns))
        st.metric("Missing Cells", int(dataframe.isna().sum().sum()))
        st.metric("Duplicate Rows", int(dataframe.duplicated().sum()))
        st.metric("Health Score", f"{readiness_score}/100")

    with arrow_col:
        st.markdown("<h1 style='text-align:center;'>➡️</h1>", unsafe_allow_html=True)

    with after_col:
        st.markdown("### After Cleaning")
        st.metric("Rows", f"{len(cleaned_data):,}")
        st.metric("Columns", len(cleaned_data.columns))
        st.metric("Missing Cells", int(cleaned_data.isna().sum().sum()))
        st.metric("Duplicate Rows", int(cleaned_data.duplicated().sum()))
        st.metric("Health Score", f"{cleaned_score}/100")

    st.progress(cleaned_score / 100)

    improvement = cleaned_score - readiness_score

    if improvement > 0:
        st.success(
            f"🎉 Dataset health improved by **{improvement} points** "
            f"({readiness_score} → {cleaned_score})."
        )
    else:
        st.info("No significant improvement was needed after preprocessing.")

    # ===============================
    # Download Section
    # ===============================
    cleaned_col, ml_col = st.columns(2)

    with cleaned_col:
        st.markdown("### Cleaned dataset")
        st.write(
            "Human-readable data with duplicates, constants, and missingness handled."
        )
        st.download_button(
            "Download cleaned CSV",
            cleaned_data.to_csv(index=False).encode("utf-8"),
            "data_doctor_cleaned.csv",
            "text/csv",
            use_container_width=True,
        )

    with ml_col:
        st.markdown("### ML-ready dataset")
        st.write(
            "Encoded and scaled numeric features with the selected target preserved."
        )

        if selected_target is None:
            st.info("Select a target column first.")
        elif st.button("Build ML-ready artifacts", use_container_width=True):
            with st.spinner("Building a memory-safe preprocessing pipeline..."):
                try:
                    result = create_ml_ready_dataset(dataframe, selected_target)
                except (ValueError, MemoryError) as error:
                    st.warning(str(error))
                else:
                    st.session_state.pipeline_result = result
                    st.session_state.pipeline_target = selected_target

    result = st.session_state.get("pipeline_result")

    if (
        result is not None
        and st.session_state.get("pipeline_target") == selected_target
    ):
        st.success(
            f"Created {result.transformed_data.shape[1]:,} ML-ready columns. "
            f"Excluded {len(result.high_cardinality_columns)} identifier-like columns."
        )

        st.dataframe(
            result.transformed_data.head(20),
            use_container_width=True,
        )

        first, second = st.columns(2)

        first.download_button(
            "Download ML-ready CSV",
            result.transformed_data.to_csv(index=False).encode("utf-8"),
            "data_doctor_ml_ready.csv",
            "text/csv",
            use_container_width=True,
        )

        second.download_button(
            "Download pipeline code",
            generate_pipeline_code(selected_target),
            "data_doctor_pipeline.py",
            "text/x-python",
            use_container_width=True,
        )

with validation_tab:
    st.subheader("Leakage-safe before vs after validation")
    if selected_target is None:
        st.info("Select a target to enable validation.")
    elif task_type == "unsupported":
        st.warning("This target type is not supported yet.")
    else:
        st.write(
            "Both approaches use identical folds. Every imputer, encoder, scaler, "
            "and model is fitted only on each training fold."
        )
        if st.button("Run model validation", type="primary"):
            with st.spinner("Training baseline models. Large datasets may take a minute..."):
                try:
                    validation = (
                        validate_classification(dataframe, selected_target)
                        if task_type == "classification"
                        else validate_regression(dataframe, selected_target)
                    )
                except Exception as error:
                    st.warning(f"Validation could not finish: {error}")
                else:
                    st.session_state.validation = validation
                    st.session_state.validation_target = selected_target

        validation = st.session_state.get("validation")
        if (
            validation is not None
            and st.session_state.get("validation_target") == selected_target
        ):
            st.success(
                f"Completed {validation.folds}-fold {validation.task_type} validation "
                f"on {validation.rows_used:,} rows."
            )
            st.dataframe(validation.metrics, use_container_width=True, hide_index=True)
            comparison_metric = "F1" if validation.task_type == "classification" else "R²"
            chart = validation.metrics.pivot(
                index="Model", columns="Approach", values=comparison_metric
            )
            st.bar_chart(chart)
            st.caption(
                "Small differences can be normal fold-to-fold variation; validation "
                "does not guarantee the intelligent pipeline always wins."
            )

with export_tab:
    validation = st.session_state.get("validation")
    valid_metrics = (
        validation.metrics
        if validation is not None
        and st.session_state.get("validation_target") == selected_target
        else None
    )
    payload = build_report_payload(
        dataframe,
        selected_target,
        readiness_score,
        score_explanation,
        findings,
        recommendations,
        valid_metrics,
    )
    st.subheader("🩺 Dataset Certification")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Certification Grade", certificate.grade)
        st.metric("Health Score", f"{certificate.score}/100")
        st.metric("Risk Level", certificate.risk)

    with col2:
        status = "✅ Certified" if certificate.ml_ready else "❌ Not Certified"

        st.metric("Status", status)

        st.code(certificate.certificate_id)

        st.info(certificate.summary)
    st.subheader("Portable analysis report")
    st.write(
        "Download a readable HTML report for your portfolio or JSON for another system."
    )
    html_col, json_col = st.columns(2)
    html_col.download_button(
        "Download HTML report",
        report_as_html(payload),
        "data_doctor_report.html",
        "text/html",
        use_container_width=True,
    )
    json_col.download_button(
        "Download JSON report",
        report_as_json(payload),
        "data_doctor_report.json",
        "application/json",
        use_container_width=True,
    )
