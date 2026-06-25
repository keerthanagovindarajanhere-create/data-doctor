import pandas as pd

from src.detectors import (
    detect_class_imbalance,
    detect_high_correlations,
    detect_near_constant_features,
    detect_outliers,
    detect_rare_categories,
)
from src.ml_pipeline import create_ml_ready_dataset, generate_pipeline_code
from src.preprocessor import create_cleaned_dataset
from src.profiler import build_column_report, calculate_readiness_score
from src.recommender import build_recommendations


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, None, 30.0],
            "city": ["Chennai", "Delhi", None, "Delhi"],
            "constant": ["same", "same", "same", "same"],
        }
    )


def test_column_report_contains_every_column():
    report = build_column_report(sample_dataframe())
    assert report["Column"].tolist() == ["age", "city", "constant"]


def test_recommendations_are_ranked():
    recommendations = build_recommendations(sample_dataframe())
    priorities = [recommendation["Priority"] for recommendation in recommendations]
    assert priorities == sorted(priorities, reverse=True)


def test_cleaning_removes_missing_values_and_constant_columns():
    cleaned = create_cleaned_dataset(sample_dataframe())
    assert cleaned.isna().sum().sum() == 0
    assert "constant" not in cleaned.columns


def test_readiness_score_stays_in_range():
    score, explanations = calculate_readiness_score(sample_dataframe())
    assert 0 <= score <= 100
    assert explanations


def test_outlier_detector_finds_extreme_value():
    dataframe = pd.DataFrame({"amount": list(range(1, 21)) + [1000]})
    findings = detect_outliers(dataframe)
    assert findings[0]["Issue"] == "Possible outliers"


def test_near_constant_detector_finds_dominant_value():
    dataframe = pd.DataFrame({"flag": ["no"] * 19 + ["yes"]})
    findings = detect_near_constant_features(dataframe)
    assert findings[0]["Column(s)"] == "flag"


def test_rare_category_detector_finds_small_group():
    dataframe = pd.DataFrame({"city": ["Chennai"] * 100 + ["Pune"]})
    findings = detect_rare_categories(dataframe)
    assert findings[0]["Issue"] == "Rare categories"


def test_correlation_detector_finds_feature_pair():
    dataframe = pd.DataFrame(
        {"distance_km": range(20), "distance_m": [value * 1000 for value in range(20)]}
    )
    findings = detect_high_correlations(dataframe)
    assert findings[0]["Issue"] == "High correlation"


def test_class_imbalance_detector_uses_selected_target():
    dataframe = pd.DataFrame({"approved": ["yes"] * 95 + ["no"] * 5})
    findings = detect_class_imbalance(dataframe, "approved")
    assert findings[0]["Severity"] == "High"


def test_ml_pipeline_creates_numeric_output_and_preserves_target():
    dataframe = pd.DataFrame(
        {
            "age": [20.0, None, 40.0, 50.0],
            "city": ["Chennai", "Delhi", None, "Chennai"],
            "constant": ["same"] * 4,
            "approved": ["yes", "no", "yes", "no"],
        }
    )
    result = create_ml_ready_dataset(dataframe, "approved")
    features = result.transformed_data.drop(columns=["approved"])

    assert features.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(features[column]) for column in features)
    assert result.transformed_data["approved"].tolist() == dataframe["approved"].tolist()
    assert result.removed_columns == ["constant"]


def test_pipeline_code_contains_selected_target():
    code = generate_pipeline_code("approved")
    assert "target_column = 'approved'" in code
