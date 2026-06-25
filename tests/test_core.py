import pandas as pd

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

