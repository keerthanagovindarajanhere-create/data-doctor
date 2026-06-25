from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class ValidationResult:
    """Summary returned by the leakage-safe classification comparison."""

    metrics: pd.DataFrame
    folds: int
    rows_used: int
    class_count: int


def is_classification_target(target: pd.Series) -> bool:
    """Use target cardinality to suggest whether classification is appropriate."""
    clean_target = target.dropna()
    unique_count = clean_target.nunique()
    return 2 <= unique_count <= max(20, int(len(clean_target) * 0.05))


def _feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def _minimal_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Perform only the transformations required to make models run."""
    numeric, categorical = _feature_groups(features)
    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers, remainder="drop")


def _intelligent_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Add robust imputation and scaling while remaining fold-local."""
    numeric, categorical = _feature_groups(features)
    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=0.01,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers, remainder="drop")


def _scoring(class_count: int) -> dict[str, object]:
    scorers: dict[str, object] = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }
    scorers["roc_auc"] = "roc_auc" if class_count == 2 else "roc_auc_ovr_weighted"
    return scorers


def validate_classification(
    dataframe: pd.DataFrame,
    target_column: str,
    requested_folds: int = 5,
) -> ValidationResult:
    """Compare minimal and intelligent preprocessing with identical CV folds."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    prepared = dataframe.drop_duplicates().dropna(subset=[target_column]).copy()
    if len(prepared) < 20:
        raise ValueError("At least 20 usable rows are required for validation.")

    target = prepared[target_column]
    if not is_classification_target(target):
        raise ValueError(
            "The selected target does not currently look like a classification target."
        )

    features = prepared.drop(columns=[target_column])
    constant_columns = [
        column for column in features.columns if features[column].nunique() <= 1
    ]
    features = features.drop(columns=constant_columns)
    if features.shape[1] == 0:
        raise ValueError("No usable feature columns remain for model validation.")

    encoded_target = LabelEncoder().fit_transform(target.astype(str))
    class_counts = pd.Series(encoded_target).value_counts()
    folds = min(requested_folds, int(class_counts.min()))
    if folds < 2:
        raise ValueError("Every target class needs at least two rows for validation.")

    split_strategy = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )
    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        ),
    }
    preprocessing_factories = {
        "Minimal baseline": _minimal_preprocessor,
        "Data Doctor": _intelligent_preprocessor,
    }

    rows: list[dict[str, object]] = []
    for approach, preprocessor_factory in preprocessing_factories.items():
        for model_name, model_factory in model_factories.items():
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor_factory(features)),
                    ("model", model_factory()),
                ]
            )
            scores = cross_validate(
                pipeline,
                features,
                encoded_target,
                cv=split_strategy,
                scoring=_scoring(len(class_counts)),
                error_score="raise",
                n_jobs=1,
            )

            rows.append(
                {
                    "Approach": approach,
                    "Model": model_name,
                    "Accuracy": scores["test_accuracy"].mean(),
                    "Precision": scores["test_precision"].mean(),
                    "Recall": scores["test_recall"].mean(),
                    "F1": scores["test_f1"].mean(),
                    "ROC-AUC": scores["test_roc_auc"].mean(),
                    "Training time (s)": scores["fit_time"].mean(),
                    "Inference time (s)": scores["score_time"].mean(),
                }
            )

    metrics = pd.DataFrame(rows)
    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    metrics[metric_columns] = metrics[metric_columns].round(4)
    metrics[["Training time (s)", "Inference time (s)"]] = metrics[
        ["Training time (s)", "Inference time (s)"]
    ].round(4)

    return ValidationResult(
        metrics=metrics,
        folds=folds,
        rows_used=len(prepared),
        class_count=len(class_counts),
    )

