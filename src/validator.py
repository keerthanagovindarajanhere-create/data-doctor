from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class ValidationResult:
    """Summary returned by the leakage-safe classification comparison."""

    metrics: pd.DataFrame
    folds: int
    rows_used: int
    class_count: int
    task_type: str = "classification"


def is_classification_target(target: pd.Series) -> bool:
    """Use target cardinality to suggest whether classification is appropriate."""
    clean_target = target.dropna()
    unique_count = clean_target.nunique()
    return 2 <= unique_count <= max(20, int(len(clean_target) * 0.05))


def infer_task_type(target: pd.Series) -> str:
    if is_classification_target(target):
        return "classification"
    if pd.api.types.is_numeric_dtype(target):
        return "regression"
    return "unsupported"


def _feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def _safe_features(features: pd.DataFrame) -> pd.DataFrame:
    constant = [column for column in features if features[column].nunique() <= 1]
    identifiers = [
        column
        for column in features.select_dtypes(exclude="number").columns
        if features[column].nunique(dropna=True) > 5_000
        or (
            features[column].nunique(dropna=True) > 100
            and features[column].nunique(dropna=True)
            / max(features[column].notna().sum(), 1)
            > 0.95
        )
    ]
    return features.drop(columns=constant + identifiers, errors="ignore")


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
                                max_categories=50,
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
    requested_folds: int = 3,
) -> ValidationResult:
    """Compare minimal and intelligent preprocessing with identical CV folds."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    prepared = dataframe.drop_duplicates().dropna(subset=[target_column]).copy()
    MAX_VALIDATION_ROWS = 10000

    if len(prepared) > MAX_VALIDATION_ROWS:
        prepared = prepared.sample(
            n=MAX_VALIDATION_ROWS,
            random_state=42,
        )
    if len(prepared) < 20:
        raise ValueError("At least 20 usable rows are required for validation.")

    target = prepared[target_column]
    if not is_classification_target(target):
        raise ValueError(
            "The selected target does not currently look like a classification target."
        )

    features = _safe_features(prepared.drop(columns=[target_column]))
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
            n_estimators=50,
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
        task_type="classification",
    )


def validate_regression(
    dataframe: pd.DataFrame,
    target_column: str,
    requested_folds: int = 3,
) -> ValidationResult:
    """Compare preprocessing approaches for a continuous numeric target."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    prepared = dataframe.drop_duplicates().dropna(subset=[target_column]).copy()
    MAX_VALIDATION_ROWS = 10000

    if len(prepared) > MAX_VALIDATION_ROWS:
        prepared = prepared.sample(
            n=MAX_VALIDATION_ROWS,
            random_state=42,
        )
    if len(prepared) < 20:
        raise ValueError("At least 20 usable rows are required for validation.")
    if infer_task_type(prepared[target_column]) != "regression":
        raise ValueError("The selected target does not look like a regression target.")

    features = _safe_features(prepared.drop(columns=[target_column]))
    if features.shape[1] == 0:
        raise ValueError("No usable feature columns remain for model validation.")

    target = pd.to_numeric(prepared[target_column], errors="coerce")
    valid_rows = target.notna()
    features, target = features.loc[valid_rows], target.loc[valid_rows]
    folds = min(requested_folds, len(target) // 5)
    if folds < 2:
        raise ValueError("Not enough usable rows for regression validation.")

    split_strategy = KFold(n_splits=folds, shuffle=True, random_state=42)
    model_factories = {
        "Linear Regression": LinearRegression,
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=50, random_state=42, n_jobs=1
        ),
    }
    preprocessing_factories = {
        "Minimal baseline": _minimal_preprocessor,
        "Data Doctor": _intelligent_preprocessor,
    }
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
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
                target,
                cv=split_strategy,
                scoring=scoring,
                error_score="raise",
                n_jobs=1,
            )
            rows.append(
                {
                    "Approach": approach,
                    "Model": model_name,
                    "MAE": -scores["test_mae"].mean(),
                    "RMSE": -scores["test_rmse"].mean(),
                    "R²": scores["test_r2"].mean(),
                    "Training time (s)": scores["fit_time"].mean(),
                    "Inference time (s)": scores["score_time"].mean(),
                }
            )

    metrics = pd.DataFrame(rows)
    numeric_columns = ["MAE", "RMSE", "R²", "Training time (s)", "Inference time (s)"]
    metrics[numeric_columns] = metrics[numeric_columns].round(4)
    return ValidationResult(
        metrics=metrics,
        folds=folds,
        rows_used=len(target),
        class_count=0,
        task_type="regression",
    )


def recommend_models(
    dataframe: pd.DataFrame, target_column: str
) -> list[dict[str, str]]:
    """Return concise model recommendations based on task and data shape."""
    task = infer_task_type(dataframe[target_column])
    rows, columns = dataframe.shape
    if task == "classification":
        suggestions = [
            ("Logistic Regression", "Strong interpretable baseline for encoded data."),
            ("Random Forest", "Handles nonlinear relationships and mixed feature effects."),
        ]
        if rows >= 1_000:
            suggestions.append(
                ("Gradient Boosting", "Often strong on medium-to-large tabular datasets.")
            )
    elif task == "regression":
        suggestions = [
            ("Linear Regression", "Fast, interpretable baseline for continuous targets."),
            ("Random Forest Regressor", "Captures nonlinear feature relationships."),
        ]
        if rows >= 1_000:
            suggestions.append(
                ("Gradient Boosting Regressor", "Usually competitive on tabular regression.")
            )
    else:
        return [{"Model": "Manual review", "Why": "Target type is not yet supported."}]

    return [
        {"Model": model, "Why": reason + f" Dataset shape: {rows:,} × {columns}."}
        for model, reason in suggestions
    ]
