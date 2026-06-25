from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PipelineResult:
    """Artifacts created after fitting the preprocessing pipeline."""

    transformed_data: pd.DataFrame
    pipeline: ColumnTransformer
    numeric_columns: list[str]
    categorical_columns: list[str]
    removed_columns: list[str]
    dropped_target_rows: int


def _split_numeric_strategies(
    dataframe: pd.DataFrame, numeric_columns: list[str]
) -> tuple[list[str], list[str]]:
    """Use median for skewed columns and mean for other numeric columns."""
    median_columns: list[str] = []
    mean_columns: list[str] = []

    for column in numeric_columns:
        skewness = dataframe[column].skew()
        if pd.notna(skewness) and abs(skewness) > 1:
            median_columns.append(column)
        else:
            mean_columns.append(column)

    return median_columns, mean_columns


def build_preprocessing_pipeline(
    features: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    """Create an unfitted Scikit-learn transformer from feature characteristics."""
    constant_columns = [
        column for column in features.columns if features[column].nunique() <= 1
    ]
    usable_features = features.drop(columns=constant_columns)

    numeric_columns = usable_features.select_dtypes(include="number").columns.tolist()
    categorical_columns = usable_features.select_dtypes(exclude="number").columns.tolist()
    median_columns, mean_columns = _split_numeric_strategies(
        usable_features, numeric_columns
    )

    transformers = []

    if median_columns:
        transformers.append(
            (
                "skewed_numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                median_columns,
            )
        )

    if mean_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                mean_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns remain after removing constants.")

    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pipeline, numeric_columns, categorical_columns, constant_columns


def create_ml_ready_dataset(
    dataframe: pd.DataFrame, target_column: str
) -> PipelineResult:
    """Fit the transformer and return encoded features with the target preserved."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    prepared = dataframe.drop_duplicates().copy()
    dropped_target_rows = int(prepared[target_column].isna().sum())
    prepared = prepared.dropna(subset=[target_column])

    target = prepared[target_column].reset_index(drop=True)
    features = prepared.drop(columns=[target_column])

    pipeline, numeric_columns, categorical_columns, removed_columns = (
        build_preprocessing_pipeline(features)
    )
    transformed = pipeline.fit_transform(features)
    feature_names = pipeline.get_feature_names_out()

    transformed_data = pd.DataFrame(transformed, columns=feature_names)
    transformed_data[target_column] = target

    return PipelineResult(
        transformed_data=transformed_data,
        pipeline=pipeline,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        removed_columns=removed_columns,
        dropped_target_rows=dropped_target_rows,
    )


def generate_pipeline_code(target_column: str) -> str:
    """Export readable starter code that rebuilds the preprocessing approach."""
    return f'''import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("your_dataset.csv").drop_duplicates()
target_column = {target_column!r}
df = df.dropna(subset=[target_column])

X = df.drop(columns=[target_column])
y = df[target_column]

constant_columns = [column for column in X.columns if X[column].nunique() <= 1]
X = X.drop(columns=constant_columns)

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

X_ready = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
ml_ready_df = pd.DataFrame(X_ready, columns=feature_names)
ml_ready_df[target_column] = y.reset_index(drop=True)

ml_ready_df.to_csv("ml_ready_dataset.csv", index=False)
'''

