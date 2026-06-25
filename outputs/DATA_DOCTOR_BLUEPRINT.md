# Data Doctor — Product and Engineering Blueprint

## 1. Product thesis

Data Doctor is an explainable ML-readiness platform, not an automatic CSV cleaner.

Its core loop is:

1. Diagnose the dataset.
2. Recommend transformations with evidence and confidence.
3. Let the user approve or override each recommendation.
4. Compile approved actions into a reproducible preprocessing DAG.
5. Validate whether the resulting data improves downstream model performance.

The portfolio differentiator is the combination of explainability, reproducibility,
leakage-aware validation, and measurable before/after evidence.

## 2. Product boundaries

### First supported problem types

- Binary classification
- Multiclass classification
- Regression

### Explicit non-goals for the first release

- Time-series forecasting
- Images, audio, or unstructured text pipelines
- Distributed processing for very large datasets
- Fully autonomous model deployment
- Claiming that every detected anomaly is an error

These can be added later without weakening the initial product.

## 3. User workflow

### A. Import

- Upload CSV.
- Preview inferred schema.
- Select a target or accept ranked target suggestions.
- Select or confirm the task type.
- Optionally identify ID, timestamp, group, or protected columns.

### B. Diagnose

The profiling engine produces typed feature profiles and dataset-level findings.
Each finding contains:

- Severity
- Confidence
- Evidence
- Affected columns
- Possible ML impact
- Candidate actions

### C. Review recommendations

Each recommendation shows:

- Proposed action
- Reason it was selected
- Alternatives considered
- Expected benefit
- Trade-offs and risks
- Priority
- Confidence
- Whether target information was used

The user can approve, reject, or modify it.

### D. Build and execute

Approved recommendations compile into a preprocessing DAG and then into a fitted
Scikit-learn-compatible pipeline.

### E. Validate

The system compares a safe minimal baseline with the approved intelligent pipeline
using identical cross-validation splits. It reports predictive quality, robustness,
training cost, inference cost, dimensionality, and readiness changes.

### F. Export

- Processed dataset
- HTML/PDF/JSON report
- Serialized fitted pipeline
- Standalone Python pipeline code
- Configuration manifest
- Validation results

## 4. System architecture

```text
React application
      |
FastAPI API
      |
Job orchestration layer
      |
      +-- Dataset ingestion and schema inference
      +-- Profiling engine
      +-- Issue detectors
      +-- Recommendation engine
      +-- Pipeline DAG compiler
      +-- Validation engine
      +-- Reporting/export engine
      |
PostgreSQL metadata + object storage for dataset artifacts
```

Long-running profiling, SHAP, and validation work should run as background jobs.
For an MVP, FastAPI background tasks are sufficient. A production version should
use a worker queue such as Celery, Dramatiq, or RQ with Redis.

Uploaded data should not be stored directly in PostgreSQL. Store metadata, findings,
recommendations, and audit history in PostgreSQL; store files and generated artifacts
in local object storage for development and S3-compatible storage in production.

## 5. Recommended repository structure

```text
data-doctor/
  apps/
    api/
    web/
  packages/
    core/
      domain/
      profiling/
      detectors/
      recommendations/
      pipelines/
      validation/
      reporting/
    sdk/
  tests/
    unit/
    integration/
    golden_datasets/
  infrastructure/
    docker/
    migrations/
  docs/
```

The `core` package should remain independent from FastAPI and React so it can later
be exposed through a CLI, Python SDK, notebook extension, or hosted API.

## 6. Core domain model

### DatasetProfile

- Row and column counts
- Task type and target
- Feature profiles
- Dataset-level statistics
- Sampling and profiling metadata

### FeatureProfile

- Semantic type and physical type
- Missingness
- Cardinality and uniqueness
- Distribution and skew
- Quantiles and outlier statistics
- Category frequencies
- Target association
- Stability and leakage indicators

### Finding

- Detector ID and version
- Severity and confidence
- Evidence
- Affected features
- Human-readable explanation

### Recommendation

- Strategy ID and version
- Priority score
- Proposed transformation
- Explanation
- Alternatives
- Preconditions
- Expected impact
- User decision

### PipelineNode

- Transformer specification
- Input and output features
- Dependencies
- Fit scope
- Execution status

### ValidationRun

- Split strategy and random seed
- Candidate models
- Per-fold metrics
- Timing and resource measurements
- Statistical comparison

## 7. Profiling and issue detection

| Issue | Detection approach | Important guardrail |
|---|---|---|
| Missing values | Null rates and missingness patterns | Distinguish structural missingness from random missingness |
| Duplicates | Exact row hashes; optional similarity matching | Exclude target only when the selected duplicate definition requires it |
| Wrong types | Parse success ratios and semantic inference | Preserve identifiers with leading zeroes |
| Outliers | IQR/MAD plus optional Isolation Forest | Flag; do not automatically delete |
| Skew | Robust skewness and distribution diagnostics | Transform only when useful for the selected model family |
| Class imbalance | Class ratios and minority support | Use resampling only inside training folds |
| Correlation | Pearson/Spearman, Cramér's V, mutual information | Correlation is not automatically redundancy |
| Multicollinearity | Correlation groups and VIF where valid | VIF is mainly useful for numeric linear-model contexts |
| Constant features | Unique count | Safe to remove |
| Near-constant features | Dominant value ratio and variance | Keep rare but predictive indicators |
| Rare categories | Frequency and fold support | Learn grouping thresholds only from training data |
| Leakage | Name, uniqueness, target association, temporal and train/test checks | Treat as a high-risk warning requiring evidence |
| Similar features | Correlation graph and optional Union-Find grouping | Present representative-selection rationale |

## 8. Recommendation engine

Use a deterministic, versioned rule engine first. It is easier to test, explain,
and trust than an LLM-driven decision maker.

Example rule:

```text
IF feature is numeric
AND missing_rate is between 0 and 0.40
AND absolute_skewness is above 1.0
THEN recommend median imputation
BECAUSE the median is less sensitive to the observed skew and extreme values
CONFIDENCE based on sample size, skew strength, and missingness stability
```

Recommendations enter a max-heap using a priority score:

```text
priority =
  severity_weight
  × affected_row_fraction
  × model_risk
  × confidence
  × reversibility_factor
```

Rules should return structured evidence, not only prose. Explanation text is rendered
from that evidence so UI language and exported reports remain consistent.

## 9. Meaningful DSA integration

- **DAG:** represents transformer dependencies and guarantees valid execution order
  through topological sorting.
- **Rule tree:** selects strategies from feature and dataset characteristics.
- **Hash maps:** provide constant-time access to feature profiles, decisions, and
  transformation history.
- **Priority queue:** ranks high-impact recommendations for review.
- **Graphs:** represent feature dependencies; centrality and connected components
  reveal redundant feature clusters.
- **Union-Find:** efficiently groups highly similar or redundant features.

These structures belong in the actual execution path and should have direct unit tests.

## 10. Model Readiness Score

The score must be transparent and diagnostic, not a mysterious quality badge.

Suggested composition:

| Dimension | Weight |
|---|---:|
| Completeness | 20 |
| Schema and type validity | 10 |
| Target suitability | 15 |
| Leakage risk | 20 |
| Feature usability | 15 |
| Distribution and anomaly risk | 10 |
| Redundancy and dimensional health | 10 |

Each dimension returns:

- Score
- Penalties
- Evidence
- Recommended actions

A leakage warning should cap the overall score until reviewed. Dataset size alone
should not be treated as quality, though insufficient statistical support should
lower confidence in the score.

## 11. Validation design

Comparing a raw dirty dataset directly against a processed dataset is often impossible
because many models cannot consume missing values or strings. Use:

- **Minimal baseline:** only transformations required to make the model run.
- **Data Doctor pipeline:** the approved intelligent preprocessing plan.

Both pipelines must use:

- Identical folds
- The same model hyperparameters
- Fold-local fitting of all imputers, encoders, scalers, selectors, and samplers
- Stratified CV for classification where valid
- Group-aware or time-aware splitting when applicable

Metrics:

- Classification: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC,
  PR-AUC, log loss, training time, and inference latency.
- Regression: MAE, RMSE, R², MAPE when valid, training time, and inference latency.

Report fold distributions and confidence intervals, not only average scores. Avoid
claiming an improvement when differences are within normal cross-validation variance.

Initial model set:

- Linear/logistic model
- Random forest
- Gradient-boosted trees

XGBoost and LightGBM should be optional integrations so the core application still
runs without them.

## 12. Feature importance and selection

Use a tiered approach:

1. Fast filter statistics for immediate feedback.
2. Permutation importance on held-out folds for model-aware importance.
3. SHAP for supported tree models as an advanced explanation.

Selection should occur inside cross-validation. Never select features on the full
dataset before evaluating the model.

## 13. Leakage detection

Leakage requires multiple detectors:

- Suspicious column names such as `outcome`, `status_after`, or `approved`.
- Near-perfect association with the target.
- Features derived from or identical to the target.
- Unique identifiers acting as memorization keys.
- Timestamp ordering that reveals future information.
- Duplicated entities spread across train and validation folds.
- Precomputed aggregates that include validation rows.

The UI should label these as suspected leakage with evidence. Automatic deletion is
too risky.

## 14. API surface for the MVP

```text
POST   /datasets
GET    /datasets/{id}/preview
POST   /datasets/{id}/target-suggestions
POST   /datasets/{id}/profile
GET    /jobs/{id}
GET    /profiles/{id}
GET    /profiles/{id}/recommendations
PATCH  /recommendations/{id}
POST   /profiles/{id}/pipelines
GET    /pipelines/{id}/dag
POST   /pipelines/{id}/execute
POST   /pipelines/{id}/validate
GET    /validation-runs/{id}
GET    /runs/{id}/artifacts
```

## 15. UI information architecture

- **Dataset Intake:** upload, preview, target selection, task confirmation.
- **Health Overview:** readiness score, severe findings, feature summary.
- **Feature Lab:** searchable column profiles and distributions.
- **Recommendations:** ranked approval queue with explanations and alternatives.
- **Pipeline Studio:** editable DAG and execution status.
- **Validation:** before/after metrics, fold distributions, timing, and model ranking.
- **Export Center:** datasets, reports, manifests, code, and fitted pipeline.

The recommendation review should be the centerpiece. A polished approval experience
will distinguish the product more than adding a large number of charts.

## 16. Delivery roadmap

### Phase 1 — Vertical MVP

- CSV upload and safe parsing
- Schema inference
- Classification/regression target selection
- Core profiling
- Five issue detectors
- Rule-based recommendations
- Approval workflow
- Scikit-learn pipeline generation
- Cleaned CSV and JSON/HTML report export
- Minimal baseline versus processed validation

### Phase 2 — Intelligence

- Readiness score
- Leakage detectors
- Feature selection
- Feature importance
- Correlation graph
- Target suggestions
- Pipeline code export

### Phase 3 — Production quality

- Authentication and projects
- Background workers
- PostgreSQL and object storage
- Dataset retention controls
- Resource limits and cancellation
- Audit history and rule versioning
- Docker deployment and CI/CD
- Security and privacy hardening

### Phase 4 — Distinctive extensions

- Time-aware datasets
- Data drift comparison
- Custom organizational rules
- Notebook/CLI SDK
- Dataset contracts
- Human feedback learning for recommendation ranking

## 17. First implementation milestone

Build one complete, demoable path before broadening detector coverage:

1. Upload the Titanic or Adult Income dataset.
2. Select the target.
3. Profile numeric and categorical features.
4. Detect missing values, wrong types, constant columns, rare categories, and skew.
5. Produce evidence-backed recommendations.
6. Approve recommendations.
7. Compile and execute a `ColumnTransformer` pipeline.
8. Compare minimal and intelligent pipelines with cross-validation.
9. Export the report, processed CSV, and Python code.

This milestone demonstrates the product thesis end to end and is stronger than a
wide dashboard filled with unfinished analyses.

## 18. Resume-ready framing

> Built an explainable ML-readiness platform that profiles tabular datasets,
> prioritizes evidence-backed preprocessing recommendations, compiles approved
> transformations into reproducible Scikit-learn DAGs, detects leakage risks, and
> quantifies downstream impact through leakage-safe cross-validation.

Strong portfolio evidence should include:

- A hosted demo
- A 90-second product video
- Architecture and pipeline diagrams
- Golden-dataset test cases
- Before/after validation reports
- Clear privacy and leakage safeguards
- A public roadmap and documented design decisions

