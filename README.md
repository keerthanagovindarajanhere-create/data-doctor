# Data Doctor

**An explainable dataset preprocessing and ML-readiness platform.**

Data Doctor diagnoses tabular data, ranks evidence-backed preprocessing
recommendations, produces human-readable and ML-ready exports, and verifies the
impact through leakage-safe model validation.

## Highlights

- Missingness, duplicates, outliers, skew, rare categories, near-constant features,
  class imbalance, and high-correlation detection
- Transparent 0–100 readiness score
- Ranked recommendations with reasons
- Memory-safe categorical encoding and identifier detection
- Classification and regression validation
- Logistic/linear and random-forest baselines
- Accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, R², and timing metrics
- Clean CSV, ML-ready CSV, reusable Scikit-learn code, HTML report, and JSON report
- Docker and GitHub Actions support

## How it works

```text
Upload CSV
  → Profile and diagnose
  → Review recommendations
  → Export cleaned or ML-ready data
  → Validate baseline vs Data Doctor
  → Download reproducible report and pipeline
```

All preprocessing used during validation is fitted inside each training fold to
avoid data leakage.

## Run the project on Windows

Open this folder in VS Code, then open **Terminal → New Terminal**.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Install the libraries

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This direct command also works when PowerShell blocks environment activation.

### 3. Start the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 4. Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Streamlit will print a local address, usually `http://localhost:8501`.
Open it in a browser and upload a CSV file.

## Project structure

```text
app.py                 Streamlit interface
src/profiler.py        Dataset profiling and readiness score
src/detectors.py       Outlier, rarity, correlation, and imbalance detection
src/recommender.py     Ranked explainable recommendations
src/preprocessor.py    Approved cleaning transformations
src/ml_pipeline.py     Reusable Scikit-learn preprocessing pipeline
src/validator.py       Baseline-versus-processed ML validation
src/reporting.py       Portable HTML and JSON reports
tests/test_core.py     Automated tests
.github/workflows/     Continuous integration
Dockerfile             Reproducible deployment container
```

## Docker

```powershell
docker build -t data-doctor .
docker run -p 8501:8501 data-doctor
```

## Deployment

The simplest hosted path is Streamlit Community Cloud:

1. Sign in with GitHub at `share.streamlit.io`.
2. Select this repository and the `main` branch.
3. Set the main file to `app.py`.
4. Deploy and add the generated URL to the GitHub repository description.

Render or Railway can deploy the included Dockerfile when more control is needed.

## Current scope

The production-ready portfolio release supports CSV-based tabular classification
and regression. Time-series, distributed processing, authentication, and persistent
project history are planned extensions rather than hidden partial features.
