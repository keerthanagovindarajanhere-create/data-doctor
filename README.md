# Data Doctor

Data Doctor is an explainable dataset preprocessing and ML-readiness platform.
The current prototype uploads a CSV, profiles its columns, calculates a transparent
readiness score, ranks preprocessing recommendations, and exports cleaned data.

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

### 4. Run the automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Streamlit will print a local address, usually `http://localhost:8501`.
Open it in a browser and upload a CSV file.

## Current project structure

```text
app.py                 Streamlit interface
src/profiler.py        Dataset profiling and readiness score
src/recommender.py     Ranked explainable recommendations
src/preprocessor.py    Approved cleaning transformations
tests/test_core.py     Automated tests
```
