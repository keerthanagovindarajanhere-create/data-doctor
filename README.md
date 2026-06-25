# Data Doctor

Data Doctor is a beginner-friendly intelligent dataset preprocessing project.
Version 0.1 uploads a CSV, profiles its columns, explains basic preprocessing
recommendations, and exports a cleaned dataset.

## Run the project on Windows

Open this folder in VS Code, then open **Terminal → New Terminal**.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should now begin with `(.venv)`.

### 3. Install the libraries

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.
Open it in a browser and upload a CSV file.

## What version 0.1 teaches

- Reading CSV data with Pandas
- Working with DataFrames
- Identifying missing and duplicate data
- Distinguishing numeric and categorical columns
- Building explainable preprocessing rules
- Creating a simple Streamlit user interface

