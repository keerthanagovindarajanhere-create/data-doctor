# 🩺 Data Doctor

**Data Doctor** is an intelligent dataset diagnosis and preprocessing platform that helps data scientists, students, and machine learning practitioners analyze dataset quality, identify issues, prepare data for modeling, and validate preprocessing pipelines—all through an interactive Streamlit interface.

Instead of only cleaning datasets, Data Doctor explains **what is wrong, why it matters, and how it affects machine learning performance**, making preprocessing more transparent and explainable.

---

## ✨ Features

### 📊 Dataset Profiling

* Comprehensive dataset overview
* Missing value analysis
* Duplicate detection
* Column profiling
* ML readiness score

### 🔍 Intelligent Diagnostics

* Outlier detection (IQR-based)
* Rare category detection
* High correlation analysis
* Near-constant feature detection
* Class imbalance detection

### 💡 Explainable Recommendations

* Priority-ranked preprocessing suggestions
* Severity classification
* Evidence-backed reasoning
* Human-readable explanations

### 🧹 Automated Preprocessing

* Missing value imputation
* Duplicate removal
* Constant feature removal
* ML-ready feature transformation
* Safe handling of high-cardinality categorical features

### 🤖 Machine Learning Validation

* Automatic task inference (Classification/Regression)
* Leakage-safe preprocessing pipelines
* Cross-validation benchmarking
* Baseline vs Data Doctor comparison
* Performance metrics including Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE and R²

### 📦 Export Options

* Cleaned dataset
* ML-ready dataset
* Reusable Scikit-learn preprocessing pipeline
* HTML analysis report
* JSON report

---

## 🛠 Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* HTML/CSS
* Git & GitHub

---

## 🏗 Architecture

Data Doctor is designed as a modular application.

```
app.py
│
├── profiler.py
├── detectors.py
├── recommender.py
├── preprocessor.py
├── ml_pipeline.py
├── validator.py
├── reporting.py
└── ...
```

Each module is responsible for a single stage of the data preparation workflow, making the project scalable and easy to maintain.

---

## 🚀 Workflow

1. Upload a CSV dataset
2. Profile dataset quality
3. Detect statistical issues
4. Generate preprocessing recommendations
5. Create a cleaned dataset
6. Build an ML-ready dataset
7. Validate preprocessing using machine learning models
8. Export reports and reusable artifacts

---

## 🎯 Why Data Doctor?

Most dataset profiling tools only describe the data.

**Data Doctor goes a step further** by combining profiling, explainable diagnostics, preprocessing recommendations, ML pipeline generation, and validation into a single end-to-end workflow designed for real machine learning projects.

---

## 📸 Screenshots

> Add screenshots of:

* Overview Dashboard
* Diagnostics
* Preprocessing
* Validation Results
* Export Page

---

## 🔮 Future Improvements

* Dataset Quality Certification
* Feature Importance Analysis
* Enhanced Model Recommendation Engine
* Professional PDF Reports
* Interactive Visual Analytics

---

## 👩‍💻 Author

**Keerthana G**

If you found this project useful, consider giving it a ⭐ on GitHub!
