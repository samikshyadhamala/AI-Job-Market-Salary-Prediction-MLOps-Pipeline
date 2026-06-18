# AI Job Market Salary Prediction MLOps Pipeline

## 🚀 Project Overview

This project implements an end-to-end **MLOps pipeline** to predict salaries for approximately **15,000 AI job roles** using the **"Global AI Job Market and Salary Trends 2025"** dataset from Kaggle.

The system automates the complete machine learning lifecycle, from **data ingestion and validation to model deployment and real-time monitoring**, ensuring reproducibility, scalability, and production readiness.

The project performs a supervised regression task to predict:

**Target Variable:** `salary_usd`

based on features such as:

- Job title
- Experience level
- Location
- Required skills
- Employment details

The final **XGBoost regression model** achieved:

- **Test R² Score:** 0.8831
- **RMSE:** $20,652


---

 Data source link : https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025 

# 🏗️ Pipeline Architecture

The MLOps workflow is divided into five operational stages:

```
Data Source (CSV)
        |
        v
Data Ingestion
        |
        v
Data Validation (Great Expectations)
        |
        v
MariaDB ColumnStore
        |
        v
Data Preprocessing
        |
        v
Feature Engineering + Transformation
        |
        v
XGBoost Model Training
        |
        v
MLflow Tracking & Model Registry
        |
        v
FastAPI Deployment
        |
        v
Redis Caching
        |
        v
Monitoring (Evidently + Grafana)
```

---

# 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Workflow Orchestration | Apache Airflow |
| Experiment Tracking | MLflow |
| Model Registry | MLflow |
| Data Storage | MariaDB ColumnStore, CSV, Parquet |
| Data Validation | Great Expectations |
| Machine Learning | XGBoost, Scikit-learn |
| API Deployment | FastAPI + Uvicorn |
| Request Validation | Pydantic |
| Caching | Redis |
| Monitoring | Evidently AI |
| Visualization | Grafana |
| Data Profiling | ydata-profiling |
| UI | Streamlit |
| Environment | Conda + Docker |

---

# 📌 Pipeline Stages

## 1. Data Ingestion

The ingestion pipeline:

- Loads raw Kaggle CSV data
- Standardizes column names and schemas
- Performs automated data quality validation
- Runs **19 Great Expectations checks**
- Stores validated data in MariaDB ColumnStore

---

## 2. Data Preprocessing

The preprocessing stage:

- Handles missing values
- Detects and manages outliers
- Performs feature engineering
- Creates **21 model features**
- Applies:
  - Scaling
  - Encoding
  - Transformation

A reusable Scikit-learn preprocessing pipeline is generated for training and inference.

Output:

```
train.parquet
test.parquet
preprocessor.pkl
```

---

## 3. Model Development

The modeling pipeline:

- Loads processed Parquet datasets
- Performs train/test evaluation
- Trains an XGBoost Regressor
- Uses 5-fold cross-validation
- Applies Optuna tuning when required
- Logs all experiments to MLflow

Tracked metrics:

- RMSE
- R² Score
- Training parameters
- Model artifacts

Final Model Performance:

```
R² Score : 0.8831
RMSE     : $20,652
```

---

## 4. Model Deployment

The trained model is deployed using **FastAPI**.

Deployment workflow:

1. Client sends prediction request
2. FastAPI validates input using Pydantic
3. Checks Redis cache
4. If cached → returns stored prediction
5. If not cached:
   - Loads preprocessing pipeline
   - Transforms input
   - Runs XGBoost prediction
   - Stores result in Redis

Cache duration:

```
TTL = 1 hour
```

---

## 5. Model Monitoring

The monitoring pipeline uses **Evidently AI**.

It monitors:

### Data Quality

Checks:

- Missing values
- Feature consistency
- Data changes

### Data Drift

Detects changes in:

- Job locations
- Skills
- Experience levels
- Salary distributions

### Prediction Drift

Tracks changes in model outputs over time.

Monitoring metrics are visualized using:

- Grafana Dashboard

Current system health:

```
Data Quality: 98.5%
```

---

# 📂 Project Structure

```
.
├── airflow
│   ├── dags
│   │   ├── __pycache__/
│   │   │   ├── deployment_dag.cpython-310.pyc
│   │   │   ├── full_dag.cpython-310.pyc
│   │   │   ├── full_dag.cpython-312.pyc
│   │   │   ├── full_dag.cpython-313.pyc
│   │   │   ├── ingestion_dag.cpython-310.pyc
│   │   │   ├── preprocessing_dag.cpython-310.pyc
│   │   │   ├── training_dag.cpython-310.pyc
│   │   │   └── training_dag.cpython-312.pyc
│   │   ├── deployment_dag.py
│   │   ├── full_dag.py
│   │   ├── ingestion_dag.py
│   │   ├── preprocessing_dag.py
│   │   └── training_dag.py
│   └── mlflow.db
├── data
│   ├── monitoring_alerts/
│   ├── monitoring_reports/
│   │   ├── data_quality.html
│   │   ├── feature_drift.html
│   │   └── prediction_drift.html
│   ├── processed/
│   │   ├── model_artifacts/
│   │   │   ├── feature_importance.csv
│   │   │   ├── model_metrics.json
│   │   │   ├── model_parameters.json
│   │   │   ├── predictions_vs_actual.png
│   │   │   └── xgboost_final_model.pkl
│   │   ├── preprocessing_metadata.json
│   │   ├── preprocessor.pkl
│   │   ├── profiling_report.html
│   │   ├── test.parquet
│   │   ├── train.parquet
│   │   └── user_inputs.parquet
│   ├── raw/
│   │   └── ai_jobs.csv
│   └── stagging/
│       └── ai_jobs_staged.csv
├── grafana
│   └── provisioning
│       ├── dashboards/
│       │   ├── dashboards.yml
│       │   └── drift_monitoring_dashboard.json
│       ├── datasources/
│       │   └── mariadb.yml
│       └── init.sql
├── logs/
│   └── app.log
├── mlartifacts/
├── mlflow_store/
│   └── mlflow.db
├── mlruns/
├── pictures_proof/
│   ├── figure_48.html ... figure_59.html
│   └── Screenshot from 2026-06-18 :png files
├── scripts/
│   └── import_grafana_dashboard.py
├── src/
│   ├── data/
│   │   ├── data/
│   │   │   └── processed/
│   │   ├── logs/
│   │   ├── mlartifacts/
│   │   ├── mlflow/
│   │   ├── app.py
│   │   ├── fastapi_deployment.py
│   │   ├── ingestion.py
│   │   ├── __init__.py, _init_.py
│   │   ├── mlflow.db
│   │   ├── monitoring.py
│   │   ├── preprocessing.py
│   │   └── training_process.py
│   └── __pycache__/:config, logger pyc files
├── databseinto.ipynb
├── eda.ipynb
├── .env
├── mlflow.db
├── README.md
├── requirements.txt
├── SamikshyaDhamala25123833mlops.pdf
└── start.sh
```

---

# ⚙️ Setup Instructions

## 1. Create Conda Environment

```bash
conda create -n ai_salary_mlops python=3.11

conda activate ai_salary_mlops
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Start Docker Services

Start required services:

```bash
docker compose up -d
```

Services started:

- MariaDB ColumnStore
- Redis
- Grafana

---

## 3. Run Airflow

Start Airflow scheduler:

```bash
airflow scheduler
```

Start Airflow webserver:

```bash
airflow webserver
```

---

## 4. Start MLflow Server

```bash
mlflow serveror mlflow ui
```

Access:

```
http://localhost:5000
```

---

## 5. Run FastAPI Deployment

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

API documentation:

```
http://localhost:8000/docs
```

---

## 6. Run Streamlit UI

```bash
streamlit run app.py
```

---

# 📊 Monitoring Dashboard

Grafana provides:

- API health monitoring
- Prediction statistics
- Data quality metrics
- Drift detection results

---

# 🔮 Future Improvements

Planned improvements:

- Add SHAP-based model explainability
- Implement automatic retraining triggers
- Add drift-based retraining pipeline
- Improve feature store integration
- Expand dataset coverage
- Add CI/CD automation

---

# 🎯 Key Highlights

✅ End-to-end production ML pipeline  
✅ Automated data validation  
✅ Experiment tracking with MLflow  
✅ Real-time API deployment  
✅ Redis prediction caching  
✅ Model drift monitoring  
✅ Containerized infrastructure  
✅ Reproducible ML workflow  
