# AI Jobs ML Pipeline — Airflow DAG
# Stages: Ingestion → Preprocessing → Training → Deployment (health check)

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

BASE_DIR = "/home/samiksya/ai_job_market"
SRC_DIR  = f"{BASE_DIR}/src"

default_args = {
    "owner": "samiksya",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ai_jobs_ml_pipeline",
    description="Ingest → Preprocess → Train → Deploy",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["ai_jobs", "ml"],
) as dag:

    # ── Stage 1: Unified Ingestion (NOW includes DB load) ───────────────
    ingest = BashOperator(
        task_id="stage_1_ingestion",
        bash_command=f"cd {BASE_DIR} && python {SRC_DIR}/data/ingestion.py",
    )

    # ── Stage 2: Preprocessing ───────────────────────────────────────────
    preprocess = BashOperator(
        task_id="stage_2_preprocessing",
        bash_command=f"cd {BASE_DIR} && python {SRC_DIR}/data/preprocessing.py",
    )

    # ── Stage 3: Model Training ──────────────────────────────────────────
    train = BashOperator(
        task_id="stage_3_training",
        bash_command=f"cd {BASE_DIR} && python {SRC_DIR}/data/training_process.py",
    )

    def check_api_health():
        import requests

        base_url = "http://127.0.0.1:8000"
        streamlit_url = "http://127.0.0.1:8501"

        resp = requests.get(f"{base_url}/health", timeout=10)
        resp.raise_for_status()

        print("===================================")
        print("🚀 System Status Check")

        print("\n🔗 FastAPI Links:")
        print(f"- Base URL: {base_url}")
        print(f"- Health: {base_url}/health")
        print(f"- Docs: {base_url}/docs")

        print("\n📊 Streamlit Dashboard:")
        print(f"- UI: {streamlit_url}")

        print("\nAPI Response:", resp.json())
        print("===================================")


    health_check = PythonOperator(
        task_id="stage_4_api_health_check",
        python_callable=check_api_health,
    )

    # ── Pipeline order ───────────────────────────────────────────────────
    ingest >> preprocess >> train >> health_check   
    