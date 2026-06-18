# from datetime import datetime, timedelta
# import sys

# from airflow import DAG
# from airflow.operators.empty import EmptyOperator
# from airflow.operators.python import PythonOperator


# sys.path.insert(0, "/home/samiksya/ai_job_market/src/data")

# from training_process import main as run_xgboost_training


# default_args = {
#     "owner": "samikshya",
#     "depends_on_past": False,
#     "retries": 1,
#     "retry_delay": timedelta(minutes=5),
# }


# with DAG(
#     dag_id="ai_jobs_model_training",
#     default_args=default_args,
#     start_date=datetime(2026, 1, 1),
#     schedule=None,
#     catchup=False,
#     tags=["mlops", "training", "xgboost"],
# ) as dag:

#     start = EmptyOperator(task_id="start")

#     train_xgboost_model = PythonOperator(
#         task_id="train_xgboost_model",
#         python_callable=run_xgboost_training,
#     )

#     end = EmptyOperator(task_id="end")

#     start >> train_xgboost_model >> end