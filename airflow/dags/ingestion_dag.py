# from airflow import DAG
# from airflow.operators.python import PythonOperator, ShortCircuitOperator
# from datetime import datetime
# import sys
# sys.path.insert(0, "/home/samiksya/ai_job_market/src")

# from data.ingestion import run_ingestion
# from data.validation import run_validation
# from data.storage import load_to_mariadb
# from pathlib import Path
# import pandas as pd

# STAGING_PATH = Path("/home/samiksya/ai_job_market/data/stagging/ai_jobs_staged.csv")

# def ingest():
#     run_ingestion()

# # def validate():
# #     df = pd.read_csv(STAGING_PATH)
# #     return run_validation(df)   # returns True/False → ShortCircuitOperator stops DAG if False

# # AFTER
# def validate():
#     return run_validation()  

# def load():
#     load_to_mariadb() 

# with DAG(
#     dag_id="ai_jobs_ingestion",
#     start_date=datetime(2025, 1, 1),
#     schedule=None,
#     catchup=False,
# ) as dag:

#     t1 = PythonOperator(task_id="ingest_csv", python_callable=ingest)

#     t2 = ShortCircuitOperator(task_id="validate_data", python_callable=validate)

#     t3 = PythonOperator(task_id="load_to_mariadb", python_callable=load)

#     t1 >> t2 >> t3


#     #try67