# """
# DAG: ai_jobs_deployment
# Stage 4 — Health → Model Info → Prediction → Cache → DB Logging → Drift → Clear Cache

# Changes from previous DAG:
#   - Payload updated to match JobFeaturesInput (added required fields:
#     job_title, employee_residence, salary_currency, required_skills, days_open;
#     removed job_description_length which is not in the schema)
#   - verify_drift_data task implemented (was empty/missing)
#   - default_args added (owner, retries, retry_delay, execution_timeout)
#   - on_task_failure callback added (mirrors preprocessing DAG pattern)
#   - All task callables moved outside the DAG context manager (Airflow best practice)
#   - XCom used to pass predicted salary between test_prediction → test_cache
#     so both tasks share the same payload deterministically
#   - HTTP errors now raise explicitly instead of relying only on assert
#   - t5 (verify_db_logging) and t6 (verify_drift_data) run in parallel after t4
#   - t7 (clear_cache) runs after both t5 and t6 succeed
# """

# from __future__ import annotations

# import json
# import logging
# from datetime import datetime, timedelta

# import requests
# from airflow import DAG
# from airflow.exceptions import AirflowException
# from airflow.operators.python import PythonOperator
# from airflow.utils.trigger_rule import TriggerRule

# log = logging.getLogger(__name__)

# API_BASE = "http://127.0.0.1:8000"

# # Canonical payload — matches JobFeaturesInput exactly
# # (all required fields present, no extra fields)
# TEST_PAYLOAD = {
#     "years_experience": 5,
#     "remote_ratio": 100,
#     "benefits_score": 8.5,
#     "experience_level": "MI",
#     "employment_type": "FT",
#     "job_title": "Machine Learning Engineer",
#     "company_location": "United States",
#     "employee_residence": "United States",
#     "company_size": "M",
#     "education_required": "Master",
#     "industry": "Technology",
#     "salary_currency": "USD",
#     "required_skills": ["Python", "Machine Learning", "SQL"],
#     "days_open": 30,
# }

# # ─────────────────────────────────────────────
# # DEFAULT ARGS
# # ─────────────────────────────────────────────

# default_args = {
#     "owner":             "samiksya",
#     "retries":           1,
#     "retry_delay":       timedelta(minutes=2),
#     "execution_timeout": timedelta(minutes=10),
# }

# # ─────────────────────────────────────────────
# # FAILURE CALLBACK
# # ─────────────────────────────────────────────

# def on_task_failure(context):
#     """
#     Called automatically on any task failure.
#     Extend to send Slack / email / PagerDuty alerts.
#     """
#     ti      = context["task_instance"]
#     dag_id  = context["dag"].dag_id
#     task_id = ti.task_id
#     exc     = context.get("exception", "unknown error")
#     log_url = ti.log_url

#     log.error(
#         "\n"
#         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
#         "  DAG TASK FAILED\n"
#         "  DAG   : %s\n"
#         "  Task  : %s\n"
#         "  Error : %s\n"
#         "  Logs  : %s\n"
#         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
#         dag_id, task_id, exc, log_url,
#     )
#     # ── Extend here ───────────────────────────
#     # send_slack_alert(dag_id, task_id, exc)
#     # send_email_alert(dag_id, task_id, exc)


# # ─────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────

# def _get(path: str, timeout: int = 10) -> dict:
#     """GET helper — raises AirflowException on non-2xx."""
#     url = f"{API_BASE}{path}"
#     try:
#         resp = requests.get(url, timeout=timeout)
#     except requests.exceptions.ConnectionError as exc:
#         raise AirflowException(
#             f"Cannot reach API at {url} — is the FastAPI server running? ({exc})"
#         )
#     if not resp.ok:
#         raise AirflowException(
#             f"GET {url} returned HTTP {resp.status_code}: {resp.text[:300]}"
#         )
#     return resp.json()


# def _post(path: str, payload: dict, timeout: int = 20) -> dict:
#     """POST helper — raises AirflowException on non-2xx."""
#     url = f"{API_BASE}{path}"
#     try:
#         resp = requests.post(url, json=payload, timeout=timeout)
#     except requests.exceptions.ConnectionError as exc:
#         raise AirflowException(
#             f"Cannot reach API at {url} — is the FastAPI server running? ({exc})"
#         )
#     if not resp.ok:
#         raise AirflowException(
#             f"POST {url} returned HTTP {resp.status_code}: {resp.text[:300]}"
#         )
#     return resp.json()


# # ─────────────────────────────────────────────
# # TASK CALLABLES
# # ─────────────────────────────────────────────

# def task_health_check(**ctx):
#     """
#     STEP 1 — GET /health
#     Asserts: status=healthy, model_loaded=True, preprocessor_loaded=True.
#     Also logs redis_connected and drift_parquet_exists for observability.
#     """
#     data = _get("/health")
#     log.info("Health response: %s", json.dumps(data, indent=2))

#     if data.get("status") != "healthy":
#         raise AirflowException(
#             f"API health check failed — status={data.get('status')}"
#         )
#     if not data.get("model_loaded"):
#         raise AirflowException("API reports model_loaded=False")
#     if not data.get("preprocessor_loaded"):
#         raise AirflowException("API reports preprocessor_loaded=False")

#     log.info(
#         "Health OK | redis_connected=%s | drift_parquet_exists=%s",
#         data.get("redis_connected"),
#         data.get("drift_parquet_exists"),
#     )

#     # Push redis status for downstream observability
#     ctx["ti"].xcom_push(key="redis_connected", value=data.get("redis_connected"))


# def task_model_info_check(**ctx):
#     """
#     STEP 2 — GET /model/info
#     Asserts preprocessor is loaded and logs raw + transformed feature lists.
#     """
#     data = _get("/model/info")
#     log.info("Model info: %s", json.dumps(data, indent=2))

#     if not data.get("preprocessor_loaded"):
#         raise AirflowException("model/info reports preprocessor_loaded=False")

#     ctx["ti"].xcom_push(key="model_version",        value=data.get("model_version"))
#     ctx["ti"].xcom_push(key="raw_feature_count",    value=len(data.get("raw_model_features", [])))
#     ctx["ti"].xcom_push(key="transformed_feat_count", value=len(data.get("transformed_features", [])))

#     log.info(
#         "Model version=%s | raw_features=%d | transformed_features=%d",
#         data.get("model_version"),
#         len(data.get("raw_model_features", [])),
#         len(data.get("transformed_features", [])),
#     )


# def task_test_prediction(**ctx):
#     """
#     STEP 3 — POST /predict (cache MISS path)
#     Uses TEST_PAYLOAD which matches JobFeaturesInput exactly.
#     Asserts: HTTP 200, predicted_salary_usd > 0, from_cache=False.
#     Pushes predicted_salary to XCom for the cache test.
#     """
#     data = _post("/predict", TEST_PAYLOAD)
#     log.info("Prediction response: %s", json.dumps(data, indent=2))

#     salary = data.get("predicted_salary_usd", 0)
#     if salary <= 0:
#         raise AirflowException(
#             f"Prediction returned non-positive salary: {salary}"
#         )
#     if data.get("from_cache") is True:
#         log.warning(
#             "Expected a cache MISS on first call but got from_cache=True. "
#             "Redis may already hold this key from a prior run — this is not a failure."
#         )

#     ctx["ti"].xcom_push(key="predicted_salary", value=salary)
#     ctx["ti"].xcom_push(key="processing_time_ms", value=data.get("processing_time_ms"))
#     log.info("Predicted salary: $%.2f in %.1f ms", salary, data.get("processing_time_ms", 0))


# def task_test_cache(**ctx):
#     """
#     STEP 4 — POST /predict again with identical payload (cache HIT path)
#     Asserts: from_cache=True.
#     Compares salary to the value pushed by task_test_prediction.
#     """
#     data = _post("/predict", TEST_PAYLOAD)
#     log.info("Cache-hit response: %s", json.dumps(data, indent=2))

#     if not data.get("from_cache"):
#         raise AirflowException(
#             "Expected from_cache=True on second identical request — "
#             "Redis caching may not be working."
#         )

#     prior_salary = ctx["ti"].xcom_pull(
#         task_ids="test_prediction",
#         key="predicted_salary"
#     )
#     if prior_salary and abs(data["predicted_salary_usd"] - prior_salary) > 0.01:
#         raise AirflowException(
#             f"Cache returned different salary than original prediction: "
#             f"cached={data['predicted_salary_usd']} original={prior_salary}"
#         )

#     log.info(
#         "Cache HIT confirmed | salary=$%.2f | processing_time=%.1f ms",
#         data["predicted_salary_usd"],
#         data.get("processing_time_ms", 0),
#     )


# def task_verify_db_logging(**ctx):
#     """
#     STEP 5 — GET /predictions/history?limit=5
#     Asserts at least one prediction record exists in MariaDB.
#     Logs the most recent prediction's timestamp and salary.
#     """
#     data = _get("/predictions/history?limit=5")
#     log.info("DB history (%d rows returned): %s", len(data), json.dumps(data, indent=2))

#     if len(data) == 0:
#         raise AirflowException(
#             "No prediction records found in MariaDB — "
#             "PredictionLogger.log_prediction may have failed silently."
#         )

#     latest = data[0]
#     log.info(
#         "Latest DB record | salary=%s | timestamp=%s | from_cache=%s",
#         latest.get("predicted_salary"),
#         latest.get("prediction_timestamp"),
#         latest.get("from_cache"),
#     )
#     ctx["ti"].xcom_push(key="db_record_count", value=len(data))


# def task_verify_drift_data(**ctx):
#     """
#     STEP 6 — GET /drift/status
#     Asserts drift parquet exists and has at least one record.
#     Drift parquet is written by save_input_for_drift() as a background task
#     so we allow a brief poll with retries (handled by Airflow's task-level retry).
#     """
#     data = _get("/drift/status")
#     log.info("Drift status: %s", json.dumps(data, indent=2))

#     if not data.get("exists"):
#         raise AirflowException(
#             "Drift parquet does not exist yet — "
#             "save_input_for_drift background task may not have completed. "
#             "Task will retry automatically."
#         )

#     records = data.get("records", 0)
#     if records == 0:
#         raise AirflowException(
#             "Drift parquet exists but contains 0 records — "
#             "background write may have failed silently."
#         )

#     log.info(
#         "Drift OK | records=%d | columns=%d | latest=%s",
#         records,
#         data.get("columns", 0),
#         data.get("latest"),
#     )
#     ctx["ti"].xcom_push(key="drift_record_count", value=records)


# def task_clear_cache(**ctx):
#     """
#     STEP 7 — POST /cache/clear
#     Clears all prediction_* keys from Redis.
#     Asserts status=cleared.
#     """
#     data = _post("/cache/clear", {})
#     log.info("Cache clear response: %s", json.dumps(data, indent=2))

#     if data.get("status") != "cleared":
#         raise AirflowException(
#             f"Cache clear returned unexpected status: {data.get('status')}"
#         )

#     log.info("Cache cleared at %s", data.get("timestamp"))


# # ─────────────────────────────────────────────
# # DAG DEFINITION
# # ─────────────────────────────────────────────

# with DAG(
#     dag_id="ai_jobs_deployment",
#     description=(
#         "Stage 4: health → model info → predict → cache → "
#         "db logging ↕ drift data → clear cache"
#     ),
#     start_date=datetime(2025, 1, 1),
#     schedule_interval=None,
#     catchup=False,
#     default_args=default_args,
#     on_failure_callback=on_task_failure,
#     tags=["stage4", "deployment", "ai_jobs"],
# ) as dag:

#     t1_health = PythonOperator(
#         task_id="health_check",
#         python_callable=task_health_check,
#     )

#     t2_model_info = PythonOperator(
#         task_id="model_info_check",
#         python_callable=task_model_info_check,
#     )

#     t3_predict = PythonOperator(
#         task_id="test_prediction",
#         python_callable=task_test_prediction,
#     )

#     t4_cache = PythonOperator(
#         task_id="test_cache",
#         python_callable=task_test_cache,
#     )

#     # t5 and t6 run in parallel — both must pass before cache is cleared
#     t5_db = PythonOperator(
#         task_id="verify_db_logging",
#         python_callable=task_verify_db_logging,
#         trigger_rule=TriggerRule.ALL_SUCCESS,
#         # Extra retry: background DB write may still be in flight
#         retries=2,
#         retry_delay=timedelta(seconds=15),
#         on_failure_callback=on_task_failure,
#     )

#     t6_drift = PythonOperator(
#         task_id="verify_drift_data",
#         python_callable=task_verify_drift_data,
#         trigger_rule=TriggerRule.ALL_SUCCESS,
#         # Extra retry: background parquet write may still be in flight
#         retries=2,
#         retry_delay=timedelta(seconds=15),
#         on_failure_callback=on_task_failure,
#     )

#     t7_clear = PythonOperator(
#         task_id="clear_cache",
#         python_callable=task_clear_cache,
#         # Only clear after BOTH db and drift checks pass
#         trigger_rule=TriggerRule.ALL_SUCCESS,
#     )

#     # ── DAG flow ──────────────────────────────
#     #
#     #  t1 → t2 → t3 → t4 → t5 ──┐
#     #                        └→ t6 ──┴→ t7
#     #
#     t1_health >> t2_model_info >> t3_predict >> t4_cache >> [t5_db, t6_drift] >> t7_clear