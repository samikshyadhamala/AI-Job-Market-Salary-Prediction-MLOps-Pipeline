# """
# DAG: ai_jobs_preprocessing
# Stage 2 — Load → Profile → Clean → Engineer → Split → Transform → Validate → Save → MLflow

# Validation failure handling:
#   - Raises AirflowException with a detailed failure report
#   - Triggers on_failure_callback (extend to Slack/email)
#   - Retries once before giving up
#   - All failures stored as XCom for post-mortem

# Changes from previous DAG:
#   - Uses DataTransformer.transform() (replaces transform_and_save)
#   - ArtifactSaver.save() is now a separate dedicated step (STEP 8)
#   - MLflowLogger.log_run() is STEP 9 (was STEP 8)
#   - task_build_and_transform: transform + save are now decoupled
#   - task_save_artifacts: new dedicated task
#   - task_log_mlflow: unchanged logic, now t9
#   - Validation expanded: min row counts, target variance, no-negatives checks
#   - IQRCapper is fitted inside the pipeline on train data only
#   - boolean_cols / ordinal_cols are now tracked in pipeline metadata
# """

# from __future__ import annotations

# import logging
# from datetime import datetime, timedelta
# from pathlib import Path

# from airflow import DAG
# from airflow.exceptions import AirflowException
# from airflow.operators.python import PythonOperator
# from airflow.utils.trigger_rule import TriggerRule

# import sys
# sys.path.insert(0, "/home/samiksya/ai_job_market/src")

# log = logging.getLogger(__name__)

# PROCESSED_DIR = Path("/home/samiksya/ai_job_market/data/processed")
# TARGET_COL    = "salary_usd"

# # ─────────────────────────────────────────────
# # DEFAULT ARGS
# # ─────────────────────────────────────────────

# default_args = {
#     "owner":             "samiksya",
#     "retries":           1,
#     "retry_delay":       timedelta(minutes=2),
#     "execution_timeout": timedelta(minutes=30),
# }

# # ─────────────────────────────────────────────
# # FAILURE CALLBACK
# # ─────────────────────────────────────────────

# def on_task_failure(context):
#     """
#     Called automatically on any task failure.
#     Extend this to send Slack / email / PagerDuty alerts.
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

# def _load_and_prepare():
#     """
#     Shared helper: load → clean → engineer features.
#     Called by several tasks to avoid persisting intermediate
#     state between steps (keeps each task idempotent).
#     """
#     from data.preprocessing import DataLoader, DataCleaner
#     df = DataLoader.load_from_mariadb()
#     df = DataCleaner.clean(df)
#     df = DataCleaner.engineer_features(df)
#     return df


# # ─────────────────────────────────────────────
# # TASK CALLABLES
# # ─────────────────────────────────────────────

# def task_load_data(**ctx):
#     """STEP 1 — Load raw data from MariaDB."""
#     from data.preprocessing import DataLoader
#     df = DataLoader.load_from_mariadb()
#     ctx["ti"].xcom_push(key="raw_row_count", value=len(df))
#     log.info("Loaded %d rows from MariaDB", len(df))


# def task_profile_data(**ctx):
#     """STEP 2 — Generate profiling report (HTML or JSON fallback)."""
#     from data.preprocessing import DataLoader, DataProfiler
#     df = DataLoader.load_from_mariadb()
#     DataProfiler.run_profiling(df)
#     log.info("Profiling complete")


# def task_clean_data(**ctx):
#     """STEP 3 — Remove duplicates, drop metadata columns, drop null critical rows."""
#     from data.preprocessing import DataLoader, DataCleaner
#     df = DataLoader.load_from_mariadb()
#     df = DataCleaner.clean(df)
#     ctx["ti"].xcom_push(key="clean_row_count", value=len(df))
#     log.info("Clean row count: %d", len(df))


# def task_engineer_features(**ctx):
#     """
#     STEP 4 — Create engineered model features:
#       days_open, num_skills, experience_encoded,
#       emp_* dummies, remote_* dummies,
#       same_country, exp_x_skills, benefits_x_exp.
#     """
#     from data.preprocessing import DataLoader, DataCleaner
#     df = DataLoader.load_from_mariadb()
#     df = DataCleaner.clean(df)
#     df = DataCleaner.engineer_features(df)
#     ctx["ti"].xcom_push(key="feature_cols", value=list(df.columns))
#     log.info("Feature engineering done. Columns: %s", list(df.columns))


# def task_split_data(**ctx):
#     """STEP 5 — Train/test split (80/20, random_state=42)."""
#     from data.preprocessing import DataSplitter
#     df = _load_and_prepare()
#     train_df, test_df = DataSplitter.split_data(df)
#     ctx["ti"].xcom_push(key="train_count", value=len(train_df))
#     ctx["ti"].xcom_push(key="test_count",  value=len(test_df))
#     log.info("Split → train=%d  test=%d", len(train_df), len(test_df))


# def task_transform(**ctx):
#     """
#     STEP 6 — Build sklearn pipeline on train data, transform both splits.

#     Pipeline stages (per PreprocessingPipeline.build):
#       numerical  → SimpleImputer(median) → IQRCapper (fitted on train only)
#       ordinal    → SimpleImputer(median)
#       categorical→ SimpleImputer(most_frequent) → OrdinalEncoder
#       boolean    → SimpleImputer(most_frequent)
#       all        → StandardScaler

#     Pushes train/test shapes to XCom. Artifacts are saved in the next task.
#     """
#     from data.preprocessing import DataSplitter, DataTransformer
#     df = _load_and_prepare()
#     train_df, test_df = DataSplitter.split_data(df)

#     # DataTransformer.transform() fits on train, transforms both — does NOT save
#     train_final, test_final, pipeline = DataTransformer.transform(
#         train_df, test_df
#     )

#     # Stash shapes for downstream tasks
#     ctx["ti"].xcom_push(key="train_shape", value=list(train_final.shape))
#     ctx["ti"].xcom_push(key="test_shape",  value=list(test_final.shape))
#     log.info(
#         "Transform done → train=%s  test=%s",
#         train_final.shape,
#         test_final.shape,
#     )

#     # NOTE: train_final / test_final / pipeline are not serialised to XCom
#     # (too large). The next task re-runs transform to get them — this is
#     # intentional: transform is fast and keeps each task self-contained.


# def task_save_artifacts(**ctx):
#     """
#     STEP 7 — Fit pipeline and persist all artifacts:
#       - train.parquet
#       - test.parquet
#       - preprocessor.pkl
#       - preprocessing_metadata.json

#     ArtifactSaver.save() writes all four atomically.
#     IQRCapper bounds are serialised inside preprocessor.pkl.
#     """
#     from data.preprocessing import DataSplitter, DataTransformer, ArtifactSaver
#     df = _load_and_prepare()
#     train_df, test_df = DataSplitter.split_data(df)
#     train_final, test_final, pipeline = DataTransformer.transform(
#         train_df, test_df
#     )
#     ArtifactSaver.save(train_final, test_final, pipeline)

#     ctx["ti"].xcom_push(key="train_shape", value=list(train_final.shape))
#     ctx["ti"].xcom_push(key="test_shape",  value=list(test_final.shape))
#     log.info(
#         "Artifacts saved → train=%s  test=%s",
#         train_final.shape,
#         test_final.shape,
#     )


# def task_validate_outputs(**ctx):
#     """
#     STEP 8 — Validate saved artifacts.

#     Runs a full suite of checks. On ANY failure:
#       - Pushes a detailed failure report to XCom
#       - Raises AirflowException  →  task turns RED in UI
#       - Triggers on_failure_callback  →  alert fires
#       - Retries once (per task-level retries) before giving up
#     """
#     import numpy as np
#     import pandas as pd

#     failed_checks = []
#     passed_checks = []

#     # ── 1. Artifact file existence ──────────────────────────────────────────
#     required_files = [
#         "train.parquet",
#         "test.parquet",
#         "preprocessor.pkl",
#         "preprocessing_metadata.json",
#     ]
#     missing_files = [
#         f for f in required_files if not (PROCESSED_DIR / f).exists()
#     ]
#     if missing_files:
#         msg = f"Missing artifact files: {missing_files}"
#         log.error(msg)
#         ctx["ti"].xcom_push(
#             key="validation_failure_report",
#             value={"missing_files": missing_files},
#         )
#         raise AirflowException(msg)

#     # ── 2. Load parquets ────────────────────────────────────────────────────
#     train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
#     test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")

#     # ── 3. Define all checks ────────────────────────────────────────────────
#     checks = {
#         # Shape checks
#         "train_not_empty": (
#             len(train) > 0,
#             "Train is empty (0 rows)",
#         ),
#         "test_not_empty": (
#             len(test) > 0,
#             "Test is empty (0 rows)",
#         ),
#         "train_min_rows": (
#             len(train) >= 100,
#             f"Train has only {len(train)} rows — suspiciously small",
#         ),
#         "test_min_rows": (
#             len(test) >= 20,
#             f"Test has only {len(test)} rows — suspiciously small",
#         ),

#         # Schema checks
#         "columns_match": (
#             list(train.columns) == list(test.columns),
#             f"Column mismatch: train={list(train.columns)} test={list(test.columns)}",
#         ),
#         "target_exists": (
#             TARGET_COL in train.columns,
#             f"Target column '{TARGET_COL}' missing from train",
#         ),

#         # Data quality checks
#         "train_no_nulls": (
#             train.isnull().sum().sum() == 0,
#             f"Train has {train.isnull().sum().sum()} null values",
#         ),
#         "test_no_nulls": (
#             test.isnull().sum().sum() == 0,
#             f"Test has {test.isnull().sum().sum()} null values",
#         ),
#         "train_all_numeric": (
#             train.select_dtypes("number").shape[1] == train.shape[1],
#             f"Train has non-numeric columns: "
#             f"{list(train.select_dtypes(exclude='number').columns)}",
#         ),
#         "test_all_numeric": (
#             test.select_dtypes("number").shape[1] == test.shape[1],
#             f"Test has non-numeric columns: "
#             f"{list(test.select_dtypes(exclude='number').columns)}",
#         ),

#         # Numeric sanity checks
#         "train_no_inf": (
#             np.isfinite(train.select_dtypes("number").values).all(),
#             "Train contains infinite values",
#         ),
#         "test_no_inf": (
#             np.isfinite(test.select_dtypes("number").values).all(),
#             "Test contains infinite values",
#         ),

#         # Target sanity checks
#         "target_no_negatives": (
#             (train[TARGET_COL] >= 0).all() if TARGET_COL in train.columns else True,
#             f"'{TARGET_COL}' contains negative values in train",
#         ),
#         "target_has_variance": (
#             (train[TARGET_COL].std() > 0) if TARGET_COL in train.columns else True,
#             f"'{TARGET_COL}' has zero variance — all values identical",
#         ),
#     }

#     # ── 4. Run all checks ───────────────────────────────────────────────────
#     for check_name, (result, error_msg) in checks.items():
#         if result:
#             passed_checks.append(check_name)
#             log.info("  ✓ PASS  %s", check_name)
#         else:
#             failed_checks.append({"check": check_name, "reason": error_msg})
#             log.error("  ✗ FAIL  %s — %s", check_name, error_msg)

#     # ── 5. Push full report to XCom ─────────────────────────────────────────
#     report = {
#         "timestamp":    datetime.utcnow().isoformat(),
#         "passed":       passed_checks,
#         "failed":       failed_checks,
#         "total_checks": len(checks),
#         "pass_count":   len(passed_checks),
#         "fail_count":   len(failed_checks),
#         "train_shape":  list(train.shape),
#         "test_shape":   list(test.shape),
#     }
#     ctx["ti"].xcom_push(key="validation_report", value=report)

#     # ── 6. Raise on any failure ─────────────────────────────────────────────
#     if failed_checks:
#         summary = "\n".join(
#             f"  ✗ {f['check']}: {f['reason']}" for f in failed_checks
#         )
#         raise AirflowException(
#             f"\n\nVALIDATION FAILED — {len(failed_checks)}/{len(checks)} "
#             f"checks failed:\n{summary}\n\n"
#             "Check XCom key 'validation_report' on task 'validate_outputs' "
#             "for full details."
#         )

#     log.info("All %d validation checks passed.", len(checks))


# def task_log_mlflow(**ctx):
#     """
#     STEP 9 — Log params, metrics, and artifacts to MLflow.

#     Reads saved parquets (written by task_save_artifacts) so this task
#     has no dependency on in-memory pipeline state.

#     Logged items:
#       params  : train_size, test_size, feature_count, target
#       metrics : train_salary_mean, train_salary_std
#       artifacts: preprocessor.pkl, preprocessing_metadata.json,
#                  profiling_report.html / .json (whichever exists)
#     """
#     import pandas as pd
#     from data.preprocessing import MLflowLogger

#     train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
#     test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")
#     MLflowLogger.log_run(train, test)
#     log.info("MLflow logging complete")


# # ─────────────────────────────────────────────
# # DAG DEFINITION
# # ─────────────────────────────────────────────

# with DAG(
#     dag_id="ai_jobs_preprocessing",
#     description=(
#         "Stage 2: load → profile → clean → engineer → split → "
#         "transform → save → validate → MLflow"
#     ),
#     start_date=datetime(2025, 1, 1),
#     schedule_interval=None,
#     catchup=False,
#     default_args=default_args,
#     on_failure_callback=on_task_failure,
#     tags=["stage2", "preprocessing", "ai_jobs"],
# ) as dag:

#     t1_load = PythonOperator(
#         task_id="load_data",
#         python_callable=task_load_data,
#     )

#     t2_profile = PythonOperator(
#         task_id="profile_data",
#         python_callable=task_profile_data,
#     )

#     t3_clean = PythonOperator(
#         task_id="clean_data",
#         python_callable=task_clean_data,
#     )

#     t4_engineer = PythonOperator(
#         task_id="engineer_features",
#         python_callable=task_engineer_features,
#     )

#     t5_split = PythonOperator(
#         task_id="split_data",
#         python_callable=task_split_data,
#     )

#     t6_transform = PythonOperator(
#         task_id="transform",
#         python_callable=task_transform,
#     )

#     t7_save = PythonOperator(
#         task_id="save_artifacts",
#         python_callable=task_save_artifacts,
#         trigger_rule=TriggerRule.ALL_SUCCESS,
#     )

#     t8_validate = PythonOperator(
#         task_id="validate_outputs",
#         python_callable=task_validate_outputs,
#         trigger_rule=TriggerRule.ALL_SUCCESS,
#         retries=1,
#         retry_delay=timedelta(seconds=30),
#         on_failure_callback=on_task_failure,
#     )

#     t9_mlflow = PythonOperator(
#         task_id="log_to_mlflow",
#         python_callable=task_log_mlflow,
#         trigger_rule=TriggerRule.ALL_SUCCESS,
#     )

#     (
#         t1_load
#         >> t2_profile
#         >> t3_clean
#         >> t4_engineer
#         >> t5_split
#         >> t6_transform
#         >> t7_save
#         >> t8_validate
#         >> t9_mlflow
#     )