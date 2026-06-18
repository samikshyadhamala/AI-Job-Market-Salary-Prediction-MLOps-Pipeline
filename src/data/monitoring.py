"""
MONITORING STAGE — Independent script (not an Airflow DAG)

Reference data : train.parquet         → has salary_usd + 21 features
Current data   : user_inputs.parquet → has predicted_salary + 21 features + timestamp

Flow:
1. Load reference (train.parquet) and current (prediction logs)
2. Align columns for Evidently (rename predicted_salary → salary_usd in current)
3. Run Evidently:
      a) Data Quality   — nulls, ranges, schema
      b) Feature Drift  — distribution shift on 21 feature columns
      c) Prediction Drift — salary_usd (ref) vs predicted_salary (current)
4. Write metrics to MariaDB (monitoring.drift_metrics)
5. Threshold check → log alert + save alert file if drift detected
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

# ── Evidently imports ────────────────────────────────────────────────────────
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    REFERENCE_DATA_PATH  = Path("/home/samiksya/ai_job_market/data/processed/train.parquet")
    PREDICTION_LOGS_PATH = Path("/home/samiksya/ai_job_market/data/processed/user_inputs.parquet")
    REPORTS_DIR          = Path("/home/samiksya/ai_job_market/data/monitoring_reports")
    ALERTS_DIR           = Path("/home/samiksya/ai_job_market/data/monitoring_alerts")

    # ── Column names ─────────────────────────────────────────────────────────
    TARGET_COL           = "salary_usd"        # column name in train.parquet
    PREDICTION_COL       = "predicted_salary"  # column name in prediction logs
    TIMESTAMP_COL        = "timestamp"

    # ── 21 shared feature columns ─────────────────────────────────────────────
    FEATURE_COLS = [
        "years_experience", "remote_ratio", "benefits_score", "days_open",
        "num_skills", "same_country", "exp_x_skills", "benefits_x_exp",
        "experience_encoded", "job_title_encoded", "company_location_encoded",
        "company_size_encoded", "employee_residence_encoded",
        "education_required_encoded", "industry_encoded",
        "salary_currency_encoded", "emp_full_time", "emp_part_time",
        "emp_freelance", "remote_onsite", "remote_remote",
    ]

    # ── Sliding window ────────────────────────────────────────────────────────
    LAST_N_DAYS          = 7   # use last 7 days of prediction logs

    # ── MariaDB ───────────────────────────────────────────────────────────────
    DB_HOST              = "127.0.0.1"
    DB_PORT              = 3307          # your host port for mymcs
    DB_NAME              = "monitoring"
    DB_USER              = "mariadbuser"
    DB_PASSWORD          = "Samikshya@123"

    # ── Drift thresholds ──────────────────────────────────────────────────────
    FEATURE_DRIFT_SHARE_THRESHOLD   = 0.30   # alert if >30% of features drifted
    PREDICTION_DRIFT_SCORE_THRESHOLD = 0.15  # alert if prediction drift score > 0.15


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_reference() -> pd.DataFrame:
    """Load train.parquet as reference. Keep features + salary_usd only."""
    logger.info("Loading reference data from %s", Config.REFERENCE_DATA_PATH)

    if not Config.REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference file not found: {Config.REFERENCE_DATA_PATH}")

    df = pd.read_parquet(Config.REFERENCE_DATA_PATH)

    keep_cols = Config.FEATURE_COLS + [Config.TARGET_COL]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Reference data missing columns: {missing}")

    df = df[keep_cols].copy()
    logger.info("Reference data shape: %s", df.shape)
    return df


def load_current() -> pd.DataFrame:
    """
    Load prediction logs from PREDICTION_LOGS_PATH.
    The path may be one parquet file or a directory containing parquet files.
    Filters to last N days using timestamp column.
    Renames predicted_salary → salary_usd so Evidently can compare them.
    """
    logger.info("Loading prediction logs from %s", Config.PREDICTION_LOGS_PATH)

    if not Config.PREDICTION_LOGS_PATH.exists():
        raise FileNotFoundError(f"Prediction logs path not found: {Config.PREDICTION_LOGS_PATH}")

    if Config.PREDICTION_LOGS_PATH.is_file():
        log_files = [Config.PREDICTION_LOGS_PATH]
    else:
        log_files = sorted(Config.PREDICTION_LOGS_PATH.glob("*.parquet"))
        if not log_files:
            raise FileNotFoundError(f"No .parquet files found in {Config.PREDICTION_LOGS_PATH}")

    dfs = [pd.read_parquet(f) for f in log_files]
    current = pd.concat(dfs, ignore_index=True)
    logger.info("Total prediction log rows before filtering: %d", len(current))

    # Filter to last N days
    if Config.TIMESTAMP_COL in current.columns:
        current[Config.TIMESTAMP_COL] = pd.to_datetime(
            current[Config.TIMESTAMP_COL],
            format="mixed",
            errors="coerce",
        )
        invalid_timestamps = current[Config.TIMESTAMP_COL].isna().sum()
        if invalid_timestamps > 0:
            logger.warning(
                "Dropping %d prediction log rows with invalid timestamps",
                invalid_timestamps,
            )
            current = current.dropna(subset=[Config.TIMESTAMP_COL]).copy()

        if current.empty:
            raise ValueError("No valid prediction logs remain after timestamp parsing")

        cutoff = current[Config.TIMESTAMP_COL].max() - pd.Timedelta(days=Config.LAST_N_DAYS)
        current = current[current[Config.TIMESTAMP_COL] >= cutoff].copy()
        logger.info("Rows after last-%d-days filter: %d", Config.LAST_N_DAYS, len(current))

    # Rename predicted_salary → salary_usd for Evidently comparison
    current = current.rename(columns={Config.PREDICTION_COL: Config.TARGET_COL})

    # Keep only relevant columns
    keep_cols = Config.FEATURE_COLS + [Config.TARGET_COL]
    missing = [c for c in keep_cols if c not in current.columns]
    if missing:
        raise ValueError(f"Prediction logs missing columns: {missing}")

    current = current[keep_cols].copy()
    logger.info("Current data shape: %s", current.shape)
    return current


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — RUN EVIDENTLY REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def run_evidently(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Run three Evidently reports:
      1. Data Quality  — nulls, ranges, schema issues
      2. Feature Drift — distribution shift across 21 features
      3. Prediction Drift — salary_usd distribution shift
    Returns a flat dict of extracted metrics.
    """
    Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    column_mapping = ColumnMapping(
        target=Config.TARGET_COL,
        numerical_features=Config.FEATURE_COLS,
    )

    # ── 1. Data Quality ───────────────────────────────────────────────────────
    logger.info("Running data quality report...")
    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    quality_report.save_html(str(Config.REPORTS_DIR / "data_quality.html"))

    # ── 2. Feature Drift ──────────────────────────────────────────────────────
    logger.info("Running feature drift report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    drift_report.save_html(str(Config.REPORTS_DIR / "feature_drift.html"))

    # ── 3. Prediction Drift ───────────────────────────────────────────────────
    logger.info("Running prediction drift report...")
    pred_report = Report(metrics=[ColumnDriftMetric(column_name=Config.TARGET_COL)])
    pred_report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    pred_report.save_html(str(Config.REPORTS_DIR / "prediction_drift.html"))

    logger.info("HTML reports saved to %s", Config.REPORTS_DIR)

    # ── Extract metrics from JSON ─────────────────────────────────────────────
    quality_json = json.loads(quality_report.json())
    drift_json   = json.loads(drift_report.json())
    pred_json    = json.loads(pred_report.json())

    # Data quality metrics
    dq = quality_json["metrics"][0]["result"]
    current_quality = dq.get("current", {})
    missing_values  = int(current_quality.get("number_of_missing_values", 0))
    schema_errors   = int(current_quality.get("number_of_columns_with_missing_values", 0))
    # overall quality score: fraction of non-missing cells
    total_cells     = current.shape[0] * current.shape[1]
    quality_score   = round(1.0 - (missing_values / max(total_cells, 1)), 4)

    # Feature drift metrics
    dd           = drift_json["metrics"][0]["result"]
    n_drifted    = int(dd.get("number_of_drifted_columns", 0))
    n_total      = int(dd.get("number_of_columns", len(Config.FEATURE_COLS)))
    drift_share  = round(n_drifted / max(n_total, 1), 4)
    feat_drift   = bool(dd.get("dataset_drift", False))

    # Prediction drift metrics
    pd_result    = pred_json["metrics"][0]["result"]
    pred_score   = float(pd_result.get("drift_score", 0.0))
    pred_drift   = bool(pd_result.get("drift_detected", False))

    metrics = {
        "data_quality_score":         quality_score,
        "missing_values_count":       missing_values,
        "schema_errors":              schema_errors,
        "feature_drift_detected":     feat_drift,
        "n_drifted_features":         n_drifted,
        "feature_drift_share":        drift_share,
        "prediction_drift_score":     pred_score,
        "prediction_drift_detected":  pred_drift,
    }

    logger.info("── Evidently Results ──────────────────────────────")
    logger.info("  Data quality score   : %.4f", quality_score)
    logger.info("  Missing values       : %d",   missing_values)
    logger.info("  Drifted features     : %d / %d (%.1f%%)", n_drifted, n_total, drift_share * 100)
    logger.info("  Feature drift flag   : %s",   feat_drift)
    logger.info("  Prediction drift score: %.4f", pred_score)
    logger.info("  Prediction drift flag : %s",  pred_drift)
    logger.info("───────────────────────────────────────────────────")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — THRESHOLD CHECK + ALERT
# ══════════════════════════════════════════════════════════════════════════════

def check_thresholds(metrics: dict) -> tuple:
    """
    Returns (drift_detected: bool, alert_triggered: bool, alert_message: str)
    """
    reasons = []

    if metrics["feature_drift_share"] > Config.FEATURE_DRIFT_SHARE_THRESHOLD:
        reasons.append(
            f"Feature drift share {metrics['feature_drift_share']:.1%} "
            f"> threshold {Config.FEATURE_DRIFT_SHARE_THRESHOLD:.1%}"
        )

    if metrics["prediction_drift_score"] > Config.PREDICTION_DRIFT_SCORE_THRESHOLD:
        reasons.append(
            f"Prediction drift score {metrics['prediction_drift_score']:.4f} "
            f"> threshold {Config.PREDICTION_DRIFT_SCORE_THRESHOLD}"
        )

    if metrics["missing_values_count"] > 0:
        reasons.append(f"Missing values detected: {metrics['missing_values_count']}")

    drift_detected  = len(reasons) > 0
    alert_triggered = drift_detected

    if alert_triggered:
        alert_msg = "DRIFT ALERT | " + " | ".join(reasons)
        logger.warning("=" * 60)
        logger.warning("⚠  %s", alert_msg)
        logger.warning("   Action: Retraining recommended. Data scientist review needed.")
        logger.warning("=" * 60)
        _save_alert_file(alert_msg, metrics)
    else:
        alert_msg = "healthy"
        logger.info("✅ Model is HEALTHY — no significant drift detected.")

    return drift_detected, alert_triggered, alert_msg


def _save_alert_file(alert_msg: str, metrics: dict) -> None:
    """Save alert details to a JSON file in ALERTS_DIR."""
    Config.ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = Config.ALERTS_DIR / f"alert_{ts}.json"

    payload = {
        "timestamp":     datetime.utcnow().isoformat(),
        "alert_message": alert_msg,
        "metrics":       metrics,
        "action":        "Retraining recommended. Data scientist review needed.",
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Alert saved to %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — WRITE TO MARIADB
# ══════════════════════════════════════════════════════════════════════════════

def ensure_table_exists(conn) -> None:
    """Create drift_metrics table if it doesn't exist yet."""
    sql = """
    CREATE TABLE IF NOT EXISTS drift_metrics (
        id                        INT AUTO_INCREMENT PRIMARY KEY,
        run_timestamp             DATETIME NOT NULL,
        n_reference_rows          INT,
        n_current_rows            INT,
        data_quality_score        FLOAT,
        missing_values_count      INT,
        schema_errors             INT,
        feature_drift_detected    BOOLEAN,
        n_drifted_features        INT,
        feature_drift_share       FLOAT,
        prediction_drift_score    FLOAT,
        prediction_drift_detected BOOLEAN,
        drift_detected            BOOLEAN,
        alert_triggered           BOOLEAN,
        notes                     TEXT
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def write_to_db(
    metrics: dict,
    n_ref: int,
    n_cur: int,
    drift_detected: bool,
    alert_triggered: bool,
    notes: str = "",
) -> None:
    logger.info("Writing metrics to MariaDB...")
    try:
        conn = pymysql.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            db=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            connect_timeout=10,
        )
        ensure_table_exists(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO drift_metrics (
                    run_timestamp, n_reference_rows, n_current_rows,
                    data_quality_score, missing_values_count, schema_errors,
                    feature_drift_detected, n_drifted_features, feature_drift_share,
                    prediction_drift_score, prediction_drift_detected,
                    drift_detected, alert_triggered, notes
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    datetime.utcnow(),
                    n_ref, n_cur,
                    metrics["data_quality_score"],
                    metrics["missing_values_count"],
                    metrics["schema_errors"],
                    int(metrics["feature_drift_detected"]),
                    metrics["n_drifted_features"],
                    metrics["feature_drift_share"],
                    metrics["prediction_drift_score"],
                    int(metrics["prediction_drift_detected"]),
                    int(drift_detected),
                    int(alert_triggered),
                    notes,
                ),
            )
        conn.commit()
        conn.close()
        logger.info("Metrics written to MariaDB successfully.")

    except pymysql.Error as e:
        logger.error("MariaDB write failed: %s", e)
        logger.warning("Continuing without DB write — metrics were logged above.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("STARTING MONITORING STAGE")
    logger.info("=" * 60)

    # 1. Load data
    reference = load_reference()
    current   = load_current()

    # 2. Run Evidently
    metrics = run_evidently(reference, current)

    # 3. Threshold check + alert
    drift_detected, alert_triggered, alert_msg = check_thresholds(metrics)

    # 4. Write to MariaDB
    write_to_db(
        metrics,
        n_ref=len(reference),
        n_cur=len(current),
        drift_detected=drift_detected,
        alert_triggered=alert_triggered,
        notes=alert_msg,
    )

    logger.info("=" * 60)
    logger.info("MONITORING STAGE COMPLETE")
    logger.info("Status: %s", "⚠ DRIFT DETECTED" if drift_detected else "✅ HEALTHY")
    logger.info("=" * 60)

    return {
        "status":          "DRIFT_DETECTED" if drift_detected else "HEALTHY",
        "drift_detected":  drift_detected,
        "alert_triggered": alert_triggered,
        "metrics":         metrics,
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
