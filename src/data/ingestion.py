"""
Unified Data Ingestion Pipeline
Flow: Load CSV → Standardize Schema → Validate → Store to MariaDB

This replaces: ingestion.py, validation.py, storage.py
Provides a single source of truth for data ingestion with proper error handling.
"""

import pandas as pd
import uuid
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import logging

try:
    import great_expectations as gx
except ImportError:
    gx = None

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────

RAW_PATH = Path("/home/samiksya/ai_job_market/data/raw/ai_jobs.csv")
STAGING_PATH = Path("/home/samiksya/ai_job_market/data/stagging/ai_jobs_staged.csv")

DB_HOST = os.getenv("MARIADB_HOST", "127.0.0.1").strip()
DB_PORT = int(os.getenv("MARIADB_PORT", "3307"))
DB_USER = os.getenv("MARIADB_USER", "mariadbuser")
DB_PASSWORD = os.getenv("MARIADB_PASSWORD", "Samikshya@123")
DB_NAME = os.getenv("MARIADB_DB", "ai_jobs_raw")
DB_TABLE = os.getenv("MARIADB_TABLE", "raw_ai_jobs")

# MariaDB ColumnStore connection
def get_db_engine(with_db=True):
    """Create SQLAlchemy engine for MariaDB ColumnStore."""
    password = quote_plus(DB_PASSWORD)
    if with_db:
        return create_engine(
            f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )
    else:
        return create_engine(
            f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/",
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )


def quote_identifier(identifier: str) -> str:
    """Quote a MariaDB identifier."""
    return f"`{identifier.replace('`', '``')}`"


def sql_type_for_series(series: pd.Series) -> str:
    """Map pandas dtypes to MariaDB-friendly column types."""
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DATETIME"
    return "VARCHAR(1024)"


def create_table_if_not_exists(conn, df: pd.DataFrame) -> None:
    """Create the raw jobs table without relying on pandas.to_sql."""
    columns_sql = [
        f"{quote_identifier(column)} {sql_type_for_series(df[column])}"
        for column in df.columns
    ]
    create_sql = (
        f"CREATE TABLE IF NOT EXISTS {quote_identifier(DB_TABLE)} "
        f"({', '.join(columns_sql)})"
    )
    conn.execute(text(create_sql))


def insert_dataframe(conn, df: pd.DataFrame, chunksize: int = 500) -> None:
    """Bulk insert a dataframe using SQLAlchemy executemany."""
    if df.empty:
        return

    columns = list(df.columns)
    quoted_columns = ", ".join(quote_identifier(column) for column in columns)
    placeholders = ", ".join(f":{column}" for column in columns)
    insert_sql = text(
        f"INSERT INTO {quote_identifier(DB_TABLE)} "
        f"({quoted_columns}) VALUES ({placeholders})"
    )

    clean_df = df.astype(object).where(pd.notna(df), None)
    records = clean_df.to_dict(orient="records")

    for start in range(0, len(records), chunksize):
        conn.execute(insert_sql, records[start:start + chunksize])


# ─────────────────────────────────────────────
# STEP 1: LOAD & STANDARDIZE
# ─────────────────────────────────────────────

def load_and_standardize() -> pd.DataFrame:
    """
    STEP 1: Load raw CSV and standardize schema.
    
    Returns:
        pd.DataFrame: Standardized dataframe with lowercase, snake_case columns
    
    Raises:
        FileNotFoundError: If RAW_PATH doesn't exist
        ValueError: If CSV is empty
    """
    log.info(f"Loading raw CSV from {RAW_PATH}")
    
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw CSV not found at {RAW_PATH}")
    
    df = pd.read_csv(RAW_PATH)
    
    if len(df) == 0:
        raise ValueError("Raw CSV is empty")
    
    log.info(f"Loaded {len(df)} rows from raw CSV")
    
    # Standardize schema: lowercase, strip whitespace, replace spaces with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    log.info(f"Standardized columns: {list(df.columns)}")
    
    return df


# ─────────────────────────────────────────────
# STEP 2: ADD METADATA
# ─────────────────────────────────────────────

def add_ingestion_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 2: Add batch tracking metadata.
    
    Args:
        df: Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with batch_id and ingested_at columns
    """
    df["batch_id"] = str(uuid.uuid4())
    df["ingested_at"] = datetime.utcnow().isoformat()
    
    log.info(f"Added metadata | batch_id={df['batch_id'].iloc[0]} | ingested_at={df['ingested_at'].iloc[0]}")
    
    return df


# ─────────────────────────────────────────────
# STEP 3: VALIDATE WITH GREAT EXPECTATIONS
# ─────────────────────────────────────────────

class PandasValidator:
    """Compatibility validator for Great Expectations versions without from_pandas."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def _result(success: bool, **details):
        return SimpleNamespace(success=success, result=details)

    def _unexpected_result(self, column: str, unexpected_mask: pd.Series):
        unexpected = self.df.loc[unexpected_mask, column].head(20).tolist()
        unexpected_count = int(unexpected_mask.sum())
        total = len(self.df)
        return {
            "element_count": total,
            "unexpected_count": unexpected_count,
            "unexpected_percent": (unexpected_count / total * 100) if total else 0,
            "partial_unexpected_list": unexpected,
        }

    def expect_column_values_to_not_be_null(self, column: str):
        unexpected_mask = self.df[column].isna()
        return self._result(
            not unexpected_mask.any(),
            **self._unexpected_result(column, unexpected_mask),
        )

    def expect_column_values_to_be_in_set(self, column: str, value_set: list):
        values = self.df[column]
        unexpected_mask = values.notna() & ~values.isin(value_set)
        return self._result(
            not unexpected_mask.any(),
            **self._unexpected_result(column, unexpected_mask),
        )

    def expect_column_values_to_be_between(self, column: str, min_value, max_value):
        values = pd.to_numeric(self.df[column], errors="coerce")
        unexpected_mask = self.df[column].notna() & (
            values.isna() | (values < min_value) | (values > max_value)
        )
        return self._result(
            not unexpected_mask.any(),
            **self._unexpected_result(column, unexpected_mask),
        )

    def expect_column_values_to_be_unique(self, column: str):
        unexpected_mask = self.df[column].duplicated(keep=False)
        return self._result(
            not unexpected_mask.any(),
            **self._unexpected_result(column, unexpected_mask),
        )

    def expect_table_row_count_to_be_between(self, min_value: int, max_value: int):
        row_count = len(self.df)
        return self._result(
            min_value <= row_count <= max_value,
            observed_value=row_count,
            min_value=min_value,
            max_value=max_value,
        )


def get_dataframe_validator(df: pd.DataFrame):
    """Return a dataframe validator that works across Great Expectations versions."""
    if gx is not None and hasattr(gx, "from_pandas"):
        return gx.from_pandas(df)

    if gx is None:
        log.warning("Great Expectations is not installed; using pandas validation fallback")
    else:
        version = getattr(gx, "__version__", "unknown")
        log.warning(
            "Great Expectations %s does not expose from_pandas; using pandas validation fallback",
            version,
        )

    return PandasValidator(df)


def validate_data(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    STEP 3: Validate dataframe using Great Expectations.
    
    Uses direct validation without context persistence to avoid "must save" errors.
    
    Args:
        df: Input dataframe to validate
    
    Returns:
        tuple: (success: bool, results: dict with details)
    
    Raises:
        Exception: If validation fails
    """
    log.info(f"Starting validation on {len(df)} rows")
    
    try:
        # Use direct validation when available; otherwise use the pandas fallback above.
        validator = get_dataframe_validator(df)
        
        failed_checks = []
        passed_checks = 0
        
        # ── Required columns (not null) ──────────────────────
        required_cols = [
            "job_id", "job_title", "salary_usd", "experience_level",
            "employment_type", "company_location", "batch_id", "ingested_at"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            try:
                result = validator.expect_column_values_to_not_be_null(column=col)
                if result.success:
                    passed_checks += 1
                    log.info(f"  ✓ Column not null: {col}")
                else:
                    failed_checks.append({
                        "expectation": f"expect_column_values_to_not_be_null({col})",
                        "details": str(result.result)
                    })
                    log.error(f"  ✗ Column has nulls: {col}")
            except Exception as e:
                log.error(f"  ✗ Error checking {col}: {str(e)}")
                failed_checks.append({"expectation": f"Column {col}", "details": str(e)})
        
        # ── Categorical columns (valid values) ─────────────────
        try:
            result = validator.expect_column_values_to_be_in_set(
                column="experience_level", 
                value_set=["EN", "MI", "SE", "EX"]
            )
            if result.success:
                passed_checks += 1
                log.info("  ✓ experience_level values valid")
            else:
                failed_checks.append({"expectation": "experience_level values", "details": str(result.result)})
                log.error(f"  ✗ experience_level has invalid values")
        except Exception as e:
            log.warning(f"  ⚠ experience_level check skipped: {str(e)}")
        
        try:
            result = validator.expect_column_values_to_be_in_set(
                column="employment_type", 
                value_set=["FT", "PT", "CT", "FL"]
            )
            if result.success:
                passed_checks += 1
                log.info("  ✓ employment_type values valid")
            else:
                failed_checks.append({"expectation": "employment_type values", "details": str(result.result)})
        except Exception as e:
            log.warning(f"  ⚠ employment_type check skipped: {str(e)}")
        
        try:
            result = validator.expect_column_values_to_be_in_set(
                column="company_size", 
                value_set=["S", "M", "L"]
            )
            if result.success:
                passed_checks += 1
                log.info("  ✓ company_size values valid")
            else:
                failed_checks.append({"expectation": "company_size values", "details": str(result.result)})
        except Exception as e:
            log.warning(f"  ⚠ company_size check skipped: {str(e)}")
        
        try:
            result = validator.expect_column_values_to_be_in_set(
                column="remote_ratio", 
                value_set=[0, 50, 100]
            )
            if result.success:
                passed_checks += 1
                log.info("  ✓ remote_ratio values valid")
            else:
                failed_checks.append({"expectation": "remote_ratio values", "details": str(result.result)})
        except Exception as e:
            log.warning(f"  ⚠ remote_ratio check skipped: {str(e)}")
        
        # ── Numeric ranges ────────────────────────────────────
        try:
            result = validator.expect_column_values_to_be_between(
                column="salary_usd", 
                min_value=5000, 
                max_value=1000000
            )
            if result.success:
                passed_checks += 1
                log.info("  ✓ salary_usd in valid range")
            else:
                failed_checks.append({"expectation": "salary_usd range", "details": str(result.result)})
                log.error(f"  ✗ salary_usd out of range")
        except Exception as e:
            log.warning(f"  ⚠ salary_usd check skipped: {str(e)}")
        
        try:
            result = validator.expect_column_values_to_be_between(
                column="years_experience", 
                min_value=0, 
                max_value=50
            )
            if result.success:
                passed_checks += 1
                log.info("  ✓ years_experience in valid range")
            else:
                failed_checks.append({"expectation": "years_experience range", "details": str(result.result)})
        except Exception as e:
            log.warning(f"  ⚠ years_experience check skipped: {str(e)}")
        
        try:
            result = validator.expect_column_values_to_be_between(
                column="benefits_score", 
                min_value=0.0, 
                max_value=10.0
            )
            if result.success:
                passed_checks += 1
                log.info("  ✓ benefits_score in valid range")
            else:
                failed_checks.append({"expectation": "benefits_score range", "details": str(result.result)})
        except Exception as e:
            log.warning(f"  ⚠ benefits_score check skipped: {str(e)}")
        
        try:
            result = validator.expect_column_values_to_be_between(
                column="job_description_length", 
                min_value=10, 
                max_value=50000
            )
            if result.success:
                passed_checks += 1
                log.info("  ✓ job_description_length in valid range")
            else:
                failed_checks.append({"expectation": "job_description_length range", "details": str(result.result)})
        except Exception as e:
            log.warning(f"  ⚠ job_description_length check skipped: {str(e)}")
        
        # ── Uniqueness & completeness ─────────────────────────
        try:
            result = validator.expect_column_values_to_be_unique(column="job_id")
            if result.success:
                passed_checks += 1
                log.info("  ✓ job_id is unique")
            else:
                failed_checks.append({"expectation": "job_id uniqueness", "details": str(result.result)})
                log.error(f"  ✗ job_id has duplicates")
        except Exception as e:
            log.warning(f"  ⚠ job_id uniqueness check skipped: {str(e)}")
        
        try:
            result = validator.expect_table_row_count_to_be_between(
                min_value=100, 
                max_value=500000
            )
            if result.success:
                passed_checks += 1
                log.info(f"  ✓ Row count {len(df)} in valid range")
            else:
                failed_checks.append({"expectation": "row count", "details": str(result.result)})
        except Exception as e:
            log.warning(f"  ⚠ Row count check skipped: {str(e)}")
        
        # ── Schema check ──────────────────────────────────────
        expected_columns = {
            "job_id", "job_title", "salary_usd", "salary_currency",
            "experience_level", "employment_type", "company_location",
            "company_size", "employee_residence", "remote_ratio",
            "required_skills", "education_required", "years_experience",
            "industry", "posting_date", "application_deadline",
            "job_description_length", "benefits_score", "company_name",
            "batch_id", "ingested_at"
        }
        
        actual_columns = set(df.columns)
        if actual_columns == expected_columns:
            passed_checks += 1
            log.info("  ✓ Schema matches exactly")
        else:
            missing = expected_columns - actual_columns
            extra = actual_columns - expected_columns
            schema_msg = f"Missing: {missing}, Extra: {extra}"
            failed_checks.append({"expectation": "schema match", "details": schema_msg})
            log.error(f"  ✗ Schema mismatch: {schema_msg}")
        
        # ── Summary ────────────────────────────────────────────
        total_checks = passed_checks + len(failed_checks)
        success = len(failed_checks) == 0
        
        validation_report = {
            "success": success,
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": len(failed_checks),
            "failed_details": failed_checks
        }
        
        if success:
            log.info(f"\n✓ Validation PASSED | {passed_checks}/{total_checks} checks passed")
        else:
            log.error(f"\n✗ Validation FAILED | {len(failed_checks)} checks failed")
            raise ValueError(f"Data validation failed: {len(failed_checks)} checks failed")
        
        return success, validation_report
    
    except Exception as e:
        log.error(f"Validation error: {str(e)}")
        raise


# ─────────────────────────────────────────────
# STEP 4: SAVE TO STAGING (LOCAL)
# ─────────────────────────────────────────────

def save_to_staging(df: pd.DataFrame) -> Path:
    """
    STEP 4: Save validated data to local staging CSV.
    
    Args:
        df: Validated dataframe
    
    Returns:
        Path: Path to saved file
    """
    log.info(f"Saving {len(df)} rows to staging CSV")
    
    STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGING_PATH, index=False)
    
    log.info(f"✓ Saved to {STAGING_PATH}")
    
    return STAGING_PATH


# ─────────────────────────────────────────────
# STEP 5: LOAD TO MARIADB
# ─────────────────────────────────────────────

def load_to_mariadb(df: pd.DataFrame = None) -> dict:
    """
    STEP 5: Load validated data to MariaDB ColumnStore.
    
    - Creates database if not exists
    - Creates table if not exists
    - Truncates and reloads data
    - Returns load statistics
    
    Args:
        df: Dataframe to load. If None, reads from staging CSV.
    
    Returns:
        dict: Load statistics (rows_loaded, batch_id, timestamp)
    
    Raises:
        Exception: If DB connection fails or load fails
    """
    if df is None:
        log.info(f"Reading from staging CSV: {STAGING_PATH}")
        df = pd.read_csv(STAGING_PATH)
    
    log.info(f"Loading {len(df)} rows to MariaDB")
    
    try:
        # Create database
        log.info("Creating database if not exists")
        engine_no_db = get_db_engine(with_db=False)
        with engine_no_db.begin() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
        engine_no_db.dispose()
        
        # Create table and load data
        log.info(f"Loading data to {DB_TABLE} table")
        engine = get_db_engine(with_db=True)
        
        with engine.begin() as conn:
            create_table_if_not_exists(conn, df)

            # Try to truncate if table exists
            try:
                conn.execute(text(f"TRUNCATE TABLE {quote_identifier(DB_TABLE)}"))
                log.info(f"Truncated existing {DB_TABLE} table")
            except Exception as e:
                log.info(f"Table doesn't exist yet (will be created): {str(e)}")

            insert_dataframe(conn, df, chunksize=500)
        
        engine.dispose()
        
        batch_id = df["batch_id"].iloc[0] if len(df) > 0 else "unknown"
        load_stats = {
            "rows_loaded": len(df),
            "batch_id": batch_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
        log.info(f"✓ Loaded {len(df)} rows to MariaDB | batch_id={batch_id}")
        
        return load_stats
    
    except Exception as e:
        log.error(f"Failed to load to MariaDB: {str(e)}")
        raise


# ─────────────────────────────────────────────
# MAIN PIPELINE: ORCHESTRATE ALL STEPS
# ─────────────────────────────────────────────

def run_ingestion_pipeline() -> dict:
    """
    MAIN PIPELINE: Execute complete ingestion workflow.
    
    Flow:
      1. Load & Standardize CSV
      2. Add Metadata (batch_id, ingested_at)
      3. Validate with Great Expectations
      4. Save to Staging CSV (if validation passes)
      5. Load to MariaDB ColumnStore
    
    Returns:
        dict: Pipeline execution results
    
    Raises:
        Exception: If any step fails
    """
    log.info("="*60)
    log.info("Starting Data Ingestion Pipeline")
    log.info("="*60)
    
    pipeline_results = {}
    
    try:
        # STEP 1: Load & Standardize
        log.info("\n[STEP 1/5] Loading and standardizing CSV...")
        df = load_and_standardize()
        pipeline_results["step_1_load"] = {"rows": len(df), "status": "success"}
        
        # STEP 2: Add Metadata
        log.info("\n[STEP 2/5] Adding ingestion metadata...")
        df = add_ingestion_metadata(df)
        pipeline_results["step_2_metadata"] = {"status": "success"}
        
        # STEP 3: Validate
        log.info("\n[STEP 3/5] Validating data with Great Expectations...")
        success, validation_report = validate_data(df)
        pipeline_results["step_3_validation"] = validation_report
        
        if not success:
            raise ValueError("Data validation failed — aborting pipeline")
        
        # STEP 4: Save to Staging
        log.info("\n[STEP 4/5] Saving validated data to staging...")
        staging_path = save_to_staging(df)
        pipeline_results["step_4_staging"] = {"path": str(staging_path), "status": "success"}
        
        # STEP 5: Load to MariaDB
        log.info("\n[STEP 5/5] Loading to MariaDB ColumnStore...")
        load_stats = load_to_mariadb(df)
        pipeline_results["step_5_mariadb"] = load_stats
        
        # Success summary
        log.info("\n" + "="*60)
        log.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        log.info("="*60)
        log.info(f"Total rows ingested: {len(df)}")
        log.info(f"Batch ID: {load_stats['batch_id']}")
        log.info(f"Timestamp: {load_stats['timestamp']}")
        
        pipeline_results["overall_status"] = "success"
        
        return pipeline_results
    
    except Exception as e:
        log.error("\n" + "="*60)
        log.error("✗ PIPELINE FAILED")
        log.error("="*60)
        log.error(f"Error: {str(e)}")
        
        pipeline_results["overall_status"] = "failed"
        pipeline_results["error"] = str(e)
        
        raise


# ─────────────────────────────────────────────
# BACKWARDS COMPATIBILITY (for existing DAG)
# ─────────────────────────────────────────────

def run_ingestion():
    """Wrapper for existing DAG compatibility."""
    return run_ingestion_pipeline()


def run_validation() -> bool:
    """
    Wrapper for existing DAG compatibility.
    Validates the staging CSV.
    """
    try:
        df = pd.read_csv(STAGING_PATH)
        success, _ = validate_data(df)
        return success
    except Exception as e:
        log.error(f"Validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    try:
        results = run_ingestion_pipeline()
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        import json
        print(json.dumps(results, indent=2))
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        sys.exit(1)
