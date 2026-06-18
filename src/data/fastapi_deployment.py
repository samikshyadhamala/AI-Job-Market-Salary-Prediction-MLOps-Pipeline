"""
STAGE 4: MODEL DEPLOYMENT - FastAPI with Redis Caching & MLflow Integration

Architecture:
1. Client sends POST /predict request (JSON with job features)
2. FastAPI receives request
3. Check Redis cache (key = hash of input features, TTL = 1 hour)
4. If cache HIT → return cached result (< 10ms)
5. If cache MISS → continue:
   - Load preprocessor (pickle)
   - Transform features
   - Load XGBoost model (pickle)
   - Make prediction
   - Store in Redis cache
   - Save transformed input to parquet (drift detection)
   - Log to MariaDB (monitoring)
   - Return prediction (JSON)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import json
import logging
import redis
import hashlib
import os
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import mlflow
import mlflow.xgboost
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Required when loading data/processed/preprocessor.pkl because it contains
# the custom IQRCapper class from the preprocessing module.
from data.preprocessing import IQRCapper  # noqa: F401


class PreprocessorUnpickler(pickle.Unpickler):
    """Load old preprocessors that were pickled from preprocessing.py as __main__."""

    def find_class(self, module, name):
        if module == "__main__" and name == "IQRCapper":
            return IQRCapper
        if module == "preprocessing" and name == "IQRCapper":
            return IQRCapper
        return super().find_class(module, name)

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Deployment configuration"""

    # === PATHS ===
    MODEL_PATH = Path("/home/samiksya/ai_job_market/data/processed/model_artifacts/xgboost_final_model.pkl")
    PREPROCESSOR_PATH = Path("/home/samiksya/ai_job_market/data/processed/preprocessor.pkl")
    PREPROCESSING_METADATA_PATH = Path("/home/samiksya/ai_job_market/data/processed/preprocessing_metadata.json")
    TRAIN_DATA_PATH = Path("/home/samiksya/ai_job_market/data/processed/train.parquet")
    DRIFT_DATA_PATH = Path("/home/samiksya/ai_job_market/data/processed/user_inputs.parquet")
    TARGET_COL = "salary_usd"

    # === FEATURE SCHEMA EXPECTED BY src/data/preprocessing.py ===
    NUMERICAL_COLS = [
        "years_experience",
        "remote_ratio",
        "benefits_score",
        "days_open",
        "num_skills",
        "same_country",
        "exp_x_skills",
        "benefits_x_exp",
    ]
    CATEGORICAL_COLS = [
        "job_title",
        "company_location",
        "company_size",
        "employee_residence",
        "education_required",
        "industry",
        "salary_currency",
    ]
    ORDINAL_COLS = ["experience_encoded"]
    BOOLEAN_COLS = [
        "emp_full_time",
        "emp_part_time",
        "emp_freelance",
        "remote_onsite",
        "remote_remote",
    ]
    RAW_MODEL_FEATURES = (
        NUMERICAL_COLS
        + CATEGORICAL_COLS
        + ORDINAL_COLS
        + BOOLEAN_COLS
    )
    TRANSFORMED_FEATURES = (
        NUMERICAL_COLS
        + ORDINAL_COLS
        + [f"{col}_encoded" for col in CATEGORICAL_COLS]
        + BOOLEAN_COLS
    )

    # === REDIS ===
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 9000
    REDIS_DB = 0
    REDIS_TTL_SECONDS = 3600  # 1 hour cache

    # === MLFLOW ===
    USE_MLFLOW_ARTIFACTS = True
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    MLFLOW_EXPERIMENT_NAME = "ai-job-salary-prediction"
    MLFLOW_PREPROCESSING_EXPERIMENT_NAME = "ai_jobs_preprocessing"
    MLFLOW_MODEL_ARTIFACT_PATH = "model"
    MLFLOW_PREPROCESSOR_ARTIFACT_PATH = "preprocessor.pkl"

    # === DATABASE (MariaDB) - for prediction monitoring ===
    DB_HOST = "127.0.0.1"
    DB_PORT = 3307
    DB_USER = "mariadbuser"
    DB_PASSWORD = "Samikshya@123"
    DB_NAME = "ai_jobs_raw"
    PREDICTIONS_TABLE = "salary_predictions"

    # === API ===
    API_HOST = "127.0.0.1"
    API_PORT = 8000


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Job Salary Prediction API",
    description="Predicts salary for AI/ML job positions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class JobFeaturesInput(BaseModel):
    """Input features used to recreate the training-time feature schema."""
    years_experience: int = Field(..., ge=0, le=50, description="Years of experience")
    remote_ratio: int = Field(..., ge=0, le=100, description="Remote work percentage (0, 50, 100)")
    benefits_score: float = Field(..., ge=0.0, le=10.0, description="Benefits score (0-10)")
    experience_level: str = Field(..., description="EN/MI/SE/EX")
    employment_type: str = Field(..., description="FT/PT/CT/FL")
    job_title: str = Field(..., description="Job title")
    company_location: str = Field(..., description="Country name")
    employee_residence: str = Field(..., description="Employee country/residence")
    company_size: str = Field(..., description="S/M/L")
    education_required: str = Field(..., description="Bachelor/Master/PhD/Associate")
    industry: str = Field(..., description="Industry name")
    salary_currency: str = Field(default="USD", description="Original salary currency")
    required_skills: List[str] = Field(default_factory=list, description="Required skills list")
    days_open: Optional[int] = Field(default=None, ge=0, le=3650, description="Posting duration in days")
    posting_date: Optional[str] = Field(default=None, description="YYYY-MM-DD posting date")
    application_deadline: Optional[str] = Field(default=None, description="YYYY-MM-DD application deadline")

    model_config = {
        "json_schema_extra": {
            "example": {
                "years_experience": 5,
                "remote_ratio": 100,
                "benefits_score": 8.5,
                "experience_level": "MI",
                "employment_type": "FT",
                "job_title": "Machine Learning Engineer",
                "company_location": "United States",
                "employee_residence": "United States",
                "company_size": "M",
                "education_required": "Master",
                "industry": "Technology",
                "salary_currency": "USD",
                "required_skills": ["Python", "Machine Learning", "SQL"],
                "days_open": 30
            }
        }
    }


class PredictionResponse(BaseModel):
    """API response"""
    predicted_salary_usd: float
    model_version: str
    timestamp: str
    confidence_score: Optional[float] = None
    from_cache: bool
    processing_time_ms: float


# ============================================================
# FEATURE PREPARATION
# ============================================================

class FeatureBuilder:
    """Build the exact raw feature frame expected by the fitted preprocessor."""

    EXPERIENCE_MAP = {
        "EN": 0,
        "entry_level": 0,
        "MI": 1,
        "mid_level": 1,
        "SE": 2,
        "senior": 2,
        "lead": 3,
        "EX": 4,
        "executive": 4,
    }

    @staticmethod
    def prepare(features: JobFeaturesInput) -> pd.DataFrame:
        raw = features.model_dump()

        num_skills = FeatureBuilder._count_skills(raw.get("required_skills", []))
        days_open = FeatureBuilder._resolve_days_open(
            raw.get("days_open"),
            raw.get("posting_date"),
            raw.get("application_deadline"),
        )
        experience_encoded = FeatureBuilder.EXPERIENCE_MAP.get(
            str(raw["experience_level"])
        ) 
        
        if experience_encoded is None:
            raise ValueError(
                "Unsupported experience_level. Use one of: EN, MI, SE, EX, "
                "entry_level, mid_level, senior, lead, executive"
            )

        employment_type = str(raw["employment_type"])
        remote_ratio = int(raw["remote_ratio"])
        years_experience = int(raw["years_experience"])
        benefits_score = float(raw["benefits_score"])

        row = {
            "years_experience": years_experience,
            "remote_ratio": remote_ratio,
            "benefits_score": benefits_score,
            "days_open": days_open,
            "num_skills": num_skills,
            "same_country": int(raw["company_location"] == raw["employee_residence"]),
            "exp_x_skills": years_experience * num_skills,
            "benefits_x_exp": benefits_score * years_experience,
            "job_title": raw["job_title"],
            "company_location": raw["company_location"],
            "company_size": raw["company_size"],
            "employee_residence": raw["employee_residence"],
            "education_required": raw["education_required"],
            "industry": raw["industry"],
            "salary_currency": raw["salary_currency"],
            "experience_encoded": experience_encoded,
            "emp_full_time": int(employment_type in ["FT", "full_time"]),
            "emp_part_time": int(employment_type in ["PT", "part_time"]),
            "emp_freelance": int(employment_type in ["FL", "freelance"]),
            "remote_onsite": int(remote_ratio == 0),
            "remote_remote": int(remote_ratio == 100),
        }

        features_df = pd.DataFrame([row], columns=Config.RAW_MODEL_FEATURES)
        FeatureBuilder.validate_schema(features_df)
        return features_df

    @staticmethod
    def validate_schema(features_df: pd.DataFrame) -> None:
        actual_cols = list(features_df.columns)
        expected_cols = Config.RAW_MODEL_FEATURES

        if actual_cols != expected_cols:
            raise ValueError(
                "Feature schema mismatch. "
                f"Expected {expected_cols}, got {actual_cols}"
            )

        missing_values = features_df.isnull().sum()
        missing_cols = missing_values[missing_values > 0].index.tolist()
        if missing_cols:
            raise ValueError(f"Missing values found in model features: {missing_cols}")

    @staticmethod
    def transformed_feature_names() -> List[str]:
        if Config.PREPROCESSING_METADATA_PATH.exists():
            try:
                with open(Config.PREPROCESSING_METADATA_PATH, "r") as f:
                    metadata = json.load(f)

                numerical_cols = metadata.get("numerical_cols", [])
                encoded_cols = metadata.get("encoded_cols", [])
                boolean_cols = metadata.get("boolean_cols", [])
                names = numerical_cols + encoded_cols + boolean_cols

                if names:
                    return names
            except Exception as e:
                logger.warning(f"Failed to read preprocessing metadata: {str(e)}")

        return Config.TRANSFORMED_FEATURES

    @staticmethod
    def drift_feature_names() -> List[str]:
        if Config.TRAIN_DATA_PATH.exists():
            try:
                train_columns = pd.read_parquet(Config.TRAIN_DATA_PATH).columns.tolist()
                names = [col for col in train_columns if col != Config.TARGET_COL]
                if names:
                    return names
            except Exception as e:
                logger.warning(f"Failed to read train schema for drift: {str(e)}")

        return FeatureBuilder.transformed_feature_names()

    @staticmethod
    def _count_skills(required_skills: List[str]) -> int:
        if required_skills is None:
            return 0

        if isinstance(required_skills, str):
            return len([s for s in required_skills.split(",") if s.strip()])

        return len([s for s in required_skills if str(s).strip()])

    @staticmethod
    def _resolve_days_open(
        days_open: Optional[int],
        posting_date: Optional[str],
        application_deadline: Optional[str],
    ) -> int:
        if days_open is not None:
            return int(days_open)

        if posting_date and application_deadline:
            start = pd.to_datetime(posting_date, errors="coerce")
            end = pd.to_datetime(application_deadline, errors="coerce")
            if pd.isna(start) or pd.isna(end):
                raise ValueError("Invalid posting_date or application_deadline")

            delta = int((end - start).days)
            if delta < 0:
                raise ValueError("application_deadline must be on or after posting_date")
            return delta

        raise ValueError(
            "Provide either days_open or both posting_date and application_deadline"
        )


# ============================================================
# REDIS CACHE
# ============================================================

class RedisCache:
    """Redis cache manager"""

    def __init__(self):
        try:
            self.client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"✓ Connected to Redis: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
            self.connected = True
        except Exception as e:
            logger.warning(f"✗ Redis connection failed: {str(e)}")
            self.connected = False

    def get_cache_key(self, features: Dict) -> str:
        feature_str = json.dumps(features, sort_keys=True)
        return f"prediction_{hashlib.md5(feature_str.encode()).hexdigest()}"

    def get(self, key: str) -> Optional[Dict]:
        if not self.connected:
            return None
        try:
            value = self.client.get(key)
            if value:
                logger.info(f"✓ Cache HIT: {key}")
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache GET error: {str(e)}")
            return None

    def set(self, key: str, value: Dict, ttl: int = Config.REDIS_TTL_SECONDS):
        if not self.connected:
            return False
        try:
            self.client.setex(key, ttl, json.dumps(value))
            logger.info(f"✓ Cache SET: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache SET error: {str(e)}")
            return False

    def clear(self):
        if not self.connected:
            return False
        try:
            keys = self.client.keys("prediction_*")
            if keys:
                self.client.delete(*keys)
                logger.info(f"✓ Cleared {len(keys)} cache entries")
            return True
        except Exception as e:
            logger.warning(f"Cache CLEAR error: {str(e)}")
            return False


redis_cache = RedisCache()


# ============================================================
# MODEL LOADER
# ============================================================

class ModelLoader:
    """Load the latest MLflow artifacts, with local files as fallback."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_version = None
        self.model_source = None
        self.preprocessor_source = None
        self._latest_training_run_id = None
        self._load_model()
        self._load_preprocessor()

    def _load_model(self):
        if Config.USE_MLFLOW_ARTIFACTS and self._load_model_from_mlflow():
            return

        try:
            if not Config.MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {Config.MODEL_PATH}")
            with open(Config.MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            self.model_source = str(Config.MODEL_PATH)
            logger.info(f"✓ Model loaded: {Config.MODEL_PATH}")

            # Load version from metadata
            metadata_path = Config.MODEL_PATH.parent / "model_metrics.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.model_version = metadata.get('timestamp', 'unknown')
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")

    def _load_preprocessor(self):
        if Config.USE_MLFLOW_ARTIFACTS and self._load_preprocessor_from_mlflow():
            self._validate_preprocessor_schema()
            self._validate_model_feature_count()
            return

        try:
            if not Config.PREPROCESSOR_PATH.exists():
                raise FileNotFoundError(f"Preprocessor not found: {Config.PREPROCESSOR_PATH}")
            with open(Config.PREPROCESSOR_PATH, 'rb') as f:
                self.preprocessor = PreprocessorUnpickler(f).load()
            self.preprocessor_source = str(Config.PREPROCESSOR_PATH)
            logger.info(f"✓ Preprocessor loaded: {Config.PREPROCESSOR_PATH}")
            self._validate_preprocessor_schema()
            self._validate_model_feature_count()
        except Exception as e:
            logger.error(f"✗ Failed to load preprocessor: {str(e)}")

    def _load_model_from_mlflow(self) -> bool:
        runs = self._latest_finished_runs(Config.MLFLOW_EXPERIMENT_NAME)
        if not runs:
            logger.warning(
                "No finished MLflow model run found in experiment %s",
                Config.MLFLOW_EXPERIMENT_NAME,
            )
            return False

        for run in runs:
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/{Config.MLFLOW_MODEL_ARTIFACT_PATH}"
            try:
                mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
                self.model = mlflow.xgboost.load_model(model_uri)
                self.model_version = run_id
                self.model_source = model_uri
                self._latest_training_run_id = run_id
                logger.info("✓ Model loaded from MLflow: %s", model_uri)
                return True
            except Exception as e:
                logger.warning("Could not load MLflow model %s: %s", model_uri, e)

            try:
                local_path = self._download_mlflow_artifact(run_id, Config.MODEL_PATH.name)
                with open(local_path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_version = run_id
                self.model_source = f"runs:/{run_id}/{Config.MODEL_PATH.name}"
                self._latest_training_run_id = run_id
                logger.info("✓ Pickle model loaded from MLflow artifact: %s", self.model_source)
                return True
            except Exception as e:
                logger.warning("Could not load pickle model from MLflow run %s: %s", run_id, e)

        return False

    def _load_preprocessor_from_mlflow(self) -> bool:
        candidate_runs = []
        if self._latest_training_run_id:
            candidate_runs.append(
                (self._latest_training_run_id, Config.MLFLOW_EXPERIMENT_NAME)
            )

        for preprocessing_run in self._latest_finished_runs(
            Config.MLFLOW_PREPROCESSING_EXPERIMENT_NAME
        ):
            candidate_runs.append(
                (
                    preprocessing_run.info.run_id,
                    Config.MLFLOW_PREPROCESSING_EXPERIMENT_NAME,
                )
            )

        for run_id, experiment_name in candidate_runs:
            try:
                local_path = self._download_mlflow_artifact(
                    run_id,
                    Config.MLFLOW_PREPROCESSOR_ARTIFACT_PATH,
                )
                with open(local_path, "rb") as f:
                    self.preprocessor = PreprocessorUnpickler(f).load()
                self.preprocessor_source = (
                    f"runs:/{run_id}/{Config.MLFLOW_PREPROCESSOR_ARTIFACT_PATH}"
                )
                logger.info(
                    "✓ Preprocessor loaded from MLflow experiment %s: %s",
                    experiment_name,
                    self.preprocessor_source,
                )
                return True
            except Exception as e:
                logger.warning(
                    "Could not load preprocessor from MLflow run %s (%s): %s",
                    run_id,
                    experiment_name,
                    e,
                )

        return False

    @staticmethod
    def _latest_finished_runs(experiment_name: str):
        try:
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient(Config.MLFLOW_TRACKING_URI)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                return []
            return client.search_runs(
                [experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["metrics.rmse ASC"],
                max_results=10,
            )
        except Exception as e:
            logger.warning("Could not query MLflow experiment %s: %s", experiment_name, e)
            return []

    @staticmethod
    def _download_mlflow_artifact(run_id: str, artifact_path: str) -> str:
        client = mlflow.tracking.MlflowClient(Config.MLFLOW_TRACKING_URI)
        dst_path = tempfile.mkdtemp(prefix="ai_job_mlflow_")
        return client.download_artifacts(run_id, artifact_path, dst_path)

    def predict(self, features_df: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor not loaded. Restart FastAPI and check that "
                "data/processed/preprocessor.pkl can be imported with "
                "data.preprocessing.IQRCapper."
            )
        FeatureBuilder.validate_schema(features_df)
        features_transformed = self.preprocessor.transform(features_df)
        predictions = self.model.predict(features_transformed)
        return float(predictions[0])

    def _validate_preprocessor_schema(self):
        expected = self._get_preprocessor_input_features()
        if not expected:
            logger.warning("Could not inspect preprocessor input schema")
            return

        if expected != Config.RAW_MODEL_FEATURES:
            logger.warning(
                "Preprocessor input schema differs from API schema. "
                f"preprocessor={expected}, api={Config.RAW_MODEL_FEATURES}"
            )
        else:
            logger.info("✓ API feature schema matches preprocessor input schema")

    def _get_preprocessor_input_features(self) -> List[str]:
        if self.preprocessor is None:
            return []

        if hasattr(self.preprocessor, "named_steps"):
            feature_step = self.preprocessor.named_steps.get("features")
            if feature_step is not None and hasattr(feature_step, "feature_names_in_"):
                return list(feature_step.feature_names_in_)

        if hasattr(self.preprocessor, "feature_names_in_"):
            return list(self.preprocessor.feature_names_in_)

        return []

    def _validate_model_feature_count(self):
        if self.model is None:
            return

        model_feature_count = getattr(self.model, "n_features_in_", None)
        if model_feature_count is None:
            return

        transformed_feature_count = len(FeatureBuilder.transformed_feature_names())
        if model_feature_count != transformed_feature_count:
            logger.warning(
                "Model feature count differs from transformed preprocessing features. "
                f"model={model_feature_count}, preprocessor={transformed_feature_count}"
            )
        else:
            logger.info("✓ Model feature count matches transformed preprocessor output")


# Initialize model loader
model_loader = ModelLoader()


# ============================================================
# DRIFT DETECTION - SAVE TRANSFORMED INPUT TO PARQUET
# ============================================================

def save_input_for_drift(features_df: pd.DataFrame, prediction: float):
    """
    Save preprocessed user input to parquet for drift detection.
    Same transformed features as train.parquet so distributions can be compared directly.
    """
    try:
        if model_loader.preprocessor is None:
            logger.warning("Preprocessor not loaded — skipping drift save")
            return

        # Transform using same preprocessor as training
        transformed = model_loader.preprocessor.transform(features_df)

        all_cols = FeatureBuilder.transformed_feature_names()
        if transformed.shape[1] != len(all_cols):
            all_cols = [f"feature_{i}" for i in range(transformed.shape[1])]

        # Build dataframe
        input_df = pd.DataFrame(transformed, columns=all_cols)
        drift_feature_cols = FeatureBuilder.drift_feature_names()
        missing_drift_cols = [
            col for col in drift_feature_cols if col not in input_df.columns
        ]
        if missing_drift_cols:
            logger.warning(
                "Drift row is missing training features; falling back to "
                f"preprocessor output columns. Missing={missing_drift_cols}"
            )
            drift_feature_cols = all_cols

        input_df['predicted_salary'] = prediction
        input_df['timestamp'] = datetime.utcnow().isoformat()
        drift_columns = drift_feature_cols + ['predicted_salary', 'timestamp']
        input_df = input_df.reindex(columns=drift_columns)

        # Append to existing parquet or create new
        drift_path = Config.DRIFT_DATA_PATH
        if drift_path.exists():
            existing = pd.read_parquet(drift_path)
            extra_cols = [col for col in existing.columns if col not in drift_columns]
            missing_cols = [col for col in drift_columns if col not in existing.columns]
            if extra_cols or missing_cols:
                logger.warning(
                    "Aligning existing drift parquet schema before append. "
                    f"Dropping extra columns={extra_cols}; adding missing columns={missing_cols}"
                )
            existing = existing.reindex(columns=drift_columns)
            combined = pd.concat([existing, input_df], ignore_index=True)
        else:
            combined = input_df

        combined = combined.reindex(columns=drift_columns)
        combined.to_parquet(drift_path, index=False)
        logger.info(f"✓ Drift parquet updated: {len(combined)} total records")

    except Exception as e:
        logger.warning(f"Drift save failed: {str(e)}")


# ============================================================
# PREDICTION LOGGER - MARIADB MONITORING ONLY
# ============================================================

class PredictionLogger:
    """Log predictions to MariaDB for monitoring"""

    @staticmethod
    def log_prediction(
        input_features: Dict,
        prediction: float,
        model_version: str,
        processing_time_ms: float,
        from_cache: bool
    ):
        try:
            password = quote_plus(Config.DB_PASSWORD)
            engine = create_engine(
                f"mysql+pymysql://{Config.DB_USER}:{password}@"
                f"{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
            )
            with engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {Config.PREDICTIONS_TABLE} (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        input_features JSON,
                        predicted_salary FLOAT,
                        model_version VARCHAR(100),
                        processing_time_ms FLOAT,
                        from_cache TINYINT(1),
                        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        KEY idx_timestamp (prediction_timestamp)
                    )
                """))
                conn.execute(text(f"""
                    INSERT INTO {Config.PREDICTIONS_TABLE}
                    (input_features, predicted_salary, model_version, processing_time_ms, from_cache)
                    VALUES (:features, :salary, :version, :time, :cache)
                """), {
                    'features': json.dumps(input_features),
                    'salary': float(prediction),
                    'version': model_version,
                    'time': float(processing_time_ms),
                    'cache': 1 if from_cache else 0
                })
                conn.commit()
                logger.info(f"✓ Prediction logged to MariaDB")
        except Exception as e:
            logger.warning(f"Failed to log to MariaDB: {str(e)}")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health", tags=["Health"])
async def health_check() -> Dict:
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
        "model_source": model_loader.model_source,
        "preprocessor_source": model_loader.preprocessor_source,
        "redis_connected": redis_cache.connected,
        "drift_parquet_exists": Config.DRIFT_DATA_PATH.exists()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    features: JobFeaturesInput,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """
    Predict salary.
    Cache HIT → returns in < 10ms
    Cache MISS → preprocesses, predicts, caches, saves drift data, logs to DB
    """
    import time
    start_time = time.time()

    try:
        features_dict = features.model_dump()
        cache_key = redis_cache.get_cache_key(features_dict)

        # Check cache first
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            cached_result['processing_time_ms'] = (time.time() - start_time) * 1000
            cached_result['from_cache'] = True
            return PredictionResponse(**cached_result)

        # Build dataframe matching the raw training-time preprocessor schema
        features_df = FeatureBuilder.prepare(features)

        # Predict
        prediction = model_loader.predict(features_df)

        response_dict = {
            'predicted_salary_usd': prediction,
            'model_version': model_loader.model_version or 'unknown',
            'timestamp': datetime.utcnow().isoformat(),
            'confidence_score': 0.90,
            'from_cache': False,
            'processing_time_ms': (time.time() - start_time) * 1000
        }

        # Cache result
        redis_cache.set(cache_key, response_dict)

        # Background: save transformed input to parquet for drift detection
        background_tasks.add_task(save_input_for_drift, features_df, prediction)

        # Background: log to MariaDB for monitoring
        # background_tasks.add_task(
        #     PredictionLogger.log_prediction,
        #     features_df.iloc[0].to_dict(), prediction,
        #     model_loader.model_version,
        #     response_dict['processing_time_ms'], False
        # )

        return PredictionResponse(**response_dict)

    except ValueError as e:
        logger.warning(f"Invalid prediction input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/predictions/history", tags=["History"])
# async def get_prediction_history(limit: int = 100) -> List[Dict]:
#     """Get recent predictions from MariaDB"""
#     try:
#         password = quote_plus(Config.DB_PASSWORD)
#         engine = create_engine(
#             f"mysql+pymysql://{Config.DB_USER}:{password}@"
#             f"{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
#         )
#         with engine.connect() as conn:
#             result = conn.execute(text(f"""
#                 SELECT * FROM {Config.PREDICTIONS_TABLE}
#                 ORDER BY prediction_timestamp DESC
#                 LIMIT :limit
#             """), {'limit': limit})
#             columns = result.keys()
#             return [dict(zip(columns, row)) for row in result.fetchall()]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/status", tags=["Drift"])
async def get_drift_status() -> Dict:
    """Check drift parquet status"""
    drift_path = Config.DRIFT_DATA_PATH
    if not drift_path.exists():
        return {"exists": False, "records": 0}
    df = pd.read_parquet(drift_path)
    return {
        "exists": True,
        "records": len(df),
        "columns": len(df.columns),
        "latest": df['timestamp'].max() if 'timestamp' in df.columns else None
    }


@app.post("/cache/clear", tags=["Cache"])
async def clear_cache() -> Dict:
    """Clear Redis cache"""
    success = redis_cache.clear()
    return {'status': 'cleared' if success else 'failed', 'timestamp': datetime.utcnow().isoformat()}


@app.get("/model/info", tags=["Model"])
async def get_model_info() -> Dict:
    """Get model info"""
    return {
        'model_type': 'XGBRegressor',
        'model_version': model_loader.model_version,
        'model_path': str(Config.MODEL_PATH),
        'model_source': model_loader.model_source,
        'preprocessor_path': str(Config.PREPROCESSOR_PATH),
        'preprocessor_source': model_loader.preprocessor_source,
        'preprocessor_loaded': model_loader.preprocessor is not None,
        'mlflow_enabled': Config.USE_MLFLOW_ARTIFACTS,
        'mlflow_tracking_uri': Config.MLFLOW_TRACKING_URI,
        'mlflow_training_experiment': Config.MLFLOW_EXPERIMENT_NAME,
        'mlflow_preprocessing_experiment': Config.MLFLOW_PREPROCESSING_EXPERIMENT_NAME,
        'raw_model_features': Config.RAW_MODEL_FEATURES,
        'transformed_features': FeatureBuilder.transformed_feature_names(),
        'cache_ttl_seconds': Config.REDIS_TTL_SECONDS,
        'drift_data_path': str(Config.DRIFT_DATA_PATH)
    }


# ============================================================
# STARTUP & SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 70)
    logger.info("STARTING FASTAPI DEPLOYMENT SERVER")
    logger.info("=" * 70)
    logger.info(f"✓ Model loaded: {model_loader.model is not None}")
    logger.info(f"✓ Preprocessor loaded: {model_loader.preprocessor is not None}")
    logger.info(f"✓ Redis connected: {redis_cache.connected}")
    logger.info(f"✓ Drift path: {Config.DRIFT_DATA_PATH}")
    logger.info(f"✓ Docs: http://{Config.API_HOST}:{Config.API_PORT}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("SHUTTING DOWN API")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_deployment:app", host=Config.API_HOST, port=Config.API_PORT)



