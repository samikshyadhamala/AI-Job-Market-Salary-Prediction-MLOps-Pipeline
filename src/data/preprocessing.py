"""
STAGE 2: DATA PREPROCESSING PIPELINE (STANDALONE)
Load from MariaDB → Profile → Clean → Engineer → Validate → Save

NO CONFIG.YAML REQUIRED - All settings hardcoded

Features:
- Data profiling with ydata-profiling (fallback to JSON)
- Outlier handling (IQR method)
- Feature engineering (domain-specific)
- Train/test split
- Sklearn preprocessing pipeline
- Data validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import quote_plus
from datetime import datetime
import logging
import json
import os
import pickle
import sys
from typing import Tuple
import mlflow 


from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ============================================================
# HARDCODED CONFIGURATION (No config.yaml needed!)
# ============================================================

class Config:
    """Standalone configuration - modify these values for your setup"""
    
    # === DATABASE SETTINGS ===
    DB_HOST = "127.0.0.1"              # Your Docker host
    DB_PORT = 3307                # Your MariaDB port
    DB_USER = "mariadbuser"            # Your MariaDB user
    DB_PASSWORD = "Samikshya@123"         # Your MariaDB password
    DB_NAME = "ai_jobs_raw"            # Your database name
    DB_TABLE = "raw_ai_jobs"           # Your raw table name
    
    # === DATA PATHS ===
    RAW_DATA_DIR = "/home/samiksya/ai_job_market/data/raw"
    PROCESSED_DATA_DIR = "/home/samiksya/ai_job_market/data/processed"
    LOGS_DIR = "logs"
    
    # === COLUMN NAMES (Match your actual database columns) ===
    TARGET_COL = "salary_usd"
    
    NUMERICAL_COLS = [
    'years_experience',
    'remote_ratio',
    'benefits_score',
    'days_open',
    'num_skills',
    'same_country',
    'exp_x_skills',
    'benefits_x_exp'
]
    
    CATEGORICAL_COLS = [
    'job_title',
    'company_location',
    'company_size',
    'employee_residence',
    'education_required',
    'industry',
    'salary_currency'
]
    
    # Columns to drop (metadata, not needed for ML)
    ORDINAL_COLS = [
    'experience_encoded'
    ]

    CATEGORICAL_ENCODED_COLS = [
    f"{col}_encoded"
    for col in CATEGORICAL_COLS
]

    ENCODED_COLS = ORDINAL_COLS + CATEGORICAL_ENCODED_COLS
    BOOLEAN_COLS = [
    'emp_full_time',
    'emp_part_time',
    'emp_freelance',
    'remote_onsite',
    'remote_remote'
    ]
    COLS_TO_DROP = [
    "job_id",
    "job_description_length",
    "batch_id",
    "company_name",
    "ingested_at"
]
    MODEL_FEATURES = (
    NUMERICAL_COLS
    + CATEGORICAL_COLS
    + ORDINAL_COLS
    + BOOLEAN_COLS
)

    # === PREPROCESSING SETTINGS ===
    TEST_SIZE = 0.2                    # 20% test, 80% train
    RANDOM_STATE = 42                  # For reproducibility
    
    # === OUTPUT FILES ===
    PROFILE_REPORT_PATH = Path(PROCESSED_DATA_DIR) / "profiling_report.html"
    PREPROCESSOR_PATH = Path(PROCESSED_DATA_DIR) / "preprocessor.pkl"
    METADATA_PATH = Path(PROCESSED_DATA_DIR) / "preprocessing_metadata.json"
    TRAIN_PARQUET_PATH = Path(PROCESSED_DATA_DIR) / "train.parquet"
    TEST_PARQUET_PATH = Path(PROCESSED_DATA_DIR) / "test.parquet"
    # === MLFLOW ===
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    MLFLOW_EXPERIMENT_NAME = "ai_jobs_preprocessing"


# ============================================================
# DATA LOADER
# ============================================================

class DataLoader:
    """Load data from MariaDB"""
    
    @staticmethod
    def load_from_mariadb() -> pd.DataFrame:
        """Load raw data from MariaDB"""
        try:
            logger.info("Connecting to MariaDB...")
            logger.info(f"  Host: {Config.DB_HOST}:{Config.DB_PORT}")
            logger.info(f"  Database: {Config.DB_NAME}")
            logger.info(f"  Table: {Config.DB_TABLE}")
            
            # Build connection string
            password = quote_plus(Config.DB_PASSWORD)
            connection_string = (
                f"mysql+pymysql://{Config.DB_USER}:{password}@"
                f"{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
            )
            
            # Connect and load data
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {Config.DB_TABLE}"))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.info(f"✓ Successfully loaded {len(df):,} rows from MariaDB")
            logger.info(f"  Columns ({len(df.columns)}): {list(df.columns)}")
            
            return df
        
        except Exception as e:
            logger.error(f"✗ Failed to load from MariaDB: {str(e)}")
            raise



# ============================================================
# DATA PROFILER
# ============================================================

class DataProfiler:
    """Generate data profiling report"""
    
    @staticmethod
    def run_profiling(df: pd.DataFrame):
        """Generate profiling report"""
        try:
            # Try to use ydata-profiling if available
            from ydata_profiling import ProfileReport
            
            logger.info("Generating HTML profiling report (ydata-profiling)...")
            profile = ProfileReport(
                df,
                title="AI Jobs Dataset - Profiling Report",
                minimal=True
            )
            profile.to_file(Config.PROFILE_REPORT_PATH)
            logger.info(f"✓ Profiling report saved to {Config.PROFILE_REPORT_PATH}")
        
        except ImportError:
            logger.warning("ydata-profiling not installed. Creating JSON report instead...")
            logger.info("  To install: pip install ydata-profiling")
            
            # Fallback: Generate JSON profiling report
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'shape': list(df.shape),
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isna().sum().to_dict(),
                'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict(),
                'duplicate_rows': int(df.duplicated().sum()),
                'data_types': df.dtypes.astype(str).to_dict(),
                'numeric_stats': {
                    col: {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        '25%': float(df[col].quantile(0.25)),
                        '50%': float(df[col].quantile(0.50)),
                        '75%': float(df[col].quantile(0.75))
                    }
                    for col in df.select_dtypes(include=['int64', 'float64']).columns
                }
            }
            
            json_path = Path(Config.PROCESSED_DATA_DIR) / "profiling_report.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"✓ JSON profiling report saved to {json_path}")


# ============================================================
# DATA CLEANER
# ============================================================

class DataCleaner:
    """Clean and engineer features"""
    
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset"""
        logger.info(f"Starting with: {len(df):,} rows, {len(df.columns)} columns")
        df = df.copy()
        
        # 1. Remove duplicates
        initial_rows = len(df)
        if 'job_id' in df.columns:
            df = df.drop_duplicates(subset=['job_id'])
        else:
            df = df.drop_duplicates()
        
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"✓ Removed {duplicates_removed:,} duplicate rows")
        
        # 2. Drop unnecessary columns
        cols_to_drop = [c for c in Config.COLS_TO_DROP if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"✓ Dropped {len(cols_to_drop)} metadata columns")
        
        # 3. Handle missing values in critical columns
        critical_cols = [c for c in [Config.TARGET_COL, 'experience_level', 'employment_type'] 
                        if c in df.columns]
        initial_rows = len(df)
        df = df.dropna(subset=critical_cols)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logger.info(f"✓ Dropped {rows_dropped:,} rows with missing critical values")
        
        logger.info(
            "Outlier capping will be fitted on train data inside the preprocessing pipeline"
        )
        
        logger.info(f"After cleaning: {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ONLY the features used during model training
        """
        logger.info("Engineering model features...")
        df = df.copy()

        # --------------------------------------------------
        # Feature 1: Days open
        # --------------------------------------------------
        if {'posting_date', 'application_deadline'}.issubset(df.columns):
            posting_date = pd.to_datetime(
                df['posting_date'],
                errors='coerce'
            )
            application_deadline = pd.to_datetime(
                df['application_deadline'],
                errors='coerce'
            )
            df['days_open'] = (
                application_deadline - posting_date
            ).dt.days
            df.loc[df['days_open'] < 0, 'days_open'] = np.nan

        # --------------------------------------------------
        # Feature 2: Number of skills
        # --------------------------------------------------
        if 'required_skills' in df.columns:
            df['num_skills'] = (
                df['required_skills']
                .fillna('')
                .astype(str)
                .apply(lambda x: len([s for s in x.split(',') if s.strip()]))
            )

        # --------------------------------------------------
        # Feature 3: Experience encoding
        # --------------------------------------------------
        if 'experience_level' in df.columns:
            experience_map = {
                'EN': 0,
                'entry_level': 0,
                'MI': 1,
                'mid_level': 1,
                'SE': 2,
                'senior': 2,
                'lead': 3,
                'EX': 4,
                'executive': 4
            }
            df['experience_encoded'] = (
                df['experience_level']
                .astype(str)
                .map(experience_map)
            )

        # --------------------------------------------------
        # Feature 4: Employment type dummies
        # --------------------------------------------------
        for col in [
            'emp_full_time',
            'emp_part_time',
            'emp_freelance'
        ]:
            df[col] = 0

        if 'employment_type' in df.columns:
            employment_type = df['employment_type'].astype(str)
            df['emp_full_time'] = employment_type.isin(
                ['FT', 'full_time']
            ).astype(int)
            df['emp_part_time'] = employment_type.isin(
                ['PT', 'part_time']
            ).astype(int)
            df['emp_freelance'] = employment_type.isin(
                ['FL', 'freelance']
            ).astype(int)

        # --------------------------------------------------
        # Feature 5: Remote type dummies
        # --------------------------------------------------
        df['remote_onsite'] = 0
        df['remote_remote'] = 0

        if 'remote_ratio' in df.columns:
            remote_ratio = pd.to_numeric(
                df['remote_ratio'],
                errors='coerce'
            )
            df['remote_onsite'] = (remote_ratio == 0).astype(int)
            df['remote_remote'] = (remote_ratio == 100).astype(int)

        # --------------------------------------------------
        # Feature 6: Same country
        # --------------------------------------------------
        if {'company_location', 'employee_residence'}.issubset(df.columns):
            df['same_country'] = (
                df['company_location'] == df['employee_residence']
            ).astype(int)

        # --------------------------------------------------
        # Feature 7: Experience × Skills
        # --------------------------------------------------
        if {'years_experience', 'num_skills'}.issubset(df.columns):
            df['exp_x_skills'] = (
                df['years_experience'] * df['num_skills']
            )

        # --------------------------------------------------
        # Feature 8: Benefits × Experience
        # --------------------------------------------------
        if {'benefits_score', 'years_experience'}.issubset(df.columns):
            df['benefits_x_exp'] = (
                df['benefits_score'] * df['years_experience']
            )

        logger.info("✓ Created model-engineered features")

        return df


# ============================================================
# DATA SPLITTER
# ============================================================

class DataSplitter:
    """Split data into train/test sets"""
    
    @staticmethod
    def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data"""
        logger.info(f"Splitting data: {int((1-Config.TEST_SIZE)*100)}% train, {int(Config.TEST_SIZE*100)}% test")
        
        train_df, test_df = train_test_split(
            df,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            shuffle=True
        )
        
        logger.info(f"✓ Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
        return train_df, test_df


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers using bounds learned from the training data."""

    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X_array, 25, axis=0)
        q3 = np.nanpercentile(X_array, 75, axis=0)
        iqr = q3 - q1

        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return np.clip(
            X_array,
            self.lower_bounds_,
            self.upper_bounds_
        )


sys.modules.setdefault("data.preprocessing", sys.modules[__name__])
IQRCapper.__module__ = "data.preprocessing"


class PreprocessingPipeline:
    """Sklearn preprocessing pipeline"""

    def __init__(self):
        self.preprocessor = None

        self.numerical_cols = []
        self.ordinal_cols = []
        self.boolean_cols = []
        self.categorical_cols = []
        self.encoded_cols = []

    def build(self, X_train: pd.DataFrame):

        logger.info("Building preprocessing pipeline...")

        self.numerical_cols = [
            c for c in Config.NUMERICAL_COLS
            if c in X_train.columns
        ]

        self.ordinal_cols = [
            c for c in Config.ORDINAL_COLS
            if c in X_train.columns
        ]

        self.boolean_cols = [
            c for c in Config.BOOLEAN_COLS
            if c in X_train.columns
        ]

        self.categorical_cols = [
            c for c in Config.CATEGORICAL_COLS
            if c in X_train.columns
        ]

        self.encoded_cols = (
            self.ordinal_cols
            + [
                f"{c}_encoded"
                for c in self.categorical_cols
            ]
        )

        logger.info(
            f"Numerical ({len(self.numerical_cols)}): "
            f"{self.numerical_cols}"
        )

        logger.info(
            f"Ordinal ({len(self.ordinal_cols)}): "
            f"{self.ordinal_cols}"
        )

        logger.info(
            f"Boolean ({len(self.boolean_cols)}): "
            f"{self.boolean_cols}"
        )

        logger.info(
            f"Categorical ({len(self.categorical_cols)}): "
            f"{self.categorical_cols}"
        )

        if not self.numerical_cols:
            raise ValueError("No numerical columns found for preprocessing")

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("outlier_capper", IQRCapper())
            ]
        )

        ordinal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    )
                )
            ]
        )

        boolean_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ]
        )

        feature_transformer = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, self.numerical_cols),
                ("ordinal", ordinal_pipeline, self.ordinal_cols),
                ("cat", categorical_pipeline, self.categorical_cols),
                ("bool", boolean_pipeline, self.boolean_cols)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        self.preprocessor = Pipeline(
            steps=[
                ("features", feature_transformer),
                ("scaler", StandardScaler())
            ]
        )

        self.preprocessor.fit(X_train)

        self.feature_names = self._build_feature_names()

        logger.info(
            "✓ Preprocessing pipeline built and fitted"
        )

    def transform(self, X: pd.DataFrame):
        if self.preprocessor is None:
            raise RuntimeError("Preprocessing pipeline has not been fitted")

        return self.preprocessor.transform(X)

    def save(self):

        with open(Config.PREPROCESSOR_PATH, "wb") as f:
            pickle.dump(self.preprocessor, f)

        logger.info(
            f"✓ Preprocessor saved to "
            f"{Config.PREPROCESSOR_PATH}"
        )

    def get_feature_names(self):
        return self.feature_names

    def _build_feature_names(self):
        return (
            self.numerical_cols
            + self.encoded_cols
            + self.boolean_cols
        )


# ============================================================
# DATA TRANSFORMER
# ============================================================

class DataTransformer:
    """Transform train/test data using a fitted preprocessing pipeline"""
    
    @staticmethod
    def transform(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, PreprocessingPipeline]:
        """Fit preprocessing on train data and transform train/test data."""
        logger.info("Transforming data...")
        
        # Separate features and target
        missing_cols = [
            col
            for col in Config.MODEL_FEATURES
            if col not in train_df.columns
            or col not in test_df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required model features: "
                f"{missing_cols}"
            )

        X_train = train_df[
            Config.MODEL_FEATURES
        ]

        X_test = test_df[
            Config.MODEL_FEATURES
        ]

        y_train = train_df[
            Config.TARGET_COL
        ]

        y_test = test_df[
            Config.TARGET_COL
        ]

        y_train = pd.to_numeric(
            y_train,
            errors="coerce"
        )

        y_test = pd.to_numeric(
            y_test,
            errors="coerce"
        )

        # Build pipeline
        pipeline = PreprocessingPipeline()
        pipeline.build(X_train)
        
        # Transform
        X_train_t = pipeline.transform(X_train)
        X_test_t = pipeline.transform(X_test)
        
        # Convert to DataFrame
        feature_names = pipeline.get_feature_names()
        
        train_final = pd.DataFrame(X_train_t, columns=feature_names)
        train_final[Config.TARGET_COL] = y_train.values
        
        test_final = pd.DataFrame(X_test_t, columns=feature_names)
        test_final[Config.TARGET_COL] = y_test.values
        
        return train_final, test_final, pipeline

    @staticmethod
    def transform_and_save(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, PreprocessingPipeline]:
        """Backward-compatible wrapper for older callers."""
        return DataTransformer.transform(
            train_df,
            test_df
        )


# ============================================================
# DATA VALIDATOR
# ============================================================

class DataValidator:
    """Validate preprocessed data"""
    
    @staticmethod
    def validate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate data"""
        logger.info("Validating preprocessed data...")

        numeric_train = train_df.select_dtypes(include=[np.number])
        numeric_test = test_df.select_dtypes(include=[np.number])
        
        checks = {
            'train_not_empty': len(train_df) > 0,
            'test_not_empty': len(test_df) > 0,
            'train_no_nulls': train_df.isnull().sum().sum() == 0,
            'test_no_nulls': test_df.isnull().sum().sum() == 0,
            'columns_match': list(train_df.columns) == list(test_df.columns),
            'target_exists': Config.TARGET_COL in train_df.columns
                and Config.TARGET_COL in test_df.columns,
            'train_all_numeric': len(numeric_train.columns) == len(train_df.columns),
            'test_all_numeric': len(numeric_test.columns) == len(test_df.columns),
            'train_no_infinite': np.isfinite(numeric_train.to_numpy()).all(),
            'test_no_infinite': np.isfinite(numeric_test.to_numpy()).all(),
        }
        
        all_passed = all(checks.values())
        
        for check_name, result in checks.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {check_name}: {status}")
        
        return all_passed
    
# ============================================================
# MLFLOW LOGGER
# ============================================================

class MLflowLogger:

    @staticmethod
    def log_run(train_df, test_df):

        logger.info("Logging artifacts to MLflow...")

        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

        logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}" )

        with mlflow.start_run(run_name="preprocessing"):

            # Parameters
            mlflow.log_param(
                "train_size",
                len(train_df)
            )

            mlflow.log_param(
                "test_size",
                len(test_df)
            )

            mlflow.log_param(
                "feature_count",
                len(train_df.columns) - 1
            )

            mlflow.log_param(
                "target",
                Config.TARGET_COL
            )

            # Metrics
            mlflow.log_metric(
                "train_salary_mean",
                float(train_df[Config.TARGET_COL].mean())
            )

            mlflow.log_metric(
                "train_salary_std",
                float(train_df[Config.TARGET_COL].std())
            )

            # Artifacts
            mlflow.log_artifact(
                str(Config.PREPROCESSOR_PATH)
            )

            mlflow.log_artifact(
                str(Config.METADATA_PATH)
            )

            with open(Config.METADATA_PATH) as f:
                metadata = json.load(f)

            mlflow.log_dict(
                metadata,
                "preprocessing_metadata.json"
            )

            if Config.PROFILE_REPORT_PATH.exists():
                mlflow.log_artifact(
                    str(Config.PROFILE_REPORT_PATH)
                )

            json_profile_path = (
                Path(Config.PROCESSED_DATA_DIR) / "profiling_report.json"
            )
            if json_profile_path.exists():
                mlflow.log_artifact(
                    str(json_profile_path)
                )

            logger.info(
                "✓ MLflow logging completed"
            )

# ============================================================
# ARTIFACT SAVER
# ============================================================

class ArtifactSaver:

    @staticmethod
    def save(train_df, test_df, pipeline):

        train_df.to_parquet(
            Config.TRAIN_PARQUET_PATH,
            index=False
        )

        test_df.to_parquet(
            Config.TEST_PARQUET_PATH,
            index=False
        )

        pipeline.save()

        metadata = {

            "pipeline_execution_time":
                datetime.utcnow().isoformat(),

            "target_col":
                Config.TARGET_COL,

            "model_features":
                Config.MODEL_FEATURES,

            "numerical_cols":
                pipeline.numerical_cols,

            "encoded_cols":
                pipeline.encoded_cols,

            "categorical_cols":
                pipeline.categorical_cols,

            "ordinal_cols":
                pipeline.ordinal_cols,

            "boolean_cols":
                pipeline.boolean_cols,

            "feature_count":
                len(pipeline.get_feature_names()),

            "train_shape":
                list(train_df.shape),

            "test_shape":
                list(test_df.shape)
        }

        with open(
            Config.METADATA_PATH,
            "w"
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2
            )

        logger.info(
            "✓ Artifacts saved"
        )


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Execute full preprocessing pipeline"""
    logger.info("="*70)
    logger.info("STARTING DATA PREPROCESSING PIPELINE (STANDALONE)")
    logger.info("="*70)
    
    try:
        # Create output directories
        Path(Config.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        
        # STEP 1: Load data
        logger.info("\n[STEP 1] Loading data from MariaDB...")
        df = DataLoader.load_from_mariadb()
        
        logger.info(df.columns.tolist())

        # STEP 2: Profile data
        logger.info("\n[STEP 2] Profiling data...")
        DataProfiler.run_profiling(df)
        
        # STEP 3: Clean data
        logger.info("\n[STEP 3] Cleaning data...")
        df = DataCleaner.clean(df)
        
        # STEP 4: Engineer features
        logger.info("\n[STEP 4] Engineering features...")
        df = DataCleaner.engineer_features(df)
        
        # STEP 5: Split data
        logger.info("\n[STEP 5] Splitting data...")
        train_df, test_df = DataSplitter.split_data(df)
        
        # STEP 6
        logger.info(
            "\n[STEP 6] Transforming..."
        )

        train_final, test_final, pipeline = (
            DataTransformer.transform(
                train_df,
                test_df
            )
        )

        # STEP 7
        logger.info(
            "\n[STEP 7] Validating..."
        )

        is_valid = DataValidator.validate(
            train_final,
            test_final
        )

        if not is_valid:
            raise Exception("Validation failed!")

        # STEP 8
        logger.info(
            "\n[STEP 8] Saving artifacts..."
        )

        ArtifactSaver.save(
            train_final,
            test_final,
            pipeline
        )

        # STEP 9
        logger.info(
            "\n[STEP 9] Logging to MLflow..."
        )

        MLflowLogger.log_run(
            train_final,
            test_final
        )
        
        logger.info("\n" + "="*70)
        logger.info("✓ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\nOutput files:")
        logger.info(f"  - Train: {Config.TRAIN_PARQUET_PATH}")
        logger.info(f"  - Test: {Config.TEST_PARQUET_PATH}")
        logger.info(f"  - Preprocessor: {Config.PREPROCESSOR_PATH}")
        logger.info(f"  - Metadata: {Config.METADATA_PATH}")
        
        return {
            'status': 'SUCCESS',
            'train_shape': train_final.shape,
            'test_shape': test_final.shape,
        }
    
    except Exception as e:
        logger.error(f"✗ Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    result = main()
    print("\n" + "="*70)
    print("RESULT:", result)
    print("="*70)
