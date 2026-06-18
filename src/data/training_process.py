"""
STAGE 3: XGBOOST MODEL TRAINING PROCESS

Uses the processed train/test datasets created by Stage 2.
No train/test split happens here.

Flow:
1. Load processed train/test parquet files
2. Separate features and target
3. Train baseline XGBoost with cross-validation
4. Tune XGBoost with Optuna if baseline is below threshold
5. Select the best XGBoost version
6. Evaluate on final test data
7. Save model artifacts
8. Log run to MLflow
"""

import json
import logging
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None
    MedianPruner = None
    TPESampler = None


warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Config:
    """Training configuration for one-model XGBoost pipeline."""

    TRAIN_DATA_PATH = Path("/home/samiksya/ai_job_market/data/processed/train.parquet")
    TEST_DATA_PATH = Path("/home/samiksya/ai_job_market/data/processed/test.parquet")
    PREPROCESSOR_PATH = Path("/home/samiksya/ai_job_market/data/processed/preprocessor.pkl")

    TARGET_COL = "salary_usd"

    ARTIFACTS_DIR = Path("/home/samiksya/ai_job_market/data/processed/model_artifacts")
    MODEL_PATH = ARTIFACTS_DIR / "xgboost_final_model.pkl"
    METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
    PARAMS_PATH = ARTIFACTS_DIR / "model_parameters.json"
    FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
    PREDICTION_PLOT_PATH = ARTIFACTS_DIR / "predictions_vs_actual.png"

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    MLFLOW_EXPERIMENT_NAME = "ai-job-salary-prediction"

    RANDOM_STATE = 42
    CV_FOLDS = 5
    OPTUNA_CV_FOLDS = 3

    MIN_R2_SCORE = 0.70
    MAX_RMSE = 25000

    OPTUNA_ENABLED = True
    OPTUNA_N_TRIALS = 50
    OPTUNA_TIMEOUT_SECONDS = 600

    BASELINE_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "n_jobs": -1,
    }


class DataLoader:
    """Load and validate processed train/test datasets."""

    @staticmethod
    def load() -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading processed train/test data...")

        if not Config.TRAIN_DATA_PATH.exists():
            raise FileNotFoundError(f"Train file not found: {Config.TRAIN_DATA_PATH}")
        if not Config.TEST_DATA_PATH.exists():
            raise FileNotFoundError(f"Test file not found: {Config.TEST_DATA_PATH}")

        train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
        test_df = pd.read_parquet(Config.TEST_DATA_PATH)

        DataLoader._validate(train_df, "train")
        DataLoader._validate(test_df, "test")

        logger.info("Train shape: %s", train_df.shape)
        logger.info("Test shape: %s", test_df.shape)
        return train_df, test_df

    @staticmethod
    def _validate(df: pd.DataFrame, name: str) -> None:
        if df.empty:
            raise ValueError(f"{name} dataset is empty")
        if Config.TARGET_COL not in df.columns:
            raise ValueError(f"{name} dataset missing target: {Config.TARGET_COL}")
        if df.isnull().sum().sum() > 0:
            raise ValueError(f"{name} dataset contains null values")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) != len(df.columns):
            non_numeric = sorted(set(df.columns) - set(numeric_cols))
            raise ValueError(f"{name} dataset contains non-numeric columns: {non_numeric}")

        if not np.isfinite(df.to_numpy()).all():
            raise ValueError(f"{name} dataset contains infinite values")

    @staticmethod
    def split_features_target(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=[Config.TARGET_COL])
        y = df[Config.TARGET_COL]
        return X, y


class XGBoostTrainer:
    """Train baseline and tuned XGBoost models."""

    @staticmethod
    def build_model(params: Dict) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(**params)

    @staticmethod
    def cross_validate(params: Dict, X: pd.DataFrame, y: pd.Series, folds: int) -> Dict:
        model = XGBoostTrainer.build_model(params)
        kfold = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=Config.RANDOM_STATE,
        )

        r2_scores = cross_val_score(
            model,
            X,
            y,
            cv=kfold,
            scoring="r2",
            n_jobs=-1,
        )
        rmse_scores = -cross_val_score(
            model,
            X,
            y,
            cv=kfold,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        return {
            "r2_mean": float(r2_scores.mean()),
            "r2_std": float(r2_scores.std()),
            "r2_scores": r2_scores.tolist(),
            "rmse_mean": float(rmse_scores.mean()),
            "rmse_scores": rmse_scores.tolist(),
        }

    @staticmethod
    def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[xgb.XGBRegressor, Dict]:
        logger.info("Training baseline XGBoost...")

        cv_results = XGBoostTrainer.cross_validate(
            Config.BASELINE_PARAMS,
            X_train,
            y_train,
            Config.CV_FOLDS,
        )

        model = XGBoostTrainer.build_model(Config.BASELINE_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        logger.info("Baseline CV R2: %.4f", cv_results["r2_mean"])
        logger.info("Baseline CV RMSE: %.2f", cv_results["rmse_mean"])
        return model, cv_results


class OptunaTuner:
    """Tune only XGBoost hyperparameters."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train

    def objective(self, trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 350),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3.0),
            "objective": "reg:squarederror",
            "random_state": Config.RANDOM_STATE,
            "verbosity": 0,
            "n_jobs": -1,
        }

        cv_results = XGBoostTrainer.cross_validate(
            params,
            self.X_train,
            self.y_train,
            Config.OPTUNA_CV_FOLDS,
        )
        return cv_results["rmse_mean"]

    def tune(self) -> Optional[Dict]:
        if not Config.OPTUNA_ENABLED:
            logger.info("Optuna disabled. Skipping tuning.")
            return None
        if optuna is None:
            logger.warning("Optuna is not installed. Skipping tuning.")
            return None

        logger.info("Running Optuna XGBoost tuning...")
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=Config.RANDOM_STATE),
            pruner=MedianPruner(),
        )
        study.optimize(
            self.objective,
            n_trials=Config.OPTUNA_N_TRIALS,
            timeout=Config.OPTUNA_TIMEOUT_SECONDS,
            show_progress_bar=False,
        )

        best_params = {
            **study.best_params,
            "objective": "reg:squarederror",
            "random_state": Config.RANDOM_STATE,
            "verbosity": 0,
            "n_jobs": -1,
        }

        return {
            "best_params": best_params,
            "best_rmse": float(study.best_value),
            "n_trials": len(study.trials),
        }


class ModelSelector:
    """Select baseline or tuned XGBoost based on cross-validated RMSE."""

    @staticmethod
    def select(
        baseline_model: xgb.XGBRegressor,
        baseline_cv: Dict,
        tuning_results: Optional[Dict],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[xgb.XGBRegressor, str, Dict]:
        baseline_good = (
            baseline_cv["r2_mean"] >= Config.MIN_R2_SCORE
            and baseline_cv["rmse_mean"] <= Config.MAX_RMSE
        )

        if baseline_good:
            logger.info("Baseline meets thresholds. Selecting baseline model.")
            return baseline_model, "baseline", Config.BASELINE_PARAMS

        if not tuning_results:
            logger.info("No tuning result available. Selecting baseline model.")
            return baseline_model, "baseline", Config.BASELINE_PARAMS

        if tuning_results["best_rmse"] >= baseline_cv["rmse_mean"]:
            logger.info("Tuned model did not improve RMSE. Selecting baseline model.")
            return baseline_model, "baseline", Config.BASELINE_PARAMS

        logger.info("Tuned model improves RMSE. Training final tuned model.")
        tuned_model = XGBoostTrainer.build_model(tuning_results["best_params"])
        tuned_model.fit(X_train, y_train, verbose=False)
        return tuned_model, "tuned", tuning_results["best_params"]


class Evaluator:
    """Evaluate the selected model."""

    @staticmethod
    def evaluate(
        model: xgb.XGBRegressor,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        return {
            "train_metrics": Evaluator._metrics(y_train, train_pred),
            "test_metrics": Evaluator._metrics(y_test, test_pred),
            "test_predictions": test_pred.tolist(),
            "test_actual": y_test.tolist(),
        }

    @staticmethod
    def _metrics(y_true, y_pred) -> Dict:
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        }


class ArtifactManager:
    """Save local model artifacts."""

    @staticmethod
    def save_all(
        model: xgb.XGBRegressor,
        model_type: str,
        final_params: Dict,
        baseline_cv: Dict,
        tuning_results: Optional[Dict],
        eval_results: Dict,
        feature_names,
    ) -> None:
        Config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(Config.MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        feature_importance.to_csv(Config.FEATURE_IMPORTANCE_PATH, index=False)

        ArtifactManager._save_prediction_plot(
            np.array(eval_results["test_actual"]),
            np.array(eval_results["test_predictions"]),
        )

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": "XGBRegressor",
            "model_type": model_type,
            "target": Config.TARGET_COL,
            "baseline_cv": baseline_cv,
            "tuning_results": tuning_results,
            "train_metrics": eval_results["train_metrics"],
            "test_metrics": eval_results["test_metrics"],
        }
        with open(Config.METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        with open(Config.PARAMS_PATH, "w") as f:
            json.dump(final_params, f, indent=2, default=str)

        logger.info("Artifacts saved to %s", Config.ARTIFACTS_DIR)

    @staticmethod
    def _save_prediction_plot(y_actual: np.ndarray, y_pred: np.ndarray) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_actual, y_pred, alpha=0.55, s=20)
        min_value = min(y_actual.min(), y_pred.min())
        max_value = max(y_actual.max(), y_pred.max())
        axes[0].plot([min_value, max_value], [min_value, max_value], "r--", lw=2)
        axes[0].set_xlabel("Actual Salary")
        axes[0].set_ylabel("Predicted Salary")
        axes[0].set_title("Predictions vs Actual")
        axes[0].grid(True, alpha=0.3)

        residuals = y_actual - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.55, s=20)
        axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[1].set_xlabel("Predicted Salary")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residual Plot")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Config.PREDICTION_PLOT_PATH, dpi=120, bbox_inches="tight")
        plt.close(fig)


class MLflowLogger:
    """Log training results to MLflow."""

    @staticmethod
    def log_run(
        model: xgb.XGBRegressor,
        model_type: str,
        final_params: Dict,
        baseline_cv: Dict,
        tuning_results: Optional[Dict],
        eval_results: Dict,
    ) -> Optional[str]:
        try:
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

            run_name = f"xgboost-{model_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(final_params)
                mlflow.log_param("model_family", "xgboost")
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("target", Config.TARGET_COL)
                mlflow.log_param("cv_folds", Config.CV_FOLDS)
                mlflow.log_param("optuna_enabled", tuning_results is not None)

                if tuning_results:
                    mlflow.log_param("optuna_n_trials", tuning_results["n_trials"])
                    mlflow.log_metric("optuna_best_cv_rmse", tuning_results["best_rmse"])

                mlflow.log_metric("baseline_cv_r2_mean", baseline_cv["r2_mean"])
                mlflow.log_metric("baseline_cv_r2_std", baseline_cv["r2_std"])
                mlflow.log_metric("baseline_cv_rmse_mean", baseline_cv["rmse_mean"])

                for prefix in ["train", "test"]:
                    for metric_name, metric_value in eval_results[f"{prefix}_metrics"].items():
                        mlflow.log_metric(f"{prefix}_{metric_name}", metric_value)

                mlflow.xgboost.log_model(model, "model")

                for artifact_path in [
                    Config.MODEL_PATH,
                    Config.METRICS_PATH,
                    Config.PARAMS_PATH,
                    Config.FEATURE_IMPORTANCE_PATH,
                    Config.PREDICTION_PLOT_PATH,
                    Config.PREPROCESSOR_PATH,
                ]:
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path))

                mlflow.set_tag("stage", "model_training")
                mlflow.set_tag("model", "XGBRegressor")
                mlflow.set_tag("task", "salary_prediction")

                logger.info("MLflow run logged: %s", run.info.run_id)
                return run.info.run_id
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)
            return None


def main() -> Dict:
    """Run the complete one-model XGBoost training process."""
    logger.info("=" * 70)
    logger.info("STARTING STAGE 3 XGBOOST TRAINING PROCESS")
    logger.info("=" * 70)

    train_df, test_df = DataLoader.load()
    X_train, y_train = DataLoader.split_features_target(train_df)
    X_test, y_test = DataLoader.split_features_target(test_df)

    baseline_model, baseline_cv = XGBoostTrainer.train_baseline(X_train, y_train)

    baseline_needs_tuning = (
        baseline_cv["r2_mean"] < Config.MIN_R2_SCORE
        or baseline_cv["rmse_mean"] > Config.MAX_RMSE
    )

    tuning_results = None
    if baseline_needs_tuning:
        tuning_results = OptunaTuner(X_train, y_train).tune()
    else:
        logger.info("Baseline is strong enough. Skipping Optuna tuning.")

    final_model, model_type, final_params = ModelSelector.select(
        baseline_model,
        baseline_cv,
        tuning_results,
        X_train,
        y_train,
    )

    eval_results = Evaluator.evaluate(
        final_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    ArtifactManager.save_all(
        final_model,
        model_type,
        final_params,
        baseline_cv,
        tuning_results,
        eval_results,
        X_train.columns,
    )

    run_id = MLflowLogger.log_run(
        final_model,
        model_type,
        final_params,
        baseline_cv,
        tuning_results,
        eval_results,
    )

    result = {
        "status": "SUCCESS",
        "model": "XGBRegressor",
        "model_type": model_type,
        "baseline_cv_r2": baseline_cv["r2_mean"],
        "baseline_cv_rmse": baseline_cv["rmse_mean"],
        "test_r2": eval_results["test_metrics"]["r2"],
        "test_rmse": eval_results["test_metrics"]["rmse"],
        "test_mae": eval_results["test_metrics"]["mae"],
        "model_path": str(Config.MODEL_PATH),
        "artifacts_dir": str(Config.ARTIFACTS_DIR),
        "mlflow_run_id": run_id,
    }

    logger.info("=" * 70)
    logger.info("XGBOOST TRAINING PROCESS COMPLETED")
    logger.info(json.dumps(result, indent=2))
    logger.info("=" * 70)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
