"""
MLflow Logger for SAM3 Training

Integrates MLflow tracking with SAM3 training
"""

import os
from typing import Dict, Any, Optional
import mlflow


class MLflowLogger:
    """MLflow experiment tracking logger"""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize MLflow logger

        Args:
            tracking_uri: MLflow tracking server URL (e.g., http://52.2.51.33:5000)
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for this training run
            tags: Optional tags for the run
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name or "sam3_training"
        self.tags = tags or {}

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)

    def start_run(self):
        """Start MLflow run"""
        mlflow.start_run(run_name=self.run_name)

        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

        print(f"Started MLflow run: {self.run_name}")
        print(f"Run URL: {self.tracking_uri}/#/experiments/{self.experiment_id}")

    def log_params(self, params: Dict[str, Any]):
        """Log training parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics"""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file)"""
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model_path: str, artifact_path: str = "model"):
        """Log model checkpoint"""
        mlflow.log_artifact(model_path, artifact_path)

    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()
        print("MLflow run ended")


def setup_mlflow_logging(config: Dict[str, Any]) -> Optional[MLflowLogger]:
    """
    Setup MLflow logging from config

    Args:
        config: Training configuration dict

    Returns:
        MLflowLogger instance or None if MLflow not configured
    """
    mlflow_config = config.get('mlflow', {})

    if not mlflow_config.get('enabled', False):
        return None

    tracking_uri = mlflow_config.get('tracking_uri')
    if not tracking_uri:
        print("Warning: MLflow enabled but no tracking_uri specified")
        return None

    experiment_name = mlflow_config.get('experiment_name', 'sam3-fuse-cutout')
    run_name = mlflow_config.get('run_name', 'sam3_training')

    # Add system info as tags
    tags = {
        'model': 'sam3',
        'task': 'fuse_cutout_detection',
        'framework': 'pytorch',
    }
    tags.update(mlflow_config.get('tags', {}))

    logger = MLflowLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags
    )

    return logger
