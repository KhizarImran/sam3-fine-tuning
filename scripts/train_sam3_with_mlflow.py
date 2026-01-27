"""
SAM3 Training Script with MLflow Integration

Enhanced wrapper that logs training metrics to MLflow
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import yaml
from datetime import datetime

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mlflow_logger import setup_mlflow_logging


def check_mlflow_connection(tracking_uri: str) -> bool:
    """Test connection to MLflow server"""
    print(f"\nTesting MLflow connection to {tracking_uri}...")
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        # Try to list experiments
        mlflow.search_experiments()
        print("   ✓ MLflow connection successful")
        return True
    except Exception as e:
        print(f"   ✗ MLflow connection failed: {e}")
        return False


def setup_mlflow_env(config: dict):
    """Setup environment variables for MLflow"""
    mlflow_config = config.get('mlflow', {})

    if not mlflow_config.get('enabled', False):
        return None

    tracking_uri = mlflow_config.get('tracking_uri')
    if not tracking_uri:
        print("Warning: MLflow enabled but no tracking_uri specified")
        return None

    # Check connection
    if not check_mlflow_connection(tracking_uri):
        response = input("MLflow connection failed. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return None

    # Setup environment variables that SAM3 training can use
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    os.environ['MLFLOW_EXPERIMENT_NAME'] = mlflow_config.get('experiment_name', 'sam3-fuse-cutout')

    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{mlflow_config.get('run_name', 'sam3_training')}_{timestamp}"
    os.environ['MLFLOW_RUN_NAME'] = run_name

    print(f"\n✓ MLflow configured:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Experiment: {mlflow_config.get('experiment_name')}")
    print(f"  Run name: {run_name}")
    print(f"  View at: {tracking_uri}")

    return {
        'tracking_uri': tracking_uri,
        'experiment_name': mlflow_config.get('experiment_name'),
        'run_name': run_name
    }


def create_mlflow_callback_script(output_path: Path, mlflow_info: dict):
    """Create a Python script that SAM3 can import for MLflow logging"""

    callback_code = f"""
# Auto-generated MLflow callback for SAM3 training
import mlflow
import os

# MLflow configuration
MLFLOW_TRACKING_URI = "{mlflow_info['tracking_uri']}"
MLFLOW_EXPERIMENT_NAME = "{mlflow_info['experiment_name']}"
MLFLOW_RUN_NAME = "{mlflow_info['run_name']}"

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Start run
active_run = None

def start_mlflow_run():
    global active_run
    if active_run is None:
        active_run = mlflow.start_run(run_name=MLFLOW_RUN_NAME)
        print(f"MLflow run started: {{MLFLOW_RUN_NAME}}")
        print(f"View at: {{MLFLOW_TRACKING_URI}}/#/experiments/{{mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id}}")

def log_params(params):
    start_mlflow_run()
    mlflow.log_params(params)

def log_metrics(metrics, step=None):
    start_mlflow_run()
    mlflow.log_metrics(metrics, step=step)

def log_artifact(path):
    start_mlflow_run()
    mlflow.log_artifact(path)

def end_mlflow_run():
    global active_run
    if active_run is not None:
        mlflow.end_run()
        active_run = None
        print("MLflow run ended")
"""

    output_path.write_text(callback_code)
    print(f"   Created MLflow callback: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SAM3 with MLflow logging",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        default="fuse_cutout_train",
        help="Config name in SAM3's config directory (without .yaml extension)"
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )

    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow logging"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SAM3 TRAINING WITH MLFLOW")
    print("=" * 70)

    # Load config from the configs directory for MLflow setup
    # The config name will be passed to SAM3's train.py (which uses Hydra)
    config_file = f"configs/{args.config}.yaml"
    if not Path(config_file).exists():
        # Try alternative location (SAM3's config directory)
        config_file = f"sam3/sam3/train/configs/{args.config}.yaml"

    print(f"Loading config from: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Disable MLflow if requested
    if args.disable_mlflow:
        if 'mlflow' in config:
            config['mlflow']['enabled'] = False

    # Setup MLflow
    mlflow_info = setup_mlflow_env(config)

    if mlflow_info:
        # Create MLflow callback script
        callback_path = Path("sam3/mlflow_callback.py")
        create_mlflow_callback_script(callback_path, mlflow_info)

        print("\n" + "=" * 70)
        print("STARTING TRAINING WITH MLFLOW LOGGING")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("STARTING TRAINING (MLflow disabled)")
        print("=" * 70)

    # Build training command
    train_script = "sam3/sam3/train/train.py"

    # Use uv run to ensure we're using the correct environment
    cmd = [
        "uv", "run", "python",
        train_script,
        "-c", args.config,
        "--use-cluster", "0",
        "--num-gpus", str(args.num_gpus)
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print()

    # Run training
    try:
        result = subprocess.run(cmd, check=True)

        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE")
        print("=" * 70)

        if mlflow_info:
            print(f"\nView results in MLflow: {mlflow_info['tracking_uri']}")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
