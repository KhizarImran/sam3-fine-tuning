#!/usr/bin/env python3
"""
MLflow Wrapper for SAM3 Training

Wraps train_sam3_patched.py with MLflow logging
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("ERROR: MLflow not installed. Install with: uv pip install mlflow")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train SAM3 with MLflow logging"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Config name (e.g., fuse_cutout_train)"
    )

    parser.add_argument(
        "--mlflow-uri",
        default="http://52.2.51.33:5000",
        help="MLflow tracking server URI"
    )

    parser.add_argument(
        "--experiment-name",
        default="SAM3-Fuse-Cutout",
        help="MLflow experiment name"
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name (default: auto-generated)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SAM3 TRAINING WITH MLFLOW")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"MLflow URI: {args.mlflow_uri}")
    print(f"Experiment: {args.experiment_name}")
    print("=" * 80)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_uri)

    # Test connection
    try:
        experiments = mlflow.search_experiments()
        print(f"\n✓ Connected to MLflow (found {len(experiments)} experiments)")
    except Exception as e:
        print(f"\n✗ Failed to connect to MLflow: {e}")
        response = input("Continue without MLflow? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Set or create experiment
    mlflow.set_experiment(args.experiment_name)

    # Generate run name
    run_name = args.run_name or f"{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        print(f"\n✓ Started MLflow run: {run_name}")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        print(f"   URL: {args.mlflow_uri}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}")

        # Log parameters
        mlflow.log_param("config", args.config)
        mlflow.log_param("timestamp", datetime.now().isoformat())

        # Log config file as artifact
        config_path = f"configs/{args.config}.yaml"
        if Path(config_path).exists():
            mlflow.log_artifact(config_path)

        # Build training command
        train_script = Path(__file__).parent / "train_sam3_patched.py"
        cmd = [
            sys.executable,
            str(train_script),
            "--config", args.config
        ]

        print(f"\n{'=' * 80}")
        print("STARTING TRAINING")
        print(f"{'=' * 80}")
        print(f"Command: {' '.join(cmd)}\n")

        # Run training
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output and parse for metrics
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line, end='')

                    # Parse and log metrics from output
                    # Example: "Epoch 5 | Loss: 0.123"
                    if "Trainer/epoch" in line:
                        try:
                            # Parse epoch number
                            parts = line.split("Trainer/epoch")
                            if len(parts) > 1:
                                epoch_str = parts[1].split(",")[0].strip(": ")
                                epoch = int(epoch_str)
                                mlflow.log_metric("epoch", epoch, step=epoch)
                        except:
                            pass

            process.wait()

            if process.returncode == 0:
                print(f"\n{'=' * 80}")
                print("✓ TRAINING COMPLETED SUCCESSFULLY")
                print(f"{'=' * 80}")
                mlflow.log_param("status", "completed")

                # Log checkpoint as artifact
                checkpoint_dir = Path("experiments/fuse_cutout/checkpoints")
                if checkpoint_dir.exists():
                    for ckpt_file in checkpoint_dir.glob("*.pt"):
                        print(f"\nLogging checkpoint: {ckpt_file}")
                        mlflow.log_artifact(str(ckpt_file))

            else:
                print(f"\n{'=' * 80}")
                print(f"✗ TRAINING FAILED (exit code: {process.returncode})")
                print(f"{'=' * 80}")
                mlflow.log_param("status", "failed")
                sys.exit(process.returncode)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            mlflow.log_param("status", "interrupted")
            process.terminate()
            sys.exit(1)
        except Exception as e:
            print(f"\n\nError during training: {e}")
            mlflow.log_param("status", "error")
            mlflow.log_param("error", str(e))
            raise


if __name__ == "__main__":
    main()
