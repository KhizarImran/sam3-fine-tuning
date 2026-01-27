"""
SAM3 Training Script for Fuse Cutout Detection

Wrapper script to simplify SAM3 training with custom configuration
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import yaml


def check_prerequisites():
    """Check if all prerequisites are met before training"""
    print("=" * 70)
    print("SAM3 TRAINING - PREREQUISITE CHECK")
    print("=" * 70)

    issues = []

    # Check if SAM3 is installed
    print("\n1. Checking SAM3 installation...")
    sam3_path = Path("sam3")
    if not sam3_path.exists():
        issues.append("SAM3 not found. Run: python scripts/setup_sam3.bat (or .sh)")
    else:
        print("   ✓ SAM3 directory found")

    # Check if training script exists
    train_script = sam3_path / "sam3" / "train" / "train.py"
    if not train_script.exists():
        issues.append(f"Training script not found: {train_script}")
    else:
        print("   ✓ Training script found")

    # Check PyTorch and CUDA
    print("\n2. Checking PyTorch and CUDA...")
    try:
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ GPU count: {torch.cuda.device_count()}")
            print(f"   ✓ GPU name: {torch.cuda.get_device_name(0)}")
        else:
            issues.append("CUDA not available. Training will be very slow on CPU.")
    except ImportError:
        issues.append("PyTorch not installed. Run: pip install torch torchvision")

    # Check if dataset exists
    print("\n3. Checking dataset...")
    dataset_path = Path("sam3_datasets/fuse-cutout-detection")
    if not dataset_path.exists():
        issues.append("Dataset not found. Run: python scripts/prepare_dataset_for_sam3.py")
    else:
        # Count images
        train_dir = dataset_path / "train"
        if train_dir.exists():
            images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
            print(f"   ✓ Dataset found: {len(images)} training images")
        else:
            issues.append("Train directory not found in dataset")

    if issues:
        print("\n" + "=" * 70)
        print("ISSUES FOUND:")
        print("=" * 70)
        for issue in issues:
            print(f"  ✗ {issue}")
        print()
        return False

    print("\n" + "=" * 70)
    print("✓ ALL CHECKS PASSED")
    print("=" * 70)
    return True


def load_config(config_path):
    """Load and validate training configuration"""
    print(f"\nLoading configuration: {config_path}")

    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        return None

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check critical paths
    warnings = []

    if 'paths' in config:
        dataset_root = config['paths'].get('roboflow_vl_100_root')
        if dataset_root and not Path(dataset_root).exists():
            warnings.append(f"Dataset root not found: {dataset_root}")

        checkpoint = config.get('model', {}).get('checkpoint')
        if checkpoint and checkpoint.startswith('path/to/'):
            warnings.append("Model checkpoint path not configured (model.checkpoint)")

    if warnings:
        print("\n⚠ Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return config


def run_training(config_path, num_gpus=1, use_cluster=False):
    """Run SAM3 training"""
    print("\n" + "=" * 70)
    print("STARTING SAM3 TRAINING")
    print("=" * 70)

    # Build training command
    train_script = "sam3/sam3/train/train.py"

    # Use uv run to ensure we're using the correct environment
    cmd = [
        "uv", "run", "python",
        train_script,
        "-c", config_path,
        "--use-cluster", "1" if use_cluster else "0",
        "--num-gpus", str(num_gpus)
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print()

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE")
        print("=" * 70)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train SAM3 on fuse cutout dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config (single GPU)
  python scripts/train_sam3.py

  # Train with custom config
  python scripts/train_sam3.py --config configs/my_config.yaml

  # Train with multiple GPUs
  python scripts/train_sam3.py --num-gpus 2

  # Skip prerequisite check (not recommended)
  python scripts/train_sam3.py --skip-checks

  # View TensorBoard during training (in another terminal):
  tensorboard --logdir experiments/fuse_cutout/tensorboard
        """
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
        help="Number of GPUs to use (default: 1)"
    )

    parser.add_argument(
        "--use-cluster",
        action="store_true",
        help="Submit to cluster via submitit (default: False)"
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip prerequisite checks (not recommended)"
    )

    args = parser.parse_args()

    # Run prerequisite checks
    if not args.skip_checks:
        if not check_prerequisites():
            print("\nPlease fix the issues above before training.")
            sys.exit(1)

    # Load and validate config
    config = load_config(args.config)
    if config is None:
        sys.exit(1)

    # Start training
    success = run_training(args.config, args.num_gpus, args.use_cluster)

    if success:
        print("\nNext steps:")
        print("  1. Check training logs in experiments/fuse_cutout/")
        print("  2. View TensorBoard: tensorboard --logdir experiments/fuse_cutout/tensorboard")
        print("  3. Evaluate model: python scripts/evaluate_sam3.py")
        print("  4. Test inference: python scripts/inference_sam3.py")
    else:
        print("\nTraining failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
