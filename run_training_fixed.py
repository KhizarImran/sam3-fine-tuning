#!/usr/bin/env python
"""
SAM3 Training Wrapper - Uses filesystem config paths instead of package resources
Solves the issue where editable SAM3 install doesn't expose configs to Hydra
"""

import os
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path("/home/ubuntu/sam3-fine-tuning")
SAM3_ROOT = PROJECT_ROOT / "sam3"
CONFIG_DIR = SAM3_ROOT / "sam3" / "train" / "configs"

# Add SAM3 to Python path
sys.path.insert(0, str(SAM3_ROOT))
os.environ["PYTHONPATH"] = f"{SAM3_ROOT}:{os.environ.get('PYTHONPATH', '')}"

# Change to project directory
os.chdir(PROJECT_ROOT)

# Now import required modules
from hydra import compose, initialize_config_dir
from argparse import Namespace
from sam3.train.utils.train_utils import register_omegaconf_resolvers
from sam3.train.train import main

# Initialize Hydra with filesystem path instead of package resource
print("=" * 70)
print("SAM3 TRAINING - FUSE NEUTRALS DATASET")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")
print(f"Config directory: {CONFIG_DIR}")
print(f"Config exists: {CONFIG_DIR.exists()}")

if CONFIG_DIR.exists():
    configs = [f for f in CONFIG_DIR.glob("*.yaml")]
    print(f"Available configs: {len(configs)}")
    for cfg in configs:
        print(f"  - {cfg.name}")

print("=" * 70)

# Initialize Hydra with filesystem config directory
initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2")

# Register OmegaConf resolvers
register_omegaconf_resolvers()

# Create arguments
args = Namespace(
    config="fuse_neutrals_train",
    use_cluster=False,
    partition=None,
    account=None,
    qos=None,
    num_gpus=1,
    num_nodes=None,
)

print(f"\nStarting training with config: {args.config}")
print(f"GPUs: {args.num_gpus}")
print(f"Use cluster: {args.use_cluster}")
print("=" * 70)
print()

# Run training
try:
    main(args)
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
except Exception as e:
    print("\n" + "=" * 70)
    print("✗ TRAINING FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
