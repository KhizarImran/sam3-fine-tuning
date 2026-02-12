#!/usr/bin/env python
"""
Simple wrapper to run SAM3 training with proper initialization
"""

import os
import sys

# Add sam3 to Python path
sam3_path = "/home/ubuntu/sam3-fine-tuning/sam3"
if sam3_path not in sys.path:
    sys.path.insert(0, sam3_path)

# Change to the project directory
os.chdir("/home/ubuntu/sam3-fine-tuning")

# Now import and run the training
from hydra import compose, initialize_config_module
from argparse import Namespace

# Initialize Hydra with sam3.train config module
initialize_config_module("sam3.train", version_base="1.2")

# Import after Hydra initialization
from sam3.train.utils.train_utils import register_omegaconf_resolvers
from sam3.train.train import main

# Register resolvers
register_omegaconf_resolvers()

# Create args
args = Namespace(
    config="fuse_neutrals_train",
    use_cluster=False,
    partition=None,
    account=None,
    qos=None,
    num_gpus=1,
    num_nodes=None,
)

# Run training
print("=" * 70)
print("STARTING SAM3 TRAINING - FUSE NEUTRALS DATASET")
print("=" * 70)
print(f"Config: {args.config}")
print(f"GPUs: {args.num_gpus}")
print(f"Use cluster: {args.use_cluster}")
print("=" * 70)

main(args)
