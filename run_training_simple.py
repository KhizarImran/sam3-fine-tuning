#!/usr/bin/env python
"""
Simple training launcher that uses local configs directory
"""

import os
import sys
import subprocess

# Change to project directory
os.chdir("/home/ubuntu/sam3-fine-tuning")

# Set PYTHONPATH to include sam3
os.environ["PYTHONPATH"] = "/home/ubuntu/sam3-fine-tuning/sam3:" + os.environ.get(
    "PYTHONPATH", ""
)

# Use the venv Python
python_exe = "/home/ubuntu/sam3-fine-tuning/.venv/bin/python"

# Build command to run train.py directly
cmd = [
    python_exe,
    "sam3/sam3/train/train.py",
    "-c",
    "fuse_neutrals_train",
    "--use-cluster",
    "0",
    "--num-gpus",
    "1",
]

print("=" * 70)
print("STARTING SAM3 TRAINING")
print("=" * 70)
print(f"Command: {' '.join(cmd)}")
print("=" * 70)

# Run the command
result = subprocess.run(cmd)
sys.exit(result.returncode)
