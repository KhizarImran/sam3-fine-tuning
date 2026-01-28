#!/usr/bin/env python3
"""
Patched SAM3 training wrapper that fixes Hydra config discovery issues.

This script modifies how Hydra finds configs by using filesystem paths
instead of relying on package resources.
"""
import os
import sys
from pathlib import Path

# Add sam3 to Python path
sam3_dir = Path(__file__).parent.parent / "sam3"
sys.path.insert(0, str(sam3_dir))

# Patch the train script before importing
import sam3.train.train as train_module

# Replace the __main__ block's initialization with filesystem-based approach
from hydra import compose, initialize_config_dir
from argparse import ArgumentParser

def main():
    # Parse arguments (same as original train.py)
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--use-cluster", type=int, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

    # Initialize Hydra with absolute path to configs directory
    configs_dir = str((Path(__file__).parent.parent / "sam3" / "sam3" / "train" / "configs").absolute())
    print(f"Using configs directory: {configs_dir}")

    # Register OmegaConf resolvers (from original train.py)
    train_module.register_omegaconf_resolvers()

    # Initialize Hydra with filesystem path instead of module
    with initialize_config_dir(version_base="1.2", config_dir=configs_dir):
        cfg = compose(config_name=args.config)

        # Make config struct mode flexible so we can modify it
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)

        # Set experiment log dir if not specified
        if not cfg.get('launcher'):
            cfg.launcher = {}
        if not cfg.launcher.get('experiment_log_dir'):
            cfg.launcher.experiment_log_dir = os.path.join(
                os.getcwd(), "sam3_logs", args.config
            )

        # Override launcher settings from command line (only if provided)
        if args.use_cluster is not None:
            cfg.launcher.use_cluster = args.use_cluster
        if args.partition is not None:
            cfg.launcher.partition = args.partition
        if args.account is not None:
            cfg.launcher.account = args.account
        if args.qos is not None:
            cfg.launcher.qos = args.qos
        if args.num_gpus is not None:
            cfg.launcher.num_gpus = args.num_gpus
        if args.num_nodes is not None:
            cfg.launcher.num_nodes = args.num_nodes

        # Set defaults if not present
        if not cfg.launcher.get('use_cluster'):
            cfg.launcher.use_cluster = False

        print("\n" + "=" * 80)
        print("SAM3 TRAINING - Configuration Loaded Successfully")
        print("=" * 80)
        print(f"Config: {args.config}")
        print(f"Dataset: {cfg.paths.roboflow_vl_100_root}")
        print(f"Experiment dir: {cfg.launcher.experiment_log_dir}")
        print(f"GPUs per node: {cfg.launcher.get('gpus_per_node', 1)}")
        print(f"Num nodes: {cfg.launcher.get('num_nodes', 1)}")
        print(f"Use cluster: {cfg.launcher.get('use_cluster', False)}")
        print("=" * 80 + "\n")

        # Call the main training function from original train.py
        # Get main_port for single node training
        main_port = cfg.get("main_port", 12355)

        if cfg.launcher.use_cluster:
            # Cluster training
            train_module.cluster_runner(cfg, main_port)
        else:
            # Single node training
            train_module.single_node_runner(cfg, main_port)

if __name__ == "__main__":
    main()
