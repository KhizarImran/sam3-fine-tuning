# SAM3 Fine-Tuning Complete Training Guide

**Last Updated:** February 12, 2026  
**Environment:** AWS EC2 with NVIDIA A10G GPU (23GB VRAM)

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Critical Issue: Hydra Config Discovery](#critical-issue-hydra-config-discovery)
4. [Step-by-Step Training Process](#step-by-step-training-process)
5. [Configuration Files](#configuration-files)
6. [Monitoring Training](#monitoring-training)
7. [Common Issues & Solutions](#common-issues--solutions)
8. [After Training](#after-training)

---

## Quick Start

If everything is already set up and you just want to start training:

```bash
cd /home/ubuntu/sam3-fine-tuning

# Clean up any previous training artifacts
rm -rf experiments/fuse_neutrals/checkpoints/*
rm -rf experiments/fuse_neutrals/logs/*
rm -rf experiments/fuse_neutrals/dumps/*

# Start training (runs in background)
nohup .venv/bin/python run_training_fixed.py > training.log 2>&1 &

# Get the process ID
echo $!

# Monitor training
tail -f training.log

# Check GPU usage
watch -n 2 nvidia-smi
```

**Important:** Always use `run_training_fixed.py` - it solves the Hydra config discovery issue.

---

## System Requirements

### Hardware
- **GPU:** NVIDIA GPU with at least 16GB VRAM (tested on A10G with 23GB)
- **RAM:** 16GB+ system RAM (we have 15GB)
- **Storage:** 100GB+ available disk space
- **CPU:** Multi-core CPU (8+ cores recommended)

### Software
- **OS:** Ubuntu 20.04+ or similar Linux distribution
- **CUDA:** 11.8+ (we have 13.0)
- **Python:** 3.10 (required by project)
- **uv:** Modern Python package manager (installed)

### GPU Memory Usage
For **fuse neutrals dataset** (24 training images):
- Batch size 2: ~17GB VRAM
- Batch size 1: ~12GB VRAM

For **fuse cutout dataset** (larger images):
- Batch size 2: May require 20GB+ VRAM
- Batch size 1: ~15GB VRAM

---

## Critical Issue: Hydra Config Discovery

### The Problem
SAM3 is installed as an **editable package** (`pip install -e sam3/`), which means:
- The package is linked to the source directory
- Config files are NOT included in package resources
- Hydra's `initialize_config_module()` cannot find configs

**Error you'll see:**
```
hydra.errors.MissingConfigException: Cannot find primary config 'fuse_neutrals_train'. 
Check that it's in your config search path.

Config search path:
	provider=hydra, path=pkg://hydra.conf
	provider=main, path=pkg://sam3.train
	provider=schema, path=structured://
```

### The Solution
Use **filesystem paths** instead of package resources with `initialize_config_dir()`.

**DO NOT use:**
```python
# ❌ This WILL NOT work with editable install
from hydra import initialize_config_module
initialize_config_module("sam3.train", version_base="1.2")
```

**USE instead:**
```python
# ✓ This works with editable install
from hydra import initialize_config_dir
CONFIG_DIR = Path("/home/ubuntu/sam3-fine-tuning/sam3/sam3/train/configs")
initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2")
```

**We've already created the solution:** `run_training_fixed.py`

---

## Step-by-Step Training Process

### 1. Verify Environment Setup

```bash
# Check you're in the right directory
cd /home/ubuntu/sam3-fine-tuning
pwd

# Check virtual environment exists
ls -la .venv/

# Verify SAM3 is installed
.venv/bin/python -c "import sys; sys.path.insert(0, 'sam3'); import sam3; print('SAM3 OK')"

# Check GPU
nvidia-smi

# Check dataset exists
ls -la sam3_datasets/fuse-neutrals/
ls -la sam3_datasets/fuse-neutrals/train/_annotations.coco.json
ls -la sam3_datasets/fuse-neutrals/valid/_annotations.coco.json
ls -la sam3_datasets/fuse-neutrals/test/_annotations.coco.json
```

### 2. Review Configuration

Check the training config:
```bash
cat sam3/sam3/train/configs/fuse_neutrals_train.yaml
```

**Key settings:**
- `batch_size: 2` - Adjust if GPU OOM (reduce to 1)
- `num_train_workers: 0` - Set to 0 to save system RAM
- `num_val_workers: 0` - Set to 0 to save system RAM
- `max_epochs: 50` - Total training epochs
- `val_epoch_freq: 5` - Validate every 5 epochs
- `save_freq: 10` - Save checkpoint every 10 epochs

### 3. Clean Up Previous Training (If Resuming)

```bash
# Remove old checkpoints and logs
rm -rf experiments/fuse_neutrals/checkpoints/*
rm -rf experiments/fuse_neutrals/logs/*
rm -rf experiments/fuse_neutrals/dumps/*

# Check disk space
df -h .
```

**Note:** The checkpoint files can be 5-6GB each, so clean up old ones to save space.

### 4. Adjust Memory Settings (If Needed)

Edit the config file if you experienced memory issues:
```bash
nano sam3/sam3/train/configs/fuse_neutrals_train.yaml
```

Change these values:
```yaml
scratch:
  batch_size: 1  # Reduce from 2 if GPU OOM
  num_train_workers: 0  # Keep at 0 to save RAM
  num_val_workers: 0  # Keep at 0 to save RAM
```

### 5. Start Training

**Method 1: Background Training (Recommended)**
```bash
nohup .venv/bin/python run_training_fixed.py > training.log 2>&1 &
echo $!  # Save this process ID
```

**Method 2: Foreground Training (For Testing)**
```bash
.venv/bin/python run_training_fixed.py
```

### 6. Verify Training Started

Wait 30-60 seconds for initialization, then check:

```bash
# Check the process is running
ps aux | grep run_training_fixed | grep -v grep

# Check GPU is being used
nvidia-smi

# Check the log file
tail -50 training.log

# You should see:
# - "SAM3 TRAINING - FUSE NEUTRALS DATASET"
# - "Total parameters 838 M"
# - "Moving components to device cuda:0"
# - Training progress: "Train Epoch: [0][ 0/24]"
```

**Expected GPU Memory Usage:**
- During initialization: 15-17GB
- During training: 16-17GB
- GPU Utilization: 2-100% (fluctuates)

### 7. Monitor Training Progress

**Option 1: Watch the log file**
```bash
tail -f training.log
```

**Option 2: Monitor GPU**
```bash
watch -n 2 nvidia-smi
```

**Option 3: Check specific log sections**
```bash
# See training metrics
grep "Train Epoch" training.log | tail -20

# See validation results
grep "val" experiments/fuse_neutrals/logs/*/log.txt | tail -20

# Check for errors
grep -i "error\|exception\|killed\|oom" training.log
```

### 8. Training Timeline

For **fuse neutrals** (24 training images, batch size 2):
- **Epoch 0 (first epoch):** ~1-2 minutes
- **Subsequent epochs:** ~45-90 seconds each
- **Total time (50 epochs):** ~1-2 hours

**What happens during training:**
- **Epoch 0-4:** Initial learning, loss should decrease rapidly
- **Epoch 5, 10, 15, etc.:** Validation runs every 5 epochs
- **Epoch 10, 20, 30, 40, 50:** Checkpoints saved
- **End of training:** Final checkpoint and validation

---

## Configuration Files

### Training Config Location
```
sam3/sam3/train/configs/fuse_neutrals_train.yaml
```

### Key Configuration Sections

**1. Paths**
```yaml
paths:
  roboflow_vl_100_root: "."  # Project root
  experiment_log_dir: "experiments/fuse_neutrals"
  bpe_path: "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
```

**2. Dataset**
```yaml
roboflow_train:
  num_images: null  # Use all available (24 training images)
  supercategory: "Find fuse neutrals.v5i.coco"  # Dataset folder name
```

**3. Training Hyperparameters**
```yaml
scratch:
  batch_size: 2  # Number of images per batch
  resolution: 1008  # Image resolution (don't change, needed for RoPE)
  num_train_workers: 0  # Data loading workers (0 = main process)
  num_val_workers: 0  # Validation workers
  max_ann_per_img: 50  # Max annotations per image

trainer:
  max_epochs: 50  # Total training epochs
  skip_first_val: false  # Run validation from start
  val_epoch_freq: 5  # Validate every N epochs
  skip_saving_ckpts: false  # Save checkpoints

checkpoint:
  save_dir: ${paths.experiment_log_dir}/checkpoints
  save_freq: 10  # Save every N epochs
  keep_last_n: 3  # Keep last N checkpoints

optimizer:
  lr: 5e-5  # Learning rate
  warmup_epochs: 5  # Warmup for first N epochs
```

### Creating a Config for a New Dataset

1. **Copy existing config:**
   ```bash
   cp sam3/sam3/train/configs/fuse_neutrals_train.yaml \
      sam3/sam3/train/configs/my_dataset_train.yaml
   ```

2. **Edit the new config:**
   ```yaml
   # Change dataset name
   roboflow_train:
     supercategory: "my-dataset.v1.coco"  # Your dataset folder name
   
   # Change output directory
   paths:
     experiment_log_dir: "experiments/my_dataset"
   
   # Adjust hyperparameters if needed
   scratch:
     batch_size: 2  # Adjust based on GPU memory
     max_ann_per_img: 50  # Increase if you have more annotations
   ```

3. **Update `run_training_fixed.py`:**
   ```python
   # Line 53: Change config name
   args = Namespace(
       config="my_dataset_train",  # ← Change this
       ...
   )
   ```

---

## Monitoring Training

### Understanding Log Output

**Initialization Phase (first 30-60 seconds):**
```
SAM3 TRAINING - FUSE NEUTRALS DATASET
======================================================================
Project root: /home/ubuntu/sam3-fine-tuning
Config directory: /home/ubuntu/sam3-fine-tuning/sam3/sam3/train/configs
Available configs: 8
  - fuse_neutrals_train.yaml
  - fuse_cutout_train.yaml
  ...
```

**Model Loading:**
```
INFO ... trainer.py: 	Total parameters 838 M
INFO ... trainer.py: 	Trainable parameters 838 M
INFO ... trainer.py:	Non-Trainable parameters 0
```

**Training Progress:**
```
INFO ... train_utils.py: Train Epoch: [0][ 0/24] | Batch Time: 12.01 (12.01) | 
  Data Time: 0.03 (0.03) | Mem (GB): 15.00 (15.00/15.00) | 
  Losses/train_all_loss: 2.67e+02 (2.67e+02)

INFO ... train_utils.py: Train Epoch: [0][10/24] | Batch Time: 1.21 (2.24) | 
  Data Time: 0.03 (0.03) | Mem (GB): 16.00 (15.91/16.00) | 
  Losses/train_all_loss: 3.18e+01 (8.32e+01)
```

**What to look for:**
- `Train Epoch: [X][Y/24]` - Epoch X, batch Y of 24
- `Batch Time: 1.21 (2.24)` - Current batch time (average time)
- `Mem (GB): 16.00` - GPU memory usage
- `Losses/train_all_loss: 3.18e+01` - Training loss (should decrease)

### Validation Output

Every 5 epochs you'll see:
```
INFO ... Evaluating on validation set...
INFO ... Val Epoch: [5] | mAP: 0.456 | Recall: 0.678 | Precision: 0.543
```

**Good signs:**
- mAP (mean Average Precision) increasing over time
- Loss decreasing over time
- No "OOM" or "killed" messages

**Warning signs:**
- Loss not decreasing after 10+ epochs
- mAP staying at 0 or very low
- GPU memory usage at 100% of available
- "CUDA out of memory" errors

### Checkpoint Files

Checkpoints are saved to:
```
experiments/fuse_neutrals/checkpoints/
├── checkpoint_epoch_10.pth  (~5.4GB)
├── checkpoint_epoch_20.pth  (~5.4GB)
├── checkpoint_epoch_30.pth  (~5.4GB)
├── checkpoint_epoch_40.pth  (~5.4GB)
└── checkpoint_epoch_50.pth  (~5.4GB)
```

**Note:** Each checkpoint is ~5.4GB. The config keeps only the last 3 by default.

### TensorBoard (Optional)

If you want to visualize training metrics:
```bash
# In a separate terminal or screen session
tensorboard --logdir experiments/fuse_neutrals/tensorboard --port 6006

# Then access via:
# http://your-ec2-public-ip:6006
```

---

## Common Issues & Solutions

### Issue 1: "Cannot find primary config" Error

**Symptom:**
```
hydra.errors.MissingConfigException: Cannot find primary config 'fuse_neutrals_train'
```

**Cause:** Using `initialize_config_module()` instead of `initialize_config_dir()`

**Solution:**
✓ Use `run_training_fixed.py` (already implements the fix)

❌ Don't use `run_training.py` or `run_training_simple.py`

---

### Issue 2: CUDA Out of Memory (GPU OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

**Option 1: Reduce batch size**
```bash
nano sam3/sam3/train/configs/fuse_neutrals_train.yaml
```
Change:
```yaml
scratch:
  batch_size: 1  # Reduce from 2
```

**Option 2: Kill other GPU processes**
```bash
# Check what's using GPU
nvidia-smi

# Kill specific process
kill -9 <PID>
```

**Option 3: Lower resolution (not recommended)**
```yaml
scratch:
  resolution: 512  # From 1008 (may affect model quality)
```

---

### Issue 3: System RAM Out of Memory

**Symptom:**
- Process killed without error message
- Training stops during checkpoint saving
- Log shows: `Killed`

**Solution:**
Reduce data loading workers to 0:
```yaml
scratch:
  num_train_workers: 0  # Don't use separate processes
  num_val_workers: 0
```

**Why this helps:**
- Each worker process loads data into RAM
- Worker 0 = main process only
- Checkpoints can be 5.4GB and need RAM to save

---

### Issue 4: Training Process Disappeared

**Check if it's still running:**
```bash
ps aux | grep python | grep training
```

**If not running, check logs:**
```bash
tail -100 training.log
```

**Common causes:**
1. **OOM (Out of Memory)** - See Issue 2 and 3
2. **User interrupted** - You pressed Ctrl+C
3. **Disk full** - Check: `df -h .`
4. **Config error** - Check log for error messages

---

### Issue 5: Training is Extremely Slow

**Expected speeds:**
- First epoch: 1-2 minutes (with model loading)
- Subsequent epochs: 45-90 seconds
- Batch processing: ~1-2 seconds per batch

**If slower than this:**

**Check GPU usage:**
```bash
nvidia-smi
```
- GPU Utilization should be 50-100% during training
- If 0-5%, something is wrong

**Check CPU:**
```bash
top
```
- Python process should use 90-100% of one CPU core

**Check if using CPU instead of GPU:**
```bash
grep "cuda:0" training.log
```
Should see: `Moving components to device cuda:0`

If not, check CUDA installation:
```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```
Should print: `True`

---

### Issue 6: Validation Loss Not Improving

**Normal pattern:**
- Epochs 0-5: Loss decreases rapidly
- Epochs 5-20: Loss decreases slowly
- Epochs 20-50: Loss plateaus or slight decrease

**If loss not decreasing after 10 epochs:**

1. **Check dataset:**
   ```bash
   # Verify annotations exist
   ls -lh sam3_datasets/fuse-neutrals/train/_annotations.coco.json
   
   # Count training images
   ls sam3_datasets/fuse-neutrals/train/*.jpg | wc -l
   ```

2. **Check learning rate:**
   ```yaml
   optimizer:
     lr: 5e-5  # Try: 1e-4 (higher) or 1e-5 (lower)
   ```

3. **Try more epochs:**
   ```yaml
   trainer:
     max_epochs: 100  # From 50
   ```

---

### Issue 7: "ModuleNotFoundError: No module named 'sam3'"

**Cause:** SAM3 not installed or wrong Python environment

**Solution:**
```bash
# Reinstall SAM3
cd /home/ubuntu/sam3-fine-tuning
uv pip install -e sam3/

# Verify installation
.venv/bin/python -c "import sys; sys.path.insert(0, 'sam3'); import sam3; print('OK')"
```

---

### Issue 8: Checkpoint File is Incomplete (.tmp file)

**Symptom:**
```
experiments/fuse_neutrals/checkpoints/checkpoint.pt.tmp  (5.4GB)
```

**Cause:** Training was interrupted during checkpoint saving

**Solution:**
```bash
# Remove incomplete checkpoint
rm experiments/fuse_neutrals/checkpoints/*.tmp

# Restart training (will resume from last saved checkpoint if available)
nohup .venv/bin/python run_training_fixed.py > training.log 2>&1 &
```

---

## After Training

### 1. Find Your Best Checkpoint

```bash
# List all checkpoints
ls -lth experiments/fuse_neutrals/checkpoints/

# The last checkpoint (epoch 50) is usually the best
# But check validation metrics in logs
```

### 2. Test on Test Set

```bash
# Run inference on test images
.venv/bin/python scripts/test_finetuned_model.py \
  --checkpoint experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
  --test-dir sam3_datasets/fuse-neutrals/test \
  --output-dir test_results_fuse_neutrals
```

### 3. Evaluate Performance

```bash
# Calculate mAP and other metrics
.venv/bin/python scripts/evaluate_sam3.py \
  --checkpoint experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
  --test-dir sam3_datasets/fuse-neutrals/test
```

### 4. Run Inference on New Images

```bash
# Predict on a single image
.venv/bin/python scripts/inference_sam3.py \
  --checkpoint experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
  --image path/to/new/image.jpg \
  --output predictions/
```

### 5. Backup Your Model

```bash
# Copy checkpoint to safe location
cp experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
   ~/models/fuse_neutrals_final.pth

# Or upload to S3
aws s3 cp experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
   s3://your-bucket/models/fuse_neutrals_final.pth
```

---

## Training Scripts Reference

### run_training_fixed.py (USE THIS)

**Purpose:** Main training script with Hydra config fix

**Key features:**
- Uses `initialize_config_dir()` for filesystem paths
- Properly adds SAM3 to Python path
- Registers OmegaConf resolvers
- Handles errors gracefully

**Usage:**
```bash
.venv/bin/python run_training_fixed.py
```

**Configuration:** Edit line 53 to change config:
```python
args = Namespace(
    config="fuse_neutrals_train",  # Config name (without .yaml)
    use_cluster=False,
    partition=None,
    account=None,
    qos=None,
    num_gpus=1,
    num_nodes=None,
)
```

### run_training.py (DON'T USE)

**Issue:** Uses `initialize_config_module()` which doesn't work with editable install

### run_training_simple.py (DON'T USE)

**Issue:** Simplified version, but also has Hydra issue

### scripts/train_sam3.py (DON'T USE DIRECTLY)

**Purpose:** Original wrapper script from SAM3 repo

**Issue:** Designed for non-editable install

---

## Advanced: Resuming Interrupted Training

If training was interrupted and you have checkpoints:

1. **Check what checkpoints exist:**
   ```bash
   ls -lth experiments/fuse_neutrals/checkpoints/
   ```

2. **The trainer should auto-resume from the latest checkpoint**
   - SAM3's trainer checks for existing checkpoints
   - It will load the latest one automatically

3. **If you want to start fresh:**
   ```bash
   # Backup old checkpoints
   mv experiments/fuse_neutrals/checkpoints experiments/fuse_neutrals/checkpoints_old
   mkdir experiments/fuse_neutrals/checkpoints
   
   # Start new training
   nohup .venv/bin/python run_training_fixed.py > training.log 2>&1 &
   ```

---

## Quick Reference Commands

### Start Training
```bash
nohup .venv/bin/python run_training_fixed.py > training.log 2>&1 &
```

### Monitor Training
```bash
tail -f training.log
watch -n 2 nvidia-smi
```

### Check Training Progress
```bash
grep "Train Epoch" training.log | tail -10
```

### Stop Training
```bash
# Find process
ps aux | grep run_training_fixed

# Kill it
kill -9 <PID>
```

### Clean Up Before Retraining
```bash
rm -rf experiments/fuse_neutrals/checkpoints/*
rm -rf experiments/fuse_neutrals/logs/*
rm -rf experiments/fuse_neutrals/dumps/*
```

### Check Disk Space
```bash
df -h .
du -sh experiments/fuse_neutrals/checkpoints/
```

---

## Summary Checklist

Before starting training, verify:

- [ ] Dataset is in `sam3_datasets/fuse-neutrals/` with train/valid/test splits
- [ ] Annotations exist: `_annotations.coco.json` in each split
- [ ] Virtual environment exists: `.venv/`
- [ ] SAM3 is installed: `uv pip install -e sam3/`
- [ ] Config file exists: `sam3/sam3/train/configs/fuse_neutrals_train.yaml`
- [ ] GPU is available: `nvidia-smi` shows your GPU
- [ ] Disk space available: `df -h .` shows 50GB+ free
- [ ] Previous training cleaned up: `rm -rf experiments/fuse_neutrals/checkpoints/*`

Then run:
```bash
nohup .venv/bin/python run_training_fixed.py > training.log 2>&1 &
tail -f training.log
```

---

## Need Help?

1. **Check the log file:** `tail -100 training.log`
2. **Search for errors:** `grep -i error training.log`
3. **Check GPU:** `nvidia-smi`
4. **Check process:** `ps aux | grep python`
5. **Review this guide's [Common Issues](#common-issues--solutions) section**

---

**Document Version:** 1.0  
**Last Tested:** February 12, 2026  
**Environment:** AWS EC2 Ubuntu 20.04, NVIDIA A10G (23GB), Python 3.10
