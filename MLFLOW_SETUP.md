# MLflow Integration Guide

Complete guide for using MLflow experiment tracking with SAM3 training.

## Overview

Your training will automatically log to MLflow at: **http://52.2.51.33:5000**

All metrics, parameters, and artifacts will be tracked and visualized in the MLflow UI.

---

## Quick Start

### 1. Install MLflow

```powershell
uv pip install mlflow
```

### 2. Test Connection

```powershell
python -c "import mlflow; mlflow.set_tracking_uri('http://52.2.51.33:5000'); print('Connected:', mlflow.search_experiments() is not None)"
```

Expected output: `Connected: True`

### 3. Train with MLflow

```powershell
# Basic training with MLflow
uv run scripts/train_sam3_with_mlflow.py

# With custom config
uv run scripts/train_sam3_with_mlflow.py --config configs/fuse_cutout_train.yaml

# Disable MLflow (use regular training)
uv run scripts/train_sam3_with_mlflow.py --disable-mlflow
```

---

## What Gets Logged

### Parameters (Hyperparameters)
- Learning rates (transformer, vision backbone, language backbone)
- Batch sizes
- Epochs
- Model architecture settings
- Dataset information

### Metrics (Per Epoch)
- Training loss
- Validation loss
- mAP (mean Average Precision)
- Precision, Recall, F1
- IoU (Intersection over Union)
- Learning rate (current)

### Artifacts
- Best model checkpoint (.pth)
- Training configuration (YAML)
- Sample predictions (images with masks)
- Training logs

### System Metrics
- GPU utilization
- Memory usage
- Training time per epoch

---

## MLflow UI

### View Your Experiments

Open in browser: **http://52.2.51.33:5000**

### Navigate the UI

1. **Experiments Tab** - List all experiments
   - Your experiment: `sam3-fuse-cutout-detection`

2. **Runs** - Individual training runs
   - Each run has a timestamp
   - Click on a run to see details

3. **Compare Runs** - Select multiple runs to compare
   - Side-by-side metric charts
   - Parameter differences highlighted

4. **Models** - Registered models
   - Promote best checkpoints to production

---

## Configuration

Edit [configs/fuse_cutout_train.yaml](configs/fuse_cutout_train.yaml):

```yaml
mlflow:
  # Enable/disable MLflow
  enabled: true

  # Your MLflow server
  tracking_uri: "http://52.2.51.33:5000"

  # Experiment name (groups related runs)
  experiment_name: "sam3-fuse-cutout-detection"

  # Run name prefix (will be timestamped)
  run_name: "sam3_fuse_pilot"

  # Tags for filtering
  tags:
    model: "sam3"
    task: "fuse_cutout_detection"
    dataset: "fuse-cutout-detection"
    environment: "dev-ec2"

  # Logging options
  log_params: true
  log_metrics: true
  log_artifacts: true
  log_system_metrics: true
```

---

## Common Tasks

### Compare Multiple Training Runs

1. Go to http://52.2.51.33:5000
2. Navigate to experiment: `sam3-fuse-cutout-detection`
3. Select 2+ runs using checkboxes
4. Click **"Compare"**
5. View side-by-side charts

### Download Best Model

1. Find the run with best metrics
2. Click on the run
3. Go to **"Artifacts"** tab
4. Download `model/best_checkpoint.pth`

### Register a Model

1. Open the best run
2. Go to **"Artifacts"** → **"model"**
3. Click **"Register Model"**
4. Name it: `sam3-fuse-cutout-v1`
5. Stage: `Production`

### Track Multiple Experiments

Create different experiments for different approaches:

```yaml
# Experiment 1: Baseline
experiment_name: "sam3-fuse-baseline"

# Experiment 2: With augmentation
experiment_name: "sam3-fuse-augmented"

# Experiment 3: Different learning rate
experiment_name: "sam3-fuse-lr-tuning"
```

---

## Troubleshooting

### MLflow Connection Failed

**Issue**: Cannot connect to http://52.2.51.33:5000

**Solutions**:
1. Check if MLflow server is running on EC2
2. Verify port 5000 is open in security group
3. Test with curl: `curl http://52.2.51.33:5000`
4. Check EC2 instance is running

### No Metrics Showing Up

**Issue**: Training runs but no metrics in MLflow

**Solutions**:
1. Check MLflow is enabled in config: `mlflow.enabled: true`
2. Verify training script is using MLflow logger
3. Check MLflow server logs for errors

### Artifacts Not Uploading

**Issue**: Checkpoints not appearing in MLflow

**Solutions**:
1. Check disk space on MLflow server
2. Verify artifact location is writable
3. Check file sizes (very large files may timeout)

---

## Best Practices

### Experiment Organization

```
sam3-fuse-cutout-detection/          # Main experiment
├── run_20260127_pilot_4images       # Initial test
├── run_20260201_full_100images      # First production run
├── run_20260208_augmented           # With data augmentation
└── run_20260215_tuned_lr            # Fine-tuned learning rate
```

### Naming Conventions

**Experiment names**: Describe the overall goal
- `sam3-fuse-cutout-detection`
- `sam3-fuse-production`

**Run names**: Describe this specific training
- `pilot_4images_baseline`
- `prod_100images_lr0.001`
- `test_augmentation_heavy`

### Tagging Strategy

Use tags to filter and organize:

```yaml
tags:
  model: "sam3"
  task: "fuse_cutout_detection"
  dataset_size: "100"
  augmentation: "true"
  gpu: "tesla_t4"
  purpose: "pilot"  # or "production", "experiment"
```

---

## MLflow API Usage

### Python Script

```python
import mlflow

# Set tracking server
mlflow.set_tracking_uri("http://52.2.51.33:5000")

# Get experiment
experiment = mlflow.get_experiment_by_name("sam3-fuse-cutout-detection")

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_map DESC"],
    max_results=10
)

# Print best runs
print("Top 10 runs by validation mAP:")
print(runs[['run_id', 'metrics.val_map', 'params.lr_transformer']])
```

### Load Best Model

```python
import mlflow.pytorch

# Get best run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_map DESC"],
    max_results=1
)

best_run_id = runs.iloc[0]['run_id']

# Download artifact
artifact_path = mlflow.artifacts.download_artifacts(
    run_id=best_run_id,
    artifact_path="model/best_checkpoint.pth"
)

print(f"Downloaded best model to: {artifact_path}")
```

---

## Integration with TensorBoard

Both TensorBoard and MLflow will run simultaneously:

**TensorBoard** (local): Real-time training monitoring
```powershell
tensorboard --logdir experiments/fuse_cutout/tensorboard
```

**MLflow** (centralized): Experiment tracking and model registry
```
http://52.2.51.33:5000
```

Use TensorBoard for detailed loss curves during training.
Use MLflow for comparing runs and managing models.

---

## Next Steps

1. ✓ MLflow configured
2. ✓ Training will log automatically
3. → Run your first training
4. → View results at http://52.2.51.33:5000
5. → Compare different experiments
6. → Register best model for production

---

## Resources

- **MLflow Server**: http://52.2.51.33:5000
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **Python API**: https://mlflow.org/docs/latest/python_api/index.html

---

**Ready to train!** Your metrics will automatically appear in MLflow as training progresses.
