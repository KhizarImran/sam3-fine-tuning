# SAM3 Fine-Tuning Guide for Fuse Cutout Detection

Complete step-by-step guide to fine-tune SAM3 on your custom fuse cutout dataset.

---

## Overview

This guide will walk you through:
1. **Environment Setup** - Install SAM3 and dependencies
2. **Dataset Preparation** - Format your Roboflow data for SAM3
3. **Model Access** - Get SAM3 checkpoint from HuggingFace
4. **Training** - Fine-tune SAM3 on your dataset
5. **Evaluation** - Test the fine-tuned model
6. **Deployment** - Copy trained model to production

**Current Status**: 4 annotated images with segmentation masks ready for pipeline testing

---

## Prerequisites

- **Python**: 3.12+
- **CUDA**: 12.6+ (recommended for GPU training)
- **GPU**: NVIDIA GPU with 16GB+ VRAM (Tesla T4 on your dev EC2)
- **Disk Space**: 20GB+ free

---

## Step 1: Environment Setup

### 1.1 Install Dependencies

```bash
# Install base requirements
uv pip install -r requirements.txt

# Setup SAM3 (Windows)
python scripts\setup_sam3.bat

# Or on Linux/Mac
bash scripts/setup_sam3.sh
```

This will:
- Clone the SAM3 repository
- Install SAM3 with training dependencies
- Verify GPU availability

### 1.2 Verify Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
CUDA available: True
```

---

## Step 2: Request SAM3 Model Access

### 2.1 Get HuggingFace Access

1. Go to: https://huggingface.co/facebook/sam3
2. Click **"Request Access"** (requires HuggingFace account)
3. Wait for approval (usually 1-2 days)

### 2.2 Download Checkpoint

Once approved:

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Download checkpoint
huggingface-cli download facebook/sam3 --local-dir sam3_checkpoints
```

### 2.3 Update Config

Edit `configs/fuse_cutout_train.yaml`:

```yaml
model:
  checkpoint: "sam3_checkpoints/sam3_checkpoint.pth"  # Update this path
```

---

## Step 3: Prepare Your Dataset

### 3.1 Current Dataset

You have:
- **4 images** with segmentation masks
- **9 fuse cutout annotations** total
- COCO JSON format from Roboflow

Location: `fuse-netrual-training-dataset/`

### 3.2 Convert to SAM3 Format

```bash
uv run scripts/prepare_dataset_for_sam3.py
```

This creates:
```
sam3_datasets/
└── fuse-cutout-detection/
    ├── train/           # 3 images
    ├── valid/           # 1 image
    ├── _annotations.coco.json
    └── categories.txt   # Text prompts for SAM3
```

### 3.3 Verify Dataset

Check the output:
- Train images: `sam3_datasets/fuse-cutout-detection/train/`
- Validation split created automatically
- COCO annotations preserved

---

## Step 4: Configure Training

### 4.1 Review Configuration

Open `configs/fuse_cutout_train.yaml` and verify:

```yaml
paths:
  roboflow_vl_100_root: "sam3_datasets"  # Dataset location
  experiment_log_dir: "experiments/fuse_cutout"  # Where to save results

model:
  checkpoint: "sam3_checkpoints/sam3_checkpoint.pth"  # ← UPDATE THIS
  do_segmentation: true  # Enable mask training

trainer:
  max_epochs: 30
  batch_size_train: 1  # For Tesla T4
  gradient_accumulation_steps: 4  # Effective batch = 4
  lr_transformer: 0.0008
```

### 4.2 Key Settings for 4-Image Testing

Since you're testing with only 4 images:

- **Epochs**: 30 (allows model to converge despite small dataset)
- **Batch size**: 1 (fits Tesla T4 memory)
- **Gradient accumulation**: 4 (simulates batch size of 4)
- **Validation frequency**: Every 5 epochs

**Expected behavior**: Model will overfit (this is normal and expected with 4 images)

---

## Step 5: Start Training

### 5.1 Run Training

```bash
# Basic training (single GPU)
uv run scripts/train_sam3.py

# With custom config
uv run scripts/train_sam3.py --config configs/fuse_cutout_train.yaml

# Multiple GPUs (if available)
uv run scripts/train_sam3.py --num-gpus 2
```

### 5.2 Monitor Training

**In another terminal**, start TensorBoard:

```bash
tensorboard --logdir experiments/fuse_cutout/tensorboard
```

Then open: http://localhost:6006

You'll see:
- Loss curves (train & validation)
- Learning rates
- Sample predictions

### 5.3 Training Duration

With 4 images on Tesla T4:
- **Total time**: ~15-30 minutes for 30 epochs
- **Per epoch**: ~30-60 seconds
- **Checkpoints**: Saved every 10 epochs

### 5.4 Expected Results

With only 4 images:
- ✓ Training loss will decrease rapidly
- ✓ Validation loss may be unstable (expected)
- ✓ Model will overfit to training data (expected)
- ✓ Pipeline validation successful!

**This is a test run** - performance will improve with 100+ images.

---

## Step 6: Check Training Results

### 6.1 Find Your Checkpoint

Trained model location:
```
experiments/fuse_cutout/
├── checkpoints/
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_20.pth
│   ├── checkpoint_epoch_30.pth
│   └── best_checkpoint.pth  ← Use this one
├── tensorboard/
└── logs/
```

### 6.2 Test Inference

```bash
uv run scripts/inference_sam3.py \
  --checkpoint experiments/fuse_cutout/checkpoints/best_checkpoint.pth \
  --image fuse-netrual-training-dataset/train/image_7.jpg \
  --text-prompt "fuse neutral" \
  --output test_results/
```

**Note**: Inference script is a template - SAM3's exact inference API needs to be added based on official documentation.

---

## Step 7: Scale Up (After Pipeline Test)

Once the pipeline works with 4 images:

### 7.1 Annotate More Images

**Target**: 100-200 images minimum

1. Upload more images to Roboflow
2. Annotate with Smart Polygon tool
3. Export new version
4. Download and prepare dataset

### 7.2 Update Configuration

For 100+ images, adjust in `configs/fuse_cutout_train.yaml`:

```yaml
trainer:
  max_epochs: 50  # More epochs for larger dataset
  val_every_n_epochs: 10
```

### 7.3 Retrain

```bash
# Prepare new dataset
uv run scripts/prepare_dataset_for_sam3.py --source new_roboflow_download/

# Train with more data
uv run scripts/train_sam3.py
```

**Expected improvement**: Much better generalization and less overfitting

---

## Step 8: Deployment to Production EC2

### 8.1 Copy Trained Model

Once you have a good model:

```bash
# From dev EC2, copy to production EC2
scp experiments/fuse_cutout/checkpoints/best_checkpoint.pth \
    user@production-ec2:/path/to/production/models/fuse_cutout_v1.pth
```

### 8.2 Update Production API

On production EC2:

1. Update model loading code to use fine-tuned checkpoint
2. Restart API service
3. Test with sample images
4. Compare results with pre-trained model

### 8.3 A/B Testing (Optional)

- Keep pre-trained model as baseline
- Run fine-tuned model on port 8001 initially
- Compare accuracy before full deployment

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in config

```yaml
trainer:
  batch_size_train: 1  # Already at minimum
  gradient_accumulation_steps: 2  # Reduce from 4
```

### Issue: Training Very Slow

**Possible causes**:
1. Running on CPU instead of GPU
2. Check: `python -c "import torch; print(torch.cuda.is_available())"`
3. If False, reinstall PyTorch with CUDA support

### Issue: Model Not Improving

With 4 images:
- This is expected - need more data
- Focus on pipeline validation, not accuracy

With 100+ images:
- Check learning rate (may be too high/low)
- Increase training epochs
- Verify annotations are correct

### Issue: SAM3 Checkpoint Access Denied

- Request access at: https://huggingface.co/facebook/sam3
- Wait for approval (1-2 days typically)
- Check email for confirmation

---

## File Structure

```
sam3-fine-tuning/
├── configs/
│   └── fuse_cutout_train.yaml         # Training configuration
├── scripts/
│   ├── setup_sam3.bat/.sh             # Environment setup
│   ├── roboflow_download.py           # Download from Roboflow
│   ├── prepare_dataset_for_sam3.py    # Format dataset
│   ├── train_sam3.py                  # Main training script
│   └── inference_sam3.py              # Test trained model
├── fuse-netrual-training-dataset/     # Your Roboflow data
├── sam3_datasets/                     # Prepared SAM3 format
│   └── fuse-cutout-detection/
├── sam3/                              # SAM3 repository (cloned)
├── sam3_checkpoints/                  # Downloaded model weights
├── experiments/                       # Training outputs
│   └── fuse_cutout/
└── requirements.txt                   # Python dependencies
```

---

## Quick Reference Commands

```bash
# Setup
python scripts\setup_sam3.bat

# Prepare dataset
uv run scripts/prepare_dataset_for_sam3.py

# Train
uv run scripts/train_sam3.py

# Monitor
tensorboard --logdir experiments/fuse_cutout/tensorboard

# Inference
uv run scripts/inference_sam3.py \
  --checkpoint experiments/fuse_cutout/checkpoints/best_checkpoint.pth \
  --image test_image.jpg
```

---

## Next Steps After 4-Image Test

1. ✓ Validate pipeline works end-to-end
2. ✓ Confirm GPU training functional
3. ✓ Check TensorBoard logging
4. → Annotate 96+ more images in Roboflow
5. → Retrain with full dataset
6. → Evaluate on test set
7. → Deploy to production EC2

---

## Success Criteria

**Pipeline Test (4 images)**:
- ✓ Training completes without errors
- ✓ Checkpoints saved successfully
- ✓ TensorBoard shows loss curves
- ✓ Can load and run inference

**Production Model (100+ images)**:
- mAP > 0.6 (pilot target)
- Precision > 0.7
- Recall > 0.8
- Better than pre-trained baseline

---

## Support & Resources

- **SAM3 GitHub**: https://github.com/facebookresearch/sam3
- **SAM3 Paper**: Check repository for paper link
- **HuggingFace**: https://huggingface.co/facebook/sam3
- **Training Docs**: sam3/README_TRAIN.md

---

## Notes

- With 4 images, expect overfitting (this validates the pipeline)
- 100-200 images minimum for production model
- Tesla T4 (16GB) can handle batch size of 1
- Training time scales linearly with dataset size
- Fine-tuned model will be ~1.7GB (.pth file)

**Good luck with your fine-tuning!** 🚀
