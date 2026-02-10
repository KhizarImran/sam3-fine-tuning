# SAM3 Fuse Neutrals Training - EC2 Setup Guide

## Project Context

This repository contains a fine-tuned SAM3 model for detecting **fuse neutrals** in electrical fuse boxes. The dataset consists of 35 labeled images with polygon segmentation masks exported from Roboflow in COCO format.

### Dataset Information
- **Location:** `sam3_datasets/fuse-neutrals/`
- **Train:** 24 images with segmentation annotations
- **Valid:** 5 images with segmentation annotations  
- **Test:** 6 images with segmentation annotations
- **Total:** 35 images @ 512x512 resolution
- **Class:** "fuse neutrals" (single class detection)
- **Format:** COCO JSON with polygon segmentation masks

### Training Configuration
- **Config file:** `configs/fuse_neutrals_train.yaml`
- **Base model:** SAM3 (Segment Anything Model 3)
- **Batch size:** 2 (reduce to 1 if OOM)
- **Epochs:** 50
- **Learning rate:** 5e-5
- **Validation frequency:** Every 5 epochs
- **Checkpoint save frequency:** Every 10 epochs
- **Output:** `experiments/fuse_neutrals/checkpoints/`

---

## EC2 Setup Instructions

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd sam3-fine-tuning
```

### 2. Verify Dataset is Present
```bash
ls -la sam3_datasets/fuse-neutrals/train/
ls -la sam3_datasets/fuse-neutrals/valid/
ls -la sam3_datasets/fuse-neutrals/test/

# Should see:
# - 24 .jpg images in train/
# - 5 .jpg images in valid/
# - 6 .jpg images in test/
# - _annotations.coco.json in each folder
```

### 3. Install SAM3 (First Time Only)
```bash
# Run the setup script
bash scripts/setup_sam3.sh

# Or manually:
git clone https://github.com/facebookresearch/sam2.git sam3
cd sam3
pip install -e .
cd ..
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt

# Key dependencies:
# - torch (with CUDA)
# - torchvision
# - hydra-core
# - opencv-python
# - pycocotools
# - numpy
# - pillow
```

### 5. Download SAM3 Base Checkpoint (Required)
```bash
# Download the pretrained SAM3 checkpoint
bash download_checkpoint.sh

# Or manually download to: checkpoints/sam3_base.pth
```

### 6. Verify Setup
```bash
python scripts/verify_setup.py

# This checks:
# - SAM3 installation
# - PyTorch/CUDA availability
# - Dataset presence
# - Config file validity
```

### 7. Start Training
```bash
python scripts/train_sam3.py -c configs/fuse_neutrals_train.yaml

# Or with specific GPU count:
python scripts/train_sam3.py -c configs/fuse_neutrals_train.yaml --num-gpus 1
```

---

## Training Monitoring

### Check Training Progress
```bash
# View latest logs
tail -f experiments/fuse_neutrals/train.log

# Check checkpoints
ls -lh experiments/fuse_neutrals/checkpoints/
```

### Expected Output
- Training logs in `experiments/fuse_neutrals/`
- Checkpoints saved every 10 epochs
- Validation runs every 5 epochs
- Training time: ~2-5 minutes per epoch (GPU-dependent)
- Total time: ~2-4 hours for 50 epochs

---

## Troubleshooting

### GPU Out of Memory
Edit `configs/fuse_neutrals_train.yaml`:
```yaml
scratch:
  batch_size: 1  # Reduce from 2 to 1
  num_train_workers: 0  # Reduce from 2 to 0
```

### Dataset Not Found
```bash
# Verify paths in config match actual structure
grep -A 5 "roboflow_train:" configs/fuse_neutrals_train.yaml

# Should show:
# supercategory: "fuse-neutrals"
```

### SAM3 Not Installed
```bash
# Check if sam3 directory exists
ls sam3/

# If not, run setup
bash scripts/setup_sam3.sh
```

### CUDA Not Available
```bash
# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## After Training

### 1. Find Best Checkpoint
```bash
ls -lth experiments/fuse_neutrals/checkpoints/
# Look for checkpoint_epoch_50.pth or best_model.pth
```

### 2. Test on Test Set
```bash
python scripts/test_finetuned_model.py \
  --checkpoint experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
  --test-dir sam3_datasets/fuse-neutrals/test \
  --output-dir test_results_fuse_neutrals
```

### 3. Visualize Results
```bash
# Check output directory for visualizations
ls test_results_fuse_neutrals/

# Should contain:
# - Predicted segmentation masks
# - Overlaid visualizations
# - Metrics JSON
```

### 4. Evaluate Performance
```bash
python scripts/run_eval_on_test.sh \
  experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
  sam3_datasets/fuse-neutrals/test

# Outputs:
# - mAP (mean Average Precision)
# - Per-class metrics
# - Confusion matrix
```

---

## Key Files Reference

### Configuration
- `configs/fuse_neutrals_train.yaml` - Main training config
- `configs/roboflow_v100/` - Base SAM3 configs (don't modify)

### Scripts
- `scripts/train_sam3.py` - Training wrapper script
- `scripts/verify_setup.py` - Environment verification
- `scripts/test_finetuned_model.py` - Model testing
- `scripts/setup_sam3.sh` - SAM3 installation

### Dataset
- `sam3_datasets/fuse-neutrals/train/_annotations.coco.json` - Train annotations
- `sam3_datasets/fuse-neutrals/valid/_annotations.coco.json` - Val annotations
- `sam3_datasets/fuse-neutrals/test/_annotations.coco.json` - Test annotations

### Output
- `experiments/fuse_neutrals/` - Training logs and metrics
- `experiments/fuse_neutrals/checkpoints/` - Model checkpoints

---

## Quick Commands Summary

```bash
# Setup
git pull origin main
pip install -r requirements.txt
bash scripts/setup_sam3.sh
bash download_checkpoint.sh

# Verify
python scripts/verify_setup.py

# Train
python scripts/train_sam3.py -c configs/fuse_neutrals_train.yaml

# Monitor
tail -f experiments/fuse_neutrals/train.log

# Test
python scripts/test_finetuned_model.py \
  --checkpoint experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
  --test-dir sam3_datasets/fuse-neutrals/test \
  --output-dir test_results
```

---

## Dataset Details

The dataset was prepared with the following steps:
1. Original 36 images from OneDrive
2. Split into 70/15/15 train/val/test ratio (25/5/6 images)
3. Labeled in Roboflow with polygon segmentation masks
4. Exported as COCO format (v5)
5. Labels cleaned to single category: "fuse neutrals"
6. All images resized to 512x512 during export

### Data Quality
- ✓ All images have annotations
- ✓ Segmentation masks are polygon-based (not bounding boxes only)
- ✓ Single class: "fuse neutrals"
- ✓ Consistent resolution: 512x512
- ✓ COCO format validated

---

## Model Performance Expectations

### Dataset Size Considerations
- **Small dataset** (35 images) - expect some overfitting
- Fine-tuning from SAM3 pretrained weights helps significantly
- Consider data augmentation if accuracy is low
- May need more diverse training images for production use

### Recommended Next Steps After First Training
1. Evaluate on test set to get baseline metrics
2. If mAP < 0.7, collect more training data
3. If overfitting, add data augmentation or reduce epochs
4. If underfitting, increase epochs or learning rate

---

## Contact & Notes

- Training was configured for AWS EC2 with GPU (g4dn.xlarge or similar)
- Expects CUDA-capable GPU with at least 8GB VRAM
- Can run on CPU but will be extremely slow (not recommended)
- Config optimized for single GPU training

Last updated: February 10, 2026
