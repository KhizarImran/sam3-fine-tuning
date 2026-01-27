# SAM3 Fine-Tuning for Fuse Cutout Detection

Fine-tune Facebook's SAM3 (Segment Anything Model 3) for detecting fuse cutouts in electrical panel images.

## Quick Start

```bash
# 1. Setup environment
python scripts\setup_sam3.bat

# 2. Get SAM3 checkpoint access
# Visit: https://huggingface.co/facebook/sam3

# 3. Prepare your dataset
uv run scripts/prepare_dataset_for_sam3.py

# 4. Start training
uv run scripts/train_sam3.py

# 5. Monitor with TensorBoard
tensorboard --logdir experiments/fuse_cutout/tensorboard
```

## Project Structure

```
sam3-fine-tuning/
├── configs/                            # Training configurations
│   └── fuse_cutout_train.yaml         # Main config file
├── scripts/                            # All scripts
│   ├── setup_sam3.bat/.sh             # Environment setup
│   ├── roboflow_download.py           # Download annotated data
│   ├── roboflow_validator.py          # Validate annotations
│   ├── visualize_annotations.py       # View annotations
│   ├── prepare_dataset_for_sam3.py    # Format for SAM3
│   ├── train_sam3.py                  # Main training script
│   └── inference_sam3.py              # Test trained model
├── fuse-netrual-training-dataset/     # Downloaded from Roboflow
├── photos/                             # Original images
└── SAM3_TRAINING_GUIDE.md             # Complete guide
```

## Current Status

✅ **Annotation Complete**: 4 images with 9 fuse cutout masks
✅ **Dataset Format**: COCO JSON with polygon segmentations
⏳ **Next Step**: SAM3 training pipeline setup

## Workflow Overview

### Phase 1: Data Annotation (DONE ✓)

1. Upload images to Roboflow
2. Annotate with Smart Polygon tool
3. Export in COCO format
4. Download to local machine

**Result**: 4 images, 9 annotations with segmentation masks

### Phase 2: Training Setup (Current)

1. Install SAM3 and dependencies
2. Request checkpoint access
3. Prepare dataset for SAM3
4. Configure training parameters

### Phase 3: Training

1. Test pipeline with 4 images (expect overfitting)
2. Validate end-to-end workflow
3. Scale up to 100-200 images
4. Retrain for production

### Phase 4: Deployment

1. Evaluate model performance
2. Copy .pth checkpoint to production EC2
3. Update production API
4. A/B test vs pre-trained model

## Key Files

- **[SAM3_TRAINING_GUIDE.md](SAM3_TRAINING_GUIDE.md)** - Complete training guide
- **[ROBOFLOW_GUIDE.md](ROBOFLOW_GUIDE.md)** - Annotation workflow
- **[configs/fuse_cutout_train.yaml](configs/fuse_cutout_train.yaml)** - Training config
- **[requirements.txt](requirements.txt)** - Python dependencies

## Requirements

- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+
- NVIDIA GPU (16GB+ VRAM recommended)
- 20GB+ free disk space

## Documentation

1. **Annotation**: See [ROBOFLOW_GUIDE.md](ROBOFLOW_GUIDE.md)
2. **Training**: See [SAM3_TRAINING_GUIDE.md](SAM3_TRAINING_GUIDE.md)
3. **Original Plan**: See [SAM3_Fine_Tuning_Plan.txt](SAM3_Fine_Tuning_Plan.txt)

## Commands Reference

### Setup

```bash
# Windows
python scripts\setup_sam3.bat

# Linux/Mac
bash scripts/setup_sam3.sh
```

### Dataset Preparation

```bash
# Download from Roboflow (if needed)
uv run scripts/roboflow_download.py

# Validate annotations
uv run scripts/roboflow_validator.py

# Visualize annotations
uv run scripts/visualize_annotations.py

# Prepare for SAM3
uv run scripts/prepare_dataset_for_sam3.py
```

### Training

```bash
# Basic training
uv run scripts/train_sam3.py

# With custom config
uv run scripts/train_sam3.py --config configs/my_config.yaml

# Multiple GPUs
uv run scripts/train_sam3.py --num-gpus 2

# Monitor training
tensorboard --logdir experiments/fuse_cutout/tensorboard
```

### Inference

```bash
# Single image
uv run scripts/inference_sam3.py \
  --checkpoint experiments/fuse_cutout/checkpoints/best_checkpoint.pth \
  --image test.jpg

# Batch processing
uv run scripts/inference_sam3.py \
  --checkpoint experiments/fuse_cutout/checkpoints/best_checkpoint.pth \
  --image-dir test_images/ \
  --output results/
```

## Training Strategy

### Pilot Phase (Current - 4 images)

- **Goal**: Validate training pipeline
- **Duration**: ~30 minutes
- **Expected**: Overfitting (normal)
- **Outcome**: Confirm workflow works

### Production Phase (Future - 100+ images)

- **Goal**: Production-ready model
- **Duration**: 4-6 hours training
- **Target Metrics**:
  - mAP > 0.6
  - Precision > 0.7
  - Recall > 0.8

## Environment

- **Dev EC2**: Training environment (Tesla T4 GPU)
- **Production EC2**: Inference API (port 8000)
- **Dataset**: S3 bucket (dno-datasets)

## Success Metrics

**Pipeline Validation** (4 images):
- ✓ Training runs without errors
- ✓ Checkpoints saved successfully
- ✓ TensorBoard logging works
- ✓ Can run inference on test images

**Production Model** (100+ images):
- Better accuracy than pre-trained SAM3
- mAP > 0.6 minimum
- Fast inference (<1s per image)
- Robust to varied lighting/angles

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce `batch_size_train` to 1 in config
- Reduce `gradient_accumulation_steps`

**Training Too Slow**
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Should return `True`

**Checkpoint Access Denied**
- Request access: https://huggingface.co/facebook/sam3
- Wait 1-2 days for approval

**Dataset Not Found**
- Run: `uv run scripts/prepare_dataset_for_sam3.py`
- Check: `sam3_datasets/fuse-cutout-detection/`

## Resources

- **SAM3 Repository**: https://github.com/facebookresearch/sam3
- **HuggingFace**: https://huggingface.co/facebook/sam3
- **Roboflow**: https://roboflow.com
- **TensorBoard**: http://localhost:6006

## License

This project uses Facebook's SAM3 model. See SAM3 repository for licensing.

## Contact

For questions about this fine-tuning setup, refer to the documentation files in this repository.

---

**Status**: Ready for training pipeline setup ✓
