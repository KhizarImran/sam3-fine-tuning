# Fuse Neutrals Training Guide

## Dataset Summary
- **Location:** `sam3_datasets/fuse-neutrals/`
- **Train:** 24 images with segmentation annotations
- **Valid:** 5 images with segmentation annotations
- **Test:** 6 images with segmentation annotations
- **Total:** 35 images
- **Class:** "fuse neutrals"

## Quick Start

### 1. Verify Setup
```bash
python scripts/verify_setup.py
```

### 2. Start Training
```bash
python scripts/train_sam3.py -c configs/fuse_neutrals_train.yaml
```

Or with specific GPU count:
```bash
python scripts/train_sam3.py -c configs/fuse_neutrals_train.yaml --num-gpus 1
```

### 3. Monitor Training
Training logs and checkpoints will be saved to:
- **Logs:** `experiments/fuse_neutrals/`
- **Checkpoints:** `experiments/fuse_neutrals/checkpoints/`

## Configuration Details

### Key Settings in `configs/fuse_neutrals_train.yaml`:
- **Dataset:** `sam3_datasets/fuse-neutrals`
- **Batch size:** 2 (adjust down to 1 if you get OOM errors)
- **Epochs:** 50
- **Validation frequency:** Every 5 epochs
- **Checkpoint save frequency:** Every 10 epochs
- **Learning rate:** 5e-5 (good for fine-tuning)

### Adjustable Parameters:

#### If you get Out Of Memory (OOM) errors:
1. Reduce `batch_size` from 2 to 1
2. Reduce `num_train_workers` from 2 to 0
3. Reduce `max_ann_per_img` from 50 to 25

#### For faster training:
1. Increase `batch_size` (if GPU memory allows)
2. Increase `num_train_workers` and `num_val_workers`

#### For better accuracy:
1. Increase `max_epochs` (try 100)
2. Add data augmentation (requires code changes)

## Testing After Training

### Run inference on test set:
```bash
python scripts/test_finetuned_model.py \
  --checkpoint experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth \
  --test-dir sam3_datasets/fuse-neutrals/test \
  --output-dir test_results
```

## Expected Training Time
- **With GPU:** ~2-5 minutes per epoch = 2-4 hours total for 50 epochs
- **Without GPU:** Much slower (not recommended)

## Troubleshooting

### Dataset not found error:
- Check that `sam3_datasets/fuse-neutrals/` exists
- Verify train/valid/test folders have images and `_annotations.coco.json`

### CUDA out of memory:
- Reduce batch_size to 1 in config
- Close other GPU-using applications

### Training not starting:
- Run `python scripts/verify_setup.py` to check prerequisites
- Ensure SAM3 is installed: check `sam3/` directory exists
- Check PyTorch and CUDA are installed

## Next Steps After Training
1. Evaluate on test set
2. Visualize predictions
3. Fine-tune hyperparameters if needed
4. Collect more training data if accuracy is low
