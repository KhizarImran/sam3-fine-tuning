# SAM3 Fine-Tuned Model Inference Guide

## Quick Start on EC2

### 1. Pull Latest Code
```bash
cd ~/sam3-fine-tuning
git pull origin main
```

### 2. Activate Environment
```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Run Inference

#### Single Image
```bash
python scripts/test_finetuned_model.py \
    --checkpoint path/to/checkpoint.pt \
    --image path/to/image.jpg \
    --text-prompt "fuse cutout" \
    --threshold 0.5 \
    --output results/
```

#### Batch Processing (Directory of Images)
```bash
python scripts/test_finetuned_model.py \
    --checkpoint path/to/checkpoint.pt \
    --image-dir path/to/images/ \
    --text-prompt "fuse cutout" \
    --threshold 0.5 \
    --output results/
```

## Available Scripts

### 1. `test_finetuned_model.py` (RECOMMENDED)
**Main inference script with full features**
- Text prompt support
- Batch processing
- Visualization with bounding boxes
- JSON results export
- Confidence thresholding

```bash
python scripts/test_finetuned_model.py \
    --checkpoint checkpoints/best_model.pt \
    --image-dir dataset/test/images/ \
    --text-prompt "fuse cutout" \
    --threshold 0.5 \
    --output test_results/ \
    --device cuda
```

### 2. `inference_sam3.py` (FIXED)
**Simplified inference script**
- Same functionality as test_finetuned_model.py
- Alternative implementation
- Fixed to use correct SAM3 API

```bash
python scripts/inference_sam3.py \
    --checkpoint checkpoints/best_model.pt \
    --image-dir dataset/test/images/ \
    --text-prompt "fuse cutout" \
    --threshold 0.5 \
    --output inference_results/ \
    --device cuda
```

### 3. `test_model_working.py`
**Optimized for low threshold testing**
- Default threshold: 0.05 (very sensitive)
- Requires BPE tokenizer file
- Good for finding all possible detections

```bash
python scripts/test_model_working.py \
    --checkpoint checkpoints/best_model.pt \
    --image-dir dataset/test/images/ \
    --text-prompt "fuse cutout" \
    --threshold 0.05 \
    --output low_threshold_results/
```

### 4. `test_model_correct.py`
**Native SAM3 forward pass (advanced)**
- Uses model's internal forward method
- More low-level control
- Good for debugging

```bash
python scripts/test_model_correct.py \
    --checkpoint checkpoints/best_model.pt \
    --image-dir dataset/test/images/ \
    --threshold 0.5 \
    --output native_results/
```

## Command Line Arguments

### Required Arguments
- `--checkpoint`: Path to fine-tuned model checkpoint (.pt file)
- `--image` or `--image-dir`: Single image or directory of images

### Optional Arguments
- `--text-prompt`: Text prompt for detection (default: "fuse cutout")
- `--threshold`: Confidence threshold 0.0-1.0 (default: 0.5)
- `--output`: Output directory for results (default: varies by script)
- `--device`: Device to use: "cuda" or "cpu" (default: "cuda")

## Expected Output

### Console Output
```
================================================================================
SAM3 FINE-TUNED MODEL TESTING
================================================================================
Checkpoint: checkpoints/best_model.pt
Text prompt: 'fuse cutout'
Confidence threshold: 0.5
Device: cuda
Output directory: test_results
================================================================================
Loading model from: checkpoints/best_model.pt
   ✓ Model loaded on cuda

Found 10 images to process

[1/10] Processing: image001.jpg
   ✓ Found 2 detections
   ✓ Average confidence: 0.847
   ✓ Saved visualization: image001_result.jpg
...
```

### Output Files

#### Visualizations
- `{image_name}_result.jpg` - Image with bounding boxes and confidence scores

#### Results JSON (test_finetuned_model.py only)
```json
[
  {
    "image_path": "dataset/test/images/image001.jpg",
    "num_detections": 2,
    "boxes": [[120.5, 340.2, 250.8, 450.6], ...],
    "scores": [0.847, 0.792],
    "image_size": [1920, 1080]
  },
  ...
]
```

## Troubleshooting

### Error: "SAM3 not found"
```bash
# Install SAM3
cd sam3
pip install -e .
```

### Error: "Checkpoint not found"
```bash
# Check checkpoint path
ls -lh checkpoints/
# Use absolute path if needed
python scripts/test_finetuned_model.py --checkpoint /absolute/path/to/checkpoint.pt ...
```

### Error: "CUDA out of memory"
```bash
# Use CPU instead
python scripts/test_finetuned_model.py --device cpu ...
```

### Error: "BPE tokenizer not found" (test_model_working.py)
```bash
# Check if tokenizer exists
ls -l sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

### Low detection count
```bash
# Try lower threshold
python scripts/test_finetuned_model.py --threshold 0.1 ...

# Or use very low threshold script
python scripts/test_model_working.py --threshold 0.05 ...
```

## Understanding Confidence Thresholds

- **0.7-0.9**: High precision, may miss some objects
- **0.5-0.7**: Balanced (recommended for production)
- **0.3-0.5**: Higher recall, more false positives
- **0.05-0.3**: Maximum sensitivity (testing/debugging)

## Best Practices

1. **Start with default threshold (0.5)** then adjust based on results
2. **Use test_finetuned_model.py** for production inference
3. **Check JSON output** for automated processing
4. **Visually inspect** a few results before batch processing
5. **Monitor GPU memory** with `nvidia-smi` during inference

## Performance Expectations

### On AWS EC2 g4dn.xlarge (Tesla T4 16GB)
- Single image: ~2-3 seconds
- Batch 10 images: ~20-30 seconds
- Batch 100 images: ~3-5 minutes

### On AWS EC2 g5.xlarge (A10G 24GB)
- Single image: ~1-2 seconds
- Batch 10 images: ~10-15 seconds
- Batch 100 images: ~2-3 minutes

## Example Production Workflow

```bash
# 1. Pull latest model
cd ~/sam3-fine-tuning
git pull origin main

# 2. Run inference on test set
python scripts/test_finetuned_model.py \
    --checkpoint checkpoints/best_model_v2.pt \
    --image-dir /data/production/batch_001/ \
    --text-prompt "fuse cutout" \
    --threshold 0.6 \
    --output /results/batch_001/ \
    --device cuda

# 3. Review results
ls -lh /results/batch_001/
cat /results/batch_001/results.json

# 4. Process next batch
python scripts/test_finetuned_model.py \
    --checkpoint checkpoints/best_model_v2.pt \
    --image-dir /data/production/batch_002/ \
    --text-prompt "fuse cutout" \
    --threshold 0.6 \
    --output /results/batch_002/ \
    --device cuda
```

## Contact & Support

For issues or questions:
- GitHub: https://github.com/KhizarImran/sam3-fine-tuning
- Check RESEARCH_PAPER.txt for methodology details
- Review SAM3_Fine_Tuning_Plan.txt for training information
