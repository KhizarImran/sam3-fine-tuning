# Testing Your Trained SAM3 Model

## Quick Start

Once training is complete, run the test script:

```bash
# Simple test with defaults
uv run python scripts/test_fuse_neutral.py

# Or with explicit paths
uv run python scripts/test_fuse_neutral.py \
  --checkpoint experiments/fuse_cutout/checkpoints/checkpoint.pt \
  --images photos/test_fuse_neutral \
  --prompt "fuse neutral" \
  --threshold 0.05
```

## What It Does

The script will:
1. ✅ Load your trained checkpoint
2. ✅ Process all images in `photos/test_fuse_neutral/` (5 images)
3. ✅ Detect "fuse neutral" objects using text prompt
4. ✅ Draw bounding boxes on detections
5. ✅ Save visualizations to `test_results/`
6. ✅ Save JSON results with all detection data

## Test Images Available

```
photos/test_fuse_neutral/
├── 20210624_120633.jpg (2.8MB)
├── 20210630_093348.jpg (2.2MB)
├── 9f5d5994-7f8a-4ef4-a8a2-49ee4212a20c.jpg (60KB)
├── IMG_4815.jpg (2.0MB)
└── f62b5edb-ef65-4b7c-8596-a7452300c900.jpg (76KB)
```

## Output

Results will be saved to `test_results/`:
- `{image_name}_detected.jpg` - Visualizations with bounding boxes
- `inference_results.json` - Structured detection data

## Adjusting Detection Sensitivity

If you're not seeing detections:

```bash
# Try lower threshold (more sensitive, may have false positives)
uv run python scripts/test_fuse_neutral.py --threshold 0.01

# Or very low threshold to see all predictions
uv run python scripts/test_fuse_neutral.py --threshold 0.001
```

If you're seeing too many false positives:

```bash
# Higher threshold (less sensitive, only high-confidence detections)
uv run python scripts/test_fuse_neutral.py --threshold 0.3
```

## Expected Output

```
================================================================================
SAM3 FUSE NEUTRAL DETECTION - INFERENCE
================================================================================
Checkpoint: experiments/fuse_cutout/checkpoints/checkpoint.pt
Text Prompt: 'fuse neutral'
Confidence Threshold: 0.05
================================================================================
✓ Using GPU: NVIDIA A10G

📦 Loading model from checkpoint...
✓ Model loaded successfully
✓ Processor initialized (threshold=0.05)

📸 Found 5 test images
================================================================================

[1/5] Processing: 20210624_120633.jpg
   Image size: 4032x3024
   ✓ Found 2 fuse neutral(s)
   ✓ Score range: 0.456 - 0.823
   ✓ Average score: 0.640
   ✓ Saved visualization: 20210624_120633_detected.jpg

[2/5] Processing: 20210630_093348.jpg
   ...
```

## Troubleshooting

### "Checkpoint not found"
Training hasn't finished yet. Wait for training to complete (check with `tail -f experiments/fuse_cutout/training_clean.log`)

### "No detections found"
- Lower the threshold: `--threshold 0.01`
- Check that training completed successfully
- Verify the text prompt matches training: `--prompt "fuse neutral"`

### "Out of memory"
The model uses ~15GB GPU memory. If inference fails, ensure training is stopped first.

## After Testing

If results look good, you can:
1. Test on new images by pointing `--images` to a different directory
2. Integrate this into your pipeline using the `test_checkpoint_inference()` function
3. Save the checkpoint for deployment

## Training Status

Check training progress:
```bash
# Watch training logs
tail -f experiments/fuse_cutout/training_clean.log

# Check for completed checkpoint
ls -lh experiments/fuse_cutout/checkpoints/
```

Training is configured for 30 epochs with validation every 10 epochs.
