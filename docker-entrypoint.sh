#!/bin/bash
# Docker entrypoint for batch inference
# Processes all images in /app/input and saves results to /app/output

set -e

echo "=========================================="
echo "SAM3 Fuse Neutrals Detection - Batch Mode"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected, running on CPU (slower)"
fi

echo ""

# Configuration
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/app/model/checkpoint.pth}"
INPUT_DIR="${INPUT_DIR:-/app/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/output}"
TEXT_PROMPT="${TEXT_PROMPT:-fuse cutout}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.5}"
DEVICE="${DEVICE:-cuda}"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Input directory: $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Text prompt: '$TEXT_PROMPT'"
echo "  Confidence threshold: $CONFIDENCE_THRESHOLD"
echo "  Device: $DEVICE"
echo ""

# Count input images
num_images=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
echo "Found $num_images images to process"
echo ""

if [ "$num_images" -eq 0 ]; then
    echo "ERROR: No images found in $INPUT_DIR"
    exit 1
fi

# Run inference
echo "Starting inference..."
python3 /app/scripts/test_finetuned_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --image-dir "$INPUT_DIR" \
    --text-prompt "$TEXT_PROMPT" \
    --threshold "$CONFIDENCE_THRESHOLD" \
    --output "$OUTPUT_DIR" \
    --device "$DEVICE"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Batch processing completed successfully"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo "Timestamp: $(date)"
else
    echo ""
    echo "=========================================="
    echo "✗ Batch processing failed with exit code $exit_code"
    echo "=========================================="
    exit $exit_code
fi
