#!/bin/bash

# SAM3 Quick Inference Script
# Usage: ./run_inference.sh [checkpoint_path] [image_dir] [optional: threshold]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check arguments
if [ "$#" -lt 2 ]; then
    echo -e "${RED}Usage: $0 <checkpoint_path> <image_dir> [threshold]${NC}"
    echo ""
    echo "Examples:"
    echo "  $0 checkpoints/best_model.pt dataset/test/images/"
    echo "  $0 checkpoints/best_model.pt dataset/test/images/ 0.6"
    exit 1
fi

CHECKPOINT=$1
IMAGE_DIR=$2
THRESHOLD=${3:-0.5}  # Default to 0.5 if not provided
OUTPUT_DIR="inference_results_$(date +%Y%m%d_%H%M%S)"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo -e "${RED}Error: Image directory not found: $IMAGE_DIR${NC}"
    exit 1
fi

# Check for images in directory
IMAGE_COUNT=$(find "$IMAGE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No images found in $IMAGE_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}SAM3 Fine-Tuned Inference${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "Checkpoint:  ${YELLOW}$CHECKPOINT${NC}"
echo -e "Image dir:   ${YELLOW}$IMAGE_DIR${NC}"
echo -e "Images:      ${YELLOW}$IMAGE_COUNT${NC}"
echo -e "Threshold:   ${YELLOW}$THRESHOLD${NC}"
echo -e "Output:      ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo -e "${YELLOW}Attempting to activate...${NC}"

    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    else
        echo -e "${RED}Error: Virtual environment not found at .venv/${NC}"
        echo -e "${RED}Please run: python -m venv .venv${NC}"
        exit 1
    fi
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA available${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    DEVICE="cuda"
else
    echo -e "${YELLOW}⚠ CUDA not available, using CPU${NC}"
    DEVICE="cpu"
fi
echo ""

# Run inference
echo -e "${GREEN}Starting inference...${NC}"
echo ""

python scripts/test_finetuned_model.py \
    --checkpoint "$CHECKPOINT" \
    --image-dir "$IMAGE_DIR" \
    --text-prompt "fuse cutout" \
    --threshold "$THRESHOLD" \
    --output "$OUTPUT_DIR" \
    --device "$DEVICE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}✓ Inference completed successfully!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "Results saved to: ${YELLOW}$OUTPUT_DIR/${NC}"
    echo ""
    echo "View results:"
    echo "  ls -lh $OUTPUT_DIR/"
    echo "  cat $OUTPUT_DIR/results.json"
else
    echo ""
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}✗ Inference failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}======================================${NC}"
    exit $EXIT_CODE
fi
