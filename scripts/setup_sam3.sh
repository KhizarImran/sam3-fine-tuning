#!/bin/bash
# SAM3 Setup Script for Fine-tuning Environment

set -e

echo "======================================================================"
echo "SAM3 FINE-TUNING ENVIRONMENT SETUP"
echo "======================================================================"

# Check Python version
echo ""
echo "1. Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "   ERROR: Python 3.12+ required, found $python_version"
    echo "   Please install Python 3.12 or higher"
    exit 1
fi
echo "   Python version: $python_version"

# Check CUDA
echo ""
echo "2. Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "   CUDA version: $cuda_version"
else
    echo "   WARNING: CUDA not found. Training will be CPU-only (very slow)"
fi

# Clone SAM3 repository
echo ""
echo "3. Cloning SAM3 repository..."
if [ -d "sam3" ]; then
    echo "   sam3/ directory already exists, skipping clone"
else
    git clone https://github.com/facebookresearch/sam3.git
    echo "   Repository cloned successfully"
fi

# Install SAM3 with training dependencies
echo ""
echo "4. Installing SAM3 with training dependencies..."
cd sam3
pip install -e ".[train,dev]"
cd ..
echo "   SAM3 installed successfully"

# Install additional requirements
echo ""
echo "5. Installing additional dependencies..."
pip install -r requirements.txt
echo "   Additional dependencies installed"

# Download BPE vocab file (if not already present)
echo ""
echo "6. Checking BPE vocabulary file..."
bpe_file="sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
if [ -f "$bpe_file" ]; then
    echo "   BPE vocab file already exists"
else
    echo "   BPE vocab file should be included with SAM3 repository"
    echo "   If missing, check sam3/sam3/assets/ directory"
fi

# Check GPU availability
echo ""
echo "7. Checking GPU availability..."
python -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else '   No GPU detected')"

echo ""
echo "======================================================================"
echo "SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Request access to SAM3 checkpoints: https://huggingface.co/facebook/sam3"
echo "  2. Prepare your dataset: python scripts/prepare_dataset_for_sam3.py"
echo "  3. Start training: python scripts/train_sam3.py"
echo ""
