@echo off
REM SAM3 Setup Script for Fine-tuning Environment (Windows)

echo ======================================================================
echo SAM3 FINE-TUNING ENVIRONMENT SETUP
echo ======================================================================

REM Check Python version
echo.
echo 1. Checking Python version...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR: Python not found
    echo    Please install Python 3.12 or higher
    exit /b 1
)

REM Check CUDA
echo.
echo 2. Checking CUDA availability...
nvcc --version 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    CUDA detected
) else (
    echo    WARNING: CUDA not found. Training will be CPU-only (very slow)
)

REM Clone SAM3 repository
echo.
echo 3. Cloning SAM3 repository...
if exist sam3 (
    echo    sam3\ directory already exists, skipping clone
) else (
    git clone https://github.com/facebookresearch/sam3.git
    if %ERRORLEVEL% NEQ 0 (
        echo    ERROR: Failed to clone repository
        exit /b 1
    )
    echo    Repository cloned successfully
)

REM Install SAM3 with training dependencies
echo.
echo 4. Installing SAM3 with training dependencies...
cd sam3
pip install -e ".[train,dev]"
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR: Failed to install SAM3
    exit /b 1
)
cd ..
echo    SAM3 installed successfully

REM Install additional requirements
echo.
echo 5. Installing additional dependencies...
pip install -r requirements.txt
echo    Additional dependencies installed

REM Check GPU availability
echo.
echo 6. Checking GPU availability...
python -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else '   No GPU detected')"

echo.
echo ======================================================================
echo SETUP COMPLETE!
echo ======================================================================
echo.
echo Next steps:
echo   1. Request access to SAM3 checkpoints: https://huggingface.co/facebook/sam3
echo   2. Prepare your dataset: python scripts\prepare_dataset_for_sam3.py
echo   3. Start training: python scripts\train_sam3.py
echo.
pause
