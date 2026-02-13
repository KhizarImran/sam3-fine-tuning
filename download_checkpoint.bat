@echo off
REM ============================================================================
REM Download SAM3 Trained Checkpoint from EC2
REM ============================================================================
REM
REM INSTRUCTIONS:
REM 1. Edit the variables below with your EC2 details
REM 2. Run this script: download_checkpoint.bat
REM 3. Wait for the 9.4GB checkpoint to download
REM
REM ============================================================================

REM EDIT THESE VARIABLES:
set EC2_USER=ubuntu
set EC2_HOST=ec2-13-40-83-241.eu-west-2.compute.amazonaws.com
set SSH_KEY_PATH=object-detection-keypair.pem
set REMOTE_CHECKPOINT_PATH=~/sam3-fine-tuning/experiments/fuse_cutout/checkpoints/checkpoint.pt

REM Don't edit below this line
REM ============================================================================

echo.
echo ============================================================================
echo SAM3 CHECKPOINT DOWNLOAD
echo ============================================================================
echo.
echo EC2 Host: %EC2_USER%@%EC2_HOST%
echo SSH Key: %SSH_KEY_PATH%
echo Remote Path: %REMOTE_CHECKPOINT_PATH%
echo Local Path: experiments\fuse_cutout\checkpoints\
echo.
echo Checkpoint Size: ~9.4GB (this will take a while)
echo ============================================================================
echo.

REM Create local directory
if not exist "experiments\fuse_cutout\checkpoints\" (
    echo Creating local checkpoint directory...
    mkdir "experiments\fuse_cutout\checkpoints\"
)

REM Download checkpoint using scp
echo Starting download...
echo.
scp -i "%SSH_KEY_PATH%" %EC2_USER%@%EC2_HOST%:%REMOTE_CHECKPOINT_PATH% experiments\fuse_cutout\checkpoints\checkpoint.pt

if %errorlevel% equ 0 (
    echo.
    echo ============================================================================
    echo SUCCESS! Checkpoint downloaded
    echo ============================================================================
    echo.
    echo Checkpoint location: experiments\fuse_cutout\checkpoints\checkpoint.pt
    echo.
    echo You can now run inference:
    echo   uv run python scripts\test_fuse_neutral.py
    echo.
) else (
    echo.
    echo ============================================================================
    echo ERROR: Download failed
    echo ============================================================================
    echo.
    echo Possible issues:
    echo   1. Incorrect EC2 IP/hostname
    echo   2. Wrong SSH key path
    echo   3. SSH key permissions
    echo   4. EC2 instance not running
    echo   5. Firewall blocking SSH (port 22)
    echo.
    echo Please check your settings and try again.
    echo.
)

pause
