#!/bin/bash
# ============================================================================
# Download SAM3 Trained Checkpoint from EC2
# ============================================================================
#
# USAGE:
#   ./download_checkpoint.sh <ec2-ip-address>
#
# EXAMPLE:
#   ./download_checkpoint.sh 54.123.45.67
#
# ============================================================================

set -e

# Configuration (automatically uses PEM key in project directory)
EC2_USER="ubuntu"
SSH_KEY_PATH="object-detection-keypair.pem"
REMOTE_CHECKPOINT_PATH="/home/ubuntu/sam3-fine-tuning/experiments/fuse_neutrals/checkpoints/checkpoint.pt"
LOCAL_DIR="checkpoints_from_ec2"

# Get EC2 IP from command line argument
if [ -z "$1" ]; then
    echo "Error: EC2 IP address required"
    echo "Usage: ./download_checkpoint.sh <ec2-ip-address>"
    echo "Example: ./download_checkpoint.sh 54.123.45.67"
    exit 1
fi

EC2_HOST="$1"

# Verify PEM key exists
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "Error: SSH key not found: $SSH_KEY_PATH"
    echo "Make sure object-detection-keypair.pem is in the project directory"
    exit 1
fi

# Set correct permissions on PEM key
chmod 400 "$SSH_KEY_PATH" 2>/dev/null || true

echo ""
echo "============================================================================"
echo "SAM3 CHECKPOINT DOWNLOAD - FUSE NEUTRALS"
echo "============================================================================"
echo ""
echo "EC2 Host:      $EC2_USER@$EC2_HOST"
echo "SSH Key:       $SSH_KEY_PATH"
echo "Remote Path:   $REMOTE_CHECKPOINT_PATH"
echo "Local Path:    $LOCAL_DIR/checkpoint.pt"
echo ""
echo "Checkpoint Size: ~9.4GB (this will take 10-30 minutes)"
echo "============================================================================"
echo ""

# Create local directory
mkdir -p "$LOCAL_DIR"

# Check if file exists on EC2
echo "Checking if checkpoint exists on EC2..."
if ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "test -f $REMOTE_CHECKPOINT_PATH"; then
    echo "✓ Checkpoint found on EC2"
    
    # Get file size
    FILE_SIZE=$(ssh -i "$SSH_KEY_PATH" "$EC2_USER@$EC2_HOST" "ls -lh $REMOTE_CHECKPOINT_PATH | awk '{print \$5}'")
    echo "  File size: $FILE_SIZE"
    echo ""
else
    echo "✗ Checkpoint not found on EC2"
    echo "  Looking for: $REMOTE_CHECKPOINT_PATH"
    exit 1
fi

# Download using rsync (supports resume if interrupted)
echo "Starting download with rsync (resumable)..."
echo ""

rsync -avz --progress \
    -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no" \
    "$EC2_USER@$EC2_HOST:$REMOTE_CHECKPOINT_PATH" \
    "$LOCAL_DIR/"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "SUCCESS! Checkpoint downloaded"
    echo "============================================================================"
    echo ""
    echo "Checkpoint location: $LOCAL_DIR/checkpoint.pt"
    echo ""
    echo "Checkpoint size:"
    ls -lh "$LOCAL_DIR/checkpoint.pt"
    echo ""
    echo "Next steps:"
    echo "  1. Test the checkpoint:"
    echo "     python scripts/test_inference.py --checkpoint $LOCAL_DIR/checkpoint.pt"
    echo ""
    echo "  2. Backup to Azure for production:"
    echo "     az storage blob upload --account-name stfusedetection \\"
    echo "       --container-name models --name checkpoint.pt \\"
    echo "       --file $LOCAL_DIR/checkpoint.pt"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "ERROR: Download failed"
    echo "============================================================================"
    echo ""
    echo "Possible issues:"
    echo "  1. Incorrect EC2 IP address: $EC2_HOST"
    echo "  2. EC2 instance not running or not accessible"
    echo "  3. SSH port 22 blocked by firewall"
    echo "  4. Wrong remote path (check EC2 directory structure)"
    echo ""
    echo "To troubleshoot, try connecting manually:"
    echo "  ssh -i $SSH_KEY_PATH $EC2_USER@$EC2_HOST"
    echo ""
    exit 1
fi
