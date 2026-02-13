# Downloading Checkpoint from EC2

This guide shows you how to download your trained SAM3 checkpoint (9.4 GB) from your EC2 instance.

## Quick Start

### 1. Get Your EC2 IP Address
First, find your EC2 instance's public IP address:
- Log into AWS Console
- Go to EC2 → Instances
- Copy the **Public IPv4 address** (e.g., `54.123.45.67`)

### 2. Run the Download Script

```bash
# Make sure you're in the project directory
cd sam3-fine-tuning

# Run the download script with your EC2 IP
./download_checkpoint.sh 54.123.45.67
```

Replace `54.123.45.67` with your actual EC2 IP address.

### 3. Wait for Download
- **Size:** 9.4 GB
- **Time:** 10-30 minutes (depends on your internet speed)
- **Progress:** You'll see real-time progress with rsync

### 4. Find Your Checkpoint
After download completes:
```bash
ls -lh checkpoints_from_ec2/checkpoint.pt
```

---

## What This Script Does

1. ✓ Uses the `object-detection-keypair.pem` file in your project
2. ✓ Connects to your EC2 instance via SSH
3. ✓ Downloads: `/home/ubuntu/sam3-fine-tuning/experiments/fuse_neutrals/checkpoints/checkpoint.pt`
4. ✓ Saves to: `checkpoints_from_ec2/checkpoint.pt` (local)
5. ✓ Shows progress during download
6. ✓ Supports resume if interrupted (using rsync)

---

## Troubleshooting

### Error: "Permission denied (publickey)"
Your PEM key permissions might be wrong. Fix it:
```bash
chmod 400 object-detection-keypair.pem
```

### Error: "Connection timed out"
Your EC2 instance might be:
- Stopped (start it in AWS Console)
- Security group blocking SSH (allow port 22 from your IP)
- Wrong IP address (check AWS Console)

### Error: "Checkpoint not found on EC2"
The checkpoint might be in a different location. SSH to EC2 and check:
```bash
ssh -i object-detection-keypair.pem ubuntu@YOUR_EC2_IP
find ~ -name "checkpoint.pt" -type f
```

### Download Interrupted?
No problem! Just run the script again - rsync will resume from where it stopped:
```bash
./download_checkpoint.sh YOUR_EC2_IP
```

---

## Alternative Methods

### Method 1: Manual SCP (Simple)
```bash
scp -i object-detection-keypair.pem \
  ubuntu@YOUR_EC2_IP:/home/ubuntu/sam3-fine-tuning/experiments/fuse_neutrals/checkpoints/checkpoint.pt \
  ./checkpoints_from_ec2/
```

### Method 2: Via S3 (Recommended for Backup)
On EC2:
```bash
aws s3 cp experiments/fuse_neutrals/checkpoints/checkpoint.pt \
  s3://your-bucket/models/fuse_neutrals/checkpoint.pt
```

On your local machine:
```bash
aws s3 cp s3://your-bucket/models/fuse_neutrals/checkpoint.pt ./
```

### Method 3: Via Azure Blob (For Production)
On EC2:
```bash
az storage blob upload \
  --account-name stfusedetection \
  --container-name models \
  --name fuse_neutrals_checkpoint.pt \
  --file experiments/fuse_neutrals/checkpoints/checkpoint.pt
```

On your local machine:
```bash
az storage blob download \
  --account-name stfusedetection \
  --container-name models \
  --name fuse_neutrals_checkpoint.pt \
  --file ./checkpoint.pt
```

---

## After Download

### Test the Checkpoint
```bash
python scripts/test_inference.py \
  --checkpoint checkpoints_from_ec2/checkpoint.pt \
  --image test_image.jpg
```

### Backup to Azure (Recommended)
```bash
az storage blob upload \
  --account-name stfusedetection \
  --container-name models \
  --name fuse_neutrals_checkpoint.pt \
  --file checkpoints_from_ec2/checkpoint.pt
```

### Check File Integrity
```bash
# On your local machine
ls -lh checkpoints_from_ec2/checkpoint.pt

# Compare size with EC2
ssh -i object-detection-keypair.pem ubuntu@YOUR_EC2_IP \
  "ls -lh /home/ubuntu/sam3-fine-tuning/experiments/fuse_neutrals/checkpoints/checkpoint.pt"
```

---

## Security Notes

- ✓ PEM key is **already in .gitignore** (won't be committed to Git)
- ✓ Keep your PEM key secure (permissions set to 400)
- ✓ Never share your PEM key or commit it to Git
- ✓ The PEM key gives full access to your EC2 instance

---

## File Locations

- **PEM Key:** `object-detection-keypair.pem` (in project root)
- **Download Script:** `download_checkpoint.sh` (in project root)
- **Downloaded Checkpoint:** `checkpoints_from_ec2/checkpoint.pt`
- **EC2 Checkpoint:** `/home/ubuntu/sam3-fine-tuning/experiments/fuse_neutrals/checkpoints/checkpoint.pt`

---

## Need Help?

1. **Check if EC2 is running:**
   ```bash
   ssh -i object-detection-keypair.pem ubuntu@YOUR_EC2_IP
   ```

2. **Test connection:**
   ```bash
   ssh -i object-detection-keypair.pem ubuntu@YOUR_EC2_IP "ls -lh ~/sam3-fine-tuning/experiments/fuse_neutrals/checkpoints/"
   ```

3. **Check your internet speed:**
   - 9.4 GB download times:
   - 100 Mbps: ~13 minutes
   - 50 Mbps: ~25 minutes
   - 25 Mbps: ~50 minutes

---

## Quick Reference

```bash
# Download checkpoint
./download_checkpoint.sh YOUR_EC2_IP

# Check download
ls -lh checkpoints_from_ec2/checkpoint.pt

# Test checkpoint
python scripts/test_inference.py --checkpoint checkpoints_from_ec2/checkpoint.pt --image test.jpg

# Backup to Azure
az storage blob upload --account-name stfusedetection --container-name models \
  --name fuse_neutrals_checkpoint.pt --file checkpoints_from_ec2/checkpoint.pt
```
