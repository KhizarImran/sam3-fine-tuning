# Production Deployment - Quick Start Guide
## SAM3 Fuse Neutrals Detection for M Group Energy Data Insight Ltd.

**Target**: 100-1000 images/day | Azure Infrastructure | £40-80/month estimated cost

---

## Recommended Approach: Scheduled GPU VM (Simplest)

This is the **easiest and most cost-effective** solution for your volume:

### Why This Approach?
✅ Simple to setup and maintain  
✅ Easy to debug and monitor  
✅ Cost-effective (only pay for 1-2 hours/day)  
✅ No complex orchestration needed  
✅ Perfect for 100-1000 images/day  

### Architecture
```
Supplier → Azure Blob Storage → Scheduled VM (GPU) → Process Images → Output Storage
                                   ↑
                               Runs 4x daily
                            (Auto-starts/stops)
```

---

## Setup Steps (2-3 hours total)

### Step 1: Azure Storage Setup (30 minutes)

```bash
# Create resource group and storage
RESOURCE_GROUP="rg-fuse-detection-prod"
STORAGE_ACCOUNT="stfusedetection"  # Change to unique name
LOCATION="uksouth"

az group create --name $RESOURCE_GROUP --location $LOCATION

az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

# Get storage key
STORAGE_KEY=$(az storage account keys list \
  --resource-group $RESOURCE_GROUP \
  --account-name $STORAGE_ACCOUNT \
  --query '[0].value' -o tsv)

# Create containers
az storage container create --name input-images --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
az storage container create --name output-images --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
az storage container create --name model-checkpoints --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY

# Upload your trained model
az storage blob upload \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --container-name model-checkpoints \
  --name checkpoint.pth \
  --file experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth
```

### Step 2: Build Docker Container (1 hour)

```bash
# On your EC2 instance or local machine with Docker

# Build the image
docker build -f Dockerfile.production -t fuse-neutrals-detector:v1.0 .

# Test locally first (optional but recommended)
docker run --gpus all \
  -v $(pwd)/Find\ fuse\ neutrals.v5i.coco/test:/app/input \
  -v $(pwd)/test_output:/app/output \
  -e CHECKPOINT_PATH=/app/model/checkpoint.pth \
  fuse-neutrals-detector:v1.0

# Create Azure Container Registry
ACR_NAME="acrfusedetection"  # Change to unique name
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic

# Push to ACR
az acr login --name $ACR_NAME
docker tag fuse-neutrals-detector:v1.0 $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0
docker push $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0
```

### Step 3: Create GPU VM (30 minutes)

```bash
# Create GPU VM
VM_NAME="vm-fuse-detector"

az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --size Standard_NC6s_v3 \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Get VM IP
VM_IP=$(az vm show -d \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --query publicIps -o tsv)

echo "VM created: $VM_IP"
```

### Step 4: Setup VM for Processing (30 minutes)

```bash
# SSH to VM
ssh azureuser@$VM_IP

# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to ACR
az acr login --name $ACR_NAME

# Pull your Docker image
docker pull $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0

# Create processing script
cat > /home/azureuser/process_batch.sh <<'EOF'
#!/bin/bash
set -e

STORAGE_ACCOUNT="stfusedetection"
STORAGE_KEY="<YOUR_STORAGE_KEY>"
ACR_NAME="acrfusedetection"

echo "========================================"
echo "Fuse Neutrals Batch Processing Started"
echo "Time: $(date)"
echo "========================================"

# Run the batch processor
docker run --gpus all \
  --rm \
  $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0 \
  python3 /app/scripts/azure_batch_processor.py \
    --storage-account $STORAGE_ACCOUNT \
    --storage-key $STORAGE_KEY \
    --input-container input-images \
    --output-container output-images \
    --checkpoint /app/model/checkpoint.pth \
    --batch-size 100 \
    --device cuda

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Processing completed successfully"
else
  echo "✗ Processing failed with exit code $EXIT_CODE"
fi
echo "Time: $(date)"
echo "========================================"

# Auto-shutdown after processing (optional)
# sudo shutdown -h +5
EOF

chmod +x /home/azureuser/process_batch.sh

# Test run
./process_batch.sh
```

### Step 5: Schedule Automated Processing (30 minutes)

**Option A: Azure Automation (Recommended)**

```bash
# Create automation account
az automation account create \
  --resource-group $RESOURCE_GROUP \
  --name automation-fuse-detection \
  --location $LOCATION

# Create runbook to start VM, process, stop VM
# See full guide in AZURE_PRODUCTION_DEPLOYMENT.md
```

**Option B: Cron Job on VM (Simpler)**

```bash
# On the VM, setup cron to run 4x daily
crontab -e

# Add these lines (runs at 2am, 8am, 2pm, 8pm):
0 2,8,14,20 * * * /home/azureuser/process_batch.sh >> /home/azureuser/batch.log 2>&1
```

**Option C: Manual Trigger (For Testing)**

```bash
# Just manually SSH and run:
ssh azureuser@$VM_IP './process_batch.sh'
```

---

## Supplier Integration

### Give Supplier Access to Upload Images

```bash
# Generate SAS token (valid for 1 year)
END_DATE=$(date -u -d "1 year" '+%Y-%m-%dT%H:%MZ')

SAS_TOKEN=$(az storage container generate-sas \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --name input-images \
  --permissions acdlrw \
  --expiry $END_DATE \
  --https-only \
  --output tsv)

# Share this URL with supplier:
echo "Upload URL: https://$STORAGE_ACCOUNT.blob.core.windows.net/input-images?$SAS_TOKEN"
```

### Supplier Upload Methods

**Method 1: Azure Storage Explorer (Easiest)**
1. Download [Azure Storage Explorer](https://azure.microsoft.com/products/storage/storage-explorer/)
2. Connect using the SAS URL above
3. Drag and drop images

**Method 2: AzCopy Command-Line**
```bash
azcopy copy "local_folder/*" "https://$STORAGE_ACCOUNT.blob.core.windows.net/input-images?$SAS_TOKEN"
```

**Method 3: Python Script**
```python
# supplier_upload.py
from azure.storage.blob import BlobServiceClient
from pathlib import Path

connection_string = "BlobEndpoint=https://$STORAGE_ACCOUNT.blob.core.windows.net/;SharedAccessSignature=$SAS_TOKEN"
blob_service = BlobServiceClient.from_connection_string(connection_string)
container = blob_service.get_container_client("input-images")

for file in Path("./images").glob("*.jpg"):
    print(f"Uploading: {file.name}")
    with open(file, "rb") as data:
        container.upload_blob(name=file.name, data=data)
```

---

## Daily Operation

### Morning: Supplier Uploads Images
- Supplier uploads 100-500 images to `input-images` container
- Can batch upload anytime during the day

### Processing: Automated (4x daily)
- VM starts automatically at 2am, 8am, 2pm, 8pm
- Processes all pending images
- Generates annotated images + JSON metadata
- Saves to `output-images/2026-02-12/` folder
- VM auto-shuts down (saves cost)

### Review: M Group Team
- Download results from `output-images` container
- Review annotated images
- Use JSON metadata for analysis

---

## Monitoring and Alerts

### Check Processing Status

```bash
# List recent outputs
az storage blob list \
  --container-name output-images \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --prefix "$(date +%Y-%m-%d)/" \
  --query "[].name" -o tsv

# Check VM logs
ssh azureuser@$VM_IP 'tail -100 /home/azureuser/batch.log'
```

### Setup Email Alerts

```bash
# Alert when VM fails to process
az monitor metrics alert create \
  --name "fuse-detection-failed" \
  --resource-group $RESOURCE_GROUP \
  --scopes "/subscriptions/<SUB_ID>/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Compute/virtualMachines/$VM_NAME" \
  --condition "avg Percentage CPU < 5" \
  --window-size 1h \
  --evaluation-frequency 30m \
  --action email your-email@mgroupenergy.com
```

---

## Cost Breakdown

### Monthly Costs (500 images/day):

| Component | Usage | Cost |
|-----------|-------|------|
| **Azure Storage** | 100GB blob storage | £5/month |
| **GPU VM (NC6s_v3)** | 1 hour/day x 30 days | £90/month |
| **Container Registry** | Basic tier | £4/month |
| **Data Transfer** | Outbound 50GB | £3/month |
| **Total** | | **~£102/month** |

### Cost Optimization Tips:

1. **Use Spot Instances**: Save 60-90% on VM costs
   ```bash
   az vm create ... --priority Spot --max-price 0.50
   ```

2. **Process Once Daily**: Instead of 4x daily, process once at night
   - Reduces to 30 hours/month = **£28/month**

3. **Use Smaller GPU**: If inference is fast enough
   - NC4as_T4_v3 @ £0.53/hour = **£16/month** (70% savings)

4. **Reserved Instance**: Commit 1 year for 30% discount
   - If processing daily, worth it

**Optimized cost: £35-50/month** with spot + daily processing

---

## Troubleshooting

### Issue: No images being processed
```bash
# Check input container has images
az storage blob list --container-name input-images --account-name $STORAGE_ACCOUNT

# Check VM is running
az vm show -d --resource-group $RESOURCE_GROUP --name $VM_NAME

# Check Docker container logs
ssh azureuser@$VM_IP 'docker logs <container_id>'
```

### Issue: GPU not detected
```bash
ssh azureuser@$VM_IP
nvidia-smi  # Should show GPU

# If not, install drivers:
sudo apt update
sudo apt install nvidia-driver-515
sudo reboot
```

### Issue: Model loading fails
```bash
# Verify checkpoint exists in blob storage
az storage blob show \
  --container-name model-checkpoints \
  --name checkpoint.pth \
  --account-name $STORAGE_ACCOUNT

# Test Docker image locally
docker run --gpus all --rm $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0 \
  ls -lh /app/model/
```

---

## Next Steps

### This Week:
1. [ ] Setup Azure Storage and upload model checkpoint
2. [ ] Build and test Docker container locally
3. [ ] Create VM and test single batch processing
4. [ ] Give supplier upload access

### Next Week:
1. [ ] Process first production batch with real supplier data
2. [ ] Setup automated scheduling (cron or Azure Automation)
3. [ ] Configure monitoring and alerts
4. [ ] Document procedures for team

### Ongoing:
- Monitor costs weekly
- Review output quality monthly
- Retrain model quarterly with new data

---

## Quick Reference Commands

```bash
# Start VM manually
az vm start --resource-group $RESOURCE_GROUP --name $VM_NAME

# Stop VM manually
az vm deallocate --resource-group $RESOURCE_GROUP --name $VM_NAME

# Run processing manually
ssh azureuser@$VM_IP './process_batch.sh'

# Download today's results
az storage blob download-batch \
  --destination ./results \
  --source output-images \
  --account-name $STORAGE_ACCOUNT \
  --pattern "$(date +%Y-%m-%d)/*"

# Check costs
az consumption usage list \
  --start-date $(date -d "1 month ago" +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d)
```

---

## Support

- **Full Deployment Guide**: See `AZURE_PRODUCTION_DEPLOYMENT.md`
- **Docker Files**: `Dockerfile.production`, `docker-entrypoint.sh`
- **Batch Processor**: `scripts/azure_batch_processor.py`
- **Azure Documentation**: https://docs.microsoft.com/azure/

---

**Last Updated**: February 12, 2026  
**Version**: 1.0  
**M Group Energy Data Insight Ltd.**
