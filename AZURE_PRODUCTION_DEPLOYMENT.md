# Azure Production Deployment Guide
## SAM3 Fuse Neutrals Detection System for M Group Energy Data Insight Ltd.

**Target Architecture:** Azure Batch Processing Pipeline  
**Expected Volume:** 100-1000 images/day  
**Latency:** Flexible (batch processing)  
**Input Method:** Azure Blob Storage  
**Output Format:** Annotated images + JSON metadata  

---

## Architecture Overview

```
Supplier Images 
    ↓
Azure Blob Storage (input-images container)
    ↓
Azure Function (Blob trigger) OR Manual trigger
    ↓
Azure Batch (GPU Pool: NC6s_v3 instances)
    ↓
Docker Container running SAM3 inference
    ↓
Azure Blob Storage (output-images container)
    ↓
Email/Teams notification with summary
```

---

## Cost Estimate (Monthly)

### For 500 images/day (15,000/month):
- **Azure Blob Storage**: £5-10/month (100GB storage)
- **Azure Batch GPU Compute**: £30-60/month (~1 hour/day of NC6s_v3 @ £3.06/hour)
- **Azure Functions**: £0 (free tier sufficient for <1M executions)
- **Data Transfer**: £5-10/month
- **Total: ~£40-80/month** (scales down to £0 on days with no images)

### Comparison to Always-On GPU:
- Always-on NC6s_v3: £2,203/month (24/7 @ £3.06/hour)
- **Savings: 96-98%** by using batch processing with auto-scaling

---

## Phase 1: Azure Storage Setup (30 minutes)

### 1.1 Create Storage Account

```bash
# Variables
RESOURCE_GROUP="rg-fuse-detection-prod"
LOCATION="uksouth"  # UK South for M Group
STORAGE_ACCOUNT="stfusedetection"  # Must be globally unique, lowercase, no hyphens

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create storage account
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS \
  --kind StorageV2

# Get storage key
STORAGE_KEY=$(az storage account keys list \
  --resource-group $RESOURCE_GROUP \
  --account-name $STORAGE_ACCOUNT \
  --query '[0].value' -o tsv)

echo "Storage account created: $STORAGE_ACCOUNT"
echo "Storage key: $STORAGE_KEY"
```

### 1.2 Create Blob Containers

```bash
# Create input container (supplier uploads here)
az storage container create \
  --name input-images \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --public-access off

# Create output container (annotated results)
az storage container create \
  --name output-images \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --public-access off

# Create model container (store checkpoint)
az storage container create \
  --name model-checkpoints \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --public-access off
```

### 1.3 Upload Fine-tuned Model

```bash
# Upload your trained checkpoint to Azure
az storage blob upload \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --container-name model-checkpoints \
  --name checkpoint_epoch_50.pth \
  --file experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth

echo "Model checkpoint uploaded"
```

### 1.4 Grant Supplier Access (Shared Access Signature)

```bash
# Generate SAS token for supplier to upload images (valid for 1 year)
END_DATE=$(date -u -d "1 year" '+%Y-%m-%dT%H:%MZ')

az storage container generate-sas \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --name input-images \
  --permissions acdlrw \
  --expiry $END_DATE \
  --https-only \
  --output tsv

# Share this SAS URL with supplier:
# https://$STORAGE_ACCOUNT.blob.core.windows.net/input-images?<SAS_TOKEN>
```

---

## Phase 2: Docker Container Setup (1 hour)

### 2.1 Build Production Docker Image

On your local machine or EC2 instance:

```bash
# Build the image
docker build -f Dockerfile.production -t fuse-neutrals-detector:v1.0 .

# Test locally (optional)
docker run --gpus all \
  -v $(pwd)/test_data:/app/input \
  -v $(pwd)/test_output:/app/output \
  fuse-neutrals-detector:v1.0
```

### 2.2 Push to Azure Container Registry

```bash
# Create Azure Container Registry
ACR_NAME="acrfusedetection"  # Must be globally unique, alphanumeric only

az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic

# Login to ACR
az acr login --name $ACR_NAME

# Tag and push image
docker tag fuse-neutrals-detector:v1.0 $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0
docker push $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0

echo "Docker image pushed to: $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0"
```

---

## Phase 3: Azure Batch Setup (1 hour)

### 3.1 Create Azure Batch Account

```bash
# Create Batch account
BATCH_ACCOUNT="batchfusedetection"

az batch account create \
  --name $BATCH_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Link storage account for auto-storage
az batch account set \
  --name $BATCH_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --storage-account $STORAGE_ACCOUNT

# Get Batch account endpoint
BATCH_ENDPOINT=$(az batch account show \
  --name $BATCH_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query accountEndpoint -o tsv)

echo "Batch account: $BATCH_ACCOUNT"
echo "Batch endpoint: $BATCH_ENDPOINT"
```

### 3.2 Create GPU Pool for Processing

```bash
# Login to Batch
az batch account login \
  --name $BATCH_ACCOUNT \
  --resource-group $RESOURCE_GROUP

# Create pool configuration file
cat > pool-config.json <<EOF
{
  "id": "gpu-pool-fuse-detection",
  "displayName": "GPU Pool for Fuse Neutral Detection",
  "vmSize": "Standard_NC6s_v3",
  "virtualMachineConfiguration": {
    "imageReference": {
      "publisher": "microsoft-azure-batch",
      "offer": "ubuntu-server-container",
      "sku": "20-04-lts",
      "version": "latest"
    },
    "nodeAgentSkuId": "batch.node.ubuntu 20.04",
    "containerConfiguration": {
      "type": "dockercompatible",
      "containerImageNames": [
        "$ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0"
      ],
      "containerRegistries": [
        {
          "registryServer": "$ACR_NAME.azurecr.io",
          "userName": "$ACR_NAME",
          "password": "$(az acr credential show -n $ACR_NAME --query passwords[0].value -o tsv)"
        }
      ]
    }
  },
  "targetDedicatedNodes": 0,
  "targetLowPriorityNodes": 0,
  "enableAutoScale": true,
  "autoScaleFormula": "
    // Scale up if tasks are waiting
    \$samples = \$ActiveTasks.GetSamplePercent(TimeInterval_Minute * 5);
    \$tasks = \$samples < 70 ? max(0, \$ActiveTasks.GetSample(1)) : max(\$ActiveTasks.GetSample(1), avg(\$ActiveTasks.GetSample(TimeInterval_Minute * 5)));
    \$targetVMs = \$tasks > 0 ? min(5, ceil(\$tasks / 1)) : 0;
    \$TargetDedicatedNodes = \$targetVMs;
    \$TargetLowPriorityNodes = 0;
    \$NodeDeallocationOption = taskcompletion;
  ",
  "autoScaleEvaluationInterval": "PT5M"
}
EOF

# Create the pool
az batch pool create \
  --json-file pool-config.json \
  --account-name $BATCH_ACCOUNT \
  --resource-group $RESOURCE_GROUP
```

**Pool Configuration Explained:**
- **VM Size**: `Standard_NC6s_v3` - 1x V100 GPU, 6 vCPUs, 112GB RAM
- **Auto-scaling**: Scales 0-5 nodes based on task queue
  - 0 nodes when idle (zero cost)
  - Spins up nodes when tasks arrive
  - Scales down after completion
- **Container**: Pulls your Docker image from ACR
- **GPU drivers**: Pre-installed on Ubuntu container image

---

## Phase 4: Batch Job Submission (Automated)

### 4.1 Create Batch Job Script

```bash
cat > submit-batch-job.sh <<'EOF'
#!/bin/bash
# Submit batch job to process images from Azure Blob Storage

BATCH_ACCOUNT="batchfusedetection"
RESOURCE_GROUP="rg-fuse-detection-prod"
POOL_ID="gpu-pool-fuse-detection"
STORAGE_ACCOUNT="stfusedetection"

# Generate unique job ID
JOB_ID="fuse-detection-$(date +%Y%m%d-%H%M%S)"

echo "Creating batch job: $JOB_ID"

# Create job
az batch job create \
  --id $JOB_ID \
  --pool-id $POOL_ID \
  --account-name $BATCH_ACCOUNT \
  --resource-group $RESOURCE_GROUP

# List unprocessed images in input container
IMAGES=$(az storage blob list \
  --container-name input-images \
  --account-name $STORAGE_ACCOUNT \
  --query "[].name" -o tsv)

TASK_NUM=0
for IMAGE in $IMAGES; do
  TASK_NUM=$((TASK_NUM + 1))
  
  echo "Adding task $TASK_NUM: $IMAGE"
  
  # Create task for this image
  az batch task create \
    --job-id $JOB_ID \
    --task-id "task-$TASK_NUM" \
    --account-name $BATCH_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --json-file - <<TASK_JSON
{
  "commandLine": "python3 /app/scripts/azure_batch_processor.py --storage-account $STORAGE_ACCOUNT --storage-key \$STORAGE_KEY --input-container input-images --output-container output-images --checkpoint /app/model/checkpoint.pth --batch-size 1",
  "containerSettings": {
    "imageName": "acrfusedetection.azurecr.io/fuse-neutrals-detector:v1.0"
  },
  "environmentSettings": [
    {
      "name": "STORAGE_KEY",
      "value": "\$STORAGE_KEY"
    }
  ]
}
TASK_JSON
done

echo "Job $JOB_ID created with $TASK_NUM tasks"
EOF

chmod +x submit-batch-job.sh
```

### 4.2 Schedule with Cron or Azure Logic Apps

**Option A: Cron Job (Simple)**
```bash
# Run every 6 hours
crontab -e
# Add line:
0 */6 * * * /path/to/submit-batch-job.sh
```

**Option B: Azure Logic App (Recommended for production)**
- Trigger: Recurrence (every 6 hours) OR Blob created event
- Action: Run Azure Batch job via REST API
- Better monitoring, alerting, and error handling

---

## Phase 5: Alternative - Simple Scheduled VM (Easier Setup)

If Azure Batch seems complex, here's a simpler alternative:

### 5.1 Create GPU VM with Auto-Shutdown

```bash
# Create a single GPU VM
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name vm-fuse-detector \
  --size Standard_NC6s_v3 \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --admin-username azureuser \
  --generate-ssh-keys

# Install Docker and run your processor
ssh azureuser@<VM_IP>

# Pull and run container
docker pull $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0

# Run batch processor (processes all images, then exits)
docker run --gpus all \
  -e STORAGE_ACCOUNT=$STORAGE_ACCOUNT \
  -e STORAGE_KEY=$STORAGE_KEY \
  $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0 \
  python3 /app/scripts/azure_batch_processor.py \
    --storage-account $STORAGE_ACCOUNT \
    --storage-key $STORAGE_KEY \
    --input-container input-images \
    --output-container output-images \
    --checkpoint /app/model/checkpoint.pth \
    --batch-size 100

# Auto-shutdown after processing
sudo shutdown -h now
```

### 5.2 Schedule VM Start/Stop with Azure Automation

- **Daily schedule**: Start VM at 2am, process images, auto-shutdown
- **Cost**: Only pay for ~1-2 hours/day instead of 24/7
- **Simpler than Batch**: Just one VM, easier to debug

---

## Phase 6: Monitoring and Notifications (30 minutes)

### 6.1 Azure Monitor Alerts

```bash
# Alert when processing completes
az monitor metrics alert create \
  --name "fuse-detection-complete" \
  --resource-group $RESOURCE_GROUP \
  --scopes "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Batch/batchAccounts/$BATCH_ACCOUNT" \
  --condition "avg TaskCompleteEvent > 0" \
  --description "Alert when batch processing completes"
```

### 6.2 Email Notification Script

```python
# Add to azure_batch_processor.py after processing completes
import smtplib
from email.mime.text import MIMEText

def send_completion_email(summary):
    msg = MIMEText(f"""
    Fuse Neutral Detection - Batch Complete
    
    Processed: {summary['processed']} images
    Detections: {summary['total_detections']}
    Failed: {summary['failed']}
    
    Results available in Azure Blob Storage: output-images container
    """)
    
    msg['Subject'] = f"Fuse Detection Complete - {summary['processed']} images"
    msg['From'] = "azure-batch@mgroupenergy.com"
    msg['To'] = "team@mgroupenergy.com"
    
    # Use Azure SendGrid or Office 365 SMTP
    with smtplib.SMTP('smtp.office365.com', 587) as server:
        server.starttls()
        server.login('azure-batch@mgroupenergy.com', 'password')
        server.send_message(msg)
```

---

## Phase 7: Supplier Integration (15 minutes)

### Option 1: Azure Storage Explorer (Easiest)

1. Give supplier the SAS URL from Phase 1.4
2. Supplier uses [Azure Storage Explorer](https://azure.microsoft.com/en-us/products/storage/storage-explorer/) (free GUI tool)
3. Drag and drop images to `input-images` container

### Option 2: Python Upload Script for Supplier

```python
# supplier_upload.py - Give this to your supplier
from azure.storage.blob import BlobServiceClient

# Provided by M Group
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=...;SharedAccessSignature=..."

def upload_images(folder_path):
    blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container = blob_service.get_container_client("input-images")
    
    for file in Path(folder_path).glob("*.jpg"):
        print(f"Uploading: {file.name}")
        with open(file, "rb") as data:
            container.upload_blob(name=file.name, data=data, overwrite=False)
    
    print("Upload complete!")

# Usage: python supplier_upload.py /path/to/images
```

### Option 3: Automated FTP to Azure Sync

If supplier only supports FTP:
- Setup Azure Files with FTP access
- Use Azure Data Factory to sync FTP → Blob Storage
- More complex but fully automated

---

## Production Workflow (Day-to-Day)

### Daily Operation:

1. **Morning (Supplier)**:
   - Supplier uploads 100-500 images to `input-images` container
   - Can be done anytime, batched upload

2. **Processing (Automated)**:
   - Scheduled job runs at 2am, 8am, 2pm, 8pm (4x daily)
   - Azure Batch pool spins up GPU VMs (0 → N nodes)
   - Processes all pending images in parallel
   - Generates annotated images + JSON metadata
   - Saves to `output-images/2026-02-12/` folder
   - Scales back down to 0 nodes (zero cost)

3. **Review (M Group Team)**:
   - Email notification: "Processed 237 images, 189 detections found"
   - Download results from `output-images` container
   - Review annotated images for quality
   - Use JSON metadata for analysis/reporting

### Monthly Operation:

1. **Model Updates**:
   - When new training data available, retrain on EC2
   - Upload new `checkpoint_epoch_XX.pth` to `model-checkpoints` container
   - Update Docker image with new checkpoint
   - Push to ACR: `docker push acrfusedetection.azurecr.io/fuse-neutrals-detector:v1.1`
   - Update Batch pool to use new image

2. **Cost Review**:
   - Check Azure Cost Management dashboard
   - Expected: £40-80/month for 15,000 images
   - If higher, optimize batch size or VM type

---

## Troubleshooting

### Issue: "Pool not scaling up"
```bash
# Check pool status
az batch pool show --id gpu-pool-fuse-detection --account-name $BATCH_ACCOUNT

# Check auto-scale formula evaluation
az batch pool autoscale evaluate \
  --pool-id gpu-pool-fuse-detection \
  --account-name $BATCH_ACCOUNT
```

### Issue: "Container fails to start"
```bash
# Check task logs
az batch task file list \
  --job-id $JOB_ID \
  --task-id task-1 \
  --account-name $BATCH_ACCOUNT

# Download stderr
az batch task file download \
  --job-id $JOB_ID \
  --task-id task-1 \
  --file-path stderr.txt \
  --destination ./stderr.txt \
  --account-name $BATCH_ACCOUNT
```

### Issue: "Model loading fails"
- Check checkpoint is uploaded to `model-checkpoints` container
- Verify Docker image has correct paths in Dockerfile
- Test locally first with `docker run`

---

## Security Best Practices

1. **Access Control**:
   - Use Azure AD authentication for team access
   - Generate time-limited SAS tokens for supplier (expire yearly)
   - Rotate storage keys annually

2. **Data Privacy**:
   - Enable encryption at rest (default in Azure Storage)
   - Use HTTPS only for all transfers
   - Set blob lifecycle policies to auto-delete old images after 90 days

3. **Network Security**:
   - Enable Azure Private Link for Batch and Storage
   - Restrict storage access to specific IPs if needed
   - Use VNet integration for production

4. **Monitoring**:
   - Enable Azure Monitor for all resources
   - Set up alerts for failures, high costs, or unusual activity
   - Log all access to sensitive containers

---

## Next Steps

### Week 1: Basic Setup
- [ ] Create Azure Storage account and containers
- [ ] Upload model checkpoint
- [ ] Build and test Docker image locally
- [ ] Push Docker image to ACR

### Week 2: Infrastructure
- [ ] Create Azure Batch account and pool OR single VM
- [ ] Test batch job with 10 sample images
- [ ] Verify outputs in `output-images` container
- [ ] Setup monitoring and email notifications

### Week 3: Integration
- [ ] Provide supplier with SAS URL or upload tool
- [ ] Test end-to-end flow with real supplier data
- [ ] Schedule automated processing (cron or Logic App)
- [ ] Document operational procedures

### Week 4: Production Launch
- [ ] Process first production batch
- [ ] Monitor costs and performance
- [ ] Gather feedback from team
- [ ] Plan model improvements based on results

---

## Cost Optimization Tips

1. **Use Low-Priority VMs**: 80% cheaper, acceptable for batch processing
2. **Batch Size**: Process images in batches of 50-100 to minimize VM spin-up overhead
3. **Off-Peak Processing**: Schedule processing during off-peak hours for better availability
4. **Storage Lifecycle**: Auto-delete processed images after 90 days to save storage costs
5. **Reserved Instances**: If processing daily, consider 1-year reserved instance for 30% discount

---

## Support and Maintenance

### Weekly Tasks:
- Review processing logs for errors
- Check output quality on sample images
- Monitor Azure costs

### Monthly Tasks:
- Review model performance metrics
- Collect new training data from edge cases
- Update model if accuracy degrades

### Quarterly Tasks:
- Retrain model with accumulated new data
- Review infrastructure costs and optimize
- Security audit (rotate keys, review access)

---

## Contacts and Resources

- **Azure Support**: [https://azure.microsoft.com/support/](https://azure.microsoft.com/support/)
- **Azure Batch Documentation**: [https://docs.microsoft.com/azure/batch/](https://docs.microsoft.com/azure/batch/)
- **SAM3 Repository**: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **Internal Team**: data-insight@mgroupenergy.com

---

## Appendix: Alternative Architectures

### Option 1: Azure Container Instances (Simpler, More Expensive)

**When to use**: Very low volume (< 50 images/day), simplest setup

```bash
az container create \
  --resource-group $RESOURCE_GROUP \
  --name fuse-detector-instance \
  --image $ACR_NAME.azurecr.io/fuse-neutrals-detector:v1.0 \
  --cpu 4 \
  --memory 16 \
  --restart-policy Never \
  --environment-variables \
    STORAGE_ACCOUNT=$STORAGE_ACCOUNT \
    STORAGE_KEY=$STORAGE_KEY
```

**Cost**: ~£0.10/hour, but no GPU support (CPU inference only)

### Option 2: Azure Machine Learning Batch Endpoints

**When to use**: Need MLOps integration, model versioning, A/B testing

- Managed endpoints with auto-scaling
- Built-in monitoring and logging
- More expensive (£0.20/inference + compute)
- Better for enterprise with multiple models

### Option 3: Azure Functions with Durable Functions

**When to use**: Event-driven, need orchestration, sporadic workload

- Trigger on blob upload
- Fan-out to multiple function instances
- CPU-only (no GPU), slower inference
- Very cheap for low volumes

---

**Document Version**: 1.0  
**Last Updated**: February 12, 2026  
**Author**: M Group Energy Data Insight Ltd.  
**Review Date**: May 2026
