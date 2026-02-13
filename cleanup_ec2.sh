#!/bin/bash
# EC2 Disk Space Cleanup Script

echo "=== Current Disk Usage ==="
df -h

echo -e "\n=== Checking directory sizes ==="
du -sh ~/sam3-fine-tuning/* 2>/dev/null | sort -h

echo -e "\n=== Clearing pip cache ==="
rm -rf ~/.cache/pip/*
echo "Pip cache cleared"

echo -e "\n=== Clearing apt cache ==="
sudo apt-get clean
echo "APT cache cleared"

echo -e "\n=== Clearing old journal logs ==="
sudo journalctl --vacuum-time=3d
echo "Journal logs cleared"

echo -e "\n=== Finding large files in home directory ==="
find ~ -type f -size +100M 2>/dev/null | head -10

echo -e "\n=== Disk usage after cleanup ==="
df -h

echo -e "\n=== Top 10 largest directories in home ==="
du -h ~ 2>/dev/null | sort -rh | head -10
