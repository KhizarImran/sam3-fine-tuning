#!/bin/bash
# Quick test script before pushing to GitHub or after pulling on EC2

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                    SAM3 PRE-PUSH VERIFICATION${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo -e "${RED}[ERROR] Virtual environment not found at .venv/${NC}"
    echo "Please create one with: python3 -m venv .venv"
    echo "Then install dependencies: .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}[1/4] Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}      ✓ Virtual environment activated${NC}"
echo ""

# Run verification script
echo -e "${YELLOW}[2/4] Running setup verification...${NC}"
python scripts/verify_setup.py
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR] Setup verification failed!${NC}"
    echo "Please fix the issues above before proceeding."
    exit 1
fi
echo ""

# Check git status
echo -e "${YELLOW}[3/4] Checking git status...${NC}"
git status --short
echo ""

# List changed files
echo -e "${YELLOW}[4/4] Files ready to commit:${NC}"
echo ""
git status --short | grep "^[AM]" || echo "No changes to commit"
echo ""

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}                       VERIFICATION COMPLETE${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo -e "${GREEN}Everything looks good! Ready to push to GitHub.${NC}"
echo ""
echo "Next steps:"
echo "  1. Review changed files above"
echo "  2. git add ."
echo "  3. git commit -m 'Fix SAM3 inference scripts and add documentation'"
echo "  4. git push origin main"
echo "  5. Pull on EC2: git pull origin main"
echo "  6. Run inference on EC2!"
echo ""
echo -e "${GREEN}================================================================================${NC}"
