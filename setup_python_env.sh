#!/bin/bash
# ============================================================
# Wisconsin Retirement AI — Python Environment Setup
# Run this once from the project root directory.
# Creates a conda environment named 'wi_retirement_ai'
# ============================================================

set -e  # Exit on any error

ENV_NAME="wi_retirement_ai"
PYTHON_VERSION="3.12"

echo "=============================================="
echo "Setting up Python environment: $ENV_NAME"
echo "=============================================="

# Create conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate
source activate $ENV_NAME || conda activate $ENV_NAME

# Install packages
pip install -r requirements_python.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

echo ""
echo "=============================================="
echo "Environment setup complete."
echo "Activate with: conda activate $ENV_NAME"
echo "=============================================="
