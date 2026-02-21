#!/bin/bash

# YOLO Instance Segmentation Training Script
# This script sets up and runs the complete training pipeline

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
VENV_DIR="$DATA_DIR/venv"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "YOLO Instance Segmentation Training Pipeline"
echo "============================================================"
echo "Script Directory: $SCRIPT_DIR"
echo "Data Directory: $DATA_DIR"
echo ""

# Step 1: Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Step 1: Creating virtual environment..."
    echo "------------------------------------------------------------"
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created at: $VENV_DIR"
    echo ""
else
    echo "✅ Step 1: Virtual environment already exists."
    echo ""
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo ""

# Step 2: Install dependencies
echo "📦 Step 2: Installing dependencies..."
echo "------------------------------------------------------------"
pip install --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"
echo "✅ Dependencies installed."
echo ""

# Step 3: Download dataset
if [ ! -f "$DATA_DIR/coco_dataset/train/_annotations.coco.json" ]; then
    echo "📥 Step 3: Downloading COCO dataset..."
    echo "------------------------------------------------------------"
    python "$SCRIPT_DIR/download_dataset.py"
    echo ""
else
    echo "✅ Step 3: COCO dataset already exists. Skipping download."
    echo ""
fi

# Step 4: Convert COCO to YOLO format
if [ ! -f "$DATA_DIR/dataset/data.yaml" ]; then
    echo "🔄 Step 4: Converting COCO to YOLO format..."
    echo "------------------------------------------------------------"
    python "$SCRIPT_DIR/convert_coco_to_yolo.py"
    echo ""
else
    echo "✅ Step 4: YOLO dataset already exists. Skipping conversion."
    echo ""
fi

# Step 5: Start training
echo "🏋️  Step 5: Starting model training..."
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/train_segmentation.py"

echo ""
echo "============================================================"
echo "✨ Training pipeline completed!"
echo "============================================================"
echo ""
echo "Note: Virtual environment is at: $VENV_DIR"
echo "To activate it manually: source $VENV_DIR/bin/activate"
