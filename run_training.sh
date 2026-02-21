#!/bin/bash

# YOLO Instance Segmentation Training Script
# This script sets up and runs the complete training pipeline

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
VENV_DIR="$DATA_DIR/venv"
LOGS_DIR="$DATA_DIR/logs"

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$LOGS_DIR"

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOGS_DIR/training_${TIMESTAMP}.log"

echo "============================================================"
echo "YOLO Instance Segmentation Training Pipeline"
echo "============================================================"
echo "Script Directory: $SCRIPT_DIR"
echo "Data Directory: $DATA_DIR"
echo "Log File: $LOG_FILE"
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

# Step 3: Check GPU availability
echo "🖥️  Step 3: Checking GPU availability..."
echo "------------------------------------------------------------"
GPU_CHECK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

if [ "$GPU_CHECK" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo "✅ GPU detected: $GPU_NAME"
    echo "   Number of GPUs: $GPU_COUNT"
    echo "   Training will use GPU acceleration!"
else
    echo "❌ ERROR: No GPU detected!"
    echo "   YOLO training requires GPU for reasonable training times."
    echo "   Please ensure you're running on a machine with CUDA-capable GPU."
    echo ""
    echo "   If you still want to train on CPU, edit train_segmentation.py"
    echo "   and change DEVICE = 0 to DEVICE = 'cpu'"
    echo ""
    exit 1
fi
echo ""

# Step 4: Download dataset
if [ ! -f "$DATA_DIR/coco_dataset/train/_annotations.coco.json" ]; then
    echo "📥 Step 4: Downloading COCO dataset..."
    echo "------------------------------------------------------------"
    python "$SCRIPT_DIR/download_dataset.py"
    echo ""
else
    echo "✅ Step 4: COCO dataset already exists. Skipping download."
    echo ""
fi

# Step 5: Convert COCO to YOLO format
if [ ! -d "$DATA_DIR/dataset/images/train" ]; then
    echo "🔄 Step 5: Converting COCO to YOLO format..."
    echo "------------------------------------------------------------"
    python "$SCRIPT_DIR/convert_coco_to_yolo.py"
    echo ""
else
    echo "✅ Step 5: YOLO dataset already exists. Skipping conversion."
    echo ""
fi

# Step 6: Start training
echo "🏋️  Step 6: Starting model training..."
echo "------------------------------------------------------------"
echo "📝 Logging all output to: $LOG_FILE"
echo ""

# Run training and capture output to both terminal and log file
python "$SCRIPT_DIR/train_segmentation.py" 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "✨ Training pipeline completed!"
echo "============================================================"
echo ""
echo "📝 Full training log saved to: $LOG_FILE"
echo "Note: Virtual environment is at: $VENV_DIR"
echo "To activate it manually: source $VENV_DIR/bin/activate"
