#!/bin/bash

# Cross-Convolution Model Training Script for Linux/Bash
# Trains Qwen2-VL model with cross-shape convolution analysis

set -e

# Color functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo_success() { echo -e "${GREEN}[✅]${NC} $@"; }
echo_info() { echo -e "${CYAN}[ℹ️]${NC} $@"; }
echo_warning() { echo -e "${YELLOW}[⚠️]${NC} $@"; }
echo_error() { echo -e "${RED}[❌]${NC} $@"; }

# Configuration
WORKSPACE_ROOT="."
SCRIPT_DIR="$WORKSPACE_ROOT/scripts"
CONDA_ENV="ocr_env"
PYTHON_SCRIPT="cross_convolution_trainer.py"
PYTHON_BIN=""

# Parse command line arguments
SKIP_EXTRACTION=false
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE="5e-5"

# Accept workspace root as first positional argument if provided
if [[ $# -gt 0 && "$1" != --* ]]; then
    WORKSPACE_ROOT="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace-root)
            WORKSPACE_ROOT="$2"
            shift 2
            ;;
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-extraction      Skip data extraction step"
            echo "  --epochs N             Number of training epochs (default: 3)"
            echo "  --batch-size N         Batch size (default: 4)"
            echo "  --learning-rate LR     Learning rate (default: 5e-5)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo_warning "Unknown option: $1"
            shift
            ;;
    esac
done

WORKSPACE_ROOT="$(cd "$WORKSPACE_ROOT" && pwd)"
SCRIPT_DIR="$WORKSPACE_ROOT/scripts"

echo_info ""
echo_info "🚀 Cross-Convolution Model Training Script"
echo_info "=========================================="
echo_info ""

# Step 1: Verify installation
echo_info "[Step 1/4] Verifying installation..."

if [ -x "$WORKSPACE_ROOT/ocr_env/bin/python" ]; then
    PYTHON_BIN="$WORKSPACE_ROOT/ocr_env/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
else
    echo_error "Python not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo_info "  Python: $PYTHON_VERSION"

if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version 2>&1 | awk '{print $3}')
    echo_info "  Conda: $CONDA_VERSION"
else
    echo_warning "Conda not found. Will proceed without conda activation."
fi

echo_success "Installation verified"

# Step 2: Activate conda environment (optional)
echo_info ""
echo_info "[Step 2/4] Environment setup..."

if command -v conda &> /dev/null; then
    # Source conda init script if needed
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi

    if conda activate $CONDA_ENV; then
        echo_success "Environment activated: $CONDA_ENV"
    else
        echo_warning "Failed to activate conda environment: $CONDA_ENV"
        echo_warning "Continuing with Python: $PYTHON_BIN"
    fi
else
    echo_info "Using Python: $PYTHON_BIN"
fi

# Step 3: Extract and prepare data (optional)
if [ "$SKIP_EXTRACTION" = false ]; then
    echo_info ""
    echo_info "[Step 3/4] Extracting and preparing training data..."
    echo_info "  This extracts 20 samples from fine-tuning data"
    echo_info "  Creates colored box visualizations"
    echo_info "  Copies model weights to init folder"
    
    EXTRACTION_SCRIPT="$SCRIPT_DIR/extract_and_prepare.py"
    
    if [ -f "$EXTRACTION_SCRIPT" ]; then
        echo_info "  Running extraction script..."
        if "$PYTHON_BIN" "$EXTRACTION_SCRIPT"; then
            echo_success "Data extraction completed"
        else
            echo_warning "Data extraction encountered an issue"
            echo_warning "Continuing with training anyway..."
        fi
    else
        echo_warning "Extraction script not found at $EXTRACTION_SCRIPT"
        echo_warning "Skipping extraction..."
    fi
else
    echo_info ""
    echo_info "[Step 3/4] Skipping data extraction (--skip-extraction flag used)"
fi

# Step 4: Run training
echo_info ""
echo_info "[Step 4/4] Starting model training..."
echo_info "  Epochs: $EPOCHS"
echo_info "  Batch Size: $BATCH_SIZE"
echo_info "  Learning Rate: $LEARNING_RATE"
echo_info ""
echo_info "Training configuration:"
echo_info "  - Cross-convolution analysis enabled"
echo_info "  - Multi-region processing (Top/Center/Bottom)"
echo_info "  - Colored visualization boxes"
echo_info "  - Enhanced XY coordinate analysis"
echo_info ""

TRAINING_SCRIPT="$SCRIPT_DIR/$PYTHON_SCRIPT"

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo_error "Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

MODEL_PATH="$WORKSPACE_ROOT/models/Qwen2-VL-7B-Instruct-KoDocOCR"
DATA_DIR="$WORKSPACE_ROOT/Test"
OUTPUT_DIR="$WORKSPACE_ROOT/init/training_output"

echo_info "Running: $PYTHON_BIN $TRAINING_SCRIPT"
echo_info ""

# Run training
if "$PYTHON_BIN" "$TRAINING_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE"; then
    
    echo_info ""
    echo_success "=========================================="
    echo_success "✅ Training completed successfully!"
    echo_success "=========================================="
    echo_info ""
    echo_info "📊 Training Results:"
    echo_info "  Output Directory: $OUTPUT_DIR"
    echo_info "  Model saved at: $OUTPUT_DIR/final_model"
    echo_info ""
    echo_info "📝 Next steps:"
    echo_info "  1. Verify results: Check $OUTPUT_DIR for model files"
    echo_info "  2. Evaluate: Test on new images"
    echo_info "  3. Deploy: Use trained model in production"
else
    echo_error ""
    echo_error "=========================================="
    echo_error "Training failed"
    echo_error "=========================================="
    exit 1
fi

echo_info ""
echo_info "Done! 🎉"
