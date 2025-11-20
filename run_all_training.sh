#!/bin/bash

# This script trains all models for all anatomical structures and datasets.
# author: chiaradivece
# date: 2025-11-19

# GPU selection (default: GPU 6)
# Override by setting CUDA_VISIBLE_DEVICES before running this script
# Example: CUDA_VISIBLE_DEVICES=7 ./run_all_training.sh
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=6
    echo "Using default GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Define the list of datasets and anatomical structures
datasets=("FP" "UCL" "MULTICENTRE")
structures=("brain" "femur" "abdomen")

# Add HC18 only for brain
datasets_brain=("FP" "HC18" "UCL" "MULTICENTRE")

# Base directories
BASE_DIR="/cdivece/workspace/fetalbiometry_paper/HRNet-Facial-Landmark-Detection-Dev"
EXPERIMENTS_DIR="$BASE_DIR/experiments/fetal"
OUTPUT_DIR="$BASE_DIR/output/FETAL"
LOGS_DIR="$OUTPUT_DIR/training_logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

echo "================================================================================"
echo "TRAINING ALL MODELS"
echo "================================================================================"
echo "Base directory: $BASE_DIR"
echo "Experiments directory: $EXPERIMENTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Logs directory: $LOGS_DIR"
echo ""

# Function to clean up intermediate checkpoints
cleanup_checkpoints() {
    local MODEL_DIR=$1
    local MODEL_NAME=$2
    
    echo "  Cleaning up intermediate checkpoints in $MODEL_DIR..."
    
    # Keep only these files:
    # - checkpoint_199.pth (last epoch)
    # - current_pred.pth
    # - model_best.pth
    # - final_state.pth
    # - predictions.pth
    
    # Remove all checkpoint_*.pth except checkpoint_199.pth
    find "$MODEL_DIR" -name "checkpoint_*.pth" ! -name "checkpoint_199.pth" -type f -delete 2>/dev/null
    
    # Count remaining files
    local remaining=$(ls -1 "$MODEL_DIR"/*.pth 2>/dev/null | wc -l)
    echo "  ✓ Cleanup complete. Remaining .pth files: $remaining"
}

# Counter for tracking progress
total_models=0
completed_models=0

# Count total models to train
for STRUCTURE in "${structures[@]}"; do
    if [ "$STRUCTURE" == "brain" ]; then
        total_models=$((total_models + ${#datasets_brain[@]}))
    else
        total_models=$((total_models + ${#datasets[@]}))
    fi
done

echo "Total models to train: $total_models"
echo ""

# Loop over each anatomical structure
for STRUCTURE in "${structures[@]}"
do
    echo "================================================================================"
    echo "TRAINING STRUCTURE: $STRUCTURE"
    echo "================================================================================"
    echo ""
    
    # Use different dataset list for brain (includes HC18)
    if [ "$STRUCTURE" == "brain" ]; then
        CURRENT_DATASETS=("${datasets_brain[@]}")
    else
        CURRENT_DATASETS=("${datasets[@]}")
    fi
    
    # Loop over each dataset
    for DATASET in "${CURRENT_DATASETS[@]}"
    do
        completed_models=$((completed_models + 1))
        
        # Construct the paths for the configuration file
        CFG_FILE="$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_${DATASET}_${STRUCTURE}.yaml"
        MODEL_NAME="fetal_landmark_hrnet_w18_${DATASET}_${STRUCTURE}"
        MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
        LOG_FILE="$LOGS_DIR/${MODEL_NAME}_train.log"
        
        echo "--------------------------------------------------------------------------------"
        echo "[$completed_models/$total_models] Training: $MODEL_NAME"
        echo "--------------------------------------------------------------------------------"
        echo "  Configuration: $CFG_FILE"
        echo "  Output directory: $MODEL_DIR"
        echo "  Log file: $LOG_FILE"
        echo ""
        
        # Check if the configuration file exists
        if [ ! -f "$CFG_FILE" ]; then
            echo "  ❌ Configuration file not found: $CFG_FILE"
            echo "  Skipping..."
            echo ""
            continue
        fi
        
        # Run the training script
        echo "  Starting training..."
        python tools/train.py --cfg "$CFG_FILE" > "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Training completed successfully"
            
            # Clean up intermediate checkpoints
            if [ -d "$MODEL_DIR" ]; then
                cleanup_checkpoints "$MODEL_DIR" "$MODEL_NAME"
            fi
        else
            echo "  ❌ Training failed. Check log file: $LOG_FILE"
        fi
        
        echo ""
    done
done

echo "================================================================================"
echo "TRAINING SUMMARY"
echo "================================================================================"
echo "Total models trained: $completed_models/$total_models"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Training logs: $LOGS_DIR"
echo ""
echo "Intermediate checkpoints have been removed to save space."
echo "Kept files per model:"
echo "  - checkpoint_199.pth (last epoch)"
echo "  - current_pred.pth"
echo "  - model_best.pth"
echo "  - final_state.pth"
echo "  - predictions.pth"
echo ""
echo "================================================================================"
echo "TRAINING COMPLETE"
echo "================================================================================"
