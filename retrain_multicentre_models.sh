#!/bin/bash
# Retrain MULTICENTRE models after recreating the dataset CSV files
# author: chiaradivece
# date: 2025-11-20

# GPU selection (default: GPU 6)
# Override by setting CUDA_VISIBLE_DEVICES before running this script
# Example: CUDA_VISIBLE_DEVICES=7 ./retrain_multicentre_models.sh
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=6
    echo "Using default GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Base directories
BASE_DIR="/cdivece/workspace/fetalbiometry_paper/HRNet-Facial-Landmark-Detection-Dev"
EXP_DIR="$BASE_DIR/experiments/fetal"
OUTPUT_DIR="$BASE_DIR/output/FETAL"
LOG_DIR="$OUTPUT_DIR/training_logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "RETRAINING MULTICENTRE MODELS"
echo "================================================================================"
echo "Base directory: $BASE_DIR"
echo "Experiments directory: $EXP_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Logs directory: $LOG_DIR"
echo ""
echo "Models to retrain: 3 (brain, abdomen, femur)"
echo ""

# Array of structures to train
structures=("brain" "abdomen" "femur")

# Counter
count=0
total=3

# Train each MULTICENTRE model
for structure in "${structures[@]}"; do
    count=$((count + 1))
    
    model_name="fetal_landmark_hrnet_w18_MULTICENTRE_${structure}"
    config_file="$EXP_DIR/${model_name}.yaml"
    output_path="$OUTPUT_DIR/${model_name}"
    log_file="$LOG_DIR/${model_name}_retrain.log"
    
    echo "--------------------------------------------------------------------------------"
    echo "[$count/$total] Training: $model_name"
    echo "--------------------------------------------------------------------------------"
    echo "  Configuration: $config_file"
    echo "  Output directory: $output_path"
    echo "  Log file: $log_file"
    echo ""
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        echo "  ❌ Configuration file not found: $config_file"
        echo ""
        continue
    fi
    
    # Remove old model files to ensure fresh training
    if [ -d "$output_path" ]; then
        echo "  Removing old model files..."
        rm -rf "$output_path"
        echo "  ✓ Old model files removed"
    fi
    
    # Train the model
    echo "  Starting training..."
    cd "$BASE_DIR"
    python tools/train.py --cfg "$config_file" > "$log_file" 2>&1
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "  ✓ Training completed successfully"
        
        # Clean up intermediate checkpoints
        echo "  Cleaning up intermediate checkpoints in $output_path..."
        
        # Keep only: checkpoint_199.pth, current_pred.pth, model_best.pth, final_state.pth, predictions.pth
        cd "$output_path"
        
        # Count .pth files before cleanup
        before_count=$(find . -name "*.pth" | wc -l)
        
        # Remove intermediate checkpoints (checkpoint_0.pth through checkpoint_198.pth)
        for i in {0..198}; do
            if [ -f "checkpoint_${i}.pth" ]; then
                rm "checkpoint_${i}.pth"
            fi
        done
        
        # Count .pth files after cleanup
        after_count=$(find . -name "*.pth" | wc -l)
        
        echo "  ✓ Cleanup complete. Remaining .pth files: $after_count (was $before_count)"
        
        cd "$BASE_DIR"
    else
        echo "  ❌ Training failed. Check log file: $log_file"
    fi
    
    echo ""
done

echo "================================================================================"
echo "TRAINING SUMMARY"
echo "================================================================================"
echo "Total models retrained: $total/$total"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Training logs: $LOG_DIR"
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
echo "NEXT STEPS"
echo "================================================================================"
echo "1. Run cross-validation tests: ./run_all_tests.sh"
echo "2. Update cross-validation table with new results"
echo "3. Regenerate boxplots if needed"
echo ""
echo "================================================================================"
echo "RETRAINING COMPLETE"
echo "================================================================================"
echo ""

