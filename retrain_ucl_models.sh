#!/bin/bash

# Script to retrain UCL models that failed due to bad initialization
# Author: Assistant helping chiaradivece
# Date: 2025-11-24

# Activate conda environment
source ~/.bashrc
conda activate fetalbiometry

# Base directories
BASE_DIR="/cdivece/workspace/fetalbiometry_paper/HRNet-Facial-Landmark-Detection-Dev"
EXPERIMENTS_DIR="$BASE_DIR/experiments/fetal"
OUTPUT_DIR="$BASE_DIR/output/FETAL"
LOGS_DIR="$OUTPUT_DIR/training_logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Change to base directory
cd "$BASE_DIR" || exit 1

echo "================================================================================"
echo "RETRAINING FAILED UCL MODELS"
echo "================================================================================"
echo "These models failed due to bad random initialization (NME > 0.7)"
echo "We will retrain them with fresh initialization"
echo ""
echo "Base directory: $BASE_DIR"
echo "Date: $(date)"
echo ""

# Define models to retrain
# Format: "DATASET ANATOMY METRIC MODEL_DIR_NAME"
models_to_retrain=(
    "UCL brain BPD fetal_landmark_hrnet_w18_UCL_brain_BPD"
    "UCL brain OFD fetal_landmark_hrnet_w18_UCL_brain_OFD"
    "UCL abdomen TAD fetal_landmark_hrnet_w18_UCL_abdomen_TAD"
    # "UCL abdomen APAD fetal_landmark_hrnet_w18_UCL_abdomen_APAD"  # NME 0.31 - already good!
    # "UCL femur FL fetal_landmark_hrnet_w18_UCL_femur_FL"  # Currently retraining manually
)

total_models=${#models_to_retrain[@]}
completed=0

for model_info in "${models_to_retrain[@]}"; do
    # Parse model information
    read -r DATASET ANATOMY METRIC MODEL_NAME <<< "$model_info"
    
    completed=$((completed + 1))
    
    CFG_FILE="$EXPERIMENTS_DIR/${MODEL_NAME}.yaml"
    MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
    BACKUP_DIR="$OUTPUT_DIR/${MODEL_NAME}_backup_failed"
    LOG_FILE="$LOGS_DIR/${MODEL_NAME}_retrain_$(date +%d%m%Y).log"
    
    echo "================================================================================"
    echo "[$completed/$total_models] Retraining: $MODEL_NAME"
    echo "================================================================================"
    echo "Dataset: $DATASET | Anatomy: $ANATOMY | Metric: $METRIC"
    echo "Config: $CFG_FILE"
    echo "Log: $LOG_FILE"
    echo ""
    
    # Check if config exists
    if [ ! -f "$CFG_FILE" ]; then
        echo "❌ ERROR: Config file not found: $CFG_FILE"
        echo "Skipping..."
        echo ""
        continue
    fi
    
    # Backup old model if it exists
    if [ -d "$MODEL_DIR" ]; then
        echo "📦 Backing up old model to: $BACKUP_DIR"
        mv "$MODEL_DIR" "$BACKUP_DIR"
    fi
    
    # Start training
    echo "🚀 Starting training..."
    echo "   This will take approximately 1-2 hours"
    echo ""
    
    python tools/train.py --cfg "$CFG_FILE" > "$LOG_FILE" 2>&1
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully!"
        
        # Check the final test NME
        FINAL_NME=$(grep "Test Epoch 199" "$LOG_FILE" | tail -1 | grep -oP 'nme:\K[0-9.]+')
        if [ -n "$FINAL_NME" ]; then
            echo "   Final Test NME: $FINAL_NME"
            
            # Compare with threshold
            if (( $(echo "$FINAL_NME > 0.5" | bc -l) )); then
                echo "   ⚠️  WARNING: NME still high ($FINAL_NME > 0.5)"
                echo "   Model may need another retry"
            else
                echo "   ✓ Model trained successfully (NME < 0.5)"
                
                # Remove backup if training was successful
                if [ -d "$BACKUP_DIR" ]; then
                    echo "   Removing backup (training was successful)"
                    rm -rf "$BACKUP_DIR"
                fi
            fi
        fi
    else
        echo "❌ ERROR: Training failed!"
        echo "   Check log: $LOG_FILE"
        
        # Restore backup if training failed
        if [ -d "$BACKUP_DIR" ]; then
            echo "   Restoring backup model"
            rm -rf "$MODEL_DIR"
            mv "$BACKUP_DIR" "$MODEL_DIR"
        fi
    fi
    
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo ""
done

echo "================================================================================"
echo "RETRAINING SUMMARY"
echo "================================================================================"
echo "Completed: $completed/$total_models models"
echo ""
echo "Next steps:"
echo "1. Check training logs in: $LOGS_DIR"
echo "2. Run cross-validation tests: ./run_all_tests.sh"
echo "3. Extract results: python extract_cross_validation_results.py"
echo "4. Generate LaTeX table"
echo ""
echo "================================================================================"
echo "RETRAINING COMPLETE"
echo "================================================================================"

