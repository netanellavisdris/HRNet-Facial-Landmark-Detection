#!/bin/bash

# Script to retrain UCL models in PARALLEL (faster!)
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
echo "RETRAINING FAILED UCL MODELS (PARALLEL)"
echo "================================================================================"
echo "These models will train simultaneously to save time"
echo ""
echo "Base directory: $BASE_DIR"
echo "Date: $(date)"
echo ""

# Define models to retrain
# Format: "DATASET ANATOMY METRIC MODEL_DIR_NAME GPU_ID"
models_to_retrain=(
    "UCL brain BPD fetal_landmark_hrnet_w18_UCL_brain_BPD 1"
    "UCL brain OFD fetal_landmark_hrnet_w18_UCL_brain_OFD 2"
    "UCL abdomen TAD fetal_landmark_hrnet_w18_UCL_abdomen_TAD 3"
)

total_models=${#models_to_retrain[@]}
echo "Available GPUs: 1-7"
echo "Assigning one GPU per model for parallel training"
echo ""

# Arrays to store PIDs and info
declare -a pids
declare -a model_names
declare -a log_files

echo "Starting $total_models models in parallel..."
echo ""

# Launch all trainings in parallel
for model_info in "${models_to_retrain[@]}"; do
    # Parse model information
    read -r DATASET ANATOMY METRIC MODEL_NAME GPU_ID <<< "$model_info"
    
    CFG_FILE="$EXPERIMENTS_DIR/${MODEL_NAME}.yaml"
    MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
    BACKUP_DIR="$OUTPUT_DIR/${MODEL_NAME}_backup_failed"
    LOG_FILE="$LOGS_DIR/${MODEL_NAME}_retrain_$(date +%d%m%Y_%H%M).log"
    
    echo "================================================================================"
    echo "Starting: $MODEL_NAME"
    echo "================================================================================"
    echo "Dataset: $DATASET | Anatomy: $ANATOMY | Metric: $METRIC | GPU: $GPU_ID"
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
    
    # Start training in background with specific GPU
    echo "🚀 Starting training in background on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train.py --cfg "$CFG_FILE" > "$LOG_FILE" 2>&1 &
    
    # Store PID and info
    pid=$!
    pids+=($pid)
    model_names+=("$MODEL_NAME (GPU $GPU_ID)")
    log_files+=("$LOG_FILE")
    
    echo "   PID: $pid"
    echo "   GPU: $GPU_ID"
    echo "   Monitor: tail -f $LOG_FILE"
    echo ""
done

echo "================================================================================"
echo "All $total_models models started!"
echo "================================================================================"
echo ""
echo "Process IDs:"
for i in "${!pids[@]}"; do
    echo "  ${model_names[$i]}: PID ${pids[$i]}"
done
echo ""
echo "To monitor progress:"
for i in "${!log_files[@]}"; do
    echo "  ${model_names[$i]}: tail -f ${log_files[$i]}"
done
echo ""
echo "Waiting for all trainings to complete..."
echo "(This will take approximately 1-2 hours)"
echo ""

# Wait for all background processes
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    model_name=${model_names[$i]}
    
    echo "[$((i+1))/$total_models] Waiting for $model_name (PID: $pid)..."
    wait $pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "   ✅ $model_name completed successfully!"
    else
        echo "   ❌ $model_name failed (exit code: $exit_code)"
    fi
done

echo ""
echo "================================================================================"
echo "ALL TRAININGS COMPLETED"
echo "================================================================================"
echo ""
echo "Checking results..."
echo ""

# Check results for each model
for i in "${!model_names[@]}"; do
    model_name=${model_names[$i]}
    log_file=${log_files[$i]}
    
    echo "--------------------------------------------------------------------------------"
    echo "$model_name"
    echo "--------------------------------------------------------------------------------"
    
    # Extract final NME
    if [ -f "$log_file" ]; then
        FINAL_NME=$(grep "Test Epoch 199" "$log_file" | tail -1 | grep -oP 'nme:\K[0-9.]+')
        
        if [ -n "$FINAL_NME" ]; then
            echo "Final Test NME: $FINAL_NME"
            
            # Compare with threshold
            if (( $(echo "$FINAL_NME > 0.5" | bc -l) )); then
                echo "⚠️  WARNING: NME still high ($FINAL_NME > 0.5)"
                echo "Model may need another retry"
            else
                echo "✓ Model trained successfully (NME < 0.5)"
                
                # Remove backup if training was successful
                BACKUP_DIR="$OUTPUT_DIR/${model_name}_backup_failed"
                if [ -d "$BACKUP_DIR" ]; then
                    echo "Removing backup (training was successful)"
                    rm -rf "$BACKUP_DIR"
                fi
            fi
        else
            echo "❌ Could not extract final NME from log"
        fi
    else
        echo "❌ Log file not found: $log_file"
    fi
    echo ""
done

echo "================================================================================"
echo "RETRAINING SUMMARY"
echo "================================================================================"
echo "Completed: $total_models/$total_models models"
echo ""
echo "Log files:"
for log_file in "${log_files[@]}"; do
    echo "  - $log_file"
done
echo ""
echo "Next steps:"
echo "1. Review training logs above"
echo "2. Run cross-validation tests: ./run_all_tests.sh"
echo "3. Extract results: python extract_cross_validation_results.py"
echo "4. Generate updated LaTeX table"
echo ""
echo "================================================================================"
echo "RETRAINING COMPLETE"
echo "================================================================================"

