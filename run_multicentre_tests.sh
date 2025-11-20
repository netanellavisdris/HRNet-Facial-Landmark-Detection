#!/bin/bash

# This script runs cross-validation tests for MULTICENTRE models
# Tests each MULTICENTRE model on each individual dataset (FP, HC18, UCL)
# author: chiaradivece
# date: 2025-11-16

# Define the list of datasets and anatomical structures
datasets=("FP" "HC18" "UCL")
structures=("brain" "femur" "abdomen")

# Base directories
BASE_DIR="/cdivece/workspace/fetalbiometry_paper/HRNet-Facial-Landmark-Detection-Dev"
EXPERIMENTS_DIR="$BASE_DIR/experiments/fetal"
OUTPUT_DIR="$BASE_DIR/output/FETAL"

echo "=========================================="
echo "MULTICENTRE Models Cross-Validation"
echo "=========================================="
echo ""

# Loop over each anatomical structure
for STRUCTURE in "${structures[@]}"
do
    echo "=========================================="
    echo "Testing MULTICENTRE ${STRUCTURE} model"
    echo "=========================================="
    
    # The MULTICENTRE model
    MODEL_DIR="$OUTPUT_DIR/fetal_landmark_hrnet_w18_MULTICENTRE_${STRUCTURE}"
    MODEL_FILE="$MODEL_DIR/final_state.pth"
    
    # Check if the model file exists
    if [ ! -f "$MODEL_FILE" ]; then
        echo "ERROR: Model file not found: $MODEL_FILE"
        echo "Skipping ${STRUCTURE}..."
        echo ""
        continue
    fi
    
    echo "Using MULTICENTRE model: $MODEL_FILE"
    echo ""
    
    # Test on each dataset
    for CFG_DATASET in "${datasets[@]}"
    do
        # Construct the path for the configuration file
        CFG_FILE="$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_${CFG_DATASET}_${STRUCTURE}.yaml"
        
        # Check if the configuration file exists
        if [ ! -f "$CFG_FILE" ]; then
            echo "WARNING: Configuration file not found: $CFG_FILE"
            echo "Skipping test on ${CFG_DATASET}..."
            echo ""
            continue
        fi
        
        echo "Testing on ${CFG_DATASET} test set..."
        echo "Configuration: $CFG_FILE"
        
        # Run the test script
        python tools/test.py --cfg "$CFG_FILE" --model-file "$MODEL_FILE"
        
        echo "Finished testing MULTICENTRE ${STRUCTURE} on ${CFG_DATASET}"
        echo "---------------------------------------------"
        echo ""
    done
    
    echo ""
done

echo "=========================================="
echo "All MULTICENTRE cross-validation tests completed!"
echo "=========================================="

