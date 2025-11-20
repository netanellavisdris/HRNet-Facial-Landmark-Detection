#!/bin/bash

# This script runs all cross-validation tests for all anatomical structures and datasets.
# author: chiaradivece
# date: 2024-11-13
# Define the list of datasets and anatomical structures
datasets=("FP" "HC18" "UCL" "MULTICENTRE")
structures=("brain" "femur" "abdomen")

# Base directories
# BASE_DIR="/home/chiara/workspace/fetalbiometry_paper/HRNet-Facial-Landmark-Detection-Dev" # FV-DGX
BASE_DIR="/cdivece/workspace/fetalbiometry_paper/HRNet-Facial-Landmark-Detection-Dev" # Hoover
EXPERIMENTS_DIR="$BASE_DIR/experiments/fetal"
OUTPUT_DIR="$BASE_DIR/output/FETAL"

# Loop over each anatomical structure
for STRUCTURE in "${structures[@]}"
do
    # Loop over each combination of model_dataset and cfg_dataset
    for MODEL_DATASET in "${datasets[@]}"
        do
        
        for CFG_DATASET in "${datasets[@]}"
        do
            # Construct the paths for the configuration file and the model file
            CFG_FILE="$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_${CFG_DATASET}_${STRUCTURE}.yaml"
            MODEL_DIR="$OUTPUT_DIR/fetal_landmark_hrnet_w18_${MODEL_DATASET}_${STRUCTURE}"
            MODEL_FILE="$MODEL_DIR/final_state.pth"

            # Check if the configuration file exists
            if [ -f "$CFG_FILE" ]; then
                echo "Using configuration file: $CFG_FILE"
            else
                echo "Configuration file not found: $CFG_FILE"
                continue
            fi

            # Check if the model file exists
            if [ -f "$MODEL_FILE" ]; then
                echo "Using model file: $MODEL_FILE"
            else
                echo "Model file not found: $MODEL_FILE"
                continue
            fi

            # Run the test script
            echo "Running test with MODEL_DATASET: $MODEL_DATASET, CFG_DATASET: $CFG_DATASET, STRUCTURE: $STRUCTURE"
            python tools/test.py --cfg "$CFG_FILE" --model-file "$MODEL_FILE"
            echo "Finished testing with MODEL_DATASET: $MODEL_DATASET, CFG_DATASET: $CFG_DATASET, STRUCTURE: $STRUCTURE"
            echo "---------------------------------------------"
        done
    done
done
