#!/bin/bash

# Script to train MULTICENTRE models for all anatomies
# This trains on the combined FP + HC18 + UCL datasets
# Uses nohup to run in background and save logs

BASE_DIR="/cdivece/workspace/fetalbiometry_paper/HRNet-Facial-Landmark-Detection-Dev"
EXPERIMENTS_DIR="$BASE_DIR/experiments/fetal"
LOGS_DIR="$BASE_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Change to the HRNet directory
cd "$BASE_DIR"

echo "=========================================="
echo "Training MULTICENTRE Models"
echo "=========================================="
echo "Logs will be saved to: $LOGS_DIR"
echo ""

# Train Abdomen model on GPU 0
echo "Starting MULTICENTRE Abdomen model training on GPU 0..."
echo "Log file: $LOGS_DIR/multisite_abdomen_train.log"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=0 nohup python tools/train.py --cfg "$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_MULTICENTRE_abdomen.yaml" > "$LOGS_DIR/multisite_abdomen_train.log" 2>&1 &
ABDOMEN_PID=$!
echo "Abdomen training started with PID: $ABDOMEN_PID on GPU 0"
echo ""

# Train Brain model on GPU 1
echo "Starting MULTICENTRE Brain model training on GPU 1..."
echo "Log file: $LOGS_DIR/multisite_brain_train.log"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=1 nohup python tools/train.py --cfg "$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_MULTICENTRE_brain.yaml" > "$LOGS_DIR/multisite_brain_train.log" 2>&1 &
BRAIN_PID=$!
echo "Brain training started with PID: $BRAIN_PID on GPU 1"
echo ""

# Train Femur model on GPU 2
echo "Starting MULTICENTRE Femur model training on GPU 2..."
echo "Log file: $LOGS_DIR/multisite_femur_train.log"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=2 nohup python tools/train.py --cfg "$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_MULTICENTRE_femur.yaml" > "$LOGS_DIR/multisite_femur_train.log" 2>&1 &
FEMUR_PID=$!
echo "Femur training started with PID: $FEMUR_PID on GPU 2"
echo ""

echo "=========================================="
echo "All MULTICENTRE training jobs started!"
echo "=========================================="
echo ""
echo "Process IDs and GPU assignments:"
echo "  Abdomen: PID $ABDOMEN_PID on GPU 0"
echo "  Brain:   PID $BRAIN_PID on GPU 1"
echo "  Femur:   PID $FEMUR_PID on GPU 2"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOGS_DIR/multisite_abdomen_train.log"
echo "  tail -f $LOGS_DIR/multisite_brain_train.log"
echo "  tail -f $LOGS_DIR/multisite_femur_train.log"
echo ""
echo "Check if processes are running:"
echo "  ps aux | grep train.py"
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

