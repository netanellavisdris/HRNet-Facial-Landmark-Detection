#!/usr/bin/env python3
"""
Parse cross-validation test results and update the LaTeX table.
Author: AI Assistant
Date: 2025-11-25
"""

import os
import re
import glob
from collections import defaultdict

# Base directory
test_logs_dir = "/cdivece/workspace/fetalbiometry_paper/HRNet-Facial-Landmark-Detection-Dev/output/FETAL/test_logs"

# Parse all test log files
results = {}

# Find all test log files
log_files = glob.glob(os.path.join(test_logs_dir, "*_test.log"))

print(f"Found {len(log_files)} test log files")
print("=" * 80)

for log_file in sorted(log_files):
    # Parse filename to extract model_dataset, cfg_dataset, structure, metric
    # Format: fetal_landmark_hrnet_w18_{MODEL}_{CFG}_{STRUCTURE}_{METRIC}_test.log
    basename = os.path.basename(log_file)
    
    # Remove prefix and suffix
    basename = basename.replace("fetal_landmark_hrnet_w18_", "")
    basename = basename.replace("_test.log", "")
    
    # Parse the components
    parts = basename.split("_")
    
    # Handle different cases
    if len(parts) >= 4:
        model_dataset = parts[0]
        cfg_dataset = parts[1]
        
        # Find where the metric starts (BPD, OFD, FL, TAD, APAD)
        metrics = ['BPD', 'OFD', 'FL', 'TAD', 'APAD']
        metric = None
        metric_idx = -1
        
        for i, part in enumerate(parts):
            if part in metrics:
                metric = part
                metric_idx = i
                break
        
        if metric is None:
            print(f"Skipping {basename}: no metric found")
            continue
        
        # Structure is everything between cfg_dataset and metric
        structure = "_".join(parts[2:metric_idx])
        
        # Read the log file and extract NME and STD
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for the line with NME results
            # Format: "Test Results time:0.8412 loss:0.0000 nme:0.4600 nme mean:0.4600 nme std:0.4835"
            nme_match = re.search(r'nme mean:\s*([\d.]+)', content)
            
            # Look for STD
            # Format: "nme std:0.4835"
            std_match = re.search(r'nme std:\s*([\d.]+)', content)
            
            if nme_match and std_match:
                nme = float(nme_match.group(1))
                std = float(std_match.group(1))
                
                key = (model_dataset, cfg_dataset, structure, metric)
                results[key] = (nme, std)
                
                print(f"✓ {model_dataset:12s} -> {cfg_dataset:12s} | {structure:8s} | {metric:5s} | NME: {nme:.4f} ± {std:.4f}")
            else:
                print(f"✗ {basename}: Could not find NME/STD")
                
        except Exception as e:
            print(f"✗ Error reading {basename}: {e}")

print("=" * 80)
print(f"Successfully parsed {len(results)} results")
print("=" * 80)

# Organize results by structure
brain_results = {}
abdomen_results = {}
femur_results = {}

for (model, cfg, structure, metric), (nme, std) in results.items():
    if 'brain' in structure.lower():
        if (model, cfg) not in brain_results:
            brain_results[(model, cfg)] = {}
        brain_results[(model, cfg)][metric] = (nme, std)
    elif 'abdomen' in structure.lower():
        if (model, cfg) not in abdomen_results:
            abdomen_results[(model, cfg)] = {}
        abdomen_results[(model, cfg)][metric] = (nme, std)
    elif 'femur' in structure.lower():
        if (model, cfg) not in femur_results:
            femur_results[(model, cfg)] = {}
        femur_results[(model, cfg)][metric] = (nme, std)

# Print organized results
print("\n" + "=" * 80)
print("BRAIN RESULTS:")
print("=" * 80)
for (model, cfg), metrics in sorted(brain_results.items()):
    bpd = metrics.get('BPD', (None, None))
    ofd = metrics.get('OFD', (None, None))
    print(f"{model:12s} -> {cfg:12s} | BPD: {bpd[0]:.4f}±{bpd[1]:.4f} | OFD: {ofd[0]:.4f}±{ofd[1]:.4f}")

print("\n" + "=" * 80)
print("ABDOMEN RESULTS:")
print("=" * 80)
for (model, cfg), metrics in sorted(abdomen_results.items()):
    apad = metrics.get('APAD', (None, None))
    tad = metrics.get('TAD', (None, None))
    print(f"{model:12s} -> {cfg:12s} | APAD: {apad[0]:.4f}±{apad[1]:.4f} | TAD: {tad[0]:.4f}±{tad[1]:.4f}")

print("\n" + "=" * 80)
print("FEMUR RESULTS:")
print("=" * 80)
for (model, cfg), metrics in sorted(femur_results.items()):
    fl = metrics.get('FL', (None, None))
    print(f"{model:12s} -> {cfg:12s} | FL: {fl[0]:.4f}±{fl[1]:.4f}")

# Save results to file for LaTeX generation
output_file = "/cdivece/workspace/fetalbiometry_paper/cross_validation_results.txt"
with open(output_file, 'w') as f:
    f.write("BRAIN RESULTS (BPD, OFD):\n")
    f.write("=" * 80 + "\n")
    for (model, cfg), metrics in sorted(brain_results.items()):
        bpd = metrics.get('BPD', (None, None))
        ofd = metrics.get('OFD', (None, None))
        f.write(f"{model}\t{cfg}\tBPD\t{bpd[0]:.4f}\t{bpd[1]:.4f}\n")
        f.write(f"{model}\t{cfg}\tOFD\t{ofd[0]:.4f}\t{ofd[1]:.4f}\n")
    
    f.write("\nABDOMEN RESULTS (APAD, TAD):\n")
    f.write("=" * 80 + "\n")
    for (model, cfg), metrics in sorted(abdomen_results.items()):
        apad = metrics.get('APAD', (None, None))
        tad = metrics.get('TAD', (None, None))
        f.write(f"{model}\t{cfg}\tAPAD\t{apad[0]:.4f}\t{apad[1]:.4f}\n")
        f.write(f"{model}\t{cfg}\tTAD\t{tad[0]:.4f}\t{tad[1]:.4f}\n")
    
    f.write("\nFEMUR RESULTS (FL):\n")
    f.write("=" * 80 + "\n")
    for (model, cfg), metrics in sorted(femur_results.items()):
        fl = metrics.get('FL', (None, None))
        f.write(f"{model}\t{cfg}\tFL\t{fl[0]:.4f}\t{fl[1]:.4f}\n")

print(f"\n✓ Results saved to: {output_file}")

