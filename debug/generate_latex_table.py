#!/usr/bin/env python3
"""
Generate updated LaTeX cross-validation table from test results.
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
log_files = glob.glob(os.path.join(test_logs_dir, "*_test.log"))

for log_file in sorted(log_files):
    basename = os.path.basename(log_file)
    basename = basename.replace("fetal_landmark_hrnet_w18_", "")
    basename = basename.replace("_test.log", "")
    parts = basename.split("_")
    
    if len(parts) >= 4:
        model_dataset = parts[0]
        cfg_dataset = parts[1]
        metrics = ['BPD', 'OFD', 'FL', 'TAD', 'APAD']
        metric = None
        metric_idx = -1
        
        for i, part in enumerate(parts):
            if part in metrics:
                metric = part
                metric_idx = i
                break
        
        if metric is None:
            continue
        
        structure = "_".join(parts[2:metric_idx])
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            nme_match = re.search(r'nme mean:\s*([\d.]+)', content)
            std_match = re.search(r'nme std:\s*([\d.]+)', content)
            
            if nme_match and std_match:
                nme = float(nme_match.group(1))
                std = float(std_match.group(1))
                key = (model_dataset, cfg_dataset, structure, metric)
                results[key] = (nme, std)
        except Exception as e:
            pass

# Organize results
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

def find_best_values(metric_results):
    """Find best and second-best values for a metric."""
    values = [(nme, key) for key, (nme, std) in metric_results.items()]
    values.sort()
    if len(values) >= 2:
        return values[0][1], values[1][1]  # best, second_best
    elif len(values) == 1:
        return values[0][1], None
    return None, None

def format_value(nme, std, is_best, is_second_best):
    """Format NME±STD with bold or underline."""
    formatted = f"{nme:.4f}$\\pm${std:.4f}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    elif is_second_best:
        return f"\\underline{{{formatted}}}"
    return formatted

# Collect all results for finding best/second-best
brain_bpd_all = {}
brain_ofd_all = {}
abdomen_apad_all = {}
abdomen_tad_all = {}
femur_fl_all = {}

for (model, cfg), metrics in brain_results.items():
    if 'BPD' in metrics:
        brain_bpd_all[(model, cfg)] = metrics['BPD']
    if 'OFD' in metrics:
        brain_ofd_all[(model, cfg)] = metrics['OFD']

for (model, cfg), metrics in abdomen_results.items():
    if 'APAD' in metrics:
        abdomen_apad_all[(model, cfg)] = metrics['APAD']
    if 'TAD' in metrics:
        abdomen_tad_all[(model, cfg)] = metrics['TAD']

for (model, cfg), metrics in femur_results.items():
    if 'FL' in metrics:
        femur_fl_all[(model, cfg)] = metrics['FL']

# Find best and second-best for each metric
best_bpd, second_bpd = find_best_values(brain_bpd_all)
best_ofd, second_ofd = find_best_values(brain_ofd_all)
best_apad, second_apad = find_best_values(abdomen_apad_all)
best_tad, second_tad = find_best_values(abdomen_tad_all)
best_fl, second_fl = find_best_values(femur_fl_all)

print(f"Best BPD: {best_bpd}, Second: {second_bpd}")
print(f"Best OFD: {best_ofd}, Second: {second_ofd}")
print(f"Best APAD: {best_apad}, Second: {second_apad}")
print(f"Best TAD: {best_tad}, Second: {second_tad}")
print(f"Best FL: {best_fl}, Second: {second_fl}")

# Generate LaTeX table
latex_lines = []
latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Cross-validation results showing NME $\pm$ STD for all train-test combinations across four datasets (FP, HC18, UCL, MULTICENTRE) and three anatomies. ``Ours'' models (MULTICENTRE) are trained on combined data from all available datasets. For each metric, bold indicates the best result, and underline indicates the second-best result across all train-test combinations.}")
latex_lines.append(r"\label{tab:cross_validation}")
latex_lines.append(r"\begin{tabular}{|c|c|c|c|c|}")
latex_lines.append(r"\hline")
latex_lines.append(r"\textbf{Anatomy} & \textbf{Train} & \textbf{Test} & \multicolumn{2}{c|}{\textbf{NME$\pm$STD}} \\ \cline{4-5}")
latex_lines.append(r"\textbf{} & \textbf{} & \textbf{} & \textbf{BPD} & \textbf{OFD} \\ \hline")
latex_lines.append(r"\multirow{16}{*}{\textbf{Brain}}")

# Brain section
train_order = ['FP', 'HC18', 'UCL', 'MULTICENTRE']
test_order = ['FP', 'HC18', 'UCL', 'MULTICENTRE']

for train_idx, train_ds in enumerate(train_order):
    for test_idx, test_ds in enumerate(test_order):
        if (train_ds, test_ds) in brain_results:
            metrics = brain_results[(train_ds, test_ds)]
            bpd_nme, bpd_std = metrics.get('BPD', (0, 0))
            ofd_nme, ofd_std = metrics.get('OFD', (0, 0))
            
            bpd_str = format_value(bpd_nme, bpd_std, 
                                  (train_ds, test_ds) == best_bpd,
                                  (train_ds, test_ds) == second_bpd)
            ofd_str = format_value(ofd_nme, ofd_std,
                                  (train_ds, test_ds) == best_ofd,
                                  (train_ds, test_ds) == second_ofd)
            
            # Replace MULTICENTRE with Ours in display
            train_display = "Ours" if train_ds == "MULTICENTRE" else train_ds
            test_display = "Ours" if test_ds == "MULTICENTRE" else test_ds
            
            line = f" & {train_display} & {test_display} & {bpd_str} & {ofd_str}\\\\"
            latex_lines.append(line)
        
        # Add \cline after each train dataset group (except the last)
        if test_idx == len(test_order) - 1 and train_idx < len(train_order) - 1:
            latex_lines.append(r" \cline{2-5}")

latex_lines.append(r" \hline")
latex_lines.append(r"\textbf{} & \textbf{} & \textbf{} & \textbf{APAD} & \textbf{TAD} \\ \hline")
latex_lines.append(r"\multirow{9}{*}{\textbf{Abdomen}}")

# Abdomen section (no HC18)
train_order_abd = ['FP', 'UCL', 'MULTICENTRE']
test_order_abd = ['FP', 'UCL', 'MULTICENTRE']

for train_idx, train_ds in enumerate(train_order_abd):
    for test_idx, test_ds in enumerate(test_order_abd):
        if (train_ds, test_ds) in abdomen_results:
            metrics = abdomen_results[(train_ds, test_ds)]
            apad_nme, apad_std = metrics.get('APAD', (0, 0))
            tad_nme, tad_std = metrics.get('TAD', (0, 0))
            
            apad_str = format_value(apad_nme, apad_std,
                                   (train_ds, test_ds) == best_apad,
                                   (train_ds, test_ds) == second_apad)
            tad_str = format_value(tad_nme, tad_std,
                                  (train_ds, test_ds) == best_tad,
                                  (train_ds, test_ds) == second_tad)
            
            train_display = "Ours" if train_ds == "MULTICENTRE" else train_ds
            test_display = "Ours" if test_ds == "MULTICENTRE" else test_ds
            
            line = f" & {train_display} & {test_display} & {apad_str} & {tad_str}\\\\"
            latex_lines.append(line)
        
        if test_idx == len(test_order_abd) - 1 and train_idx < len(train_order_abd) - 1:
            latex_lines.append(r" \cline{2-5}")

latex_lines.append(r" \hline")
latex_lines.append(r"\textbf{} & \textbf{} & \textbf{} & \multicolumn{2}{c|}{\textbf{FL}} \\ \hline")
latex_lines.append(r"\multirow{9}{*}{\textbf{Femur}}")

# Femur section (no HC18)
for train_idx, train_ds in enumerate(train_order_abd):
    for test_idx, test_ds in enumerate(test_order_abd):
        if (train_ds, test_ds) in femur_results:
            metrics = femur_results[(train_ds, test_ds)]
            fl_nme, fl_std = metrics.get('FL', (0, 0))
            
            fl_str = format_value(fl_nme, fl_std,
                                 (train_ds, test_ds) == best_fl,
                                 (train_ds, test_ds) == second_fl)
            
            train_display = "Ours" if train_ds == "MULTICENTRE" else train_ds
            test_display = "Ours" if test_ds == "MULTICENTRE" else test_ds
            
            line = f" & {train_display} & {test_display} & \\multicolumn{{2}}{{c|}}{{{fl_str}}}\\\\"
            latex_lines.append(line)
        
        if test_idx == len(test_order_abd) - 1 and train_idx < len(train_order_abd) - 1:
            latex_lines.append(r" \cline{2-5}")

latex_lines.append(r" \hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")
latex_lines.append(r"")
latex_lines.append(r"")

# Write to file
output_file = "/cdivece/workspace/fetalbiometry_paper/cross_validation_combined_table_NEW.tex"
with open(output_file, 'w') as f:
    f.write('\n'.join(latex_lines))

print(f"\n✓ LaTeX table generated: {output_file}")
print(f"\n✓ Total entries: {len(results)}")

