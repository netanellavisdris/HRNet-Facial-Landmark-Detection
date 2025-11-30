#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Verify and visualize all fetal biometry measurements across specified datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Files to verify
FILES_TO_VERIFY = [
    # FP dataset
    ('FP', 'Abdomen', 'Test', 'fetalbiometrydata/FP/Abdomen_Test.csv', 'fetalbiometrydata/data/FP/Abdomen', ['TAD', 'APAD']),
    ('FP', 'Abdomen', 'Train', 'fetalbiometrydata/FP/Abdomen_Train.csv', 'fetalbiometrydata/data/FP/Abdomen', ['TAD', 'APAD']),
    ('FP', 'Femur', 'Test', 'fetalbiometrydata/FP/Femur_Test.csv', 'fetalbiometrydata/data/FP/FL', ['FL']),
    ('FP', 'Femur', 'Train', 'fetalbiometrydata/FP/Femur_Train.csv', 'fetalbiometrydata/data/FP/FL', ['FL']),
    ('FP', 'Head', 'Test', 'fetalbiometrydata/FP/Head_Test.csv', 'fetalbiometrydata/data/FP/Head', ['BPD', 'OFD']),
    ('FP', 'Head', 'Train', 'fetalbiometrydata/FP/Head_Train.csv', 'fetalbiometrydata/data/FP/Head', ['BPD', 'OFD']),
    
    # HC18 dataset
    ('HC18', 'Head', 'Test', 'fetalbiometrydata/HC18/Head_Test.csv', 'fetalbiometrydata/data/HC18/Head', ['BPD', 'OFD']),
    ('HC18', 'Head', 'Train', 'fetalbiometrydata/HC18/Head_Train.csv', 'fetalbiometrydata/data/HC18/Head', ['BPD', 'OFD']),
    
    # UCL dataset
    ('UCL', 'Abdomen', 'Test', 'fetalbiometrydata/UCL/Abdomen_Test.csv', 'fetalbiometrydata/data/UCL/Abdomen', ['TAD', 'APAD']),
    ('UCL', 'Abdomen', 'Train', 'fetalbiometrydata/UCL/Abdomen_Train.csv', 'fetalbiometrydata/data/UCL/Abdomen', ['TAD', 'APAD']),
    ('UCL', 'Femur', 'Test', 'fetalbiometrydata/UCL/Femur_Test.csv', 'fetalbiometrydata/data/UCL/Femur', ['FL']),
    ('UCL', 'Femur', 'Train', 'fetalbiometrydata/UCL/Femur_Train.csv', 'fetalbiometrydata/data/UCL/Femur', ['FL']),
    ('UCL', 'Head', 'Test', 'fetalbiometrydata/UCL/Head_Test.csv', 'fetalbiometrydata/data/UCL/Head', ['BPD', 'OFD']),
    ('UCL', 'Head', 'Train', 'fetalbiometrydata/UCL/Head_Train.csv', 'fetalbiometrydata/data/UCL/Head', ['BPD', 'OFD']),
]

# Colors for measurements
COLORS = {
    'BPD': ('red', 'darkred'),
    'OFD': ('blue', 'darkblue'),
    'TAD': ('green', 'darkgreen'),
    'APAD': ('orange', 'darkorange'),
    'FL': ('purple', 'indigo')
}

def check_measurement_completeness(csv_file, measurements):
    """Check if all measurements are populated in the CSV file"""
    
    if not os.path.exists(csv_file):
        return None, f"File not found: {csv_file}"
    
    df = pd.read_csv(csv_file)
    results = {}
    
    for measurement in measurements:
        col_1_x = f'{measurement.lower()}_1_x'
        col_1_y = f'{measurement.lower()}_1_y'
        col_2_x = f'{measurement.lower()}_2_x'
        col_2_y = f'{measurement.lower()}_2_y'
        
        # Check if columns exist
        if col_1_x not in df.columns:
            results[measurement] = {
                'status': 'missing_columns',
                'populated': 0,
                'total': len(df),
                'percentage': 0.0
            }
            continue
        
        # Check how many are populated
        populated = df[[col_1_x, col_1_y, col_2_x, col_2_y]].notna().all(axis=1).sum()
        total = len(df)
        percentage = (populated / total * 100) if total > 0 else 0
        
        results[measurement] = {
            'status': 'complete' if populated == total else 'incomplete',
            'populated': populated,
            'total': total,
            'percentage': percentage
        }
    
    return results, None

def plot_measurement(ax, df_row, measurement_name, color, marker_size=10):
    """Plot a single measurement (two points and a line)"""
    m1_x = df_row[f'{measurement_name.lower()}_1_x']
    m1_y = df_row[f'{measurement_name.lower()}_1_y']
    m2_x = df_row[f'{measurement_name.lower()}_2_x']
    m2_y = df_row[f'{measurement_name.lower()}_2_y']
    
    # Check if measurements are populated (not NaN)
    if pd.isna(m1_x) or pd.isna(m1_y) or pd.isna(m2_x) or pd.isna(m2_y):
        return False
    
    # Plot markers and line
    ax.plot(m1_x, m1_y, 'o', color=color[0], markersize=marker_size, 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot(m2_x, m2_y, 'o', color=color[0], markersize=marker_size, 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot([m1_x, m2_x], [m1_y, m2_y], '-', color=color[1], linewidth=3, alpha=0.7)
    
    return True

def visualize_measurements(dataset, structure, split, csv_file, image_dir, measurements, num_samples=2):
    """Create visualization for a specific dataset/structure/split"""
    
    if not os.path.exists(csv_file):
        print(f"  ✗ File not found: {csv_file}")
        return False
    
    df = pd.read_csv(csv_file)
    
    # Find sample images
    sample_images = []
    for _, row in df.head(20).iterrows():
        img_path = os.path.join(image_dir, row['image_name'])
        if os.path.exists(img_path):
            sample_images.append(row['image_name'])
            if len(sample_images) >= num_samples:
                break
    
    if not sample_images:
        print(f"  ⚠ No images found in {image_dir}")
        return False
    
    # Create figure
    n_samples = len(sample_images)
    fig, axes = plt.subplots(1, n_samples, figsize=(8*n_samples, 8))
    if n_samples == 1:
        axes = [axes]
    
    for idx, img_name in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path)
        img_row = df[df['image_name'] == img_name].iloc[0]
        
        ax = axes[idx]
        ax.imshow(img, cmap='gray')
        
        # Plot each measurement
        plotted_measurements = []
        for measurement in measurements:
            if plot_measurement(ax, img_row, measurement, COLORS[measurement]):
                plotted_measurements.append(measurement)
        
        title = f"{dataset} - {structure} ({split})\n{img_name}\n{' + '.join(plotted_measurements)}"
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    output_file = f"viz_{dataset}_{structure}_{split}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization: {output_file}")
    return True

# Main execution
print("="*80)
print("MEASUREMENT COMPLETENESS CHECK")
print("="*80)

summary = []

for dataset, structure, split, csv_file, image_dir, measurements in FILES_TO_VERIFY:
    print(f"\n{dataset} - {structure} ({split}):")
    print(f"  File: {csv_file}")
    
    results, error = check_measurement_completeness(csv_file, measurements)
    
    if error:
        print(f"  ✗ {error}")
        summary.append((dataset, structure, split, 'ERROR', error))
        continue
    
    all_complete = True
    for measurement, info in results.items():
        status_icon = '✓' if info['status'] == 'complete' else '✗'
        print(f"  {status_icon} {measurement}: {info['populated']}/{info['total']} ({info['percentage']:.1f}%)")
        if info['status'] != 'complete':
            all_complete = False
    
    summary.append((dataset, structure, split, 'COMPLETE' if all_complete else 'INCOMPLETE', results))

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

for dataset, structure, split, csv_file, image_dir, measurements in FILES_TO_VERIFY:
    print(f"\n{dataset} - {structure} ({split}):")
    visualize_measurements(dataset, structure, split, csv_file, image_dir, measurements)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Group by dataset and structure
from collections import defaultdict
grouped = defaultdict(lambda: defaultdict(list))

for dataset, structure, split, status, info in summary:
    grouped[dataset][structure].append((split, status))

for dataset in ['HC18', 'FP', 'UCL']:
    if dataset in grouped:
        print(f"\n{dataset}:")
        for structure in sorted(grouped[dataset].keys()):
            splits_info = grouped[dataset][structure]
            status_str = ', '.join([f"{split}: {status}" for split, status in splits_info])
            print(f"  {structure}: {status_str}")

print("\n" + "="*80)
print("✓ VERIFICATION COMPLETE")
print("="*80)

