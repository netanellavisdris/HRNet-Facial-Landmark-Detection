#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Comprehensive visualization of all fetal biometry measurements across all datasets
- HC18: BPD + OFD (Head)
- FP: BPD + OFD (Head), TAD + APAD (Abdomen), FL (Femur)
- UCL: BPD + OFD (Head), TAD + APAD (Abdomen), FL (Femur)
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Configuration for each dataset and structure
DATASETS = {
    'HC18': {
        'Head': {
            'file': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/HC18/Head_Test.csv',
            'image_dir': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/HC18/Head',
            'measurements': ['BPD', 'OFD'],
            'sample_images': ['055_HC.png', '091_HC.png']
        }
    },
    'FP': {
        'Head': {
            'file': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/FP/Head_Test.csv',
            'image_dir': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/FP/Head',
            'measurements': ['BPD', 'OFD'],
            'sample_images': None  # Will auto-select
        },
        'Abdomen': {
            'file': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/FP/Abdomen_Test.csv',
            'image_dir': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/FP/Abdomen',
            'measurements': ['TAD', 'APAD'],
            'sample_images': None
        },
        'Femur': {
            'file': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/FP/Femur_Test.csv',
            'image_dir': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/FP/FL',
            'measurements': ['FL'],
            'sample_images': None
        }
    },
    'UCL': {
        'Head': {
            'file': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/UCL/Head_Test.csv',
            'image_dir': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/UCL/Head',
            'measurements': ['BPD', 'OFD'],
            'sample_images': None
        },
        'Abdomen': {
            'file': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/UCL/Abdomen_Test.csv',
            'image_dir': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/UCL/Abdomen',
            'measurements': ['TAD', 'APAD'],
            'sample_images': None
        },
        'Femur': {
            'file': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/UCL/Femur_Test.csv',
            'image_dir': '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/UCL/Femur',
            'measurements': ['FL'],
            'sample_images': None
        }
    }
}

# Colors for different measurements
COLORS = {
    'BPD': ('red', 'darkred'),
    'OFD': ('blue', 'darkblue'),
    'TAD': ('green', 'darkgreen'),
    'APAD': ('orange', 'darkorange'),
    'FL': ('purple', 'indigo')
}

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
            markeredgecolor='white', markeredgewidth=2, label=f'{measurement_name}_1')
    ax.plot(m2_x, m2_y, 'o', color=color[0], markersize=marker_size, 
            markeredgecolor='white', markeredgewidth=2, label=f'{measurement_name}_2')
    ax.plot([m1_x, m2_x], [m1_y, m2_y], '-', color=color[1], linewidth=3, 
            alpha=0.7, label=f'{measurement_name} line')
    
    return True

def visualize_dataset_structure(dataset_name, structure_name, config):
    """Visualize measurements for a specific dataset and structure"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING: {dataset_name} - {structure_name}")
    print(f"{'='*80}")
    
    # Check if file exists
    if not os.path.exists(config['file']):
        print(f"⚠ Warning: File not found: {config['file']}")
        return
    
    # Read data
    df = pd.read_csv(config['file'])
    print(f"✓ Read {len(df)} images from {os.path.basename(config['file'])}")
    
    # Check if image directory exists
    if not os.path.exists(config['image_dir']):
        print(f"⚠ Warning: Image directory not found: {config['image_dir']}")
        return
    
    # Select sample images
    sample_images = config['sample_images']
    if sample_images is None:
        # Auto-select first 2 images that exist
        sample_images = []
        for _, row in df.head(10).iterrows():
            img_path = os.path.join(config['image_dir'], row['image_name'])
            if os.path.exists(img_path):
                sample_images.append(row['image_name'])
                if len(sample_images) >= 2:
                    break
    
    if not sample_images:
        print(f"⚠ Warning: No sample images found")
        return
    
    # Create figure
    n_samples = len(sample_images)
    fig, axes = plt.subplots(1, n_samples, figsize=(8*n_samples, 8))
    if n_samples == 1:
        axes = [axes]
    
    for idx, img_name in enumerate(sample_images):
        img_path = os.path.join(config['image_dir'], img_name)
        
        if not os.path.exists(img_path):
            print(f"⚠ Warning: Image not found: {img_path}")
            continue
        
        # Load image
        img = Image.open(img_path)
        
        # Get data row
        img_row = df[df['image_name'] == img_name]
        if img_row.empty:
            print(f"⚠ Warning: No data for {img_name}")
            continue
        
        img_row = img_row.iloc[0]
        
        # Plot image
        ax = axes[idx]
        ax.imshow(img, cmap='gray')
        
        # Plot each measurement
        measurements_str = []
        for measurement in config['measurements']:
            success = plot_measurement(ax, img_row, measurement, COLORS[measurement])
            if success:
                measurements_str.append(measurement)
        
        title = f"{img_name}\n{' + '.join(measurements_str)}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')
        
        print(f"  ✓ Plotted {img_name}: {', '.join(measurements_str)}")
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"/cdivece/workspace/fetalbiometry_paper/viz_{dataset_name}_{structure_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

def check_measurement_completeness(dataset_name, structure_name, config):
    """Check if all measurements are populated"""
    
    if not os.path.exists(config['file']):
        return
    
    df = pd.read_csv(config['file'])
    
    print(f"\n{dataset_name} - {structure_name}:")
    
    for measurement in config['measurements']:
        col_1_x = f'{measurement.lower()}_1_x'
        col_1_y = f'{measurement.lower()}_1_y'
        col_2_x = f'{measurement.lower()}_2_x'
        col_2_y = f'{measurement.lower()}_2_y'
        
        # Check if columns exist
        if col_1_x not in df.columns:
            print(f"  ✗ {measurement}: Columns do not exist")
            continue
        
        # Check how many are populated
        populated = df[[col_1_x, col_1_y, col_2_x, col_2_y]].notna().all(axis=1).sum()
        total = len(df)
        percentage = (populated / total * 100) if total > 0 else 0
        
        if populated == total:
            print(f"  ✓ {measurement}: {populated}/{total} ({percentage:.1f}%) - COMPLETE")
        else:
            print(f"  ⚠ {measurement}: {populated}/{total} ({percentage:.1f}%) - INCOMPLETE")

# Main execution
print("="*80)
print("FETAL BIOMETRY MEASUREMENTS - COMPLETENESS CHECK")
print("="*80)

for dataset_name, structures in DATASETS.items():
    for structure_name, config in structures.items():
        check_measurement_completeness(dataset_name, structure_name, config)

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

for dataset_name, structures in DATASETS.items():
    for structure_name, config in structures.items():
        visualize_dataset_structure(dataset_name, structure_name, config)

print("\n" + "="*80)
print("✓ ALL VISUALIZATIONS COMPLETE")
print("="*80)
print("\nOutput files:")
for dataset_name, structures in DATASETS.items():
    for structure_name in structures.keys():
        print(f"  - viz_{dataset_name}_{structure_name}.png")

