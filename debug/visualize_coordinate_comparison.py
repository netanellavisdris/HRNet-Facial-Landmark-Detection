#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Visual comparison of BPD coordinates from different files
Overlays markers on actual ultrasound images to verify which coordinate order is correct
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Read the CSV files
bpd_test = pd.read_csv('/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/HC18/BPD_Test.csv')
biometry_split = pd.read_csv('/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/HC18/BiometryNet_split.csv')

# Image directory
image_dir = '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/HC18/Head'

# Select a few test images to visualize
test_images = ['055_HC.png', '091_HC.png', '169_HC.png']

fig, axes = plt.subplots(len(test_images), 2, figsize=(14, 6*len(test_images)))

for idx, img_name in enumerate(test_images):
    # Get image path
    img_path = os.path.join(image_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        continue
    
    # Load image
    img = Image.open(img_path)
    
    # Get data from both files
    bpd_test_row = bpd_test[bpd_test['image_name'] == img_name]
    biometry_row = biometry_split[biometry_split['image_name'] == img_name]
    
    if bpd_test_row.empty or biometry_row.empty:
        print(f"Warning: No data found for {img_name}")
        continue
    
    bpd_test_row = bpd_test_row.iloc[0]
    biometry_row = biometry_row.iloc[0]
    
    # Left plot: BiometryNet_split.csv (CORRECT - matches JSON)
    ax_left = axes[idx, 0] if len(test_images) > 1 else axes[0]
    ax_left.imshow(img, cmap='gray')
    ax_left.set_title(f'{img_name}\nBiometryNet_split.csv (CORRECT - matches JSON)', fontsize=12, fontweight='bold')
    
    # Plot BPD markers from BiometryNet_split.csv
    bpd1_x = biometry_row['bpd_1_x']
    bpd1_y = biometry_row['bpd_1_y']
    bpd2_x = biometry_row['bpd_2_x']
    bpd2_y = biometry_row['bpd_2_y']
    
    ax_left.plot(bpd1_x, bpd1_y, 'ro', markersize=15, label='BPD_1', markeredgecolor='white', markeredgewidth=2)
    ax_left.plot(bpd2_x, bpd2_y, 'bo', markersize=15, label='BPD_2', markeredgecolor='white', markeredgewidth=2)
    ax_left.plot([bpd1_x, bpd2_x], [bpd1_y, bpd2_y], 'g-', linewidth=2, label='BPD line')
    
    # Add coordinate labels
    ax_left.text(bpd1_x, bpd1_y-20, f'BPD_1\n({int(bpd1_x)}, {int(bpd1_y)})', 
                 color='red', fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_left.text(bpd2_x, bpd2_y-20, f'BPD_2\n({int(bpd2_x)}, {int(bpd2_y)})', 
                 color='blue', fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_left.legend(loc='upper right')
    ax_left.axis('off')
    
    # Right plot: BPD_Test.csv (SWAPPED coordinates)
    ax_right = axes[idx, 1] if len(test_images) > 1 else axes[1]
    ax_right.imshow(img, cmap='gray')
    ax_right.set_title(f'{img_name}\nBPD_Test.csv (SWAPPED x↔y)', fontsize=12, fontweight='bold')
    
    # Plot BPD markers from BPD_Test.csv (with swapped coordinates)
    bpd1_x_swap = bpd_test_row['bpd_1_x']
    bpd1_y_swap = bpd_test_row['bpd_1_y']
    bpd2_x_swap = bpd_test_row['bpd_2_x']
    bpd2_y_swap = bpd_test_row['bpd_2_y']
    
    ax_right.plot(bpd1_x_swap, bpd1_y_swap, 'ro', markersize=15, label='BPD_1', markeredgecolor='white', markeredgewidth=2)
    ax_right.plot(bpd2_x_swap, bpd2_y_swap, 'bo', markersize=15, label='BPD_2', markeredgecolor='white', markeredgewidth=2)
    ax_right.plot([bpd1_x_swap, bpd2_x_swap], [bpd1_y_swap, bpd2_y_swap], 'g-', linewidth=2, label='BPD line')
    
    # Add coordinate labels
    ax_right.text(bpd1_x_swap, bpd1_y_swap-20, f'BPD_1\n({int(bpd1_x_swap)}, {int(bpd1_y_swap)})', 
                  color='red', fontsize=10, fontweight='bold', ha='center',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_right.text(bpd2_x_swap, bpd2_y_swap-20, f'BPD_2\n({int(bpd2_x_swap)}, {int(bpd2_y_swap)})', 
                  color='blue', fontsize=10, fontweight='bold', ha='center',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_right.legend(loc='upper right')
    ax_right.axis('off')

plt.tight_layout()
plt.savefig('/cdivece/workspace/fetalbiometry_paper/coordinate_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: /cdivece/workspace/fetalbiometry_paper/coordinate_comparison.png")

plt.show()

print("\n" + "="*80)
print("VISUAL COMPARISON COMPLETE")
print("="*80)
print("""
Compare the two columns:
- LEFT: BiometryNet_split.csv coordinates (matches JSON ground truth)
- RIGHT: BPD_Test.csv coordinates (has x↔y swapped)

The correct coordinates should show BPD markers positioned across the widest 
part of the fetal head (biparietal diameter). If markers are positioned 
incorrectly, they will be on the wrong axis.
""")

