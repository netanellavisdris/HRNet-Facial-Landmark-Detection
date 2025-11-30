#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Visualize both BPD and OFD measurements on ultrasound images
to verify that Head_Test.csv and Head_Train.csv are correct
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Read the new Head files
head_test = pd.read_csv('/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/HC18/Head_Test.csv')
biometry_split = pd.read_csv('/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/HC18/BiometryNet_split.csv')

# Image directory
image_dir = '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/HC18/Head'

# Select sample images to visualize
test_images = ['055_HC.png', '091_HC.png', '169_HC.png', '239_HC.png']

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

for idx, img_name in enumerate(test_images):
    # Get image path
    img_path = os.path.join(image_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        continue
    
    # Load image
    img = Image.open(img_path)
    
    # Get data from Head_Test.csv
    head_row = head_test[head_test['image_name'] == img_name]
    
    if head_row.empty:
        print(f"Warning: No data found for {img_name}")
        continue
    
    head_row = head_row.iloc[0]
    
    # Plot image
    ax = axes[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'{img_name}\nBPD (red) + OFD (blue)', fontsize=14, fontweight='bold')
    
    # Plot BPD markers (RED)
    bpd1_x = head_row['bpd_1_x']
    bpd1_y = head_row['bpd_1_y']
    bpd2_x = head_row['bpd_2_x']
    bpd2_y = head_row['bpd_2_y']
    
    ax.plot(bpd1_x, bpd1_y, 'ro', markersize=12, label='BPD_1', 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot(bpd2_x, bpd2_y, 'ro', markersize=12, label='BPD_2', 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot([bpd1_x, bpd2_x], [bpd1_y, bpd2_y], 'r-', linewidth=3, 
            label='BPD line', alpha=0.7)
    
    # Plot OFD markers (BLUE)
    ofd1_x = head_row['ofd_1_x']
    ofd1_y = head_row['ofd_1_y']
    ofd2_x = head_row['ofd_2_x']
    ofd2_y = head_row['ofd_2_y']
    
    ax.plot(ofd1_x, ofd1_y, 'bo', markersize=12, label='OFD_1', 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot(ofd2_x, ofd2_y, 'bo', markersize=12, label='OFD_2', 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot([ofd1_x, ofd2_x], [ofd1_y, ofd2_y], 'b-', linewidth=3, 
            label='OFD line', alpha=0.7)
    
    # Add coordinate labels
    offset = 25
    ax.text(bpd1_x, bpd1_y-offset, f'BPD_1\n({int(bpd1_x)},{int(bpd1_y)})', 
            color='red', fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.text(bpd2_x, bpd2_y-offset, f'BPD_2\n({int(bpd2_x)},{int(bpd2_y)})', 
            color='red', fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.text(ofd1_x, ofd1_y+offset, f'OFD_1\n({int(ofd1_x)},{int(ofd1_y)})', 
            color='blue', fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.text(ofd2_x, ofd2_y+offset, f'OFD_2\n({int(ofd2_x)},{int(ofd2_y)})', 
            color='blue', fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Verify against ground truth
    biometry_row = biometry_split[biometry_split['image_name'] == img_name]
    if not biometry_row.empty:
        biometry_row = biometry_row.iloc[0]
        
        bpd_match = (head_row['bpd_1_x'] == biometry_row['bpd_1_x'] and 
                     head_row['bpd_1_y'] == biometry_row['bpd_1_y'] and
                     head_row['bpd_2_x'] == biometry_row['bpd_2_x'] and
                     head_row['bpd_2_y'] == biometry_row['bpd_2_y'])
        
        ofd_match = (head_row['ofd_1_x'] == biometry_row['ofd_1_x'] and 
                     head_row['ofd_1_y'] == biometry_row['ofd_1_y'] and
                     head_row['ofd_2_x'] == biometry_row['ofd_2_x'] and
                     head_row['ofd_2_y'] == biometry_row['ofd_2_y'])
        
        status = "✓ VERIFIED" if (bpd_match and ofd_match) else "✗ MISMATCH"
        color = 'green' if (bpd_match and ofd_match) else 'red'
        
        ax.text(0.5, 0.02, status, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                color='white')
    
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

plt.tight_layout()
output_file = '/cdivece/workspace/fetalbiometry_paper/head_measurements_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"VISUALIZATION SAVED")
print(f"{'='*80}")
print(f"✓ Output: {output_file}")

# Print verification summary
print(f"\n{'='*80}")
print(f"VERIFICATION SUMMARY")
print(f"{'='*80}")

for img_name in test_images:
    head_row = head_test[head_test['image_name'] == img_name]
    biometry_row = biometry_split[biometry_split['image_name'] == img_name]
    
    if not head_row.empty and not biometry_row.empty:
        head_row = head_row.iloc[0]
        biometry_row = biometry_row.iloc[0]
        
        bpd_match = (head_row['bpd_1_x'] == biometry_row['bpd_1_x'] and 
                     head_row['bpd_1_y'] == biometry_row['bpd_1_y'] and
                     head_row['bpd_2_x'] == biometry_row['bpd_2_x'] and
                     head_row['bpd_2_y'] == biometry_row['bpd_2_y'])
        
        ofd_match = (head_row['ofd_1_x'] == biometry_row['ofd_1_x'] and 
                     head_row['ofd_1_y'] == biometry_row['ofd_1_y'] and
                     head_row['ofd_2_x'] == biometry_row['ofd_2_x'] and
                     head_row['ofd_2_y'] == biometry_row['ofd_2_y'])
        
        print(f"\n{img_name}:")
        print(f"  BPD: {'✓ MATCH' if bpd_match else '✗ MISMATCH'}")
        print(f"  OFD: {'✓ MATCH' if ofd_match else '✗ MISMATCH'}")

print(f"\n{'='*80}")
print(f"NOTES")
print(f"{'='*80}")
print("""
Expected anatomy:
- BPD (Biparietal Diameter, RED): Measures the widest transverse diameter 
  of the fetal skull (left-right across the head)
  
- OFD (Occipitofrontal Diameter, BLUE): Measures the longest anteroposterior 
  diameter of the fetal skull (front-back of the head)
  
These two measurements should be roughly perpendicular to each other, 
forming a cross pattern on the fetal head.
""")

plt.show()

