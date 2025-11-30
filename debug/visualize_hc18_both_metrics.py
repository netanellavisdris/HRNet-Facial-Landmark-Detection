"""
Visualize HC18 images with both OFD and BPD markers to verify correctness
"""
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('fetalbiometrydata/HC18/Head.csv')
img_dir = 'fetalbiometrydata/data/HC18/Head'

# Select a few random images
sample_images = df.sample(n=6, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (_, row) in enumerate(sample_images.iterrows()):
    img_path = os.path.join(img_dir, row['image_name'])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Could not load: {img_path}")
        continue
    
    # Convert to RGB for colored markers
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Draw OFD in BLUE
    ofd_1 = (int(row['ofd_1_x']), int(row['ofd_1_y']))
    ofd_2 = (int(row['ofd_2_x']), int(row['ofd_2_y']))
    cv2.circle(img_rgb, ofd_1, 5, (0, 0, 255), -1)  # Blue
    cv2.circle(img_rgb, ofd_2, 5, (0, 0, 255), -1)
    cv2.line(img_rgb, ofd_1, ofd_2, (0, 0, 255), 2)
    
    # Draw BPD in RED
    bpd_1 = (int(row['bpd_1_x']), int(row['bpd_1_y']))
    bpd_2 = (int(row['bpd_2_x']), int(row['bpd_2_y']))
    cv2.circle(img_rgb, bpd_1, 5, (255, 0, 0), -1)  # Red
    cv2.circle(img_rgb, bpd_2, 5, (255, 0, 0), -1)
    cv2.line(img_rgb, bpd_1, bpd_2, (255, 0, 0), 2)
    
    axes[idx].imshow(img_rgb)
    axes[idx].set_title(f"{row['image_name']}\nOFD(blue) BPD(red)")
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('debug/viz_HC18_both_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: debug/viz_HC18_both_metrics.png")
plt.close()

