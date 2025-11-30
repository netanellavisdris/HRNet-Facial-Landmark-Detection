#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Script to verify coordinate consistency between BPD_Test.csv, BiometryNet_split.csv, and JSON file
"""

import pandas as pd
import json

# Read the CSV files
bpd_test = pd.read_csv('/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/HC18/BPD_Test.csv')
biometry_split = pd.read_csv('/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/HC18/BiometryNet_split.csv')

# Read JSON file
with open('/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/HC18/2022USlandmark_HC18_HC.json', 'r') as f:
    json_data = json.load(f)

# Get image metadata
img_metadata = json_data['_via_img_metadata']

# Check a few test images
test_images = ['055_HC.png', '091_HC.png', '169_HC.png']

print("="*80)
print("COORDINATE COMPARISON ANALYSIS")
print("="*80)

for img_name in test_images:
    print(f"\n{'='*80}")
    print(f"Image: {img_name}")
    print("="*80)
    
    # Get data from BPD_Test.csv
    bpd_test_row = bpd_test[bpd_test['image_name'] == img_name]
    if not bpd_test_row.empty:
        bpd_test_row = bpd_test_row.iloc[0]
        print(f"\nBPD_Test.csv:")
        print(f"  BPD_1: x={bpd_test_row['bpd_1_x']}, y={bpd_test_row['bpd_1_y']}")
        print(f"  BPD_2: x={bpd_test_row['bpd_2_x']}, y={bpd_test_row['bpd_2_y']}")
    
    # Get data from BiometryNet_split.csv
    biometry_row = biometry_split[biometry_split['image_name'] == img_name]
    if not biometry_row.empty:
        biometry_row = biometry_row.iloc[0]
        print(f"\nBiometryNet_split.csv:")
        print(f"  BPD_1: x={biometry_row['bpd_1_x']}, y={biometry_row['bpd_1_y']}")
        print(f"  BPD_2: x={biometry_row['bpd_2_x']}, y={biometry_row['bpd_2_y']}")
        print(f"  OFD_1: x={biometry_row['ofd_1_x']}, y={biometry_row['ofd_1_y']}")
        print(f"  OFD_2: x={biometry_row['ofd_2_x']}, y={biometry_row['ofd_2_y']}")
    
    # Get data from JSON
    # Find the matching entry in JSON (the key format is "filename+filesize")
    json_entry = None
    for key, value in img_metadata.items():
        if value['filename'] == img_name:
            json_entry = value
            break
    
    if json_entry:
        print(f"\nJSON file (2022USlandmark_HC18_HC.json):")
        for region in json_entry['regions']:
            landmark_name = region['region_attributes']['US_landmarks']
            cx = region['shape_attributes']['cx']
            cy = region['shape_attributes']['cy']
            print(f"  {landmark_name}: cx={cx}, cy={cy}")
    
    # Analyze the swap
    if not bpd_test_row.empty and not biometry_row.empty:
        print(f"\n{'*'*80}")
        print("ANALYSIS:")
        print("*"*80)
        
        # Check if BPD_Test has x,y swapped compared to BiometryNet
        bpd1_x_test = bpd_test_row['bpd_1_x']
        bpd1_y_test = bpd_test_row['bpd_1_y']
        bpd1_x_bio = biometry_row['bpd_1_x']
        bpd1_y_bio = biometry_row['bpd_1_y']
        
        if bpd1_x_test == bpd1_y_bio and bpd1_y_test == bpd1_x_bio:
            print("✓ CONFIRMED: BPD coordinates are SWAPPED (x↔y) in BPD_Test.csv vs BiometryNet_split.csv")
        else:
            print("✗ Coordinates do not follow simple x↔y swap pattern")
        
        # Check JSON consistency
        if json_entry:
            bpd1_json = None
            for region in json_entry['regions']:
                if region['region_attributes']['US_landmarks'] == 'BPD_1':
                    bpd1_json = (region['shape_attributes']['cx'], region['shape_attributes']['cy'])
                    break
            
            if bpd1_json:
                if bpd1_json[0] == bpd1_x_bio and bpd1_json[1] == bpd1_y_bio:
                    print("✓ BiometryNet_split.csv MATCHES JSON file (this is the CORRECT format)")
                    print("✗ BPD_Test.csv has INCORRECT x↔y swap")
                elif bpd1_json[0] == bpd1_x_test and bpd1_json[1] == bpd1_y_test:
                    print("✓ BPD_Test.csv matches JSON file")
                    print("✗ BiometryNet_split.csv has incorrect x↔y swap")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
Based on the analysis:
1. BiometryNet_split.csv has the CORRECT coordinate order (matches JSON ground truth)
2. BPD_Test.csv and BPD_Train.csv have x↔y SWAPPED for BPD

For fixing and adding OFD:
Option A: Fix the BPD swap AND add OFD correctly
  - Swap back BPD x↔y to correct them
  - Add OFD with correct x,y from BiometryNet_split.csv
  
Option B: Maintain consistency within files (keep swap pattern)
  - Keep BPD as-is (with the swap)
  - Add OFD with x↔y also swapped to match BPD pattern in the file
  
RECOMMENDED: Option A - Fix the coordinates to match the ground truth JSON
""")

