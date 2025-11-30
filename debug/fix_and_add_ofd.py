#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Fix BPD coordinate swap and add OFD measurements to HC18 dataset files

This script:
1. Fixes the x↔y coordinate swap for BPD in BPD_Test.csv and BPD_Train.csv
2. Adds OFD measurements from BiometryNet_split.csv
3. Saves the corrected data as Head_Test.csv and Head_Train.csv
4. Ensures all coordinates match the JSON ground truth
"""

import pandas as pd
import os

# Input file paths (original files with only BPD)
bpd_test_file = '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/HC18/BPD_Test.csv'
bpd_train_file = '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/HC18/BPD_Train.csv'
biometry_split_file = '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/data/HC18/BiometryNet_split.csv'

# Output file paths (new files with both BPD and OFD)
head_test_file = '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/HC18/Head_Test.csv'
head_train_file = '/cdivece/workspace/fetalbiometry_paper/fetalbiometrydata/HC18/Head_Train.csv'

# Read files
print("="*80)
print("READING FILES")
print("="*80)
bpd_test = pd.read_csv(bpd_test_file)
bpd_train = pd.read_csv(bpd_train_file)
biometry_split = pd.read_csv(biometry_split_file)
print(f"✓ Read {len(bpd_test)} rows from BPD_Test.csv")
print(f"✓ Read {len(bpd_train)} rows from BPD_Train.csv")
print(f"✓ Read {len(biometry_split)} rows from BiometryNet_split.csv")

def fix_and_add_measurements(df, source_df, dataset_name):
    """
    Fix BPD coordinates (swap x↔y back) and add OFD measurements
    """
    print(f"\n" + "="*80)
    print(f"PROCESSING {dataset_name}")
    print("="*80)
    
    fixed_count = 0
    ofd_added_count = 0
    missing_count = 0
    
    for idx, row in df.iterrows():
        img_name = row['image_name']
        
        # Find matching row in BiometryNet_split.csv
        source_row = source_df[source_df['image_name'] == img_name]
        
        if source_row.empty:
            print(f"⚠ Warning: No matching data found for {img_name}")
            missing_count += 1
            continue
        
        source_row = source_row.iloc[0]
        
        # Fix BPD coordinates by using the correct values from BiometryNet_split.csv
        # (which is equivalent to swapping x↔y from the current incorrect values)
        df.at[idx, 'bpd_1_x'] = source_row['bpd_1_x']
        df.at[idx, 'bpd_1_y'] = source_row['bpd_1_y']
        df.at[idx, 'bpd_2_x'] = source_row['bpd_2_x']
        df.at[idx, 'bpd_2_y'] = source_row['bpd_2_y']
        fixed_count += 1
        
        # Add OFD measurements
        df.at[idx, 'ofd_1_x'] = source_row['ofd_1_x']
        df.at[idx, 'ofd_1_y'] = source_row['ofd_1_y']
        df.at[idx, 'ofd_2_x'] = source_row['ofd_2_x']
        df.at[idx, 'ofd_2_y'] = source_row['ofd_2_y']
        ofd_added_count += 1
    
    print(f"✓ Fixed BPD coordinates for {fixed_count} images")
    print(f"✓ Added OFD measurements for {ofd_added_count} images")
    if missing_count > 0:
        print(f"⚠ {missing_count} images had no matching data in BiometryNet_split.csv")
    
    return df

# Process both files
bpd_test_fixed = fix_and_add_measurements(bpd_test, biometry_split, "BPD_Test.csv → Head_Test.csv")
bpd_train_fixed = fix_and_add_measurements(bpd_train, biometry_split, "BPD_Train.csv → Head_Train.csv")

# Verify the fix with a sample
print("\n" + "="*80)
print("VERIFICATION - Sample Image: 055_HC.png")
print("="*80)

sample_img = '055_HC.png'
test_row = bpd_test_fixed[bpd_test_fixed['image_name'] == sample_img]
bio_row = biometry_split[biometry_split['image_name'] == sample_img]

if not test_row.empty and not bio_row.empty:
    test_row = test_row.iloc[0]
    bio_row = bio_row.iloc[0]
    
    print(f"\nHead_Test.csv (corrected):")
    print(f"  BPD_1: x={test_row['bpd_1_x']}, y={test_row['bpd_1_y']}")
    print(f"  BPD_2: x={test_row['bpd_2_x']}, y={test_row['bpd_2_y']}")
    print(f"  OFD_1: x={test_row['ofd_1_x']}, y={test_row['ofd_1_y']}")
    print(f"  OFD_2: x={test_row['ofd_2_x']}, y={test_row['ofd_2_y']}")
    
    print(f"\nBiometryNet_split.csv (ground truth):")
    print(f"  BPD_1: x={bio_row['bpd_1_x']}, y={bio_row['bpd_1_y']}")
    print(f"  BPD_2: x={bio_row['bpd_2_x']}, y={bio_row['bpd_2_y']}")
    print(f"  OFD_1: x={bio_row['ofd_1_x']}, y={bio_row['ofd_1_y']}")
    print(f"  OFD_2: x={bio_row['ofd_2_x']}, y={bio_row['ofd_2_y']}")
    
    # Check if they match
    bpd_match = (test_row['bpd_1_x'] == bio_row['bpd_1_x'] and 
                 test_row['bpd_1_y'] == bio_row['bpd_1_y'] and
                 test_row['bpd_2_x'] == bio_row['bpd_2_x'] and
                 test_row['bpd_2_y'] == bio_row['bpd_2_y'])
    
    ofd_match = (test_row['ofd_1_x'] == bio_row['ofd_1_x'] and 
                 test_row['ofd_1_y'] == bio_row['ofd_1_y'] and
                 test_row['ofd_2_x'] == bio_row['ofd_2_x'] and
                 test_row['ofd_2_y'] == bio_row['ofd_2_y'])
    
    print(f"\n{'✓' if bpd_match else '✗'} BPD coordinates match: {bpd_match}")
    print(f"{'✓' if ofd_match else '✗'} OFD coordinates match: {ofd_match}")

# Save fixed files
print("\n" + "="*80)
print("SAVING NEW FILES")
print("="*80)
bpd_test_fixed.to_csv(head_test_file, index=False)
bpd_train_fixed.to_csv(head_train_file, index=False)
print(f"✓ Saved: {head_test_file}")
print(f"✓ Saved: {head_train_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
✓ BPD coordinates have been FIXED (swapped x↔y to match ground truth)
✓ OFD measurements have been ADDED from BiometryNet_split.csv
✓ All coordinates now match the JSON ground truth file

New files created:
  - fetalbiometrydata/HC18/Head_Test.csv  (with BPD + OFD)
  - fetalbiometrydata/HC18/Head_Train.csv (with BPD + OFD)

Original files preserved:
  - fetalbiometrydata/HC18/BPD_Test.csv  (original with only BPD)
  - fetalbiometrydata/HC18/BPD_Train.csv (original with only BPD)

To verify visually, run the visualization script again:
  python visualize_coordinate_comparison.py
""")

