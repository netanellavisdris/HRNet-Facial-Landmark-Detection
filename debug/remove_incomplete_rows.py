#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Remove rows with any missing measurement values from all dataset files
"""

import pandas as pd
import os
import shutil

# Files to clean
FILES_TO_CLEAN = [
    # FP dataset
    ('FP', 'Abdomen', 'Test', 'fetalbiometrydata/FP/Abdomen_Test.csv', ['tad', 'apad']),
    ('FP', 'Abdomen', 'Train', 'fetalbiometrydata/FP/Abdomen_Train.csv', ['tad', 'apad']),
    ('FP', 'Femur', 'Test', 'fetalbiometrydata/FP/Femur_Test.csv', ['fl']),
    ('FP', 'Femur', 'Train', 'fetalbiometrydata/FP/Femur_Train.csv', ['fl']),
    ('FP', 'Head', 'Test', 'fetalbiometrydata/FP/Head_Test.csv', ['bpd', 'ofd']),
    ('FP', 'Head', 'Train', 'fetalbiometrydata/FP/Head_Train.csv', ['bpd', 'ofd']),
    
    # HC18 dataset
    ('HC18', 'Head', 'Test', 'fetalbiometrydata/HC18/Head_Test.csv', ['bpd', 'ofd']),
    ('HC18', 'Head', 'Train', 'fetalbiometrydata/HC18/Head_Train.csv', ['bpd', 'ofd']),
    
    # UCL dataset
    ('UCL', 'Abdomen', 'Test', 'fetalbiometrydata/UCL/Abdomen_Test.csv', ['tad', 'apad']),
    ('UCL', 'Abdomen', 'Train', 'fetalbiometrydata/UCL/Abdomen_Train.csv', ['tad', 'apad']),
    ('UCL', 'Femur', 'Test', 'fetalbiometrydata/UCL/Femur_Test.csv', ['fl']),
    ('UCL', 'Femur', 'Train', 'fetalbiometrydata/UCL/Femur_Train.csv', ['fl']),
    ('UCL', 'Head', 'Test', 'fetalbiometrydata/UCL/Head_Test.csv', ['bpd', 'ofd']),
    ('UCL', 'Head', 'Train', 'fetalbiometrydata/UCL/Head_Train.csv', ['bpd', 'ofd']),
]

def clean_file(csv_file, measurements):
    """Remove rows with any missing measurement values"""
    
    if not os.path.exists(csv_file):
        return None, "File not found"
    
    # Read the file
    df = pd.read_csv(csv_file)
    original_count = len(df)
    
    # Collect all measurement columns
    all_measurement_cols = []
    for measurement in measurements:
        cols = [
            f'{measurement}_1_x',
            f'{measurement}_1_y',
            f'{measurement}_2_x',
            f'{measurement}_2_y'
        ]
        # Only add columns that exist
        existing_cols = [col for col in cols if col in df.columns]
        all_measurement_cols.extend(existing_cols)
    
    if not all_measurement_cols:
        return None, "No measurement columns found"
    
    # Find rows with ANY missing values in measurement columns
    missing_mask = df[all_measurement_cols].isna().any(axis=1)
    rows_to_remove = df[missing_mask]
    
    if len(rows_to_remove) == 0:
        return {
            'removed_count': 0,
            'original_count': original_count,
            'final_count': original_count,
            'removed_images': [],
            'action': 'none'
        }, None
    
    # Get info about removed rows
    removed_images = rows_to_remove['image_name'].tolist() if 'image_name' in df.columns else []
    removed_indices = rows_to_remove.index.tolist()
    
    # Remove rows with missing values
    df_clean = df[~missing_mask].copy()
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    result = {
        'removed_count': len(rows_to_remove),
        'original_count': original_count,
        'final_count': len(df_clean),
        'removed_images': removed_images,
        'removed_indices': removed_indices,
        'cleaned_df': df_clean,
        'action': 'cleaned'
    }
    
    return result, None

# Main execution
print("="*80)
print("REMOVING INCOMPLETE ROWS")
print("="*80)

files_with_removals = []
files_unchanged = []
total_removed = 0

for dataset, structure, split, csv_file, measurements in FILES_TO_CLEAN:
    print(f"\n{dataset} - {structure} ({split}):")
    print(f"  File: {csv_file}")
    
    result, error = clean_file(csv_file, measurements)
    
    if error:
        print(f"  ✗ ERROR: {error}")
        continue
    
    if result['action'] == 'none':
        print(f"  ✓ No missing values - file unchanged ({result['original_count']} rows)")
        files_unchanged.append((dataset, structure, split))
    else:
        print(f"  ⚠ Removing {result['removed_count']} row(s) with missing values")
        print(f"     Original: {result['original_count']} rows")
        print(f"     Final: {result['final_count']} rows")
        
        if result['removed_images']:
            print(f"     Removed images: {', '.join(result['removed_images'])}")
        
        # Create backup
        backup_file = f"{csv_file}.backup_before_removal"
        shutil.copy2(csv_file, backup_file)
        print(f"  ✓ Backup created: {backup_file}")
        
        # Save cleaned file
        result['cleaned_df'].to_csv(csv_file, index=False)
        print(f"  ✓ Cleaned file saved: {csv_file}")
        
        files_with_removals.append((dataset, structure, split, result))
        total_removed += result['removed_count']

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if files_with_removals:
    print(f"\n✓ Cleaned {len(files_with_removals)} file(s):")
    for dataset, structure, split, result in files_with_removals:
        print(f"  • {dataset} {structure} ({split}): Removed {result['removed_count']} row(s)")
        for img in result['removed_images']:
            print(f"      - {img}")
    
    print(f"\n✓ Total rows removed: {total_removed}")
else:
    print("\n✓ No files needed cleaning - all measurements are complete!")

print(f"\n✓ Files unchanged: {len(files_unchanged)}")
print(f"✓ Files cleaned: {len(files_with_removals)}")

# Verify the cleaning
print("\n" + "="*80)
print("VERIFICATION - Re-checking for missing values")
print("="*80)

for dataset, structure, split, csv_file, measurements in FILES_TO_CLEAN:
    df = pd.read_csv(csv_file)
    
    all_measurement_cols = []
    for measurement in measurements:
        cols = [f'{measurement}_1_x', f'{measurement}_1_y', 
                f'{measurement}_2_x', f'{measurement}_2_y']
        existing_cols = [col for col in cols if col in df.columns]
        all_measurement_cols.extend(existing_cols)
    
    if all_measurement_cols:
        missing_count = df[all_measurement_cols].isna().any(axis=1).sum()
        if missing_count > 0:
            print(f"  ✗ {dataset} {structure} ({split}): Still has {missing_count} incomplete rows!")
        else:
            # Only print if file was cleaned
            if any(dataset == d and structure == s and split == sp 
                   for d, s, sp, _ in files_with_removals):
                print(f"  ✓ {dataset} {structure} ({split}): Clean ({len(df)} rows)")

print("\n" + "="*80)
print("✓ CLEANUP COMPLETE")
print("="*80)

if files_with_removals:
    print("\nBackup files created (can be deleted if satisfied with results):")
    for dataset, structure, split, result in files_with_removals:
        csv_file = [f for d, s, sp, f, _ in FILES_TO_CLEAN 
                    if d == dataset and s == structure and sp == split][0]
        print(f"  - {csv_file}.backup_before_removal")

