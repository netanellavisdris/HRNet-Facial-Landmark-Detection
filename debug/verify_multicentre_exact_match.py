#!/usr/bin/env python3
"""
Verify that MULTICENTRE CSV files contain exactly the same rows and values
as the source files from FP, HC18, and UCL.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def compare_csv_files(multicentre_file, source_files, name):
    """Compare MULTICENTRE file with source files to ensure exact match."""
    print(f"\n{'='*80}")
    print(f"Verifying: {name}")
    print(f"{'='*80}")
    
    # Read MULTICENTRE file
    if not multicentre_file.exists():
        print(f"❌ ERROR: MULTICENTRE file not found: {multicentre_file}")
        return False
    
    df_multicentre = pd.read_csv(multicentre_file)
    print(f"MULTICENTRE file: {len(df_multicentre)} rows, {len(df_multicentre.columns)} columns")
    
    # Read and combine all source files
    dfs_source = []
    total_source_rows = 0
    
    for source_file in source_files:
        if not source_file.exists():
            print(f"❌ ERROR: Source file not found: {source_file}")
            return False
        
        df = pd.read_csv(source_file)
        dfs_source.append(df)
        total_source_rows += len(df)
        print(f"  Source ({source_file.parent.name}/{source_file.name}): {len(df)} rows, {len(df.columns)} columns")
    
    # Combine all source dataframes
    df_combined = pd.concat(dfs_source, ignore_index=True)
    print(f"\nCombined source files: {len(df_combined)} rows")
    
    # Check column consistency
    multicentre_cols = df_multicentre.columns.tolist()
    combined_cols = df_combined.columns.tolist()
    
    if multicentre_cols != combined_cols:
        print(f"\n⚠️  WARNING: Column mismatch!")
        print(f"  MULTICENTRE columns: {multicentre_cols}")
        print(f"  Source columns:      {combined_cols}")
        
        # Try to align columns
        common_cols = [c for c in multicentre_cols if c in combined_cols]
        if len(common_cols) < len(multicentre_cols):
            print(f"  Only {len(common_cols)}/{len(multicentre_cols)} columns match")
            return False
        
        df_multicentre = df_multicentre[common_cols]
        df_combined = df_combined[common_cols]
        print(f"  Using common columns: {common_cols}")
    else:
        print(f"✅ Column names match: {len(multicentre_cols)} columns")
    
    # Reset index column if it exists (it may have been renumbered)
    if 'index' in df_multicentre.columns:
        df_multicentre = df_multicentre.drop(columns=['index'])
    if 'index' in df_combined.columns:
        df_combined = df_combined.drop(columns=['index'])
    
    # Sort both dataframes by all columns to ensure consistent ordering
    # This allows us to compare row by row
    sort_cols = [c for c in df_multicentre.columns if c != 'index']
    df_multicentre_sorted = df_multicentre.sort_values(by=sort_cols).reset_index(drop=True)
    df_combined_sorted = df_combined.sort_values(by=sort_cols).reset_index(drop=True)
    
    # Compare row counts
    if len(df_multicentre_sorted) != len(df_combined_sorted):
        print(f"\n❌ ROW COUNT MISMATCH!")
        print(f"  MULTICENTRE: {len(df_multicentre_sorted)} rows")
        print(f"  Source combined: {len(df_combined_sorted)} rows")
        print(f"  Difference: {abs(len(df_multicentre_sorted) - len(df_combined_sorted))} rows")
        return False
    
    print(f"✅ Row counts match: {len(df_multicentre_sorted)} rows")
    
    # Normalize data types and handle NaN values for comparison
    # Convert numeric columns to float for consistent comparison, handle NaN
    def normalize_value(val):
        """Normalize a value for comparison."""
        if pd.isna(val):
            return ''
        # Convert numeric values to float for consistent representation
        try:
            if isinstance(val, (int, float)):
                return float(val)
            # Try to convert string numbers to float
            if isinstance(val, str) and val.strip():
                return float(val)
        except (ValueError, TypeError):
            pass
        # For non-numeric values, use string representation
        return str(val).strip() if val else ''
    
    def row_to_tuple(row_values, row_dict):
        """Convert row to tuple, handling NaN and normalizing numeric values."""
        return tuple(normalize_value(val) for val in row_values)
    
    multicentre_tuples = set()
    multicentre_by_image = {}
    for idx, row in df_multicentre_sorted.iterrows():
        row_tuple = row_to_tuple(row.values, row.to_dict())
        multicentre_tuples.add(row_tuple)
        # Store by image_name for error reporting (use original dataframe)
        if 'image_name' in df_multicentre_sorted.columns:
            img_name = str(row['image_name']) if pd.notna(row['image_name']) else f'row_{idx}'
            multicentre_by_image[img_name] = row_tuple
    
    combined_tuples = set()
    combined_by_image = {}
    for idx, row in df_combined_sorted.iterrows():
        row_tuple = row_to_tuple(row.values, row.to_dict())
        combined_tuples.add(row_tuple)
        # Store by image_name for error reporting (use original dataframe)
        if 'image_name' in df_combined_sorted.columns:
            img_name = str(row['image_name']) if pd.notna(row['image_name']) else f'row_{idx}'
            combined_by_image[img_name] = row_tuple
    
    missing_in_multicentre = combined_tuples - multicentre_tuples
    extra_in_multicentre = multicentre_tuples - combined_tuples
    
    if missing_in_multicentre:
        print(f"\n❌ MISSING {len(missing_in_multicentre)} rows in MULTICENTRE file")
        # Try to find image names for missing rows
        missing_images = []
        for row_tuple in list(missing_in_multicentre)[:10]:
            # Find image name in combined_by_image
            for img_name, img_tuple in combined_by_image.items():
                if img_tuple == row_tuple:
                    missing_images.append(img_name)
                    break
            else:
                # If no image name found, use first non-empty value
                missing_images.append(row_tuple[0] if row_tuple and row_tuple[0] else 'unknown')
        
        for i, img_name in enumerate(missing_images[:5], 1):
            print(f"  {i}. {img_name}")
        if len(missing_in_multicentre) > 5:
            print(f"  ... and {len(missing_in_multicentre) - 5} more")
        
        # Show a sample of the actual missing row data
        if missing_in_multicentre:
            sample_missing = list(missing_in_multicentre)[0]
            print(f"\n  Sample missing row (first 5 values): {sample_missing[:5]}")
        return False
    
    if extra_in_multicentre:
        print(f"\n⚠️  MULTICENTRE file has {len(extra_in_multicentre)} extra rows")
        # Try to find image names for extra rows
        extra_images = []
        for row_tuple in list(extra_in_multicentre)[:10]:
            # Find image name in multicentre_by_image
            for img_name, img_tuple in multicentre_by_image.items():
                if img_tuple == row_tuple:
                    extra_images.append(img_name)
                    break
            else:
                # If no image name found, use first non-empty value
                extra_images.append(row_tuple[0] if row_tuple and row_tuple[0] else 'unknown')
        
        for i, img_name in enumerate(extra_images[:5], 1):
            print(f"  {i}. {img_name}")
        if len(extra_in_multicentre) > 5:
            print(f"  ... and {len(extra_in_multicentre) - 5} more")
        return False
    
    print(f"\n✅ VERIFICATION PASSED: All rows and values match exactly!")
    return True

def main():
    base_dir = Path("fetalbiometrydata")
    multicentre_dir = base_dir / "MULTICENTRE"
    
    results = []
    
    # ========================================================================
    # ABDOMEN
    # ========================================================================
    
    results.append(compare_csv_files(
        multicentre_file=multicentre_dir / "Abdomen_Test.csv",
        source_files=[
            base_dir / "FP" / "Abdomen_Test.csv",
            base_dir / "UCL" / "Abdomen_Test.csv"
        ],
        name="Abdomen_Test.csv"
    ))
    
    results.append(compare_csv_files(
        multicentre_file=multicentre_dir / "Abdomen_Train.csv",
        source_files=[
            base_dir / "FP" / "Abdomen_Train.csv",
            base_dir / "UCL" / "Abdomen_Train.csv"
        ],
        name="Abdomen_Train.csv"
    ))
    
    # ========================================================================
    # FEMUR
    # ========================================================================
    
    results.append(compare_csv_files(
        multicentre_file=multicentre_dir / "Femur_Test.csv",
        source_files=[
            base_dir / "FP" / "Femur_Test.csv",
            base_dir / "UCL" / "Femur_Test.csv"
        ],
        name="Femur_Test.csv"
    ))
    
    results.append(compare_csv_files(
        multicentre_file=multicentre_dir / "Femur_Train.csv",
        source_files=[
            base_dir / "FP" / "Femur_Train.csv",
            base_dir / "UCL" / "Femur_Train.csv"
        ],
        name="Femur_Train.csv"
    ))
    
    # ========================================================================
    # HEAD
    # ========================================================================
    
    results.append(compare_csv_files(
        multicentre_file=multicentre_dir / "Head_Test.csv",
        source_files=[
            base_dir / "FP" / "Head_Test.csv",
            base_dir / "HC18" / "Head_Test.csv",
            base_dir / "UCL" / "Head_Test.csv"
        ],
        name="Head_Test.csv"
    ))
    
    results.append(compare_csv_files(
        multicentre_file=multicentre_dir / "Head_Train.csv",
        source_files=[
            base_dir / "FP" / "Head_Train.csv",
            base_dir / "HC18" / "Head_Train.csv",
            base_dir / "UCL" / "Head_Train.csv"
        ],
        name="Head_Train.csv"
    ))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL VERIFICATIONS PASSED!")
        print("MULTICENTRE files contain exactly the same rows and values as source files.")
        return 0
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("Please check the differences above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

