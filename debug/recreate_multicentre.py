#!/usr/bin/env python3
"""
Recreate MULTICENTRE annotations by merging FP, HC18, and UCL datasets.
This ensures the MULTICENTRE dataset uses the updated UCL file naming convention.
"""

import pandas as pd
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MULTICENTRE_DIR = BASE_DIR / "MULTICENTRE"

# Create backup of existing MULTICENTRE files
print("Creating backups of existing MULTICENTRE files...")
MULTICENTRE_DIR.mkdir(exist_ok=True)
for csv_file in MULTICENTRE_DIR.glob("*.csv"):
    backup_file = csv_file.with_suffix(".csv.backup")
    if csv_file.exists():
        csv_file.rename(backup_file)
        print(f"  Backed up: {csv_file.name} -> {backup_file.name}")

# Define datasets and anatomies to process
DATASETS = {
    "Head": ["FP", "HC18", "UCL"],
    "Abdomen": ["FP", "UCL"],
    "Femur": ["FP", "UCL"]
}

# Process each anatomy
for anatomy, datasets in DATASETS.items():
    print(f"\nProcessing {anatomy}...")
    
    # Process combined files (all data)
    dfs_all = []
    for dataset in datasets:
        csv_file = BASE_DIR / dataset / f"{anatomy}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"  Loaded {dataset}/{anatomy}.csv: {len(df)} rows")
            dfs_all.append(df)
        else:
            print(f"  Warning: {csv_file} not found")
    
    if dfs_all:
        combined_df = pd.concat(dfs_all, ignore_index=True)
        combined_df['index'] = range(len(combined_df))
        output_file = MULTICENTRE_DIR / f"{anatomy}.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"  Created {anatomy}.csv: {len(combined_df)} rows")
    
    # Process Train files
    dfs_train = []
    for dataset in datasets:
        csv_file = BASE_DIR / dataset / f"{anatomy}_Train.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"  Loaded {dataset}/{anatomy}_Train.csv: {len(df)} rows")
            dfs_train.append(df)
        else:
            print(f"  Warning: {csv_file} not found")
    
    if dfs_train:
        combined_df = pd.concat(dfs_train, ignore_index=True)
        combined_df['index'] = range(len(combined_df))
        output_file = MULTICENTRE_DIR / f"{anatomy}_Train.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"  Created {anatomy}_Train.csv: {len(combined_df)} rows")
    
    # Process Test files
    dfs_test = []
    for dataset in datasets:
        csv_file = BASE_DIR / dataset / f"{anatomy}_Test.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"  Loaded {dataset}/{anatomy}_Test.csv: {len(df)} rows")
            dfs_test.append(df)
        else:
            print(f"  Warning: {csv_file} not found")
    
    if dfs_test:
        combined_df = pd.concat(dfs_test, ignore_index=True)
        combined_df['index'] = range(len(combined_df))
        output_file = MULTICENTRE_DIR / f"{anatomy}_Test.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"  Created {anatomy}_Test.csv: {len(combined_df)} rows")

print("\n✓ MULTICENTRE annotations recreated successfully!")
print("  Old files backed up with .backup extension")

