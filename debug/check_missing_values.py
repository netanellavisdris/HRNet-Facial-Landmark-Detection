#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Check for NaN/missing values in all measurement columns across all dataset files
"""

import pandas as pd
import os

# Files to check
FILES_TO_CHECK = [
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

def check_missing_values(csv_file, measurements):
    """Check for missing values in measurement columns"""
    
    if not os.path.exists(csv_file):
        return None, f"File not found"
    
    df = pd.read_csv(csv_file)
    results = {
        'total_rows': len(df),
        'measurements': {}
    }
    
    for measurement in measurements:
        # Get column names for this measurement
        cols = [
            f'{measurement}_1_x',
            f'{measurement}_1_y',
            f'{measurement}_2_x',
            f'{measurement}_2_y'
        ]
        
        # Check if columns exist
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            results['measurements'][measurement] = {
                'status': 'columns_missing',
                'missing_columns': missing_cols,
                'missing_rows': [],
                'missing_count': len(df)
            }
            continue
        
        # Check for NaN values in each column
        missing_by_col = {}
        for col in cols:
            missing_indices = df[df[col].isna()].index.tolist()
            if missing_indices:
                missing_by_col[col] = missing_indices
        
        # Find rows where ANY coordinate is missing
        any_missing_mask = df[cols].isna().any(axis=1)
        missing_rows = df[any_missing_mask].index.tolist()
        
        # Find rows where ALL coordinates are missing
        all_missing_mask = df[cols].isna().all(axis=1)
        all_missing_rows = df[all_missing_mask].index.tolist()
        
        results['measurements'][measurement] = {
            'status': 'complete' if len(missing_rows) == 0 else 'incomplete',
            'missing_count': len(missing_rows),
            'missing_rows': missing_rows,
            'all_missing_rows': all_missing_rows,
            'missing_by_column': missing_by_col,
            'image_names': df.loc[missing_rows, 'image_name'].tolist() if 'image_name' in df.columns else []
        }
    
    return results, None

# Main execution
print("="*80)
print("MISSING VALUES CHECK - ALL DATASETS")
print("="*80)

all_issues = []
complete_files = []

for dataset, structure, split, csv_file, measurements in FILES_TO_CHECK:
    print(f"\n{dataset} - {structure} ({split}):")
    print(f"  File: {csv_file}")
    
    results, error = check_missing_values(csv_file, measurements)
    
    if error:
        print(f"  ✗ ERROR: {error}")
        all_issues.append((dataset, structure, split, 'FILE_ERROR', error))
        continue
    
    print(f"  Total rows: {results['total_rows']}")
    
    file_has_issues = False
    for measurement, info in results['measurements'].items():
        measurement_upper = measurement.upper()
        
        if info['status'] == 'columns_missing':
            print(f"  ✗ {measurement_upper}: COLUMNS MISSING - {info['missing_columns']}")
            all_issues.append((dataset, structure, split, measurement_upper, 'COLUMNS_MISSING', info))
            file_has_issues = True
        elif info['status'] == 'incomplete':
            missing_count = info['missing_count']
            total = results['total_rows']
            percentage = (missing_count / total * 100) if total > 0 else 0
            
            print(f"  ✗ {measurement_upper}: {missing_count}/{total} rows with missing values ({percentage:.1f}%)")
            
            # Show details about missing values
            if info['missing_by_column']:
                print(f"      Missing by column:")
                for col, indices in info['missing_by_column'].items():
                    print(f"        - {col}: {len(indices)} missing")
            
            if info['all_missing_rows']:
                print(f"      Completely missing (all 4 coords): {len(info['all_missing_rows'])} rows")
            
            # Show some example image names
            if info['image_names']:
                examples = info['image_names'][:3]
                print(f"      Example images: {', '.join(examples)}")
                if len(info['image_names']) > 3:
                    print(f"      ... and {len(info['image_names']) - 3} more")
            
            all_issues.append((dataset, structure, split, measurement_upper, 'INCOMPLETE', info))
            file_has_issues = True
        else:
            print(f"  ✓ {measurement_upper}: Complete (no missing values)")
    
    if not file_has_issues:
        complete_files.append((dataset, structure, split))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all_issues:
    print(f"\n⚠ FOUND {len(all_issues)} ISSUE(S):\n")
    
    for issue in all_issues:
        if issue[4] == 'COLUMNS_MISSING':
            dataset, structure, split, measurement, issue_type, info = issue
            print(f"  ✗ {dataset} {structure} ({split}) - {measurement}:")
            print(f"      Missing columns: {info['missing_columns']}")
        elif issue[4] == 'INCOMPLETE':
            dataset, structure, split, measurement, issue_type, info = issue
            print(f"  ✗ {dataset} {structure} ({split}) - {measurement}:")
            print(f"      {info['missing_count']} rows with missing values")
            if info['image_names']:
                print(f"      Images affected: {', '.join(info['image_names'][:5])}")
                if len(info['image_names']) > 5:
                    print(f"      ... and {len(info['image_names']) - 5} more")
        else:
            print(f"  ✗ {issue[0]} {issue[1]} ({issue[2]}): {issue[3]}")
else:
    print("\n✓ NO ISSUES FOUND - All measurement columns are complete!")

print(f"\n✓ {len(complete_files)}/{len(FILES_TO_CHECK)} files are completely populated")

# Create detailed report file
print("\n" + "="*80)
print("CREATING DETAILED REPORT")
print("="*80)

with open('missing_values_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("MISSING VALUES DETAILED REPORT\n")
    f.write("="*80 + "\n\n")
    
    if all_issues:
        f.write(f"FOUND {len(all_issues)} ISSUE(S):\n\n")
        
        for issue in all_issues:
            if issue[4] == 'INCOMPLETE':
                dataset, structure, split, measurement, issue_type, info = issue
                f.write(f"\n{dataset} - {structure} ({split}) - {measurement}:\n")
                f.write(f"  Missing count: {info['missing_count']}\n")
                f.write(f"  Missing rows (indices): {info['missing_rows']}\n")
                f.write(f"  Completely missing rows: {info['all_missing_rows']}\n")
                f.write(f"  Image names: {info['image_names']}\n")
                f.write(f"  Missing by column:\n")
                for col, indices in info['missing_by_column'].items():
                    f.write(f"    {col}: indices {indices}\n")
                f.write("\n")
    else:
        f.write("NO ISSUES FOUND - All measurement columns are complete!\n")

print("✓ Detailed report saved to: missing_values_report.txt")

print("\n" + "="*80)
print("✓ CHECK COMPLETE")
print("="*80)

