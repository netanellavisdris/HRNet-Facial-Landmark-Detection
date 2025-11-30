#!/usr/bin/env python3
"""
Verify that MULTICENTRE CSV files contain all data from source files.
"""

import csv
import sys
from pathlib import Path

def read_csv_rows(filepath):
    """Read all rows from a CSV file, returning as list of tuples (excluding header)."""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if row:  # Skip empty rows
                rows.append(tuple(row))
    return rows, header

def verify_merge(target_file, source_files, name):
    """Verify that target_file contains all rows from source_files."""
    print(f"\n{'='*60}")
    print(f"Verifying {name}")
    print(f"{'='*60}")
    
    # Read target file
    target_rows, target_header = read_csv_rows(target_file)
    print(f"Target file ({target_file}): {len(target_rows)} data rows")
    
    # Read all source files
    all_source_rows = []
    all_source_headers = []
    for source_file in source_files:
        if not Path(source_file).exists():
            print(f"WARNING: Source file not found: {source_file}")
            continue
        rows, header = read_csv_rows(source_file)
        all_source_rows.extend(rows)
        all_source_headers.append((source_file, header))
        print(f"  Source ({source_file}): {len(rows)} data rows")
    
    # Check if target contains all source rows
    target_set = set(target_rows)
    source_set = set(all_source_rows)
    
    missing_in_target = source_set - target_set
    extra_in_target = target_set - source_set
    
    if missing_in_target:
        print(f"\n❌ MISSING {len(missing_in_target)} rows in target file:")
        for i, row in enumerate(list(missing_in_target)[:10]):  # Show first 10
            print(f"  {i+1}. {row[0] if row else 'empty row'}")
        if len(missing_in_target) > 10:
            print(f"  ... and {len(missing_in_target) - 10} more")
    else:
        print(f"\n✅ All source rows are present in target file")
    
    if extra_in_target:
        print(f"\n⚠️  Target file has {len(extra_in_target)} extra rows not in sources")
        for i, row in enumerate(list(extra_in_target)[:5]):  # Show first 5
            print(f"  {i+1}. {row[0] if row else 'empty row'}")
        if len(extra_in_target) > 5:
            print(f"  ... and {len(extra_in_target) - 5} more")
    
    # Check expected total
    expected_total = len(source_set)
    actual_total = len(target_set)
    print(f"\nExpected unique rows: {expected_total}")
    print(f"Actual unique rows in target: {actual_total}")
    
    if not missing_in_target and actual_total >= expected_total:
        print(f"✅ VERIFICATION PASSED: {name}")
        return True
    else:
        print(f"❌ VERIFICATION FAILED: {name}")
        return False

def main():
    base_dir = Path("fetalbiometrydata/MULTICENTRE")
    
    results = []
    
    # Abdomen_Test.csv
    results.append(verify_merge(
        base_dir / "Abdomen_Test.csv",
        [
            base_dir / "FP" / "Abdomen_Test.csv",
            base_dir / "UCL" / "Abdomen_Test.csv"
        ],
        "Abdomen_Test.csv"
    ))
    
    # Abdomen_Train.csv
    results.append(verify_merge(
        base_dir / "Abdomen_Train.csv",
        [
            base_dir / "FP" / "Abdomen_Train.csv",
            base_dir / "UCL" / "Abdomen_Train.csv"
        ],
        "Abdomen_Train.csv"
    ))
    
    # Femur_Test.csv
    results.append(verify_merge(
        base_dir / "Femur_Test.csv",
        [
            base_dir / "FP" / "Femur_Test.csv",
            base_dir / "UCL" / "Femur_Test.csv"
        ],
        "Femur_Test.csv"
    ))
    
    # Femur_Train.csv
    results.append(verify_merge(
        base_dir / "Femur_Train.csv",
        [
            base_dir / "FP" / "Femur_Train.csv",
            base_dir / "UCL" / "Femur_Train.csv"
        ],
        "Femur_Train.csv"
    ))
    
    # Head_Test.csv
    results.append(verify_merge(
        base_dir / "Head_Test.csv",
        [
            base_dir / "FP" / "BPD_Test.csv",
            base_dir / "HC18" / "BPD_Test.csv",
            base_dir / "UCL" / "Head_Test.csv"
        ],
        "Head_Test.csv"
    ))
    
    # Head_Train.csv
    results.append(verify_merge(
        base_dir / "Head_Train.csv",
        [
            base_dir / "FP" / "BPD_Train.csv",
            base_dir / "HC18" / "BPD_Train.csv",
            base_dir / "UCL" / "Head_Train.csv"
        ],
        "Head_Train.csv"
    ))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All verifications passed!")
        return 0
    else:
        print("❌ Some verifications failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

