#!/usr/bin/env python3
# author: chiaradivece
# date: 2025-11-21
"""
Verify that MULTICENTRE image directories contain all images referenced in CSV files.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

def get_images_from_csv(csv_file):
    """Extract image names from a CSV file."""
    images = set()
    if not Path(csv_file).exists():
        return images
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row.get('image_name', '').strip()
            if image_name:
                images.add(image_name)
    return images

def get_images_in_directory(directory):
    """Get all image files in a directory."""
    images = set()
    if not Path(directory).exists():
        return images
    
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        for img_file in Path(directory).glob(f'*{ext}'):
            images.add(img_file.name)
    return images

def verify_images(anatomy, split, csv_file, image_dir, source_dirs, name):
    """Verify that all images referenced in CSV exist in the image directory."""
    print(f"\n{'='*60}")
    print(f"Verifying {name} ({anatomy}, {split})")
    print(f"{'='*60}")
    
    # Get images from CSV
    csv_images = get_images_from_csv(csv_file)
    print(f"Images referenced in CSV ({csv_file}): {len(csv_images)}")
    
    # Get images in MULTICENTRE directory
    mult_images = get_images_in_directory(image_dir)
    print(f"Images in MULTICENTRE directory ({image_dir}): {len(mult_images)}")
    
    # Check which CSV images are missing in MULTICENTRE
    missing_in_mult = csv_images - mult_images
    
    if missing_in_mult:
        print(f"\n❌ MISSING {len(missing_in_mult)} images in MULTICENTRE directory:")
        for i, img in enumerate(sorted(list(missing_in_mult))[:20]):  # Show first 20
            print(f"  {i+1}. {img}")
        if len(missing_in_mult) > 20:
            print(f"  ... and {len(missing_in_mult) - 20} more")
    else:
        print(f"\n✅ All CSV-referenced images are present in MULTICENTRE directory")
    
    # Check images in source directories
    all_source_images = set()
    for source_dir in source_dirs:
        if Path(source_dir).exists():
            source_images = get_images_in_directory(source_dir)
            all_source_images.update(source_images)
            print(f"  Source ({source_dir}): {len(source_images)} images")
    
    # Check if all source images are in MULTICENTRE
    missing_source = all_source_images - mult_images
    if missing_source:
        print(f"\n⚠️  {len(missing_source)} source images not in MULTICENTRE (may be expected if not in CSV)")
        for i, img in enumerate(sorted(list(missing_source))[:10]):  # Show first 10
            print(f"  {i+1}. {img}")
        if len(missing_source) > 10:
            print(f"  ... and {len(missing_source) - 10} more")
    
    # Summary
    if not missing_in_mult:
        print(f"\n✅ VERIFICATION PASSED: {name}")
        return True
    else:
        print(f"\n❌ VERIFICATION FAILED: {name}")
        return False

def main():
    base_dir = Path("fetalbiometrydata/MULTICENTRE")
    data_dir = Path("fetalbiometrydata/data/MULTICENTRE")
    
    results = []
    
    # Abdomen Test
    results.append(verify_images(
        "Abdomen", "Test",
        base_dir / "Abdomen_Test.csv",
        data_dir / "Abdomen",
        [
            Path("fetalbiometrydata/data/FP/Abdomen"),
            Path("fetalbiometrydata/data/UCL/Abdomen")
        ],
        "Abdomen_Test"
    ))
    
    # Abdomen Train
    results.append(verify_images(
        "Abdomen", "Train",
        base_dir / "Abdomen_Train.csv",
        data_dir / "Abdomen",
        [
            Path("fetalbiometrydata/data/FP/Abdomen"),
            Path("fetalbiometrydata/data/UCL/Abdomen")
        ],
        "Abdomen_Train"
    ))
    
    # Femur Test
    results.append(verify_images(
        "Femur", "Test",
        base_dir / "Femur_Test.csv",
        data_dir / "Femur",
        [
            Path("fetalbiometrydata/data/FP/FL"),  # FP uses FL for Femur
            Path("fetalbiometrydata/data/UCL/Femur")
        ],
        "Femur_Test"
    ))
    
    # Femur Train
    results.append(verify_images(
        "Femur", "Train",
        base_dir / "Femur_Train.csv",
        data_dir / "Femur",
        [
            Path("fetalbiometrydata/data/FP/FL"),  # FP uses FL for Femur
            Path("fetalbiometrydata/data/UCL/Femur")
        ],
        "Femur_Train"
    ))
    
    # Head Test
    results.append(verify_images(
        "Head", "Test",
        base_dir / "Head_Test.csv",
        data_dir / "Head",
        [
            Path("fetalbiometrydata/data/FP/BPD"),  # FP uses BPD for Head
            Path("fetalbiometrydata/data/HC18/Head"),
            Path("fetalbiometrydata/data/UCL/Head")
        ],
        "Head_Test"
    ))
    
    # Head Train
    results.append(verify_images(
        "Head", "Train",
        base_dir / "Head_Train.csv",
        data_dir / "Head",
        [
            Path("fetalbiometrydata/data/FP/BPD"),  # FP uses BPD for Head
            Path("fetalbiometrydata/data/HC18/Head"),
            Path("fetalbiometrydata/data/UCL/Head")
        ],
        "Head_Train"
    ))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All image verifications passed!")
        return 0
    else:
        print("❌ Some image verifications failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

