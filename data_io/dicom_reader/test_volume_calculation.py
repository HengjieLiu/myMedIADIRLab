#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the improved volume calculation in overview_all_rtst_folder.py

This script tests the new mask-based volume calculation approach.
"""

import sys
from pathlib import Path

# Add the current directory to the path to import our module
sys.path.insert(0, str(Path(__file__).parent))

from overview_all_rtst_folder import RTStructureOverviewScanner


def test_volume_calculation():
    """
    Test the improved volume calculation with verbose output.
    """
    print("Testing Improved Volume Calculation")
    print("=" * 50)
    
    # Example path (replace with your actual path)
    test_path = "/data/hengjie/brainmets/dicom/Data/SRS3126"
    folder_path = Path(test_path)
    
    if folder_path.exists():
        print(f"Testing with path: {test_path}")
        print("-" * 40)
        
        try:
            # Create scanner with volume calculation and verbose output
            scanner = RTStructureOverviewScanner(folder_path, calculate_volumes=True, verbose=True)
            
            # Scan for RT structure files
            overview_data = scanner.scan_rtst_files()
            
            # Print overview
            scanner.print_overview(overview_data, max_width=25)
            
            print("\n" + "="*50)
            print("Volume calculation test completed!")
            print("Check the output above for volume values and debug information.")
            
        except Exception as e:
            print(f"Error testing volume calculation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Path does not exist: {test_path}")
        print("Please update the test_path variable with your actual DICOM data path.")


if __name__ == "__main__":
    test_volume_calculation()




