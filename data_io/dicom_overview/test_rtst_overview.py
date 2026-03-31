#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for overview_all_rtst_folder.py

This script demonstrates how to use the RT Structure overview scanner
and provides examples of expected output.
"""

import sys
from pathlib import Path

# Add the current directory to the path to import our module
sys.path.insert(0, str(Path(__file__).parent))

from overview_all_rtst_folder import RTStructureOverviewScanner


def test_rtst_scanner():
    """
    Test the RT Structure scanner with example usage.
    """
    print("RT Structure Overview Scanner Test")
    print("=" * 50)
    
    # Example paths (these would be your actual paths)
    example_paths = [
        "/data/hengjie/brainmets/dicom/Data/SRS3126/2009-08__Studies",  # Single study
        "/data/hengjie/brainmets/dicom/Data/SRS3126",  # Patient folder
    ]
    
    for path in example_paths:
        folder_path = Path(path)
        
        if folder_path.exists():
            print(f"\nTesting with path: {path}")
            print("-" * 40)
            
            try:
                # Create scanner
                scanner = RTStructureOverviewScanner(folder_path)
                
                # Scan for RT structure files
                overview_data = scanner.scan_rtst_files()
                
                # Print overview
                scanner.print_overview(overview_data, max_width=25)
                
            except Exception as e:
                print(f"Error testing path {path}: {e}")
        else:
            print(f"\nPath does not exist: {path}")
            print("This is expected for the test - replace with your actual paths")


if __name__ == "__main__":
    test_rtst_scanner()




