#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RT Structure Folder Overview Script

This script scans DICOM folders specifically for RT Structure (RTSTRUCT) files and provides
a comprehensive overview of all structures contained within them. It can analyze either
a single study folder or a patient folder containing multiple studies.

Key Features:
- Counts total RT structure files in the folder
- For patient folders, shows RT structure count per study in chronological order
- Extracts all structure names from RT structure files
- Creates a comprehensive table showing which structures exist in which files
- Uses date-SeriesDescription format for file identification
- Shows structure presence with 'x' markers

Output Format:
- Summary counts of RT structure files
- Chronological study overview (for patient folders)
- Individual RT structure file analysis with structure lists
- Cross-reference table: structures (rows) vs files (columns)
- Presence indicators: 'x' for existing structures, blank for missing

CLI Usage:
    python overview_all_rtst_folder.py <path_to_folder> [options]

Examples:
    # Analyze a single study folder
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126/2009-08__Studies
    
    # Analyze a patient folder with multiple studies
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126
    
    # Save results to JSON file
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --output rtst_overview.json
    
    # Enable verbose output for debugging
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --verbose

Arguments:
    folder_path       Path to the study folder or patient folder containing RT structure files
    --output, -o      Output file path for JSON summary (optional)
    --verbose, -v     Enable verbose output for debugging
    --max-width        Maximum width for structure names in table (default: 30)

Requires:
    - pydicom
    - pathlib
    - collections
    - datetime
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, OrderedDict
from datetime import datetime
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError


class RTStructureOverviewScanner:
    """
    A class for scanning RT Structure files and generating comprehensive overviews.
    """
    
    def __init__(self, folder_path: Path):
        self.folder_path = Path(folder_path)
        self.rtst_files = []
        self.studies = OrderedDict()  # Maintain chronological order
        self.all_structures = set()
        self.structure_file_map = defaultdict(set)  # structure_name -> set of file_ids
        
    def scan_rtst_files(self) -> Dict[str, Any]:
        """
        Scan for RT Structure files and extract structure information.
        
        Returns:
            Dict containing organized RT structure information
        """
        print(f"Scanning RT Structure files in: {self.folder_path}")
        
        # First, find all RT structure files
        self._find_rtst_files()
        
        # Extract structure information from each file
        self._extract_structure_information()
        
        # Organize by studies if this is a patient folder
        self._organize_by_studies()
        
        return {
            "folder_path": str(self.folder_path),
            "total_rtst_files": len(self.rtst_files),
            "total_structures": len(self.all_structures),
            "studies": dict(self.studies),
            "all_structures": sorted(list(self.all_structures)),
            "structure_file_map": dict(self.structure_file_map),
            "scan_timestamp": datetime.now().isoformat()
        }
    
    def _find_rtst_files(self):
        """
        Recursively find all RT Structure files in the folder.
        """
        for root, dirs, files in os.walk(self.folder_path):
            folder_path = Path(root)
            
            for file in files:
                file_path = folder_path / file
                if self._is_rtst_file(file_path):
                    self.rtst_files.append(file_path)
    
    def _is_rtst_file(self, file_path: Path) -> bool:
        """
        Check if a file is an RT Structure file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is an RT Structure file, False otherwise
        """
        try:
            # Try to read the file as DICOM
            ds = dcmread(str(file_path), stop_before_pixels=True)
            modality = getattr(ds, 'Modality', '')
            return modality == 'RTSTRUCT'
        except (InvalidDicomError, Exception):
            return False
    
    def _extract_structure_information(self):
        """
        Extract structure information from all RT Structure files.
        """
        for rtst_file in self.rtst_files:
            try:
                ds = dcmread(str(rtst_file), stop_before_pixels=True)
                
                # Extract file metadata
                study_date = getattr(ds, 'StudyDate', '')
                study_time = getattr(ds, 'StudyTime', '')
                series_description = getattr(ds, 'SeriesDescription', '')
                patient_id = getattr(ds, 'PatientID', 'UNKNOWN')
                
                # Create file identifier
                file_id = f"{study_date}-{series_description}" if series_description else f"{study_date}-RTSTRUCT"
                
                # Extract structure information
                structures = self._extract_structures_from_dataset(ds)
                
                # Store file information
                file_info = {
                    "file_path": str(rtst_file),
                    "file_id": file_id,
                    "study_date": study_date,
                    "study_time": study_time,
                    "series_description": series_description,
                    "patient_id": patient_id,
                    "structures": structures,
                    "structure_count": len(structures)
                }
                
                # Add to our collections
                self.all_structures.update(structures)
                for structure in structures:
                    self.structure_file_map[structure].add(file_id)
                
                # Store file info for later organization
                if not hasattr(self, '_file_info_list'):
                    self._file_info_list = []
                self._file_info_list.append(file_info)
                
            except Exception as e:
                print(f"Error reading RT Structure file {rtst_file}: {e}")
    
    def _extract_structures_from_dataset(self, ds: pydicom.Dataset) -> List[str]:
        """
        Extract structure names from RT Structure DICOM dataset.
        
        Args:
            ds: pydicom Dataset
            
        Returns:
            List of structure names
        """
        structures = []
        
        try:
            # Get the Structure Set ROI Sequence
            roi_sequence = getattr(ds, 'StructureSetROISequence', [])
            
            for roi in roi_sequence:
                roi_name = getattr(roi, 'ROIName', '')
                if roi_name:
                    structures.append(roi_name)
            
        except Exception as e:
            print(f"Error extracting structures: {e}")
        
        return structures
    
    def _organize_by_studies(self):
        """
        Organize RT structure files by studies (chronologically ordered).
        """
        if not hasattr(self, '_file_info_list'):
            return
        
        # Group files by study date
        study_groups = defaultdict(list)
        for file_info in self._file_info_list:
            study_key = file_info['study_date']
            study_groups[study_key].append(file_info)
        
        # Sort studies chronologically and organize
        for study_date in sorted(study_groups.keys()):
            files_in_study = study_groups[study_date]
            
            # Sort files within study by time
            files_in_study.sort(key=lambda x: x['study_time'])
            
            self.studies[study_date] = {
                "study_date": study_date,
                "rtst_files": files_in_study,
                "total_rtst_files": len(files_in_study),
                "total_structures": len(set().union(*[f['structures'] for f in files_in_study]))
            }
    
    def print_overview(self, overview_data: Dict[str, Any], max_width: int = 30):
        """
        Print the RT Structure overview in a formatted table.
        
        Args:
            overview_data: Overview data dictionary
            max_width: Maximum width for structure names in table
        """
        print("\n" + "="*80)
        print("RT STRUCTURE FOLDER OVERVIEW")
        print("="*80)
        print(f"Scan Date: {overview_data['scan_timestamp']}")
        print(f"Folder: {overview_data['folder_path']}")
        print(f"Total RT Structure Files: {overview_data['total_rtst_files']}")
        print(f"Total Unique Structures: {overview_data['total_structures']}")
        print("="*80)
        
        # Print study-by-study summary if multiple studies
        if len(self.studies) > 1:
            print("\nSTUDY SUMMARY (Chronological Order):")
            print("-" * 60)
            print(f"{'Study Date':<12} {'RTST Files':<12} {'Structures':<12}")
            print("-" * 60)
            
            for study_date, study_data in self.studies.items():
                print(f"{study_date:<12} {study_data['total_rtst_files']:<12} {study_data['total_structures']:<12}")
        
        # Print detailed file information
        print(f"\nDETAILED RT STRUCTURE FILES:")
        print("-" * 80)
        
        for study_date, study_data in self.studies.items():
            print(f"\nStudy Date: {study_date}")
            print("-" * 40)
            
            for file_info in study_data['rtst_files']:
                print(f"\nFile: {file_info['file_id']}")
                print(f"  Structures ({file_info['structure_count']}): {', '.join(file_info['structures'])}")
        
        # Print cross-reference table
        self._print_cross_reference_table(overview_data, max_width)
    
    def _print_cross_reference_table(self, overview_data: Dict[str, Any], max_width: int):
        """
        Print the cross-reference table showing structure presence across files.
        
        Args:
            overview_data: Overview data dictionary
            max_width: Maximum width for structure names
        """
        print(f"\nCROSS-REFERENCE TABLE:")
        print("="*80)
        print("Structure presence across RT Structure files")
        print("'x' = structure exists, blank = structure missing")
        print("="*80)
        
        # Get all file IDs sorted
        all_file_ids = []
        for study_data in self.studies.values():
            for file_info in study_data['rtst_files']:
                all_file_ids.append(file_info['file_id'])
        
        all_file_ids.sort()
        
        # Get all structures sorted
        all_structures = sorted(list(self.all_structures))
        
        # Print header
        header = f"{'Structure':<{max_width}}"
        for file_id in all_file_ids:
            # Truncate file_id if too long
            display_id = file_id[:20] + "..." if len(file_id) > 23 else file_id
            header += f" {display_id:<23}"
        print(header)
        print("-" * len(header))
        
        # Print each structure row
        for structure in all_structures:
            # Truncate structure name if too long
            display_structure = structure[:max_width-3] + "..." if len(structure) > max_width else structure
            row = f"{display_structure:<{max_width}}"
            
            for file_id in all_file_ids:
                if file_id in self.structure_file_map[structure]:
                    row += f" {'x':<23}"
                else:
                    row += f" {'':<23}"
            
            print(row)
        
        print("-" * len(header))
        print(f"Total structures: {len(all_structures)}")
        print(f"Total files: {len(all_file_ids)}")


def main():
    """
    Main function to run the RT Structure overview scanner.
    """
    parser = argparse.ArgumentParser(
        description="Scan RT Structure files and provide comprehensive overview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single study folder
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126/2009-08__Studies
    
    # Analyze a patient folder with multiple studies
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126
    
    # Save results to JSON file
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --output rtst_overview.json
    
    # Enable verbose output for debugging
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --verbose
    
    # Set maximum width for structure names in table
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --max-width 40
        """
    )
    
    parser.add_argument(
        "folder_path",
        type=Path,
        help="Path to the study folder or patient folder containing RT structure files"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path for JSON summary (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output for debugging"
    )
    
    parser.add_argument(
        "--max-width",
        type=int,
        default=30,
        help="Maximum width for structure names in table (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not args.folder_path.exists():
        print(f"Error: Path {args.folder_path} does not exist")
        sys.exit(1)
    
    if not args.folder_path.is_dir():
        print(f"Error: Path {args.folder_path} is not a directory")
        sys.exit(1)
    
    try:
        # Create scanner and scan RT structure files
        scanner = RTStructureOverviewScanner(args.folder_path)
        overview_data = scanner.scan_rtst_files()
        
        # Print overview
        scanner.print_overview(overview_data, max_width=args.max_width)
        
        # Save to JSON if requested
        if args.output:
            import json
            args.output.parent.mkdir(parents=True, exist_ok=True)
            
            # Custom JSON encoder to handle pydicom MultiValue objects
            class DICOMJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                        return list(obj)
                    return str(obj)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(overview_data, f, indent=2, sort_keys=False, cls=DICOMJSONEncoder)
            print(f"\nOverview saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during scanning: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
