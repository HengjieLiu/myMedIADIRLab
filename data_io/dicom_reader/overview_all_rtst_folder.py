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
- Shows structure presence with 'x' markers (default)
- Optional volume calculation in mm³ for each structure

Output Format:
- Summary counts of RT structure files
- Chronological study overview (for patient folders)
- Individual RT structure file analysis with structure lists
- Cross-reference table: structures (rows) vs files (columns)
- Presence indicators: 'x' for existing structures, blank for missing (default)
- Volume display: structure volumes in mm³ when --volumes flag is used

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
    --volumes         Calculate and display structure volumes in mm³ (default: show 'x' markers)

Requires:
    - pydicom
    - pathlib
    - collections
    - datetime
    - numpy (for volume calculations)
    - SimpleITK (for volume calculations)
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
import numpy as np

# Try to import SimpleITK for volume calculations
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print("Warning: SimpleITK not available. Volume calculations will be disabled.")

# Add the third_party dicomviewer to the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "dicomviewer" / "src"))

try:
    from dicom_viewer.readers.DICOMImageReader import DICOMImageReader
    from dicom_viewer.readers.RTStructReader import RTStructReader
    DICOMVIEWER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DICOM readers: {e}")
    print("Continuing with basic pydicom functionality only.")
    DICOMVIEWER_AVAILABLE = False


class RTStructureOverviewScanner:
    """
    A class for scanning RT Structure files and generating comprehensive overviews.
    """
    
    def __init__(self, folder_path: Path, calculate_volumes: bool = False, verbose: bool = False):
        self.folder_path = Path(folder_path)
        self.rtst_files = []
        self.studies = OrderedDict()  # Maintain chronological order
        self.all_structures = set()
        self.structure_file_map = defaultdict(set)  # structure_name -> set of file_ids
        self.calculate_volumes = calculate_volumes and SITK_AVAILABLE and DICOMVIEWER_AVAILABLE
        self.structure_volumes = defaultdict(dict)  # file_id -> {structure_name: volume}
        self.verbose = verbose
        
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
                
                # Calculate volumes if requested
                structure_volumes = {}
                if self.calculate_volumes:
                    structure_volumes = self._calculate_structure_volumes(ds, rtst_file)
                
                # Store file information
                file_info = {
                    "file_path": str(rtst_file),
                    "file_id": file_id,
                    "study_date": study_date,
                    "study_time": study_time,
                    "series_description": series_description,
                    "patient_id": patient_id,
                    "structures": structures,
                    "structure_count": len(structures),
                    "structure_volumes": structure_volumes
                }
                
                # Add to our collections
                self.all_structures.update(structures)
                for structure in structures:
                    self.structure_file_map[structure].add(file_id)
                
                # Store volume information
                if self.calculate_volumes:
                    self.structure_volumes[file_id] = structure_volumes
                
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
    
    def _calculate_structure_volumes(self, ds: pydicom.Dataset, rtst_file_path: Path) -> Dict[str, float]:
        """
        Calculate volumes for all structures in the RT Structure file using mask-based approach.
        
        Args:
            ds: pydicom Dataset
            rtst_file_path: Path to the RT Structure file
            
        Returns:
            Dictionary mapping structure names to volumes in mm³
        """
        volumes = {}
        
        try:
            # Use dicomviewer's RTStructReader for proper mask generation
            rtstruct_reader = RTStructReader(ds)
            rtstruct_reader.read()
            
            # Get all structure names
            structure_names = rtstruct_reader.get_structure_names()
            
            # Find the referenced image series
            image_reader = self._find_referenced_image_series(ds, rtst_file_path)
            
            if image_reader is None:
                if self.verbose:
                    print(f"Warning: Could not find referenced image series for {rtst_file_path}")
                return volumes
            
            # Calculate volume for each structure
            for structure_name in structure_names:
                try:
                    # Generate mask for the structure
                    mask = rtstruct_reader.get_structure_mask(structure_name, image_reader)
                    
                    # Calculate volume: count voxels * voxel volume
                    voxel_count = np.sum(mask > 0)
                    
                    # Get voxel spacing from the image
                    spacing = image_reader.image.GetSpacing()
                    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³
                    
                    volume = voxel_count * voxel_volume
                    volumes[structure_name] = volume
                    
                    if self.verbose:
                        print(f"  Structure '{structure_name}': {voxel_count} voxels, "
                              f"spacing {spacing}, volume: {volume:.2f} mm³")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error calculating volume for structure '{structure_name}': {e}")
                    volumes[structure_name] = 0.0
            
        except Exception as e:
            print(f"Error calculating volumes for {rtst_file_path}: {e}")
        
        return volumes
    
    def _find_referenced_image_series(self, ds: pydicom.Dataset, rtst_file_path: Path) -> Optional[DICOMImageReader]:
        """
        Find the referenced image series for the RT Structure file.
        
        Args:
            ds: pydicom Dataset
            rtst_file_path: Path to the RT Structure file
            
        Returns:
            DICOMImageReader instance or None if not found
        """
        try:
            # Get referenced frame of reference sequence
            referenced_frame_of_reference = getattr(ds, 'ReferencedFrameOfReferenceSequence', [])
            
            for frame_ref in referenced_frame_of_reference:
                rt_referenced_study = getattr(frame_ref, 'RTReferencedStudySequence', [])
                
                for study_ref in rt_referenced_study:
                    rt_referenced_series = getattr(study_ref, 'RTReferencedSeriesSequence', [])
                    
                    for series_ref in rt_referenced_series:
                        series_instance_uid = getattr(series_ref, 'SeriesInstanceUID', None)
                        
                        if series_instance_uid:
                            if self.verbose:
                                print(f"Looking for series UID: {series_instance_uid}")
                            
                            # Look for the referenced series in the study directory
                            study_dir = rtst_file_path.parent.parent  # Go up to study directory
                            
                            if study_dir.exists():
                                # Search all subdirectories in the study
                                for subdir in study_dir.iterdir():
                                    if subdir.is_dir():
                                        try:
                                            # Try MR modality first (since these are MR studies)
                                            image_reader = DICOMImageReader(
                                                str(subdir), 
                                                modality="MR",
                                                series_instance_uid=series_instance_uid
                                            )
                                            image_reader.read()
                                            if self.verbose:
                                                print(f"Found MR series in: {subdir}")
                                            return image_reader
                                        except Exception:
                                            # Try CT modality if MR fails
                                            try:
                                                image_reader = DICOMImageReader(
                                                    str(subdir), 
                                                    modality="CT",
                                                    series_instance_uid=series_instance_uid
                                                )
                                                image_reader.read()
                                                if self.verbose:
                                                    print(f"Found CT series in: {subdir}")
                                                return image_reader
                                            except Exception:
                                                continue
            
            if self.verbose:
                print("No referenced image series found")
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"Error finding referenced image series: {e}")
            return None
    
    
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
        if self.calculate_volumes:
            print("Structure volumes across RT Structure files (mm³)")
            print("Numbers = volume in mm³, blank = structure missing")
        else:
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
                    if self.calculate_volumes:
                        # Show volume if available
                        volume = self.structure_volumes.get(file_id, {}).get(structure, 0.0)
                        if volume > 0:
                            # Format volume with appropriate precision
                            if volume >= 1000:
                                volume_str = f"{volume:.0f}"
                            elif volume >= 1:
                                volume_str = f"{volume:.1f}"
                            else:
                                volume_str = f"{volume:.2f}"
                            row += f" {volume_str:<23}"
                        else:
                            row += f" {'0':<23}"
                    else:
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
    
    # Calculate and display structure volumes
    python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --volumes
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
    
    parser.add_argument(
        "--volumes",
        action="store_true",
        help="Calculate and display structure volumes in mm³ (default: show 'x' markers)"
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
        scanner = RTStructureOverviewScanner(args.folder_path, calculate_volumes=args.volumes, verbose=args.verbose)
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
