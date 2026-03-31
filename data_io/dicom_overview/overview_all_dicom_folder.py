#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DICOM Folder Overview Script

This script scans DICOM folders and provides a comprehensive overview of the structure,
organized by patient ID and study date. It identifies different modalities (MR, CT, RTDOSE, 
RTSTRUCT, RTPLAN) and extracts metadata including series descriptions, instance counts, 
and linking relationships.

Key Features:
- Organizes data by Patient ID with studies chronologically ordered
- Identifies primary image modalities (MR, CT, PT, SPECT, US, CR, DX)
- Groups RT series (RTSTRUCT, RTPLAN, RTDOSE) separately
- Shows summary counts for each study
- Displays 4-column format: Modality, Count, SeriesDescription, FolderName
- Shows full folder names by default (with optional truncation)
- Detects and reports unlinked RT series
- Exports results to JSON format

Output Format:
- Patient ID with all studies chronologically ordered
- Study Date/Time and Primary Modality
- Summary line showing counts of each series type
- Series table with Modality, Count, SeriesDescription, and FolderName columns
- Series ordered: Image modalities first, then RTSTRUCT, RTPLAN, RTDOSE
- All series sorted chronologically within each modality type

CLI Usage:
    python overview_all_dicom_folder.py <path_to_parent_folder> [options]

Examples:
    # Basic usage - scan entire dataset
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data
    
    # Scan specific patient folder
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126
    
    # Truncate long descriptions for compact output
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data --truncate
    
    # Save detailed results to JSON file
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data --output overview.json
    
    # Enable verbose output for debugging
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data --verbose

Arguments:
    parent_folder    Path to the parent folder containing DICOM data
    --output, -o     Output file path for JSON summary (optional)
    --verbose, -v    Enable verbose output for debugging
    --truncate, -t   Truncate long folder names (default: show full folder names)

Requires:
    - pydicom
    - SimpleITK
    - numpy
    - pathlib
    - third_party/dicom_viewer (DICOMImageReader, RTDoseReader, RTStructReader)
        Yasin's dicom_viewer (https://gitlab.com/YAAF/dicomviewer)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError

# Add the third_party dicomviewer to the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "dicomviewer" / "src"))

try:
    from dicom_viewer.readers.DICOMImageReader import DICOMImageReader
    from dicom_viewer.readers.RTDoseReader import RTDoseReader
    from dicom_viewer.readers.RTStructReader import RTStructReader
    print("dicom_viewer imported successfully (but not used in this script for now)")
except ImportError as e:
    print(f"Warning: Could not import DICOM readers: {e}")
    print("Continuing with basic pydicom functionality only.")


class DICOMOverviewScanner:
    """
    A class for scanning DICOM folders and generating comprehensive overviews.
    """
    
    def __init__(self, parent_path: Path):
        self.parent_path = Path(parent_path)
        self.patients = {}
        self.unlinked_series = []
        
    def scan_all_folders(self) -> Dict[str, Any]:
        """
        Scan all DICOM folders and organize by patient ID and study date.
        
        Returns:
            Dict containing organized patient and study information
        """
        print(f"Scanning DICOM folders in: {self.parent_path}")
        
        # First, scan all folders to identify DICOM series
        all_series = self._scan_for_dicom_series()
        
        # Organize by patient ID and study date
        self._organize_by_patient_and_study(all_series)
        
        # Detect linking relationships
        self._detect_linking_relationships()
        
        return {
            "patients": self.patients,
            "unlinked_series": self.unlinked_series,
            "scan_timestamp": datetime.now().isoformat(),
            "total_patients": len(self.patients),
            "total_studies": sum(len(patient["studies"]) for patient in self.patients.values())
        }
    
    def _scan_for_dicom_series(self) -> List[Dict[str, Any]]:
        """
        Recursively scan directories for DICOM series.
        
        Returns:
            List of series information dictionaries
        """
        series_list = []
        
        for root, dirs, files in os.walk(self.parent_path):
            if not files:
                continue
                
            folder_path = Path(root)
            folder_name = folder_path.name
            
            # Check if this folder contains DICOM files
            dicom_files = []
            for file in files:
                file_path = folder_path / file
                if self._is_dicom_file(file_path):
                    dicom_files.append(file_path)
            
            if dicom_files:
                # Extract metadata from the first DICOM file
                try:
                    ds = dcmread(dicom_files[0], stop_before_pixels=True)
                    series_info = self._extract_series_info(ds, folder_path, dicom_files)
                    series_list.append(series_info)
                except Exception as e:
                    print(f"Error reading DICOM file {dicom_files[0]}: {e}")
        
        return series_list
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """
        Check if a file is a DICOM file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is a DICOM file, False otherwise
        """
        try:
            # Try to read the file as DICOM
            dcmread(str(file_path), stop_before_pixels=True)
            return True
        except (InvalidDicomError, Exception):
            return False
    
    def _extract_series_info(self, ds: pydicom.Dataset, folder_path: Path, dicom_files: List[Path]) -> Dict[str, Any]:
        """
        Extract series information from a DICOM dataset.
        
        Args:
            ds: pydicom Dataset
            folder_path: Path to the folder containing the series
            dicom_files: List of DICOM files in the folder
            
        Returns:
            Dictionary containing series information
        """
        # Extract basic information
        modality = getattr(ds, 'Modality', 'UNKNOWN')
        series_instance_uid = getattr(ds, 'SeriesInstanceUID', '')
        study_instance_uid = getattr(ds, 'StudyInstanceUID', '')
        patient_id = getattr(ds, 'PatientID', 'UNKNOWN')
        study_date = getattr(ds, 'StudyDate', '')
        study_time = getattr(ds, 'StudyTime', '')
        series_description = getattr(ds, 'SeriesDescription', '')
        
        # Extract instance count
        instance_count = len(dicom_files)
        
        # Extract additional metadata based on modality
        additional_info = self._extract_modality_specific_info(ds, modality)
        
        return {
            "folder_path": str(folder_path),
            "folder_name": folder_path.name,
            "modality": modality,
            "series_instance_uid": series_instance_uid,
            "study_instance_uid": study_instance_uid,
            "patient_id": patient_id,
            "study_date": study_date,
            "study_time": study_time,
            "series_description": series_description,
            "instance_count": instance_count,
            "dicom_files": [str(f) for f in dicom_files],
            **additional_info
        }
    
    def _extract_modality_specific_info(self, ds: pydicom.Dataset, modality: str) -> Dict[str, Any]:
        """
        Extract modality-specific information from DICOM dataset.
        
        Args:
            ds: pydicom Dataset
            modality: DICOM modality
            
        Returns:
            Dictionary containing modality-specific information
        """
        info = {}
        
        if modality == "MR":
            # Extract MR-specific information
            info["sequence_name"] = getattr(ds, 'SequenceName', '')
            info["scanning_sequence"] = getattr(ds, 'ScanningSequence', '')
            info["sequence_variant"] = getattr(ds, 'SequenceVariant', '')
            info["scan_options"] = getattr(ds, 'ScanOptions', '')
            
        elif modality == "RTDOSE":
            # Extract RTDOSE-specific information
            info["dose_units"] = getattr(ds, 'DoseUnits', '')
            info["dose_type"] = getattr(ds, 'DoseType', '')
            info["dose_grid_scaling"] = getattr(ds, 'DoseGridScaling', None)
            
        elif modality == "RTSTRUCT":
            # Extract RTSTRUCT-specific information
            info["structure_set_name"] = getattr(ds, 'StructureSetName', '')
            info["roi_count"] = len(getattr(ds, 'StructureSetROISequence', []))
            
        elif modality == "RTPLAN":
            # Extract RTPLAN-specific information
            info["plan_name"] = getattr(ds, 'RTPlanName', '')
            info["plan_label"] = getattr(ds, 'RTPlanLabel', '')
        
        return info
    
    def _organize_by_patient_and_study(self, series_list: List[Dict[str, Any]]):
        """
        Organize series by patient ID and study date.
        
        Args:
            series_list: List of series information dictionaries
        """
        for series in series_list:
            patient_id = series["patient_id"]
            study_date = series["study_date"]
            study_time = series["study_time"]
            
            # Create study key combining date and time
            study_key = f"{study_date}_{study_time}" if study_time else study_date
            
            if patient_id not in self.patients:
                self.patients[patient_id] = {
                    "patient_id": patient_id,
                    "studies": {},
                    "total_studies": 0
                }
            
            if study_key not in self.patients[patient_id]["studies"]:
                self.patients[patient_id]["studies"][study_key] = {
                    "study_date": study_date,
                    "study_time": study_time,
                    "study_instance_uid": series["study_instance_uid"],
                    "series": [],
                    "primary_image_modality": None,
                    "rt_series": []
                }
            
            self.patients[patient_id]["studies"][study_key]["series"].append(series)
        
        # Sort studies by date and identify primary image modality
        for patient_id, patient_data in self.patients.items():
            # Sort studies by date
            sorted_studies = sorted(patient_data["studies"].items(), 
                                 key=lambda x: x[1]["study_date"])
            patient_data["studies"] = dict(sorted_studies)
            patient_data["total_studies"] = len(sorted_studies)
            
            # Identify primary image modality for each study
            for study_key, study_data in patient_data["studies"].items():
                self._identify_primary_modality(study_data)
    
    def _identify_primary_modality(self, study_data: Dict[str, Any]):
        """
        Identify the primary image modality for a study.
        
        Args:
            study_data: Study data dictionary
        """
        # Look for MR, CT, or other image modalities
        image_modalities = ["MR", "CT", "PT", "US", "CR", "DX"]
        
        for series in study_data["series"]:
            if series["modality"] in image_modalities:
                if study_data["primary_image_modality"] is None:
                    study_data["primary_image_modality"] = series["modality"]
                break
        
        # Separate RT series
        rt_modalities = ["RTDOSE", "RTSTRUCT", "RTPLAN"]
        study_data["rt_series"] = [s for s in study_data["series"] 
                                 if s["modality"] in rt_modalities]
    
    def _detect_linking_relationships(self):
        """
        Detect linking relationships between RT series and primary images.
        """
        for patient_id, patient_data in self.patients.items():
            for study_key, study_data in patient_data["studies"].items():
                primary_modality = study_data["primary_image_modality"]
                rt_series = study_data["rt_series"]
                
                if primary_modality and rt_series:
                    # Check if RT series are linked to the primary image
                    for rt_series_info in rt_series:
                        if not self._is_rt_series_linked(rt_series_info, study_data["series"]):
                            self.unlinked_series.append({
                                "patient_id": patient_id,
                                "study_date": study_data["study_date"],
                                "rt_series": rt_series_info,
                                "reason": "No clear linking relationship found"
                            })
    
    def _is_rt_series_linked(self, rt_series: Dict[str, Any], all_series: List[Dict[str, Any]]) -> bool:
        """
        Check if an RT series is linked to other series in the study.
        
        Args:
            rt_series: RT series information
            all_series: All series in the study
            
        Returns:
            True if linked, False otherwise
        """
        # This is a simplified check - in practice, you'd check ReferencedStudySequence,
        # ReferencedSeriesSequence, etc.
        return len(all_series) > 1  # Simple heuristic: linked if there are other series
    
    def _get_patient_folder_paths(self, patients_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Get the folder paths for each patient by finding the common parent directory.
        
        Args:
            patients_data: Dictionary containing patient data
            
        Returns:
            Dictionary mapping patient IDs to their folder paths
        """
        patient_paths = {}
        
        for patient_id, patient_data in patients_data.items():
            # Get all folder paths for this patient
            all_paths = []
            for study_data in patient_data["studies"].values():
                for series in study_data["series"]:
                    all_paths.append(Path(series["folder_path"]))
            
            if all_paths:
                # Find the common parent directory that contains all series for this patient
                # This should be the patient folder
                common_path = all_paths[0].parent
                for path in all_paths[1:]:
                    # Find the common parent by going up the directory tree
                    while not str(path).startswith(str(common_path)):
                        common_path = common_path.parent
                        if common_path == common_path.parent:  # Reached root
                            break
                
                patient_paths[patient_id] = str(common_path)
        
        return patient_paths
    
    def _get_study_folder_paths(self, studies_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Get the folder paths for each study by finding the common parent directory.
        
        Args:
            studies_data: Dictionary containing study data
            
        Returns:
            Dictionary mapping study keys to their folder paths
        """
        study_paths = {}
        
        for study_key, study_data in studies_data.items():
            # Get all folder paths for this study
            all_paths = []
            for series in study_data["series"]:
                all_paths.append(Path(series["folder_path"]))
            
            if all_paths:
                # Find the common parent directory that contains all series for this study
                # This should be the study folder
                common_path = all_paths[0].parent
                for path in all_paths[1:]:
                    # Find the common parent by going up the directory tree
                    while not str(path).startswith(str(common_path)):
                        common_path = common_path.parent
                        if common_path == common_path.parent:  # Reached root
                            break
                
                study_paths[study_key] = str(common_path)
        
        return study_paths
    
    def print_overview(self, overview_data: Dict[str, Any], truncate_folder_names: bool = False):
        """
        Print the overview in a formatted table with 4 columns: Modality, Count, SeriesDescription, FolderName.
        
        Args:
            overview_data: Overview data dictionary
            truncate_folder_names: Whether to truncate long folder names
        """
        print("\n" + "="*80)
        print("DICOM FOLDER OVERVIEW")
        print("="*80)
        print(f"Scan Date: {overview_data['scan_timestamp']}")
        print(f"Total Patients: {overview_data['total_patients']}")
        print(f"Total Studies: {overview_data['total_studies']}")
        
        # Print patient summary table
        print("\nPatient Summary:")
        print(f"{'Patient ID':<15} {'Studies':<8}")
        print("-" * 25)
        for patient_id, patient_data in overview_data["patients"].items():
            study_count = patient_data["total_studies"]
            print(f"{patient_id:<15} {study_count:<8}")
        
        print("="*80)
        
        # Get patient folder paths for display
        patient_folder_paths = self._get_patient_folder_paths(overview_data["patients"])
        
        for idx, (patient_id, patient_data) in enumerate(overview_data["patients"].items(), 1):
            total_patients = overview_data["total_patients"]
            patient_path = patient_folder_paths.get(patient_id, "Unknown path")
            print(f"\nPatient idx {idx}/{total_patients} and path to the patient folder: {patient_path}")
            print(f"\nPatient ID: {patient_id}")
            print("-" * 60)
            
            # Get study folder paths for this patient
            study_folder_paths = self._get_study_folder_paths(patient_data["studies"])
            
            for study_idx, (study_key, study_data) in enumerate(patient_data["studies"].items(), 1):
                total_studies_for_patient = patient_data["total_studies"]
                study_path = study_folder_paths.get(study_key, "Unknown path")
                print(f"\nStudy {study_idx}/{total_studies_for_patient} and path to the study folder: {study_path}")
                print(f"\nStudy Date: {study_data['study_date']} {study_data['study_time']}")
                print(f"Primary Modality: {study_data['primary_image_modality'] or 'None'}")
                
                # Count series by type
                image_series = [s for s in study_data["series"] if s["modality"] in ["MR", "CT", "PT", "SPECT", "US", "CR", "DX"]]
                rtstruct_series = [s for s in study_data["series"] if s["modality"] == "RTSTRUCT"]
                rtplan_series = [s for s in study_data["series"] if s["modality"] == "RTPLAN"]
                rtdose_series = [s for s in study_data["series"] if s["modality"] == "RTDOSE"]
                
                # Print summary line
                summary_parts = []
                if image_series:
                    summary_parts.append(f"{len(image_series)} image folder{'s' if len(image_series) > 1 else ''}")
                if rtstruct_series:
                    summary_parts.append(f"{len(rtstruct_series)} RTSTRUCT")
                if rtplan_series:
                    summary_parts.append(f"{len(rtplan_series)} RTPLAN")
                if rtdose_series:
                    summary_parts.append(f"{len(rtdose_series)} RTDOSE")
                
                print(f"Summary: {', '.join(summary_parts)}")
                
                # Print series in sorted order
                print(f"{'Modality':<10} {'Count':<6} {'SeriesDescription':<25} {'FolderName':<40}")
                print("-" * 85)
                
                # Sort and print image series first
                for series in sorted(image_series, key=lambda x: x["study_date"] + x["study_time"]):
                    modality = series["modality"]
                    count = series["instance_count"]
                    series_description = series["series_description"] or ""
                    folder_name = series["folder_name"]
                    
                    # Truncate folder name if requested and too long
                    if truncate_folder_names and len(folder_name) > 37:
                        folder_name = folder_name[:34] + "..."
                    
                    print(f"{modality:<10} {count:<6} {series_description:<25} {folder_name}")
                
                # Sort and print RTSTRUCT series
                for series in sorted(rtstruct_series, key=lambda x: x["study_date"] + x["study_time"]):
                    modality = series["modality"]
                    count = series["instance_count"]
                    series_description = series["series_description"] or ""
                    folder_name = series["folder_name"]
                    
                    # Truncate folder name if requested and too long
                    if truncate_folder_names and len(folder_name) > 37:
                        folder_name = folder_name[:34] + "..."
                    
                    print(f"{modality:<10} {count:<6} {series_description:<25} {folder_name}")
                
                # Sort and print RTPLAN series
                for series in sorted(rtplan_series, key=lambda x: x["study_date"] + x["study_time"]):
                    modality = series["modality"]
                    count = series["instance_count"]
                    series_description = series["series_description"] or ""
                    folder_name = series["folder_name"]
                    
                    # Truncate folder name if requested and too long
                    if truncate_folder_names and len(folder_name) > 37:
                        folder_name = folder_name[:34] + "..."
                    
                    print(f"{modality:<10} {count:<6} {series_description:<25} {folder_name}")
                
                # Sort and print RTDOSE series
                for series in sorted(rtdose_series, key=lambda x: x["study_date"] + x["study_time"]):
                    modality = series["modality"]
                    count = series["instance_count"]
                    series_description = series["series_description"] or ""
                    folder_name = series["folder_name"]
                    
                    # Truncate folder name if requested and too long
                    if truncate_folder_names and len(folder_name) > 37:
                        folder_name = folder_name[:34] + "..."
                    
                    print(f"{modality:<10} {count:<6} {series_description:<25} {folder_name}")
        
        # Print unlinked series
        if overview_data["unlinked_series"]:
            print("\n" + "="*80)
            print("UNLINKED RT SERIES")
            print("="*80)
            for unlinked in overview_data["unlinked_series"]:
                print(f"Patient: {unlinked['patient_id']}, "
                      f"Study: {unlinked['study_date']}, "
                      f"RT Series: {unlinked['rt_series']['modality']} - "
                      f"{unlinked['rt_series']['series_description']}")
                print(f"Reason: {unlinked['reason']}")


def main():
    """
    Main function to run the DICOM overview scanner.
    """
    parser = argparse.ArgumentParser(
        description="Scan DICOM folders and provide comprehensive overview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data --truncate
    python overview_all_dicom_folder.py /data/hengjie/brainmets/dicom/Data --output summary.json
        """
    )
    
    parser.add_argument(
        "parent_folder",
        type=Path,
        help="Path to the parent folder containing DICOM data"
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
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--truncate",
        "-t",
        action="store_true",
        help="Truncate long folder names (default: show full folder names)"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not args.parent_folder.exists():
        print(f"Error: Path {args.parent_folder} does not exist")
        sys.exit(1)
    
    if not args.parent_folder.is_dir():
        print(f"Error: Path {args.parent_folder} is not a directory")
        sys.exit(1)
    
    try:
        # Create scanner and scan folders
        scanner = DICOMOverviewScanner(args.parent_folder)
        overview_data = scanner.scan_all_folders()
        
        # Print overview
        scanner.print_overview(overview_data, truncate_folder_names=args.truncate)
        
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
