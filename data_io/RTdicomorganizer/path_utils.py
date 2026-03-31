"""
==============================================================================
MODULE NAME: path_utils.py
==============================================================================

PURPOSE:
    This module provides utilities for constructing file system paths and 
    discovering DICOM folders within the brain metastasis study structure.
    
    Key responsibilities:
    - Convert date formats for folder naming conventions
    - Build standardized study folder paths from patient IDs and dates
    - Find RTStruct folders with specific naming patterns
    - Discover all MR imaging folders within a study
    - Find DICOM paths from inventory Excel files for RT studies

DEPENDENCIES:
    External packages:
    - pathlib: For cross-platform path manipulation
    - datetime: For date parsing and formatting
    - pydicom: For reading DICOM file metadata
    - pandas: For reading Excel/CSV inventory files
    
    Internal modules:
    - data_parser: For parsing lesion labels and converting target formats

FUNCTIONS:
    1. convert_iso_date_to_folder_format(iso_date: str) -> str
       Convert ISO date (YYYY-MM-DD) to folder format (YYYY-MM)
       
    2. build_study_folder_path(base_folder: str, patient_id: str, iso_date: str) -> Path
       Build full path to a study folder from components
       
    3. find_rtstruct_folder(study_folder: Path, folder_pattern: str) -> Path
       Find RTStruct folder matching a specific naming pattern
       
    4. find_all_mr_folders(study_folder: Path) -> list
       Find all MR image folders in a study folder
       
    5. find_all_rtstruct_folders(study_folder: Path) -> list
       Find all RTStruct folders in a study folder
       
    6. find_all_rtdose_folders(study_folder: Path) -> list
       Find all RTDOSE folders in a study folder
       
    7. count_dicom_files(folder_path: Path) -> int
       Count the number of DICOM files in a folder
       
    8. find_dicom_paths_for_rtstudy(lesion_label: str, study_type: str,
       inventory_excel_path: str, base_folder: str, debug: bool) -> dict
       Find DICOM file paths from inventory Excel file for RT study

USAGE EXAMPLE:
    ```python
    from code.RTdicomorganizer import path_utils
    
    # Convert date format
    folder_date = path_utils.convert_iso_date_to_folder_format("1997-07-18")
    # Returns: "1997-07"
    
    # Build study path
    study_path = path_utils.build_study_folder_path(
        base_folder="/database/dicom",
        patient_id="0871",
        iso_date="1997-07-18"
    )
    # Returns: Path("/database/dicom/SRS0871/1997-07__Studies")
    
    # Find RTStruct folder
    rtstruct_folder = path_utils.find_rtstruct_folder(
        study_path,
        "Brain.MS.Init.Model"
    )
    
    # Find all MR folders
    mr_folders = path_utils.find_all_mr_folders(study_path)
    
    # Find DICOM paths from inventory
    result = path_utils.find_dicom_paths_for_rtstudy(
        lesion_label='0871.01',
        study_type='initial_moving',
        inventory_excel_path='./excel/output/inventory.xlsx',
        base_folder='/database/brainmets/dicom/organized'
    )
    # Returns: dict with 'mr_path', 'rtstruct_path', 'rtdose_path', 'structure_name'
    ```

NOTES:
    - Patient IDs are always zero-padded to 4 digits (e.g., '0871', not '871')
    - Study folders follow naming convention: 'YYYY-MM__Studies'
    - RTStruct folder patterns are case-sensitive
    - MR folder detection requires reading DICOM metadata

FOLDER STRUCTURE ASSUMPTIONS:
    Base folder structure:
    /database/dicom/
    └── SRSPPPP/                    # Patient folder (PPPP = 4-digit ID)
        └── YYYY-MM__Studies/       # Study folder (date-based)
            ├── SRSPPPP_..._MR_.../ # MR series folders
            └── SRSPPPP_..._RTst_.../ # RTStruct folders

==============================================================================
"""

from pathlib import Path
from datetime import datetime
import pydicom
import pandas as pd

# Import from data_parser for lesion label parsing and target format conversion
try:
    from . import data_parser
except ImportError:
    # Handle case where module is imported directly
    import data_parser


def convert_iso_date_to_folder_format(iso_date: str) -> str:
    """
    Convert ISO date (YYYY-MM-DD) to folder format (YYYY-MM).
    
    DETAILED DESCRIPTION:
        This function transforms dates from the standard ISO 8601 format
        (YYYY-MM-DD) used in CSV files to the abbreviated year-month format
        (YYYY-MM) used in DICOM study folder names.
        
        This conversion is necessary because the DICOM folder structure uses
        month-level organization rather than day-level precision.
    
    Args:
        iso_date (str): Date in ISO format (YYYY-MM-DD)
            - Must be a valid date string
            - Example: '1997-07-18', '2010-10-05', '1999-04-02'
    
    Returns:
        str: Date in folder format (YYYY-MM)
            - Year and month only, separated by hyphen
            - Example: '1997-07', '2010-10', '1999-04'
    
    Raises:
        ValueError: If iso_date is not in valid YYYY-MM-DD format
    
    Example:
        >>> date_folder = convert_iso_date_to_folder_format('1997-07-18')
        >>> print(date_folder)
        '1997-07'
        
        >>> date_folder = convert_iso_date_to_folder_format('2010-10-05')
        >>> print(date_folder)
        '2010-10'
    
    Notes:
        - Day information is intentionally discarded
        - Output format matches the folder naming convention in the DICOM archive
        - This is a pure function with no side effects
    """
    date_obj = datetime.strptime(iso_date, "%Y-%m-%d")
    return date_obj.strftime("%Y-%m")


def build_study_folder_path(base_folder: str, patient_id: str, iso_date: str) -> Path:
    """
    Build the full path to a study folder from its components.
    
    DETAILED DESCRIPTION:
        Constructs a complete filesystem path to a DICOM study folder by
        combining:
        1. Base folder containing all patient data
        2. Patient-specific subfolder (SRS + 4-digit ID)
        3. Date-based study subfolder (YYYY-MM__Studies)
        
        This function enforces the standard folder naming conventions used
        throughout the brain metastasis DICOM archive.
    
    Args:
        base_folder (str): Base directory containing all patient folders
            - Absolute or relative path
            - Example: '/database/brainmets/dicom/organized'
            
        patient_id (str): Patient ID as string or int
            - Will be zero-padded to 4 digits automatically
            - Example: '871', '0871', '1885', 871, 1885
            
        iso_date (str): Date in ISO format (YYYY-MM-DD)
            - Will be converted to YYYY-MM folder format
            - Example: '1997-07-18', '2010-10-05'
    
    Returns:
        Path: Complete path to the study folder
            - Format: base_folder/SRSPPPP/YYYY-MM__Studies
            - Example: Path('/database/.../SRS0871/1997-07__Studies')
            - Path object, not string (for cross-platform compatibility)
    
    Example:
        >>> study_path = build_study_folder_path(
        ...     base_folder="/database/brainmets/dicom",
        ...     patient_id="871",  # Will be padded to "0871"
        ...     iso_date="1997-07-18"
        ... )
        >>> print(study_path)
        /database/brainmets/dicom/SRS0871/1997-07__Studies
        
        >>> # Works with already-padded IDs
        >>> study_path = build_study_folder_path(
        ...     base_folder="/database/brainmets/dicom",
        ...     patient_id="1885",
        ...     iso_date="2004-01-14"
        ... )
        >>> print(study_path)
        /database/brainmets/dicom/SRS1885/2004-01__Studies
    
    Notes:
        - Patient ID is always zero-padded to 4 digits for consistency
        - Folder naming follows convention: YYYY-MM__Studies (double underscore)
        - Returns Path object for easy manipulation (.exists(), .iterdir(), etc.)
        - Does NOT verify that the path actually exists on disk
    """
    # Convert date to folder format (YYYY-MM)
    folder_date = convert_iso_date_to_folder_format(iso_date)
    
    # Ensure patient_id is a 4-digit string with leading zeros
    patient_id_str = str(patient_id).zfill(4)
    
    # Build folder names following naming convention
    patient_folder = f"SRS{patient_id_str}"
    study_folder = f"{folder_date}__Studies"
    
    # Construct full path
    full_path = Path(base_folder) / patient_folder / study_folder
    
    return full_path


def find_rtstruct_folder(study_folder: Path, folder_pattern: str) -> Path:
    """
    Find RTStruct folder matching a specific naming pattern.
    
    DETAILED DESCRIPTION:
        Searches within a study folder for a subfolder containing DICOM
        RTStruct files with a specific naming pattern. This is used to
        locate the correct RTStruct file when multiple RTStructs may exist.
        
        Common patterns:
        - "Brain.MS.Init.Model" : Initial treatment planning RTStruct
        - "Brain.MS.ReTx.Model" : Follow-up/retreatment RTStruct
        
        The function ensures exactly one matching folder exists to prevent
        ambiguity in file selection.
    
    Args:
        study_folder (Path): Path to the study folder containing series subfolders
            - Should be a valid directory path
            - Example: Path('/database/.../SRS0871/1997-07__Studies')
            
        folder_pattern (str): String pattern to match in folder name
            - Case-sensitive substring match
            - Example: 'Brain.MS.Init.Model', 'Brain.MS.ReTx.Model'
    
    Returns:
        Path: Path to the matching RTStruct folder
            - Full path to the folder containing RTStruct DICOM files
            - Example: Path('/.../SRS0871_..._Brain.MS.Init.Model_n1__00000')
    
    Raises:
        ValueError: If study folder does not exist
        ValueError: If no matching folder is found
        ValueError: If multiple matching folders are found
    
    Example:
        >>> study_path = Path('/database/.../SRS0871/1997-07__Studies')
        >>> rtstruct_folder = find_rtstruct_folder(
        ...     study_path,
        ...     'Brain.MS.Init.Model'
        ... )
        >>> print(rtstruct_folder.name)
        'SRS0871_SRS0871_RTst_1997-07-18_000000_._Brain.MS.Init.Model_n1__00000'
        
        >>> # For follow-up RTStruct
        >>> followup_folder = find_rtstruct_folder(
        ...     study_path,
        ...     'Brain.MS.ReTx.Model'
        ... )
    
    Notes:
        - Pattern matching is case-sensitive
        - Only searches direct subfolders (not recursive)
        - Raises error if ambiguous (multiple matches) for safety
        - Does not verify that folder actually contains valid RTStruct files
    """
    if not study_folder.exists():
        raise ValueError(f"Study folder does not exist: {study_folder}")
    
    # Find all subfolders matching the pattern
    matching_folders = [
        f for f in study_folder.iterdir() 
        if f.is_dir() and folder_pattern in f.name
    ]
    
    if len(matching_folders) == 0:
        raise ValueError(
            f"No RTStruct folder found with pattern '{folder_pattern}' in {study_folder}"
        )
    elif len(matching_folders) > 1:
        raise ValueError(
            f"Multiple RTStruct folders found with pattern '{folder_pattern}' in {study_folder}:\n"
            + "\n".join(f"  - {f.name}" for f in matching_folders)
        )
    
    return matching_folders[0]


def find_all_mr_folders(study_folder: Path) -> list:
    """
    Find all MR image folders in a study folder.
    
    DETAILED DESCRIPTION:
        Scans a study folder and identifies all subfolders containing MR
        (Magnetic Resonance) imaging DICOM files. This is done by reading
        the DICOM metadata of the first file in each subfolder and checking
        the Modality tag.
        
        This function is useful when:
        - A study contains multiple MR sequences (T1, T2, FLAIR, etc.)
        - You need to process all available MR series
        - Building a complete inventory of imaging data
        
        The function returns folders in the order they are discovered
        (filesystem order), which may not be chronological.
    
    Args:
        study_folder (Path): Path to the study folder containing series subfolders
            - Should be a valid directory path
            - Example: Path('/database/.../SRS0871/1997-07__Studies')
    
    Returns:
        list: List of Path objects to MR folders
            - Each Path points to a folder containing MR DICOM files
            - Empty list if no MR folders found
            - Example: [Path('/.../SRS0871_..._MR_...axial_n47__00000'), ...]
    
    Example:
        >>> study_path = Path('/database/.../SRS0871/1997-07__Studies')
        >>> mr_folders = find_all_mr_folders(study_path)
        >>> print(f"Found {len(mr_folders)} MR series")
        Found 3 MR series
        
        >>> for mr_folder in mr_folders:
        ...     print(mr_folder.name)
        SRS0871_SRS0871_MR_1997-07-18_000000_._axial_n47__00000
        SRS0871_SRS0871_MR_1997-07-18_000000_._coronal_n35__00001
        SRS0871_SRS0871_MR_1997-07-18_000000_._sagittal_n32__00002
    
    Notes:
        - Only checks the first file in each folder for efficiency
        - Silently skips folders that don't contain valid DICOM files
        - Returns empty list if study_folder doesn't exist (no error raised)
        - Identifies MR by checking DICOM Modality tag == "MR"
        - Does not verify image quality or completeness
    """
    mr_folders = []
    
    if not study_folder.exists():
        print(f"WARNING: Study folder does not exist: {study_folder}")
        return mr_folders
    
    # Iterate through all subfolders
    for folder in study_folder.iterdir():
        if not folder.is_dir():
            continue
        
        # Check if folder contains MR DICOM files
        # Get list of files in the folder
        dicom_files = [f for f in folder.iterdir() if f.is_file()]
        
        # Check only the first file for efficiency
        for dicom_file in dicom_files[:1]:
            try:
                # Read DICOM metadata without loading pixel data
                ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
                
                # Check if this is an MR image
                if ds.Modality == "MR":
                    mr_folders.append(folder)
                    break  # Found an MR file, move to next folder
                    
            except Exception:
                # If file is not a valid DICOM or other error, skip it
                continue
    
    return mr_folders


def find_all_rtstruct_folders(study_folder: Path) -> list:
    """
    Find all RTStruct folders in a study folder.
    
    DETAILED DESCRIPTION:
        Scans a study folder and identifies all subfolders containing RTSTRUCT
        (radiation therapy structure set) DICOM files. This is done by reading
        the DICOM metadata of the first file in each subfolder and checking
        the Modality tag.
        
        This function is useful when:
        - A study contains multiple RTStruct files (backup, different versions)
        - Building a complete inventory of structure sets
        - Quality control requires reviewing all available structures
        
        The function returns folders in filesystem order, which may not be
        chronological or by series number.
    
    Args:
        study_folder (Path): Path to the study folder containing series subfolders
            - Should be a valid directory path
            - Example: Path('/database/.../SRS0871/1997-07__Studies')
    
    Returns:
        list: List of Path objects to RTStruct folders
            - Each Path points to a folder containing RTSTRUCT DICOM files
            - Empty list if no RTStruct folders found
            - Example: [Path('/.../SRS0871_..._Brain.MS.Init.Model_n1__00000'), ...]
    
    Example:
        >>> study_path = Path('/database/.../SRS0871/1997-07__Studies')
        >>> rtstruct_folders = find_all_rtstruct_folders(study_path)
        >>> print(f"Found {len(rtstruct_folders)} RTStruct series")
        Found 2 RTStruct series
        
        >>> for rtstruct_folder in rtstruct_folders:
        ...     print(rtstruct_folder.name)
        SRS0871_SRS0871_RTst_1997-07-18_000000_._Brain.MS.Init.Model_n1__00000
        SRS0871_SRS0871_RTst_1997-07-18_000000_._Backup_Structures_n1__00001
    
    Notes:
        - Only checks the first file in each folder for efficiency
        - Silently skips folders that don't contain valid DICOM files
        - Returns empty list if study_folder doesn't exist (no error raised)
        - Identifies RTStruct by checking DICOM Modality tag == "RTSTRUCT"
        - Does not verify structure completeness or correctness
    """
    rtstruct_folders = []
    
    if not study_folder.exists():
        print(f"WARNING: Study folder does not exist: {study_folder}")
        return rtstruct_folders
    
    # Iterate through all subfolders
    for folder in study_folder.iterdir():
        if not folder.is_dir():
            continue
        
        # Check if folder contains RTSTRUCT DICOM files
        dicom_files = [f for f in folder.iterdir() if f.is_file()]
        
        # Check only the first file for efficiency
        for dicom_file in dicom_files[:1]:
            try:
                # Read DICOM metadata without loading pixel data
                ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
                
                # Check if this is an RTSTRUCT
                if ds.Modality == "RTSTRUCT":
                    rtstruct_folders.append(folder)
                    break  # Found an RTSTRUCT file, move to next folder
                    
            except Exception:
                # If file is not a valid DICOM or other error, skip it
                continue
    
    return rtstruct_folders


def find_all_rtdose_folders(study_folder: Path) -> list:
    """
    Find all RTDOSE folders in a study folder.
    
    DETAILED DESCRIPTION:
        Scans a study folder and identifies all subfolders containing RTDOSE
        (radiation therapy dose) DICOM files. This is done by reading the
        DICOM metadata of the first file in each subfolder and checking the
        Modality tag.
        
        This function is useful when:
        - A study contains multiple dose distributions
        - Building a complete inventory of treatment plans
        - Quality control requires comparing different dose calculations
        
        The function returns folders in filesystem order, which may not be
        chronological or by series number.
    
    Args:
        study_folder (Path): Path to the study folder containing series subfolders
            - Should be a valid directory path
            - Example: Path('/database/.../SRS0871/1997-07__Studies')
    
    Returns:
        list: List of Path objects to RTDOSE folders
            - Each Path points to a folder containing RTDOSE DICOM files
            - Empty list if no RTDOSE folders found
            - Example: [Path('/.../SRS0871_..._n1__00000'), ...]
    
    Example:
        >>> study_path = Path('/database/.../SRS0871/1997-07__Studies')
        >>> rtdose_folders = find_all_rtdose_folders(study_path)
        >>> print(f"Found {len(rtdose_folders)} RTDOSE series")
        Found 1 RTDOSE series
        
        >>> for rtdose_folder in rtdose_folders:
        ...     print(rtdose_folder.name)
        SRS0871_SRS0871_RTdo_1997-07-18_000000_._Treatment_Dose_n1__00000
    
    Notes:
        - Only checks the first file in each folder for efficiency
        - Silently skips folders that don't contain valid DICOM files
        - Returns empty list if study_folder doesn't exist (no error raised)
        - Identifies RTDOSE by checking DICOM Modality tag == "RTDOSE"
        - Does not verify dose calculation accuracy
    """
    rtdose_folders = []
    
    if not study_folder.exists():
        print(f"WARNING: Study folder does not exist: {study_folder}")
        return rtdose_folders
    
    # Iterate through all subfolders
    for folder in study_folder.iterdir():
        if not folder.is_dir():
            continue
        
        # Check if folder contains RTDOSE DICOM files
        dicom_files = [f for f in folder.iterdir() if f.is_file()]
        
        # Check only the first file for efficiency
        for dicom_file in dicom_files[:1]:
            try:
                # Read DICOM metadata without loading pixel data
                ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
                
                # Check if this is an RTDOSE
                if ds.Modality == "RTDOSE":
                    rtdose_folders.append(folder)
                    break  # Found an RTDOSE file, move to next folder
                    
            except Exception:
                # If file is not a valid DICOM or other error, skip it
                continue
    
    return rtdose_folders


def count_dicom_files(folder_path: Path) -> int:
    """
    Count the number of DICOM files in a folder.
    
    DETAILED DESCRIPTION:
        This function counts the total number of valid files in a folder path.
        It does NOT validate whether files are valid DICOM format - it simply
        counts all files (not directories) present in the folder.
        
        This is useful for:
        - Quick validation that a folder contains data
        - Estimating series completeness
        - Quality control checks
        
        For more thorough DICOM validation, consider reading metadata from
        each file.
    
    Args:
        folder_path (Path): Path to folder containing DICOM files
            - Should be a valid directory path
            - Example: Path('/.../SRS0871_..._MR_...axial_n47__00000')
    
    Returns:
        int: Number of files in the folder
            - Counts only files, not subdirectories
            - Returns 0 if folder doesn't exist or is empty
            - Example: 47, 120, 1
    
    Example:
        >>> mr_folder = Path('/.../SRS0871_..._MR_...axial_n47__00000')
        >>> file_count = count_dicom_files(mr_folder)
        >>> print(f"Folder contains {file_count} files")
        Folder contains 47 files
        
        >>> # Check if folder has data
        >>> if count_dicom_files(mr_folder) == 0:
        ...     print("Warning: Folder is empty!")
    
    Notes:
        - Does NOT verify files are valid DICOM format
        - Includes all files regardless of extension
        - Returns 0 if folder doesn't exist (no error raised)
        - Does not count subdirectories
        - Fast operation, suitable for large-scale scanning
    """
    if not folder_path.exists():
        return 0
    
    # Count only files, not directories
    file_count = sum(1 for f in folder_path.iterdir() if f.is_file())
    
    return file_count


def find_dicom_paths_for_rtstudy(
    lesion_label: str,
    study_type: str,
    inventory_excel_path: str,
    base_folder: str,
    debug: bool = False
) -> dict:
    """
    Find DICOM file paths from inventory Excel file for RT study.
    
    DETAILED DESCRIPTION:
        This function parses a lesion label, loads an inventory Excel file,
        validates selections, and returns the paths to MR, RTSTRUCT, and RTDOSE
        folders along with the structure name. It is designed for radiation
        therapy (RT) studies where exactly one series of each modality must be
        selected for processing.
        
        WORKFLOW:
        1. Parse lesion_label to extract patient_id and target
        2. Validate study_type parameter
        3. Load inventory Excel/CSV file
        4. Filter inventory for matching patient_id and target
        5. Validate that exactly 1 MR, 1 RTSTRUCT, and 1 RTDOSE are selected
           for the requested study_type
        6. Extract folder names and study_date from inventory
        7. Build full paths to DICOM folders using study folder path
        8. Convert target to RTStruct naming format
        9. Return dictionary with all paths and structure name
    
    Args:
        lesion_label (str): Lesion identifier in format 'PPPP.TT'
            - PPPP: 4-digit patient ID (e.g., '0871', '1885')
            - TT: 2-digit target ID (e.g., '01', '09')
            - Example: '0871.01', '1885.09', '3126.15'
            
        study_type (str): Which timepoint to process
            - 'initial_moving': Process initial timepoint only
            - 'followup_fixed': Process follow-up timepoint only
            - Case-insensitive
            
        inventory_excel_path (str): Path to inventory Excel/CSV file
            - Expected file: prep1_step1_study_folder_inventory_manual_check_wip.xlsx
            - Must contain columns: study_id, patient_id, study_date, study_type,
              targets, modality, folder_name, selected
            - The 'selected' column should have 'x' for chosen series
            - Example: "./excel/output/prep1_step1_study_folder_inventory_manual_check_wip.xlsx"
            
        base_folder (str): Base directory containing all patient DICOM data
            - Should contain SRSPPPP subfolders
            - Example: '/database/brainmets/dicom/organized'
            
        debug (bool): If True, print detailed diagnostic information
            - Default: False
            - Shows parsing, filtering, and validation details
    
    Returns:
        dict: Dictionary containing:
            - 'mr_path' (Path): Path to MR DICOM folder
            - 'rtstruct_path' (Path): Path to RTSTRUCT DICOM folder
            - 'rtdose_path' (Path): Path to RTDOSE DICOM folder
            - 'structure_name' (str): RTStruct structure name (e.g., 'target1', 'target9')
    
    Raises:
        FileNotFoundError: If inventory Excel/CSV file doesn't exist
        ValueError: If lesion_label format is invalid
        ValueError: If study_type is not valid
        ValueError: If selections in inventory are invalid (not exactly 1 per modality)
        KeyError: If target is not found in inventory
    
    Example:
        >>> from code.RTdicomorganizer import path_utils
        
        >>> result = path_utils.find_dicom_paths_for_rtstudy(
        ...     lesion_label='0871.01',
        ...     study_type='initial_moving',
        ...     inventory_excel_path='./excel/output/inventory.xlsx',
        ...     base_folder='/database/brainmets/dicom/organized',
        ...     debug=False
        ... )
        >>> print(result['mr_path'])
        /database/brainmets/dicom/organized/SRS0871/1997-07__Studies/44-MR-3_Plane_Loc_TCORONAL_POST_GAD-73066
        >>> print(result['structure_name'])
        target1
        
        >>> # For follow-up study
        >>> result = path_utils.find_dicom_paths_for_rtstudy(
        ...     lesion_label='1885.09',
        ...     study_type='followup_fixed',
        ...     inventory_excel_path='./excel/output/inventory.xlsx',
        ...     base_folder='/database/brainmets/dicom/organized'
        ... )
    
    Notes:
        - Uses data_parser.parse_patient_target() for robust lesion label parsing
        - Uses data_parser.convert_target_to_rtstruct_format() for structure name conversion
        - Uses build_study_folder_path() to construct study folder paths
        - Validates selections before building paths to fail fast
        - Target matching: Checks if target appears anywhere in targets column
          (handles both '01' and '01/02/03' formats)
        - Returns Path objects for cross-platform compatibility
        - Does NOT verify that folders actually exist on disk
    """
    # ===================================================================
    # Step 1: Parse lesion_label
    # ===================================================================
    try:
        patient_id, target, _ = data_parser.parse_patient_target(lesion_label)
        if patient_id is None or target is None:
            raise ValueError(f"Could not parse lesion_label: '{lesion_label}'")
    except Exception as e:
        raise ValueError(
            f"Invalid lesion_label format: '{lesion_label}'\n"
            f"Expected format: 'PPPP.TT' (e.g., '0871.01')\n"
            f"Error: {e}"
        )
    
    if debug:
        print(f"Parsed lesion_label: patient_id={patient_id}, target={target}")
    
    # ===================================================================
    # Step 2: Validate study_type
    # ===================================================================
    study_type_lower = study_type.lower()
    valid_study_types = ['initial_moving', 'followup_fixed']
    
    if study_type_lower not in valid_study_types:
        raise ValueError(
            f"Invalid study_type: '{study_type}'\n"
            f"Valid options: {valid_study_types}"
        )
    
    if debug:
        print(f"Study type: {study_type_lower}")
    
    # ===================================================================
    # Step 3: Load inventory Excel/CSV file
    # ===================================================================
    inventory_path = Path(inventory_excel_path)
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Inventory Excel/CSV file not found: {inventory_excel_path}\n"
            f"Please ensure the file exists and path is correct."
        )
    
    if debug:
        print(f"Loading inventory: {inventory_path.name}...")
    
    # Read CSV or Excel based on file extension
    if inventory_path.suffix == '.csv':
        df_inventory = pd.read_csv(inventory_path)
    else:
        df_inventory = pd.read_excel(inventory_path)
    
    if debug:
        print(f"Loaded inventory: {len(df_inventory)} rows")
    
    # ===================================================================
    # Step 4: Filter inventory for matching patient_id and target
    # ===================================================================
    # Filter by patient_id
    df_filtered = df_inventory[
        df_inventory['patient_id'].astype(str).str.zfill(4) == patient_id
    ].copy()
    
    # Filter by target (check if target appears in targets column)
    # Handle both '01' and '01/02/03' formats
    df_filtered = df_filtered[
        df_filtered['targets'].astype(str).str.contains(target, regex=False, na=False)
    ]
    
    if len(df_filtered) == 0:
        available = df_inventory.groupby(['patient_id', 'targets']).size().reset_index(name='count')
        raise KeyError(
            f"No data found for patient {patient_id}, target {target}\n"
            f"Available combinations:\n{available.to_string()}"
        )
    
    if debug:
        print(f"Filtered: {len(df_filtered)} rows")
    
    # ===================================================================
    # Step 5: Validate selections for the requested timepoint
    # ===================================================================
    # Filter for this timepoint
    df_tp = df_filtered[df_filtered['study_type'] == study_type_lower].copy()
    
    if len(df_tp) == 0:
        raise ValueError(
            f"No inventory data found for {study_type_lower}\n"
            f"Available study_types: {df_filtered['study_type'].unique().tolist()}"
        )
    
    # Filter for selected rows (marked with 'x')
    df_selected = df_tp[df_tp['selected'].astype(str).str.lower() == 'x'].copy()
    
    # Count selections by modality
    selection_counts = df_selected.groupby('modality').size().to_dict()
    
    # Validate MR
    mr_count = selection_counts.get('MR', 0)
    if mr_count != 1:
        mr_folders = df_selected[df_selected['modality'] == 'MR']['folder_name'].tolist()
        raise ValueError(
            f"ERROR: Expected exactly 1 MR selected for {study_type_lower}, found {mr_count}\n"
            f"Selected MR folders: {mr_folders if mr_folders else 'None'}\n"
            f"Please update the inventory Excel file."
        )
    
    # Validate RTSTRUCT
    rtstruct_count = selection_counts.get('RTSTRUCT', 0)
    if rtstruct_count != 1:
        rtstruct_folders = df_selected[df_selected['modality'] == 'RTSTRUCT']['folder_name'].tolist()
        raise ValueError(
            f"ERROR: Expected exactly 1 RTSTRUCT selected for {study_type_lower}, found {rtstruct_count}\n"
            f"Selected RTSTRUCT folders: {rtstruct_folders if rtstruct_folders else 'None'}\n"
            f"Please update the inventory Excel file."
        )
    
    # Validate RTDOSE
    rtdose_count = selection_counts.get('RTDOSE', 0)
    if rtdose_count != 1:
        rtdose_folders = df_selected[df_selected['modality'] == 'RTDOSE']['folder_name'].tolist()
        raise ValueError(
            f"ERROR: Expected exactly 1 RTDOSE selected for {study_type_lower}, found {rtdose_count}\n"
            f"Selected RTDOSE folders: {rtdose_folders if rtdose_folders else 'None'}\n"
            f"Please update the inventory Excel file."
        )
    
    # Extract folder names and study_date
    mr_row = df_selected[df_selected['modality'] == 'MR'].iloc[0]
    rtstruct_row = df_selected[df_selected['modality'] == 'RTSTRUCT'].iloc[0]
    rtdose_row = df_selected[df_selected['modality'] == 'RTDOSE'].iloc[0]
    
    study_date = mr_row['study_date']
    mr_folder = mr_row['folder_name']
    rtstruct_folder = rtstruct_row['folder_name']
    rtdose_folder = rtdose_row['folder_name']
    
    if debug:
        print(f"Selected folders:")
        print(f"  MR:       {mr_folder}")
        print(f"  RTSTRUCT: {rtstruct_folder}")
        print(f"  RTDOSE:   {rtdose_folder}")
        print(f"  Study date: {study_date}")
    
    # ===================================================================
    # Step 6: Build full paths
    # ===================================================================
    study_folder = build_study_folder_path(base_folder, patient_id, study_date)
    
    mr_path = study_folder / mr_folder
    rtstruct_path = study_folder / rtstruct_folder
    rtdose_path = study_folder / rtdose_folder
    
    # ===================================================================
    # Step 7: Convert target to RTStruct format
    # ===================================================================
    structure_name_target     = data_parser.convert_target_to_rtstruct_format(target)
    structure_name_outer_rind = 'Brain_' + structure_name_target
    
    if debug:
        print(f"Structure name (target): {structure_name_target}")
        print(f"Structure name (outer rind incl. target): {structure_name_outer_rind}")
        print(f"MR path: {mr_path}")
        print(f"RTSTRUCT path: {rtstruct_path}")
        print(f"RTDOSE path: {rtdose_path}")
    
    # ===================================================================
    # Step 8: Return results
    # ===================================================================
    return {
        'mr_path': mr_path,
        'rtstruct_path': rtstruct_path,
        'rtdose_path': rtdose_path,
        'structure_name_target': structure_name_target,
        'structure_name_outer_rind': structure_name_outer_rind
    }

