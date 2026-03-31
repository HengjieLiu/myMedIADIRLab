"""
==============================================================================
MODULE NAME: data_analysis.py
==============================================================================

PURPOSE:
    This module provides functions for analyzing and summarizing brain
    metastasis lesion data. It transforms raw patient statistics into
    structured summaries suitable for registration planning and QC analysis.
    
    Key responsibilities:
    - Build per-lesion summary tables from raw data
    - Group lesions into unique registration pairs
    - Count and validate DICOM modalities in log text
    - Extract QC metrics from structured log files
    
    This module implements the core analytical logic for identifying which
    scans need to be registered together and tracking data completeness.

DEPENDENCIES:
    External packages:
    - pandas: For DataFrame operations and grouping
    - re: For pattern matching in log text
    
    Internal modules:
    - data_parser: Uses parse_patient_target, ensure_dates, targets_join

FUNCTIONS:
    1. build_summaries(df: pd.DataFrame) -> tuple
       Build per-lesion and per-pair summary tables from raw data
       
    2. count_modality_in_section(section_text: str, modality: str) -> int
       Count DICOM series of specific modality in log text
       
    3. check_series_description_in_section(section_text: str, 
                                          series_description: str) -> bool
       Check if series description exists in log text
       
    4. generate_study_folder_inventory(csv_path: str, base_folder: str,
                                      output_path: str, **options) -> str
       Generate comprehensive inventory of DICOM folders for all study folders

USAGE EXAMPLE:
    ```python
    from code.RTdicomorganizer import data_io, data_analysis
    
    # Read raw data
    df_raw = data_io.read_table("input/patient_stats.xlsx")
    
    # Build summaries
    per_lesion_df, per_pair_df = data_analysis.build_summaries(df_raw)
    
    print(f"Found {len(per_lesion_df)} lesions")
    print(f"Need {len(per_pair_df)} unique registrations")
    
    # Parse log text
    mr_count = data_analysis.count_modality_in_section(log_text, "MR")
    has_rtstruct = data_analysis.check_series_description_in_section(
        log_text, "Brain_MS_Init_Model"
    )
    ```

NOTES:
    - build_summaries() is the main entry point for data processing
    - Patient IDs are always zero-padded to 4 digits
    - Rows with missing dates are automatically filtered out
    - Registration pairs group lesions with same patient + same dates

==============================================================================
"""

import re
import pandas as pd
import pydicom
from pathlib import Path
from . import data_parser
from . import path_utils
from . import dicom_utils


def build_summaries(df: pd.DataFrame) -> tuple:
    """
    Build per-lesion and per-registration-pair summary tables from raw data.
    
    DETAILED DESCRIPTION:
        This is the core analytical function that transforms raw patient
        statistics into structured summaries for registration planning.
        
        PROCESS:
        1. Parse lesion identifiers to extract patient and target IDs
        2. Convert date columns to datetime format
        3. Filter out rows with missing dates
        4. Convert dates to ISO format (YYYY-MM-DD)
        5. Create per-lesion summary table
        6. Group lesions by (patient, initial_date, followup_date) to identify
           unique registration pairs
        7. Aggregate target IDs for each registration pair
        
        REGISTRATION PAIR LOGIC:
        A registration pair is uniquely identified by:
        - Patient ID (same patient)
        - Initial scan date (same moving image)
        - Follow-up scan date (same fixed image)
        
        Multiple lesions may share the same registration pair if they were
        all present at both timepoints.
    
    Args:
        df (pd.DataFrame): Raw DataFrame from input CSV/Excel file
            See INPUT FORMAT below
    
    Returns:
        tuple: (per_lesion_df, per_pair_df)
            - per_lesion_df: One row per lesion
            - per_pair_df: One row per unique registration pair
            See OUTPUT FORMATS below
    
    Raises:
        KeyError: If 'lesno' or 'lesno_1' column is not found
    
    INPUT FORMAT:
        Required columns (case-insensitive):
        - lesno or lesno_1 (str/float): Lesion identifier
          Format: PPPP.TT or numeric (e.g., 1885.09)
          Example: '0871.01', 1885.089966 (Excel float)
          
        - datepriorsrs (str/date): Initial scan date
          Any common date format
          Example: '1997-07-18', 43891
          
        - dategk (str/date): Follow-up scan date
          Any common date format
          Example: '1999-04-02', 43950
    
    OUTPUT FORMAT - Per-Lesion DataFrame:
        Columns:
        - patient_id (str): 4-digit patient ID
          Example: '0871', '1885', '3126'
          
        - target (str): 2-digit target ID
          Example: '01', '09', '15'
          
        - lesion_label (str): Combined identifier
          Format: 'PPPP.TT'
          Example: '0871.01', '1885.09'
          
        - initial_moving (str): Initial scan date
          Format: ISO YYYY-MM-DD
          Example: '1997-07-18', '2004-01-14'
          
        - followup_fixed (str): Follow-up scan date
          Format: ISO YYYY-MM-DD
          Example: '1999-04-02', '2010-10-05'
    
    OUTPUT FORMAT - Per-Pair DataFrame:
        Columns:
        - patient_id (str): 4-digit patient ID
          Example: '0871', '1885'
          
        - regpair (str): 2-digit registration pair counter (resets per patient)
          Format: '01', '02', '03', ...
          Example: '01' (first pair for this patient), '02' (second pair)
          Note: Counter resets to '01' for each new patient_id
          
        - initial_moving (str): Initial scan date (YYYY-MM-DD)
          Example: '1997-07-18'
          
        - followup_fixed (str): Follow-up scan date (YYYY-MM-DD)
          Example: '1999-04-02'
          
        - targets (str): Slash-separated target IDs
          Format: 'TT' or 'TT/TT/TT'
          Example: '01', '01/09/15', '06/07/08/11/12'
          
        - n_targets (int): Number of lesions sharing this pair
          Example: 1, 2, 5
    
    Example:
        >>> import pandas as pd
        >>> from code.RTdicomorganizer import data_analysis
        
        >>> # Raw input data
        >>> df_raw = pd.DataFrame({
        ...     'lesno': [1885.09, 1885.10, 1885.13],
        ...     'datepriorsrs': ['2004-01-14', '2005-01-03', '2006-08-23'],
        ...     'dategk': ['2010-10-05', '2015-11-12', '2010-10-05']
        ... })
        
        >>> # Build summaries
        >>> per_lesion, per_pair = data_analysis.build_summaries(df_raw)
        
        >>> print(per_lesion)
           patient_id target lesion_label initial_moving followup_fixed
        0        1885     09      1885.09     2004-01-14     2010-10-05
        1        1885     10      1885.10     2005-01-03     2015-11-12
        2        1885     13      1885.13     2006-08-23     2010-10-05
        
        >>> print(per_pair)
           patient_id regpair initial_moving followup_fixed targets  n_targets
        0        1885      01     2004-01-14     2010-10-05      09          1
        1        1885      02     2005-01-03     2015-11-12      10          1
        2        1885      03     2006-08-23     2010-10-05      13          1
        
        >>> # Example with multiple lesions in same registration pair
        >>> df_raw2 = pd.DataFrame({
        ...     'lesno': [1190.01, 1190.02],
        ...     'datepriorsrs': ['1999-07-21', '1999-07-21'],
        ...     'dategk': ['2001-06-26', '2001-06-26']
        ... })
        >>> per_lesion2, per_pair2 = data_analysis.build_summaries(df_raw2)
        >>> print(per_pair2)
           patient_id regpair initial_moving followup_fixed targets  n_targets
        0        1190      01     1999-07-21     2001-06-26   01/02          2
    
    Notes:
        - Rows with missing dates (NaT) are automatically excluded
        - Patient IDs are always 4 digits with leading zeros
        - Target IDs are always 2 digits with leading zeros
        - Results are sorted by patient_id, then by dates
        - Target IDs in per_pair are sorted numerically
    """
    df = df.copy()
    
    # ===================================================================
    # Step 1: Find the lesion number column
    # ===================================================================
    les_src = "lesno" if "lesno" in df.columns else (
        "lesno_1" if "lesno_1" in df.columns else None
    )
    if les_src is None:
        raise KeyError(
            "Expected a 'lesno' or 'lesno_1' column in the input DataFrame.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # ===================================================================
    # Step 2: Parse patient ID and target ID from lesion numbers
    # ===================================================================
    parsed = df[les_src].apply(data_parser.parse_patient_target)
    df["patient_id"] = parsed.apply(lambda x: x[0])      # 4-digit patient ID
    df["target_id_2d"] = parsed.apply(lambda x: x[1])    # 2-digit target ID
    df["lesion_label"] = parsed.apply(lambda x: x[2])    # Full label (e.g., '1885.09')
    
    # ===================================================================
    # Step 3: Convert date columns to datetime format
    # ===================================================================
    df = data_parser.ensure_dates(df, ("datepriorsrs", "dategk"))
    
    # ===================================================================
    # Step 4: Filter to keep only rows with both dates present
    # ===================================================================
    df = df.loc[df["datepriorsrs"].notna() & df["dategk"].notna()].copy()
    
    # ===================================================================
    # Step 5: Convert dates to ISO format (YYYY-MM-DD)
    # ===================================================================
    df["initial_date_iso"] = df["datepriorsrs"].dt.strftime("%Y-%m-%d")
    df["followup_date_iso"] = df["dategk"].dt.strftime("%Y-%m-%d")

    # ===================================================================
    # Step 6: Create per-lesion summary table
    # ===================================================================
    lesion_cols = [
        "patient_id",
        "target_id_2d",
        "lesion_label",
        "initial_date_iso",
        "followup_date_iso",
    ]
    per_lesion = df[lesion_cols].rename(
        columns={
            "target_id_2d": "target",
            "initial_date_iso": "initial_moving",
            "followup_date_iso": "followup_fixed",
        }
    )
    
    # ===================================================================
    # Step 7: Group lesions into registration pairs
    # ===================================================================
    # Group by patient and date pair to find unique registrations
    per_pair = (
        per_lesion.groupby(
            ["patient_id", "initial_moving", "followup_fixed"],
            dropna=False
        )
        .agg(
            targets=("target", data_parser.targets_join),  # Combine target IDs
            n_targets=("target", "nunique")                # Count unique lesions
        )
        .reset_index()
        .sort_values(["patient_id", "initial_moving", "followup_fixed"], ignore_index=True)
    )
    
    # ===================================================================
    # Step 8: Add regpair counter (resets for each patient)
    # ===================================================================
    # Count registration pairs within each patient (01, 02, 03, ...)
    per_pair['regpair'] = (
        per_pair.groupby('patient_id').cumcount() + 1
    ).apply(lambda x: str(x).zfill(2))
    
    # Reorder columns to place regpair after patient_id
    cols = ['patient_id', 'regpair', 'initial_moving', 'followup_fixed', 'targets', 'n_targets']
    per_pair = per_pair[cols]

    return per_lesion, per_pair


def count_modality_in_section(section_text: str, modality: str) -> int:
    """
    Count the number of DICOM series for a specific modality in log text.
    
    DETAILED DESCRIPTION:
        This function parses structured log text output from the DICOM
        overview script to count how many series of a specific modality
        (MR, RTDOSE, RTSTRUCT, CT, etc.) are present.
        
        The log format expected is:
        MODALITY    SERIES_NUM   DESCRIPTION   ...
        MR          44           axial         ...
        MR          45           sagittal      ...
        RTDOSE      12           dose_plan     ...
        
        This is used for QC validation to ensure required imaging data exists.
    
    Args:
        section_text (str): Text section from DICOM overview log
            - Should be multi-line string from log file
            - Contains lines formatted as: "MODALITY  SERIES_NUM  ..."
            - Example: Output from one scan's overview
            
        modality (str): DICOM modality to count
            - Case-sensitive
            - Example: "MR", "RTDOSE", "RTSTRUCT", "CT"
    
    Returns:
        int: Count of series with matching modality
            - 0 if no matches found
            - Example: 1, 3, 5
    
    LOG TEXT FORMAT:
        Expected line format:
        MODALITY    SERIES_NUM    ORIENTATION    OTHER_INFO
        MR          44            axial          T1 contrast
        MR          45            sagittal       T2
        RTDOSE      12            N/A            Treatment plan
        RTSTRUCT    8             N/A            Target contours
    
    Example:
        >>> log_text = '''
        ... MR          44     axial              T1 contrast
        ... MR          45     sagittal           T2 FLAIR
        ... RTSTRUCT    8      N/A                Brain_MS_Init_Model
        ... RTDOSE      12     N/A                Treatment dose
        ... '''
        
        >>> mr_count = count_modality_in_section(log_text, "MR")
        >>> print(mr_count)
        2
        
        >>> rtdose_count = count_modality_in_section(log_text, "RTDOSE")
        >>> print(rtdose_count)
        1
        
        >>> ct_count = count_modality_in_section(log_text, "CT")
        >>> print(ct_count)
        0
    
    Notes:
        - Pattern matching is case-sensitive
        - Matches lines where modality appears at start followed by numbers
        - Uses regex pattern: ^MODALITY\s+\d+
        - Returns 0 if no matches (not None or error)
        - Counts series, not individual slices
    """
    # Pattern: modality at start of line, followed by whitespace and number
    pattern = rf'^{modality}\s+\d+'
    matches = re.findall(pattern, section_text, re.MULTILINE)
    return len(matches)


def check_series_description_in_section(section_text: str, 
                                       series_description: str) -> bool:
    """
    Check if a specific series description exists in log text section.
    
    DETAILED DESCRIPTION:
        This function performs a simple substring search to determine if
        a particular series description (like "Brain_MS_Init_Model") appears
        anywhere in the log text section.
        
        This is used for QC validation to verify that required RTStruct
        files with specific naming conventions are present in the data.
        
        Unlike count_modality_in_section, this is a binary check (present
        or not) rather than a count.
    
    Args:
        section_text (str): Text section from DICOM overview log
            - Multi-line string from log file
            - Should contain series descriptions
            - Example: Output from RTStruct folder overview
            
        series_description (str): Exact series description to search for
            - Case-sensitive substring match
            - Example: "Brain_MS_Init_Model", "Brain_MS_ReTx_Model"
    
    Returns:
        bool: True if series description found, False otherwise
    
    LOG TEXT FORMAT:
        Series descriptions appear in log output like:
        RTSTRUCT    8      N/A    Brain_MS_Init_Model    ...
        RTSTRUCT    9      N/A    Brain_MS_ReTx_Model    ...
    
    Example:
        >>> log_text = '''
        ... RTSTRUCT    8      N/A    Brain_MS_Init_Model    Created 2020-01-15
        ... RTSTRUCT    9      N/A    Backup_structures      Archived
        ... '''
        
        >>> has_init = check_series_description_in_section(
        ...     log_text, "Brain_MS_Init_Model"
        ... )
        >>> print(has_init)
        True
        
        >>> has_retx = check_series_description_in_section(
        ...     log_text, "Brain_MS_ReTx_Model"
        ... )
        >>> print(has_retx)
        False
        
        >>> # Case-sensitive
        >>> has_lowercase = check_series_description_in_section(
        ...     log_text, "brain_ms_init_model"
        ... )
        >>> print(has_lowercase)
        False
    
    Notes:
        - Simple substring search (not regex pattern matching)
        - Case-sensitive matching
        - Returns False if section_text is empty
        - Used primarily for RTStruct model validation in QC
        - Consider using this for binary presence checks vs count_modality
          for counting series
    """
    return series_description in section_text


def generate_study_folder_inventory(
    csv_path: str,
    base_folder: str,
    output_path: str,
    include_metadata: bool = False,
    validate_rtstruct: bool = True,
    expected_rtstruct_patterns: list = None
) -> str:
    """
    Generate comprehensive inventory of DICOM folders for all study folders.
    
    DETAILED DESCRIPTION:
        This function reads a registration pair summary CSV and generates a
        detailed inventory of all MR, RTStruct, and RTDose folders found in
        each study folder. The inventory is saved as both CSV and Excel files
        with one row per DICOM series, organized by study folder.
        
        The function performs these steps:
        1. Read registration pair CSV (prep1_step1_summary_by_registrationpair.csv)
        2. For each unique study folder (both initial_moving and followup_fixed):
           - Construct the study folder path
           - Find all MR folders
           - Find all RTStruct folders
           - Find all RTDose folders
        3. For each folder found:
           - Read series description
           - Count DICOM files
           - Optionally: read image dimensions, orientation, acquisition date
           - Optionally: validate RTStruct naming patterns
        4. Build a long-format DataFrame with all series
        5. Export to CSV (for easier viewing) and Excel (for manual selection)
        
        The output files allow manual selection of which series to use for
        subsequent processing steps by marking rows with "x" in the Excel file.
        
        This is particularly useful for:
        - Quality control - reviewing all available data
        - Manual data selection - choosing the best MR series
        - RTStruct validation - ensuring correct structure sets are used
        - Documentation - tracking which data was selected for analysis
    
    Args:
        csv_path (str): Path to registration pair summary CSV
            - Expected file: prep1_step1_summary_by_registrationpair.csv
            - Must contain columns: patient_id, regpair, initial_moving, followup_fixed,
              targets, n_targets
            - patient_id should be 4-digit string with leading zeros (e.g., '0871')
            - regpair should be 2-digit string (e.g., '01', '02', '03')
              * Resets for each patient (each patient starts from '01')
              * Generated automatically by build_summaries()
            - Example: "./excel/output/prep1_step1_summary_by_registrationpair.csv"
            
        base_folder (str): Root directory containing patient DICOM data
            - Base path where patient folders (SRSXXXX) are located
            - Example: "/database/brainmets/dicom"
            
        output_path (str): Path for output Excel file
            - Will be created if doesn't exist
            - Should have .xlsx extension
            - CSV file will also be created (same name with .csv extension)
            - Example: "./excel/output/prep1_step1_study_folder_inventory.xlsx"
            
        include_metadata (bool, optional): Whether to include optional metadata
            - If True: adds columns for image dimensions, orientation, acquisition date
            - If False: only includes essential info (folder name, series description)
            - Default: False
            - Setting to True increases processing time
            
        validate_rtstruct (bool, optional): Whether to validate RTStruct names
            - If True: adds a validation flag column for RTStruct series
            - Checks if series description matches expected patterns
            - Default: True
            - Useful for automatically flagging correct structure sets
            
        expected_rtstruct_patterns (list, optional): RTStruct naming patterns to validate
            - List of strings to match against series descriptions
            - Only used if validate_rtstruct=True
            - Default: ["Brain_MS_Init_Model", "Brain_MS_ReTx_Model"]
            - Case-sensitive matching
    
    Returns:
        str: Path to the generated Excel file
            - Same as output_path parameter
            - Excel file contains one sheet with all series (for manual selection)
            - CSV file also created (same name, .csv extension, for easier viewing)
            - Both files contain identical data
    
    OUTPUT FILE FORMAT (CSV and Excel):
        Long-format table with one row per DICOM series folder
        Both CSV and Excel files contain identical data structure
        
        Required columns:
        - study_id (str): Unique identifier with regpair number and date
            * Format: {patient_id}_regpair{nn}_{type}_{YYYY-MM}
            * Example: '3126_regpair01_initial_2009-08'
            * regpair number resets for each patient (01, 02, ... per patient)
        - patient_id (str): 4-digit patient ID (e.g., '0871')
        - study_date (str): Study date in ISO format (e.g., '1997-07-18')
        - study_type (str): Either 'initial_moving' or 'followup_fixed'
        - targets (str): Slash-separated target IDs (e.g., '01', '01/02')
        - n_targets (int): Number of lesion targets
        - modality (str): DICOM modality (e.g., 'MR', 'RTSTRUCT', 'RTDOSE')
        - folder_name (str): Name of the folder containing series
        - series_description (str): Series description from DICOM
        - file_count (int): Number of files in folder
        - selected (str): Mark with 'x' for selection
            * Auto-populated with 'x' for validated RTStruct (when validate_rtstruct=True)
            * Empty for all other rows (manual entry required)
        
        Optional columns (if include_metadata=True):
        - image_dimensions (str): Image matrix size (e.g., '512x512x47')
        - orientation (str): Image orientation (e.g., 'Axial', 'Sagittal')
        - acquisition_date (str): Acquisition date from DICOM
        
        Optional columns (if validate_rtstruct=True):
        - rtstruct_validated (str): 'PASS' if matches expected pattern, else empty
        
        Sorting:
        - Rows are sorted by: registration pair order (from CSV), modality, folder_name
        - This preserves the EXACT order from the input CSV file
        - Row 1 in CSV → regpair01 (initial then followup)
        - Row 2 in CSV → regpair02 (initial then followup)
        - Maintains order even for same patient with multiple registration pairs
        
        Example rows:
        study_id                       | patient_id | study_date  | targets | modality  | series_description    | selected | rtstruct_validated
        -------------------------------|------------|-------------|---------|-----------|----------------------|----------|-------------------
        0871_regpair01_initial_1997-07 | 0871       | 1997-07-18  | 01      | MR        | T1_axial_post_gad    |          |
        0871_regpair01_initial_1997-07 | 0871       | 1997-07-18  | 01      | RTSTRUCT  | Brain_MS_Init_Model  | x        | PASS
        0871_regpair01_followup_1999-04| 0871       | 1999-04-02  | 01      | MR        | T1_axial_post_gad    |          |
        0944_regpair01_initial_1998-11 | 0944       | 1998-11-12  | 13      | MR        | T1_axial_post_gad    |          |   ← 0944's first pair
        0944_regpair01_followup_2000-07| 0944       | 2000-07-28  | 13      | MR        | T1_axial_post_gad    |          |
        0979_regpair01_initial_1998-11 | 0979       | 1998-11-23  | 03      | MR        | T1_axial_post_gad    |          |   ← 0979's first pair
        0979_regpair02_initial_1998-11 | 0979       | 1998-11-23  | 02      | MR        | T1_axial_post_gad    |          |   ← 0979's second pair
        ...
        3126_regpair01_initial_2009-08 | 3126       | 2009-08-28  | 02/05   | MR        | T1_axial_post_gad    |          |   ← 3126's first pair
        3126_regpair01_followup_2010-03| 3126       | 2010-03-18  | 02/05   | MR        | T1_axial_post_gad    |          |
        3126_regpair02_initial_2009-08 | 3126       | 2009-08-28  | 06/07/..| MR        | T1_axial_post_gad    |          |   ← 3126's second pair
        3126_regpair02_followup_2011-01| 3126       | 2011-01-07  | 06/07/..| MR        | T1_axial_post_gad    |          |
        3126_regpair03_initial_2009-12 | 3126       | 2009-12-23  | 13/14/..| MR        | T1_axial_post_gad    |          |   ← 3126's third pair
        3126_regpair04_initial_2010-03 | 3126       | 2010-03-18  | 18/19/..| MR        | T1_axial_post_gad    |          |   ← 3126's fourth pair
    
    Raises:
        FileNotFoundError: If csv_path or base_folder doesn't exist
        ValueError: If CSV is missing required columns
        
    Example:
        >>> # Basic usage
        >>> inventory_path = generate_study_folder_inventory(
        ...     csv_path="./excel/output/prep1_step1_summary_by_registrationpair.csv",
        ...     base_folder="/database/brainmets/dicom",
        ...     output_path="./excel/output/prep1_step1_study_folder_inventory.xlsx"
        ... )
        >>> print(f"Inventory saved to: {inventory_path}")
        ✓ Processing 33 unique study folders...
        ✓ Found 145 MR series, 38 RTStruct series, 33 RTDose series
        ✓ Inventory saved to CSV: ./excel/output/prep1_step1_study_folder_inventory.csv
        ✓ Inventory saved to Excel: ./excel/output/prep1_step1_study_folder_inventory.xlsx
        
        >>> # With optional metadata
        >>> inventory_path = generate_study_folder_inventory(
        ...     csv_path="./excel/output/prep1_step1_summary_by_registrationpair.csv",
        ...     base_folder="/database/brainmets/dicom",
        ...     output_path="./excel/output/prep1_step1_study_folder_inventory.xlsx",
        ...     include_metadata=True,
        ...     validate_rtstruct=True,
        ...     expected_rtstruct_patterns=["Brain_MS_Init_Model", "Brain_MS_ReTx_Model"]
        ... )
    
    Notes:
        - Processing time scales with number of study folders and series per folder
        - With include_metadata=True, expect ~1-2 seconds per study folder
        - Study folders that don't exist are skipped with a warning
        - Empty folders (no MR/RTStruct/RTDose) are skipped silently
        - The 'selected' column:
            * Auto-populated with 'x' for RTStruct that match validation patterns
            * Intentionally left empty for all other rows (manual entry required)
        - patient_id format: Always 4-digit string with leading zeros (0871, 0944, etc.)
            * Automatically zero-padded if needed (871 → 0871)
            * Preserved from input CSV with dtype specification
        - study_id format: {patient_id}_regpair{nn}_{initial|followup}_{YYYY-MM}
            * regpair number resets for each patient (each patient starts from 01)
            * Example: 0871_regpair01, 0944_regpair01, 3126_regpair01, 3126_regpair02
            * Same patient can have multiple regpair numbers for different date combinations
        - Order follows the input CSV exactly (preserves registration pair sequence)
        - RTStruct validation is case-sensitive
        - Both CSV and Excel files are created; use CSV for viewing, Excel for selection
    """
    # Set default expected patterns if not provided
    if expected_rtstruct_patterns is None:
        expected_rtstruct_patterns = ["Brain_MS_Init_Model", "Brain_MS_ReTx_Model"]
    
    # Validate input paths
    csv_path = Path(csv_path)
    base_folder = Path(base_folder)
    output_path = Path(output_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not base_folder.exists():
        raise FileNotFoundError(f"Base folder not found: {base_folder}")
    
    # Read registration pair CSV (preserve patient_id as string to keep leading zeros)
    df_pairs = pd.read_csv(csv_path, dtype={'patient_id': str})
    
    # Validate required columns
    required_cols = ["patient_id", "regpair", "initial_moving", "followup_fixed", "targets", "n_targets"]
    missing_cols = [col for col in required_cols if col not in df_pairs.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    print(f"✓ Read {len(df_pairs)} registration pairs from {csv_path.name}")
    
    # Build list of all study folders to process, keeping registration pair order
    study_folders_info = []
    
    for idx, row in df_pairs.iterrows():
        patient_id = str(row['patient_id']).zfill(4)  # Ensure 4-digit format with leading zeros
        regpair_num = str(row['regpair']).zfill(2)    # Read regpair from CSV (already formatted)
        targets = row['targets']
        n_targets = row['n_targets']
        
        # Add initial_moving study
        study_folders_info.append({
            'regpair_num': regpair_num,
            'regpair_order': idx * 2,  # For sorting: maintains CSV row order
            'patient_id': patient_id,
            'date': row['initial_moving'],
            'type': 'initial_moving',
            'targets': targets,
            'n_targets': n_targets
        })
        
        # Add followup_fixed study
        study_folders_info.append({
            'regpair_num': regpair_num,
            'regpair_order': idx * 2 + 1,  # For sorting: followup comes second
            'patient_id': patient_id,
            'date': row['followup_fixed'],
            'type': 'followup_fixed',
            'targets': targets,
            'n_targets': n_targets
        })
    
    # Create DataFrame (DO NOT drop duplicates - keep all with their regpair numbers)
    df_studies = pd.DataFrame(study_folders_info)
    
    print(f"✓ Processing {len(df_studies)} study folders (including duplicate dates with different regpair numbers)...")
    
    # Build inventory list
    inventory_rows = []
    
    for _, study in df_studies.iterrows():
        regpair_num = study['regpair_num']
        regpair_order = study['regpair_order']
        patient_id = study['patient_id']
        date_iso = study['date']
        study_type = study['type']
        targets = study['targets']
        n_targets = study['n_targets']
        
        # Build study folder path
        study_folder = path_utils.build_study_folder_path(
            str(base_folder),
            patient_id,
            date_iso
        )
        
        if not study_folder.exists():
            print(f"WARNING: Study folder not found: {study_folder}")
            continue
        
        # Build study_id: patient_regpairNN_type_YYYY-MM
        # Example: "3126_regpair01_initial_2009-08"
        date_folder = path_utils.convert_iso_date_to_folder_format(date_iso)  # YYYY-MM
        study_type_short = 'initial' if 'initial' in study_type else 'followup'
        study_id = f"{patient_id}_regpair{regpair_num}_{study_type_short}_{date_folder}"
        
        # Find all MR folders
        mr_folders = path_utils.find_all_mr_folders(study_folder)
        for mr_folder in mr_folders:
            row_data = {
                'regpair_order': regpair_order,  # For sorting
                'study_id': study_id,
                'patient_id': patient_id,
                'study_date': date_iso,
                'study_type': study_type,
                'targets': targets,
                'n_targets': n_targets,
                'modality': 'MR',
                'folder_name': mr_folder.name,
                'series_description': dicom_utils.read_series_description(mr_folder),
                'file_count': path_utils.count_dicom_files(mr_folder),
                'selected': ''  # Empty for manual entry
            }
            
            # Add optional metadata
            if include_metadata:
                metadata = _read_optional_metadata(mr_folder)
                row_data.update(metadata)
            
            inventory_rows.append(row_data)
        
        # Find all RTStruct folders
        rtstruct_folders = path_utils.find_all_rtstruct_folders(study_folder)
        for rtstruct_folder in rtstruct_folders:
            series_desc = dicom_utils.read_series_description(rtstruct_folder)
            
            # Check if RTStruct is validated
            is_valid = False
            if validate_rtstruct:
                is_valid = any(pattern in series_desc for pattern in expected_rtstruct_patterns)
            
            row_data = {
                'regpair_order': regpair_order,  # For sorting
                'study_id': study_id,
                'patient_id': patient_id,
                'study_date': date_iso,
                'study_type': study_type,
                'targets': targets,
                'n_targets': n_targets,
                'modality': 'RTSTRUCT',
                'folder_name': rtstruct_folder.name,
                'series_description': series_desc,
                'file_count': path_utils.count_dicom_files(rtstruct_folder),
                'selected': 'x' if is_valid else ''  # Auto-select if validated
            }
            
            # Add RTStruct validation column
            if validate_rtstruct:
                row_data['rtstruct_validated'] = 'PASS' if is_valid else ''
            
            # Add optional metadata
            if include_metadata:
                metadata = _read_optional_metadata(rtstruct_folder)
                row_data.update(metadata)
            
            inventory_rows.append(row_data)
        
        # Find all RTDose folders
        rtdose_folders = path_utils.find_all_rtdose_folders(study_folder)
        for rtdose_folder in rtdose_folders:
            row_data = {
                'regpair_order': regpair_order,  # For sorting
                'study_id': study_id,
                'patient_id': patient_id,
                'study_date': date_iso,
                'study_type': study_type,
                'targets': targets,
                'n_targets': n_targets,
                'modality': 'RTDOSE',
                'folder_name': rtdose_folder.name,
                'series_description': dicom_utils.read_series_description(rtdose_folder),
                'file_count': path_utils.count_dicom_files(rtdose_folder),
                'selected': ''
            }
            
            # Add optional metadata
            if include_metadata:
                metadata = _read_optional_metadata(rtdose_folder)
                row_data.update(metadata)
            
            inventory_rows.append(row_data)
    
    # Create DataFrame
    df_inventory = pd.DataFrame(inventory_rows)
    
    if len(df_inventory) == 0:
        print("WARNING: No DICOM folders found in any study folder!")
        # Create empty DataFrame with expected columns
        df_inventory = pd.DataFrame(columns=[
            'regpair_order', 'study_id', 'patient_id', 'study_date', 'study_type', 'targets', 'n_targets',
            'modality', 'folder_name', 'series_description', 'file_count', 'selected'
        ])
    else:
        # Sort by regpair_order (maintains CSV row order), then modality, then folder_name
        # This ensures we follow the exact order from the input CSV file
        df_inventory = df_inventory.sort_values(
            by=['regpair_order', 'modality', 'folder_name'],
            ignore_index=True
        )
        
        # Drop the regpair_order column after sorting (internal use only)
        df_inventory = df_inventory.drop(columns=['regpair_order'])
        
        # Print summary
        mr_count = len(df_inventory[df_inventory['modality'] == 'MR'])
        rtstruct_count = len(df_inventory[df_inventory['modality'] == 'RTSTRUCT'])
        rtdose_count = len(df_inventory[df_inventory['modality'] == 'RTDOSE'])
        
        print(f"✓ Found {mr_count} MR series, {rtstruct_count} RTStruct series, {rtdose_count} RTDose series")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate CSV path (replace .xlsx with .csv)
    csv_path = output_path.with_suffix('.csv')
    
    # Save to CSV (for easier viewing and command-line processing)
    df_inventory.to_csv(csv_path, index=False)
    print(f"✓ Inventory saved to CSV: {csv_path}")
    
    # Save to Excel (for manual selection with better formatting)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_inventory.to_excel(writer, index=False, sheet_name='Inventory')
        
        # Optional: Auto-adjust column widths
        worksheet = writer.sheets['Inventory']
        for idx, col in enumerate(df_inventory.columns, 1):
            max_length = max(
                df_inventory[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(64 + idx)].width = min(max_length + 2, 50)
    
    print(f"✓ Inventory saved to Excel: {output_path}")
    
    return str(output_path)


def _read_optional_metadata(folder_path: Path) -> dict:
    """
    Read optional DICOM metadata (dimensions, orientation, acquisition date).
    
    Helper function for generate_study_folder_inventory() to extract additional
    metadata fields when include_metadata=True.
    
    Args:
        folder_path (Path): Path to DICOM folder
    
    Returns:
        dict: Dictionary with keys:
            - 'image_dimensions': String like '512x512x47' or 'N/A'
            - 'orientation': String like 'Axial', 'Sagittal', 'Coronal', or 'N/A'
            - 'acquisition_date': String like '1997-07-18' or 'N/A'
    """
    metadata = {
        'image_dimensions': 'N/A',
        'orientation': 'N/A',
        'acquisition_date': 'N/A'
    }
    
    if not folder_path.exists():
        return metadata
    
    # Try to read first DICOM file
    for dicom_file in folder_path.iterdir():
        if not dicom_file.is_file():
            continue
        
        try:
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
            
            # Get image dimensions (for image series only)
            if ds.Modality in ['MR', 'CT']:
                if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                    # Count number of files for 3rd dimension
                    n_slices = path_utils.count_dicom_files(folder_path)
                    metadata['image_dimensions'] = f"{ds.Rows}x{ds.Columns}x{n_slices}"
            
            # Get orientation (simplified)
            if hasattr(ds, 'ImageOrientationPatient'):
                # This is a simplified orientation detection
                # For more accurate detection, would need to analyze the orientation vectors
                orientation_str = str(ds.ImageOrientationPatient)
                if 'axial' in str(folder_path).lower():
                    metadata['orientation'] = 'Axial'
                elif 'sagittal' in str(folder_path).lower():
                    metadata['orientation'] = 'Sagittal'
                elif 'coronal' in str(folder_path).lower():
                    metadata['orientation'] = 'Coronal'
                else:
                    metadata['orientation'] = 'Unknown'
            
            # Get acquisition date
            if hasattr(ds, 'AcquisitionDate') and ds.AcquisitionDate:
                # Convert YYYYMMDD to YYYY-MM-DD
                date_str = str(ds.AcquisitionDate)
                if len(date_str) == 8:
                    metadata['acquisition_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            elif hasattr(ds, 'SeriesDate') and ds.SeriesDate:
                # Fallback to SeriesDate
                date_str = str(ds.SeriesDate)
                if len(date_str) == 8:
                    metadata['acquisition_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            # Only read first file
            break
            
        except Exception:
            continue
    
    return metadata

