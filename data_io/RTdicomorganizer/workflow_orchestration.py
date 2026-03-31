"""
==============================================================================
MODULE NAME: workflow_orchestration.py
==============================================================================

PURPOSE:
    This module provides high-level workflow orchestration functions that
    coordinate multiple operations across different modules to accomplish
    complete end-to-end tasks.
    
    Key responsibilities:
    - Orchestrate multi-step processing workflows
    - Coordinate between multiple modules (I/O, path management, visualization)
    - Handle error recovery and reporting
    - Provide progress tracking for long-running operations
    
    These functions represent complete workflows that can be called from
    notebooks or scripts to perform complex multi-step operations.

DEPENDENCIES:
    External packages:
    - pandas: For DataFrame operations
    - sys: For system operations
    - pathlib (Path): For path handling
    
    Internal modules:
    - path_utils: For folder path construction and discovery
    - data_io: For reading CSV and generating reports
    - data_parser: For target format conversion
    - dicom_utils: For RTStruct verification and DICOM overview
    - visualization: For displaying images with contours
    - formatting_utils: For console output formatting

FUNCTIONS:
    1. process_all_lesions(csv_input: str, base_folder: str,
                          overview_script_path: str, log_file: str) -> None
       Main workflow to process all lesions and generate DICOM overviews
       
    2. process_and_display_lesion_v1(row: pd.Series, base_folder: str,
                                     debug: bool = False) -> None
       Process and display MR images with RTStruct overlays for single lesion (Version 1)
       Uses DataFrame row input and auto-discovers DICOM folders
       
    3. process_and_display_lesion_v2(lesion_label: str, study_type: str,
                                     inventory_excel_path: str, base_folder: str,
                                     debug: bool = False) -> None
       Process and display MR/RTStruct/RTDose using pre-validated inventory file (Version 2)
       Uses manual selection from inventory Excel and creates multi-row visualization
       
    4. process_and_display_lesion (alias for process_and_display_lesion_v1)
       Backward compatibility alias

USAGE EXAMPLE:
    ```python
    from code.RTdicomorganizer import workflow_orchestration
    
    # Process all lesions with DICOM overview
    workflow_orchestration.process_all_lesions(
        csv_input="./output/prep1_step1_summary_by_lesion.csv",
        base_folder="/database/brainmets/dicom/organized",
        overview_script_path="/path/to/overview_script.py",
        log_file="./output/processing.log"
    )
    
    # Display single lesion for QC (Version 1 - auto-discover)
    import pandas as pd
    df = pd.read_csv("./output/summary_by_lesion.csv")
    row = df.iloc[0]
    workflow_orchestration.process_and_display_lesion_v1(
        row,
        base_folder="/database/brainmets/dicom/organized",
        debug=True
    )
    
    # Display single lesion for QC (Version 2 - use inventory)
    workflow_orchestration.process_and_display_lesion_v2(
        lesion_label='0871.01',
        study_type='both',
        inventory_excel_path='./output/prep1_step1_study_folder_inventory_manual_check_wip.xlsx',
        base_folder="/database/brainmets/dicom/organized",
        debug=False
    )
    ```

NOTES:
    - These functions are designed to be called from Jupyter notebooks
    - Progress information is printed to console
    - Errors are handled gracefully to allow batch processing to continue
    - Log files capture all output for later review

==============================================================================
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import internal modules
from . import path_utils
from . import data_io
from . import data_parser
from . import dicom_utils
from . import visualization
from . import formatting_utils


def process_all_lesions(csv_input: str, base_folder: str,
                        overview_script_path: str, log_file: str) -> None:
    """
    Main workflow to process all lesions and generate DICOM overviews.
    
    DETAILED DESCRIPTION:
        This function orchestrates a complete multi-lesion processing workflow:
        
        WORKFLOW STEPS:
        1. Set up output redirection to both console and log file
        2. Validate input paths (CSV, base folder, overview script)
        3. Read lesion summary CSV
        4. For each lesion:
           a. Build initial and follow-up folder paths
           b. Verify folder existence
           c. Run DICOM overview script on both folders
           d. Display and log results
        5. Generate summary statistics
        6. Report success/failure counts
        
        This is typically the second step in the processing pipeline (prep1_step2),
        after the lesion summary CSV has been generated.
    
    Args:
        csv_input (str): Path to per-lesion summary CSV file
            - Generated by data_analysis.build_summaries()
            - Example: './output/prep1_step1_summary_by_lesion.csv'
            
        base_folder (str): Base directory containing all patient DICOM data
            - Should contain SRSPPPP subfolders
            - Example: '/database/brainmets/dicom/organized'
            
        overview_script_path (str): Path to DICOM overview Python script
            - External script that analyzes DICOM folder structure
            - Example: '/path/to/overview_all_dicom_folder.py'
            
        log_file (str): Path for output log file
            - All console output will be duplicated here
            - Example: './output/prep1_step2_overview_all_pairs.log'
    
    Returns:
        None: Prints progress to console/log, no return value
    
    CONSOLE OUTPUT:
        Prints formatted progress including:
        - Total lesions to process
        - For each lesion:
          * Lesion number and identifier
          * Initial and follow-up dates
          * Folder paths
          * DICOM overview output for both timepoints
        - Summary statistics (successful, failed, missing folders)
    
    Example:
        >>> from code.RTdicomorganizer import workflow_orchestration
        
        >>> workflow_orchestration.process_all_lesions(
        ...     csv_input="./output/prep1_step1_summary_by_lesion.csv",
        ...     base_folder="/database/brainmets/dicom/organized",
        ...     overview_script_path="/path/to/overview_script.py",
        ...     log_file="./output/overview_log.txt"
        ... )
        Output is being saved to: ./output/overview_log.txt
        ===================================...
        DICOM OVERVIEW FOR ALL REGISTRATION PAIRS
        ===================================...
        ✓ Found 46 lesions to process
        ...
        SUMMARY
        -----------------------------------...
        Total lesions processed: 46
        ✓ Successful: 46
        ❌ Failed: 0
    
    ERROR HANDLING:
        - Missing folders: Logged and skipped, processing continues
        - Script failures: Logged and marked as failed, processing continues
        - Timeouts: Logged with timeout message, processing continues
        - Final summary shows counts of successful vs failed
    
    Notes:
        - Output is duplicated to both console and log file
        - Each DICOM overview has 60-second timeout
        - Progress is shown as "N/M" format (e.g., "23/46")
        - Uses formatting_utils.TeeOutput for dual output
        - Automatically restores stdout when complete
    """
    # Set up output redirection to log file
    tee = formatting_utils.TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print(f"Output is being saved to: {log_file}")
        
        formatting_utils.print_section_header("DICOM OVERVIEW FOR ALL REGISTRATION PAIRS")
        
        # Validate paths
        if not Path(overview_script_path).exists():
            print(f"❌ Error: Overview script not found at: {overview_script_path}")
            return
        
        if not Path(csv_input).exists():
            print(f"❌ Error: CSV file not found at: {csv_input}")
            return
        
        # Read CSV (preserve patient_id and target as strings)
        df = pd.read_csv(csv_input, dtype={'patient_id': str, 'target': str})
        df['patient_id'] = df['patient_id'].astype(str).str.zfill(4)
        df['target'] = df['target'].astype(str).str.zfill(2)
        
        print(f"\n✓ Found {len(df)} lesions to process")
        print(f"✓ Base folder: {base_folder}")
        print(f"✓ Overview script: {overview_script_path}")
        
        # Process each lesion
        total_lesions = len(df)
        successful = 0
        failed = 0
        missing_folders = []
        
        for idx, row in df.iterrows():
            lesion_num = idx + 1
            patient_id = row['patient_id']
            target = row['target']
            lesion_label = row['lesion_label']
            initial_date = row['initial_moving']
            followup_date = row['followup_fixed']
            
            # Print lesion header
            formatting_utils.print_section_header(
                f"LESION {lesion_num}/{total_lesions}: {lesion_label} "
                f"(Patient {patient_id}, Target {target})"
            )
            
            print(f"\nInitial (moving):  {initial_date}")
            print(f"Followup (fixed):  {followup_date}")
            
            # Build folder paths
            initial_folder = path_utils.build_study_folder_path(
                base_folder, patient_id, initial_date
            )
            followup_folder = path_utils.build_study_folder_path(
                base_folder, patient_id, followup_date
            )
            
            print(f"\nInitial folder:  {initial_folder}")
            print(f"Followup folder: {followup_folder}")
            
            # Check if folders exist
            initial_exists = initial_folder.exists()
            followup_exists = followup_folder.exists()
            
            if not initial_exists:
                print(f"\n⚠️  Warning: Initial folder does not exist!")
                missing_folders.append((lesion_label, "initial", initial_folder))
            
            if not followup_exists:
                print(f"⚠️  Warning: Followup folder does not exist!")
                missing_folders.append((lesion_label, "followup", followup_folder))
            
            if not (initial_exists and followup_exists):
                failed += 1
                print(f"\n⏭️  Skipping this lesion due to missing folders\n")
                continue
            
            # Run overview for INITIAL scan
            formatting_utils.print_subsection_header("📁 INITIAL (MOVING) SCAN OVERVIEW")
            
            success, output, error = dicom_utils.run_overview_script(
                initial_folder, 
                overview_script_path
            )
            
            if success:
                print(output)
                if error:
                    print(f"\nWarnings/Notes:\n{error}")
            else:
                print(f"❌ Error running overview script:")
                print(f"   {error}")
                if output:
                    print(f"\nPartial output:\n{output}")
                failed += 1
                continue
            
            # Run overview for FOLLOWUP scan
            formatting_utils.print_subsection_header("📁 FOLLOWUP (FIXED) SCAN OVERVIEW")
            
            success, output, error = dicom_utils.run_overview_script(
                followup_folder,
                overview_script_path
            )
            
            if success:
                print(output)
                if error:
                    print(f"\nWarnings/Notes:\n{error}")
                successful += 1
            else:
                print(f"❌ Error running overview script:")
                print(f"   {error}")
                if output:
                    print(f"\nPartial output:\n{output}")
                failed += 1
            
            # Spacing between lesions
            print("\n" + "░" * 100 + "\n")
        
        # Print summary
        formatting_utils.print_section_header("SUMMARY")
        print(f"\nTotal lesions processed: {total_lesions}")
        print(f"✓ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if missing_folders:
            print(f"\n⚠️  Missing folders ({len(missing_folders)}):")
            for lesion_label, scan_type, folder_path in missing_folders:
                print(f"   {lesion_label} ({scan_type}): {folder_path}")
        
        print("\n" + "=" * 100)
        
    finally:
        # Restore stdout and close log file
        sys.stdout = tee.terminal
        tee.close()
        print(f"\n✓ Log file saved to: {log_file}")


def process_and_display_lesion_v1(row: pd.Series, base_folder: str,
                                  debug: bool = False) -> None:
    """
    Process and display MR images with RTStruct overlays for a single lesion (Version 1).
    
    DETAILED DESCRIPTION:
        This function provides a complete visualization workflow for a single
        lesion, showing all MR series from both initial and follow-up timepoints
        with overlaid target contours.
        
        WORKFLOW STEPS:
        1. Extract lesion information from row
        2. Convert target to RTStruct format
        3. For INITIAL timepoint:
           a. Build study folder path
           b. Find RTStruct folder with "Brain.MS.Init.Model"
           c. Verify RTStruct series description
           d. Load RTStruct file
           e. Find all MR folders
           f. Display each MR series with target contour
        4. For FOLLOWUP timepoint:
           a. Build study folder path
           b. Find RTStruct folder with "Brain.MS.ReTx.Model"
           c. Verify RTStruct series description
           d. Load RTStruct file
           e. Find all MR folders
           f. Display each MR series with target contour
        
        This is typically used in Jupyter notebooks for interactive QC.
    
    Args:
        row (pd.Series): Row from lesion summary DataFrame
            - Required columns: patient_id, target, lesion_label,
              initial_moving, followup_fixed
            - Typically obtained via df.iloc[index]
            
        base_folder (str): Base directory containing all patient DICOM data
            - Should contain SRSPPPP subfolders
            - Example: '/database/brainmets/dicom/organized'
            
        debug (bool): If True, print detailed diagnostic information
            - Default: False
            - Shows slice selection details, coordinate transformations
    
    Returns:
        None: Creates and displays matplotlib figures for each MR series
    
    INPUT ROW FORMAT:
        Required columns:
        - patient_id (str): '0871', '1885'
        - target (str): '01', '09'
        - lesion_label (str): '0871.01', '1885.09'
        - initial_moving (str): '1997-07-18'
        - followup_fixed (str): '1999-04-02'
    
    Example:
        >>> import pandas as pd
        >>> from code.RTdicomorganizer import workflow_orchestration
        
        >>> # Load lesion data
        >>> df = pd.read_csv('./output/summary_by_lesion.csv')
        
        >>> # Process first lesion
        >>> row = df.iloc[0]
        >>> workflow_orchestration.process_and_display_lesion(
        ...     row,
        ...     base_folder='/database/brainmets/dicom/organized',
        ...     debug=True
        ... )
        ===================================...
        LESION: 0871.01
        ===================================...
        Patient ID: 0871
        Target: 01 (RTStruct name: target1)
        ...
        [Displays matplotlib figures for each MR series]
    
    CONSOLE OUTPUT:
        Prints:
        - Lesion header with patient ID and target
        - Initial and follow-up dates
        - Study folder paths
        - RTStruct folder names
        - Available structures in each RTStruct
        - Number of MR series found
        - For each MR series:
          * Series name
          * Debug information (if debug=True)
          * Image loading status
        
        Displays:
        - One matplotlib figure per MR series
        - Each figure shows center slice with target contour
    
    ERROR HANDLING:
        - Missing folders: Prints error and continues to next timepoint
        - Missing RTStruct: Prints error and skips timepoint
        - RTStruct verification fails: Prints warning and continues
        - Image loading fails: Prints error and skips that series
        - Missing target: Returns middle slice with warning
    
    Notes:
        - Requires DICOMImageReader and RTStructReader classes
        - Must be imported from dicomreader package (not included here)
        - Displays figures with plt.show() - blocks until closed
        - Debug mode shows detailed slice selection information
        - Patient IDs automatically zero-padded to 4 digits
    """
    # Extract lesion information
    patient_id = row['patient_id']
    target = row['target']
    lesion_label = row['lesion_label']
    initial_date = row['initial_moving']
    followup_date = row['followup_fixed']
    
    # Ensure proper formatting
    patient_id_str = str(patient_id).zfill(4)
    target_str = str(target).zfill(2)
    
    # Convert target to RTStruct format (e.g., '01' -> 'target1')
    structure_name = data_parser.convert_target_to_rtstruct_format(target_str)
    
    print("=" * 100)
    print(f"LESION: {lesion_label}".center(100))
    print("=" * 100)
    print(f"Patient ID: {patient_id_str}")
    print(f"Target: {target_str} (RTStruct name: {structure_name})")
    print(f"Initial date: {initial_date}")
    print(f"Follow-up date: {followup_date}")
    print()
    
    # Import DICOM readers (required but not part of this package)
    try:
        from dicomreader.DICOMImageReader import DICOMImageReader
        from dicomreader.RTStructReader import RTStructReader
    except ImportError as e:
        print(f"❌ Error: Could not import DICOM readers: {e}")
        print("Please ensure dicomreader package is available in sys.path")
        return
    
    # ===================================================================
    # INITIAL (MOVING) TIMEPOINT
    # ===================================================================
    print("-" * 100)
    print("INITIAL (MOVING) TIMEPOINT")
    print("-" * 100)
    
    try:
        # Build study folder path
        init_study_folder = path_utils.build_study_folder_path(
            base_folder, patient_id_str, initial_date
        )
        print(f"Study folder: {init_study_folder}")
        
        # Find RTStruct folder
        init_rtstruct_folder = path_utils.find_rtstruct_folder(
            init_study_folder, "Brain.MS.Init.Model"
        )
        print(f"RTStruct folder: {init_rtstruct_folder.name}")
        
        # Verify series description
        is_valid = dicom_utils.verify_rtstruct_series_description(
            init_rtstruct_folder, "Brain_MS_Init_Model"
        )
        if not is_valid:
            print("WARNING: Series description verification failed!")
        
        # Load RTStruct
        init_rtstruct_reader = RTStructReader(str(init_rtstruct_folder))
        init_rtstruct_reader.read()
        print(f"RTStruct loaded successfully")
        print(f"Available structures: {init_rtstruct_reader.get_structure_names()}")
        
        # Find all MR folders
        init_mr_folders = path_utils.find_all_mr_folders(init_study_folder)
        print(f"Found {len(init_mr_folders)} MR series")
        print()
        
        # Display each MR series with contour overlay
        for idx, mr_folder in enumerate(init_mr_folders, 1):
            print(f"Loading MR series {idx}/{len(init_mr_folders)}: {mr_folder.name}")
            
            try:
                # Load MR image
                mr_reader = DICOMImageReader(str(mr_folder), modality='MR')
                mr_reader.read()
                
                # Display with contour overlay
                title = f"{lesion_label} - Initial - MR Series {idx}\n{mr_folder.name}"
                visualization.display_mr_with_contour(
                    mr_reader, 
                    init_rtstruct_reader, 
                    structure_name, 
                    title
                )
                
            except Exception as e:
                print(f"ERROR loading/displaying MR series: {e}")
            
            print()
    
    except Exception as e:
        print(f"ERROR processing initial timepoint: {e}")
        print()
    
    # ===================================================================
    # FOLLOW-UP (FIXED) TIMEPOINT
    # ===================================================================
    print("-" * 100)
    print("FOLLOW-UP (FIXED) TIMEPOINT")
    print("-" * 100)
    
    try:
        # Build study folder path
        followup_study_folder = path_utils.build_study_folder_path(
            base_folder, patient_id_str, followup_date
        )
        print(f"Study folder: {followup_study_folder}")
        
        # Find RTStruct folder
        followup_rtstruct_folder = path_utils.find_rtstruct_folder(
            followup_study_folder, "Brain.MS.ReTx.Model"
        )
        print(f"RTStruct folder: {followup_rtstruct_folder.name}")
        
        # Verify series description
        is_valid = dicom_utils.verify_rtstruct_series_description(
            followup_rtstruct_folder, "Brain_MS_ReTx_Model"
        )
        if not is_valid:
            print("WARNING: Series description verification failed!")
        
        # Load RTStruct
        followup_rtstruct_reader = RTStructReader(str(followup_rtstruct_folder))
        followup_rtstruct_reader.read()
        print(f"RTStruct loaded successfully")
        print(f"Available structures: {followup_rtstruct_reader.get_structure_names()}")
        
        # Find all MR folders
        followup_mr_folders = path_utils.find_all_mr_folders(followup_study_folder)
        print(f"Found {len(followup_mr_folders)} MR series")
        print()
        
        # Display each MR series with contour overlay
        for idx, mr_folder in enumerate(followup_mr_folders, 1):
            print(f"Loading MR series {idx}/{len(followup_mr_folders)}: {mr_folder.name}")
            
            try:
                # Load MR image
                mr_reader = DICOMImageReader(str(mr_folder), modality='MR')
                mr_reader.read()
                
                # Display with contour overlay
                title = f"{lesion_label} - Follow-up - MR Series {idx}\n{mr_folder.name}"
                visualization.display_mr_with_contour(
                    mr_reader,
                    followup_rtstruct_reader,
                    structure_name,
                    title
                )
                
            except Exception as e:
                print(f"ERROR loading/displaying MR series: {e}")
            
            print()
    
    except Exception as e:
        print(f"ERROR processing follow-up timepoint: {e}")
        print()
    
    print("=" * 100)
    print()


# Backward compatibility alias
process_and_display_lesion = process_and_display_lesion_v1


def process_and_display_lesion_v2(
    lesion_label: str,
    study_type: str,
    inventory_excel_path: str,
    base_folder: str,
    debug: bool = False
) -> None:
    """
    Process and display MR images with RTStruct and RTDose overlays using inventory file (Version 2).
    
    DETAILED DESCRIPTION:
        This function provides an enhanced visualization workflow that uses a
        pre-validated inventory Excel file to determine which DICOM series to load.
        It supports displaying both initial and follow-up timepoints based on
        user selection.
        
        WORKFLOW STEPS:
        1. Parse lesion_label to extract patient_id and target
        2. Load inventory Excel file
        3. Filter inventory for matching patient_id and target
        4. Validate that exactly 1 MR, 1 RTSTRUCT, and 1 RTDOSE are selected
           for each requested timepoint
        5. Extract folder names from inventory
        6. Load DICOM data (MR, RTSTRUCT, RTDOSE)
        7. Extract unique Z-coordinates from RTStruct contours
        8. Create multi-row visualization:
           - Row 1: MR slices (image only)
           - Row 2: MR slices + RTStruct points
           - Row 3: MR slices + RTStruct contours
           - Row 4: RTDose slices (placeholder for now)
           - Row 5: MR + RTDose overlay (placeholder for now)
        
        This is designed for quality control and detailed inspection of
        individual lesions using manually curated selections.
    
    Args:
        lesion_label (str): Lesion identifier in format 'PPPP.TT'
            - PPPP: 4-digit patient ID (e.g., '0871', '1885')
            - TT: 2-digit target ID (e.g., '01', '09')
            - Example: '0871.01', '1885.09', '3126.15'
            
        study_type (str): Which timepoint(s) to display
            - 'initial_moving': Display only initial timepoint
            - 'followup_fixed': Display only follow-up timepoint
            - 'both': Display both timepoints
            - Case-insensitive
            
        inventory_excel_path (str): Path to inventory Excel file
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
            - Shows data loading details, coordinate information
    
    Returns:
        None: Creates and displays matplotlib figures with multi-row layouts
    
    Raises:
        FileNotFoundError: If inventory Excel file doesn't exist
        ValueError: If lesion_label format is invalid
        ValueError: If study_type is not valid
        ValueError: If selections in inventory are invalid (not exactly 1 per modality)
        KeyError: If target is not found in inventory
    
    INVENTORY EXCEL FORMAT:
        Required columns:
        - study_id (str): Format 'PPPP_regpairNN_type_YYYY-MM'
          Example: '0871_regpair01_initial_1997-07'
          
        - patient_id (str): 4-digit patient ID
          Example: '0871'
          
        - study_date (str): ISO format date (YYYY-MM-DD)
          Example: '1997-07-18'
          
        - study_type (str): 'initial_moving' or 'followup_fixed'
          
        - targets (str): Slash-separated target IDs
          Example: '01', '01/02/03'
          
        - modality (str): DICOM modality
          Options: 'MR', 'RTSTRUCT', 'RTDOSE'
          
        - folder_name (str): DICOM folder name
          Example: '44-MR-3_Plane_Loc_TCORONAL_POST_GAD-73066'
          
        - selected (str): Selection marker
          'x' = selected, empty = not selected
    
    Example:
        >>> from code.RTdicomorganizer import workflow_orchestration
        
        >>> # Display both timepoints for lesion 0871.01
        >>> workflow_orchestration.process_and_display_lesion_v2(
        ...     lesion_label='0871.01',
        ...     study_type='both',
        ...     inventory_excel_path='./excel/output/prep1_step1_study_folder_inventory_manual_check_wip.xlsx',
        ...     base_folder='/database/brainmets/dicom/organized',
        ...     debug=False
        ... )
        ========================================
        LESION: 0871.01
        ========================================
        Patient ID: 0871
        Target: 01
        Study type: both
        
        ✓ Loaded inventory: 521 rows
        ✓ Filtered for patient 0871, target 01: 12 rows
        ✓ Validated selections for initial_moving
        ✓ Validated selections for followup_fixed
        
        Selected folders (initial_moving):
          MR:       44-MR-3_Plane_Loc_TCORONAL_POST_GAD-73066
          RTSTRUCT: 123-RTSTRUCT-Brain_MS_Init_Model-12345
          RTDOSE:   124-RTDOSE-dose_plan-12346
        
        [Displays multi-row matplotlib figure]
        
        >>> # Display only follow-up timepoint
        >>> workflow_orchestration.process_and_display_lesion_v2(
        ...     lesion_label='1885.09',
        ...     study_type='followup_fixed',
        ...     inventory_excel_path='./excel/output/inventory.xlsx',
        ...     base_folder='/database/dicom'
        ... )
    
    CONSOLE OUTPUT:
        Prints:
        - Lesion header with patient ID and target
        - Study type requested
        - Inventory loading status
        - Filtering results
        - Validation results (pass/fail for each timepoint)
        - Selected folder names for each modality
        - Number of Z-slices found in RTStruct
        - Plotting progress
        
        ERROR MESSAGES (if validation fails):
        - If not exactly 1 MR selected: Lists all selected MR folders
        - If not exactly 1 RTSTRUCT selected: Lists all selected RTSTRUCT folders
        - If not exactly 1 RTDOSE selected: Lists all selected RTDOSE folders
        - If target not found in targets column: Shows available targets
    
    Notes:
        - Requires DICOMImageReader and RTStructReader classes from dicomreader package
        - Target matching: Checks if target appears anywhere in targets column
          (handles both '01' and '01/02/03' formats)
        - Rows 4 and 5 (dose-related) are placeholders for future implementation
        - The function validates before loading any DICOM data to fail fast
        - Patient IDs and targets are automatically zero-padded if needed
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # ===================================================================
    # Step 1: Parse lesion_label
    # ===================================================================
    print("=" * 100)
    print(f"LESION: {lesion_label}".center(100))
    print("=" * 100)
    
    try:
        patient_id, target, _ = data_parser.parse_patient_target(lesion_label)
    except Exception as e:
        raise ValueError(
            f"Invalid lesion_label format: '{lesion_label}'\n"
            f"Expected format: 'PPPP.TT' (e.g., '0871.01')\n"
            f"Error: {e}"
        )
    
    print(f"Patient ID: {patient_id}")
    print(f"Target: {target}")
    
    # ===================================================================
    # Step 2: Validate study_type
    # ===================================================================
    study_type_lower = study_type.lower()
    valid_study_types = ['initial_moving', 'followup_fixed', 'both']
    
    if study_type_lower not in valid_study_types:
        raise ValueError(
            f"Invalid study_type: '{study_type}'\n"
            f"Valid options: {valid_study_types}"
        )
    
    print(f"Study type: {study_type_lower}")
    print()
    
    # Determine which timepoints to process
    process_initial = study_type_lower in ['initial_moving', 'both']
    process_followup = study_type_lower in ['followup_fixed', 'both']
    
    # ===================================================================
    # Step 3: Load inventory Excel file
    # ===================================================================
    inventory_path = Path(inventory_excel_path)
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Inventory Excel file not found: {inventory_excel_path}\n"
            f"Please ensure the file exists and path is correct."
        )
    
    print(f"Loading inventory: {inventory_path.name}...")
    df_inventory = pd.read_csv(inventory_path) if inventory_path.suffix == '.csv' else pd.read_excel(inventory_path)
    print(f"✓ Loaded inventory: {len(df_inventory)} rows")
    print()
    
    # ===================================================================
    # Step 4: Filter inventory for matching patient_id and target
    # ===================================================================
    print(f"Filtering for patient {patient_id}, target {target}...")
    
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
        # Show available patient_id/target combinations
        available = df_inventory.groupby(['patient_id', 'targets']).size().reset_index(name='count')
        raise KeyError(
            f"No data found for patient {patient_id}, target {target}\n"
            f"Available combinations:\n{available.to_string()}"
        )
    
    print(f"✓ Filtered: {len(df_filtered)} rows")
    print()
    
    # ===================================================================
    # Step 5: Validate selections for each timepoint
    # ===================================================================
    def validate_and_extract_folders(df: pd.DataFrame, timepoint_type: str) -> dict:
        """
        Validate that exactly 1 MR, 1 RTSTRUCT, and 1 RTDOSE are selected.
        Returns dict with folder names if valid, raises ValueError if not.
        """
        # Filter for this timepoint
        df_tp = df[df['study_type'] == timepoint_type].copy()
        
        if len(df_tp) == 0:
            raise ValueError(
                f"No inventory data found for {timepoint_type}\n"
                f"Available study_types: {df['study_type'].unique().tolist()}"
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
                f"ERROR: Expected exactly 1 MR selected for {timepoint_type}, found {mr_count}\n"
                f"Selected MR folders: {mr_folders if mr_folders else 'None'}\n"
                f"Please update the inventory Excel file."
            )
        
        # Validate RTSTRUCT
        rtstruct_count = selection_counts.get('RTSTRUCT', 0)
        if rtstruct_count != 1:
            rtstruct_folders = df_selected[df_selected['modality'] == 'RTSTRUCT']['folder_name'].tolist()
            raise ValueError(
                f"ERROR: Expected exactly 1 RTSTRUCT selected for {timepoint_type}, found {rtstruct_count}\n"
                f"Selected RTSTRUCT folders: {rtstruct_folders if rtstruct_folders else 'None'}\n"
                f"Please update the inventory Excel file."
            )
        
        # Validate RTDOSE
        rtdose_count = selection_counts.get('RTDOSE', 0)
        if rtdose_count != 1:
            rtdose_folders = df_selected[df_selected['modality'] == 'RTDOSE']['folder_name'].tolist()
            raise ValueError(
                f"ERROR: Expected exactly 1 RTDOSE selected for {timepoint_type}, found {rtdose_count}\n"
                f"Selected RTDOSE folders: {rtdose_folders if rtdose_folders else 'None'}\n"
                f"Please update the inventory Excel file."
            )
        
        # Extract folder names and study_date
        mr_row = df_selected[df_selected['modality'] == 'MR'].iloc[0]
        rtstruct_row = df_selected[df_selected['modality'] == 'RTSTRUCT'].iloc[0]
        rtdose_row = df_selected[df_selected['modality'] == 'RTDOSE'].iloc[0]
        
        return {
            'study_date': mr_row['study_date'],
            'mr_folder': mr_row['folder_name'],
            'rtstruct_folder': rtstruct_row['folder_name'],
            'rtdose_folder': rtdose_row['folder_name']
        }
    
    # Validate and extract folders for requested timepoints
    folders = {}
    
    if process_initial:
        print("Validating selections for initial_moving...")
        folders['initial'] = validate_and_extract_folders(df_filtered, 'initial_moving')
        print("✓ Validated selections for initial_moving")
        print(f"  MR:       {folders['initial']['mr_folder']}")
        print(f"  RTSTRUCT: {folders['initial']['rtstruct_folder']}")
        print(f"  RTDOSE:   {folders['initial']['rtdose_folder']}")
        print()
    
    if process_followup:
        print("Validating selections for followup_fixed...")
        folders['followup'] = validate_and_extract_folders(df_filtered, 'followup_fixed')
        print("✓ Validated selections for followup_fixed")
        print(f"  MR:       {folders['followup']['mr_folder']}")
        print(f"  RTSTRUCT: {folders['followup']['rtstruct_folder']}")
        print(f"  RTDOSE:   {folders['followup']['rtdose_folder']}")
        print()
    
    # ===================================================================
    # Step 6: Load DICOM data and create visualizations
    # ===================================================================
    
    # Import DICOM readers
    try:
        from dicomreader.DICOMImageReader import DICOMImageReader
        from dicomreader.RTStructReader import RTStructReader
    except ImportError as e:
        print(f"❌ Error: Could not import DICOM readers: {e}")
        print("Please ensure dicomreader package is available in sys.path")
        return
    
    # Convert target to RTStruct format
    structure_name = data_parser.convert_target_to_rtstruct_format(target)

    ### DEBUG [DONE]
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("structure_name: ", structure_name)
    
    # Process each timepoint
    for timepoint_key in ['initial', 'followup']:
        if timepoint_key not in folders:
            continue
        
        timepoint_type = 'initial_moving' if timepoint_key == 'initial' else 'followup_fixed'
        timepoint_label = 'INITIAL (MOVING)' if timepoint_key == 'initial' else 'FOLLOW-UP (FIXED)'
        
        print("-" * 100)
        print(timepoint_label)
        print("-" * 100)
        
        folder_info = folders[timepoint_key]
        study_date = folder_info['study_date']
        
        try:
            # Build study folder path
            study_folder = path_utils.build_study_folder_path(
                base_folder, patient_id, study_date
            )
            print(f"Study folder: {study_folder}")
            print()
            
            # Build full paths to DICOM folders
            mr_path = study_folder / folder_info['mr_folder']
            rtstruct_path = study_folder / folder_info['rtstruct_folder']
            rtdose_path = study_folder / folder_info['rtdose_folder']
            
            # Verify folders exist
            if not mr_path.exists():
                raise FileNotFoundError(f"MR folder not found: {mr_path}")
            if not rtstruct_path.exists():
                raise FileNotFoundError(f"RTSTRUCT folder not found: {rtstruct_path}")
            if not rtdose_path.exists():
                raise FileNotFoundError(f"RTDOSE folder not found: {rtdose_path}")
            
            # Load MR image
            print(f"Loading MR: {folder_info['mr_folder']}")
            mr_reader = DICOMImageReader(str(mr_path), modality='MR')
            mr_reader.read()
            print(f"✓ MR loaded: {mr_reader.image.GetSize()}")
            
            # Load RTStruct
            print(f"Loading RTSTRUCT: {folder_info['rtstruct_folder']}")
            rtstruct_reader = RTStructReader(str(rtstruct_path))
            rtstruct_reader.read()
            print(f"✓ RTSTRUCT loaded")
            print(f"  Available structures: {rtstruct_reader.get_structure_names()}")
            
            # Load RTDose
            print(f"Loading RTDOSE: {folder_info['rtdose_folder']}")
            try:
                from dicomreader.RTDoseReader import RTDoseReader
                rtdose_reader = RTDoseReader(str(rtdose_path))
                rtdose_reader.read()
                print(f"✓ RTDOSE loaded: {rtdose_reader.rtdose_image.GetSize()}")
            except Exception as e:
                print(f"⚠ Warning: Could not load RTDOSE: {e}")
                rtdose_reader = None
            print()

            return (mr_path, mr_reader), (rtstruct_path, rtstruct_reader), (rtdose_path, rtdose_reader)
            
            # ===================================================================
            # Step 7: Extract unique Z-coordinates from RTStruct
            # ===================================================================
            print(f"Extracting Z-coordinates for structure: {structure_name}")
            
            # Use the new method to get contour data in physical space
            try:
                contour_slices = rtstruct_reader.get_structure_contour_points_in_physical_space(structure_name)
                
                # Extract all Z coordinates
                z_coords = [contour[0, 2] for contour in contour_slices]
                
                # Get unique Z values and sort
                unique_z = sorted(set(z_coords))
                M = len(unique_z)
                
                print(f"✓ Found {len(contour_slices)} contour slices")
                print(f"✓ Found {M} unique Z-coordinates")
                if M > 0:
                    print(f"  Z range: [{min(unique_z):.2f}, {max(unique_z):.2f}] mm")
                print()
                
            except Exception as e:
                print(f"❌ Error extracting Z-coordinates: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                continue
            
            # ===================================================================
            # Step 8: Create multi-row visualization
            # ===================================================================
            print("Creating multi-row visualization...")
            print(f"  Showing {M} slices (based on RTStruct Z-coordinates)")
            print()
            
            try:
                import matplotlib.pyplot as plt
                import SimpleITK as sitk
                from matplotlib.patches import Polygon
                from matplotlib.collections import LineCollection
                
                # Get image array and properties
                mr_array = sitk.GetArrayFromImage(mr_reader.image)
                mr_spacing = mr_reader.image.GetSpacing()
                mr_origin = mr_reader.image.GetOrigin()
                
                # Get contour points in pixel space
                contours_pixel = rtstruct_reader.get_structure_contour_points_in_pixel_space(
                    structure_name, mr_reader
                )
                
                # Map Z coordinates to slice indices
                z_to_slice_idx = {}
                for z_coord in unique_z:
                    # Find closest slice index
                    slice_idx = int(round((z_coord - mr_origin[2]) / mr_spacing[2]))
                    if 0 <= slice_idx < mr_array.shape[0]:
                        z_to_slice_idx[z_coord] = slice_idx
                
                # Limit to valid slices
                valid_z_coords = [z for z in unique_z if z in z_to_slice_idx]
                num_slices = len(valid_z_coords)
                
                if num_slices == 0:
                    print("⚠ Warning: No valid slices found within image bounds")
                    continue
                
                # Create figure with 5 rows
                fig, axes = plt.subplots(5, num_slices, figsize=(3*num_slices, 15))
                if num_slices == 1:
                    axes = axes.reshape(-1, 1)
                
                # Get dose array if available
                dose_array = None
                if rtdose_reader is not None:
                    dose_array = rtdose_reader.dose_array
                
                # Plot each slice
                for col_idx, z_coord in enumerate(valid_z_coords):
                    slice_idx = z_to_slice_idx[z_coord]
                    mr_slice = mr_array[slice_idx, :, :]
                    
                    # Get contours for this slice
                    slice_contours = contours_pixel.get(slice_idx, [])
                    
                    # Row 1: MR image only
                    axes[0, col_idx].imshow(mr_slice, cmap='gray', origin='lower')
                    axes[0, col_idx].set_title(f'MR Slice {slice_idx}\nZ={z_coord:.1f}mm', fontsize=8)
                    axes[0, col_idx].axis('off')
                    
                    # Row 2: MR + RTStruct points
                    axes[1, col_idx].imshow(mr_slice, cmap='gray', origin='lower')
                    for contour_points in slice_contours:
                        contour_array = np.array(contour_points)
                        axes[1, col_idx].scatter(
                            contour_array[:, 1], contour_array[:, 0],
                            c='red', s=1, alpha=0.8
                        )
                    axes[1, col_idx].set_title('MR + Points', fontsize=8)
                    axes[1, col_idx].axis('off')
                    
                    # Row 3: MR + RTStruct contours
                    axes[2, col_idx].imshow(mr_slice, cmap='gray', origin='lower')
                    for contour_points in slice_contours:
                        contour_array = np.array(contour_points)
                        # Plot as closed polygon
                        axes[2, col_idx].plot(
                            np.append(contour_array[:, 1], contour_array[0, 1]),
                            np.append(contour_array[:, 0], contour_array[0, 0]),
                            'r-', linewidth=1.5, alpha=0.8
                        )
                    axes[2, col_idx].set_title('MR + Contours', fontsize=8)
                    axes[2, col_idx].axis('off')
                    
                    # Row 4: RTDose slice
                    if dose_array is not None and slice_idx < dose_array.shape[0]:
                        dose_slice = dose_array[slice_idx, :, :]
                        im = axes[3, col_idx].imshow(
                            dose_slice, cmap='jet', origin='lower',
                            vmin=0, vmax=np.percentile(dose_array, 99)
                        )
                        axes[3, col_idx].set_title('Dose', fontsize=8)
                        axes[3, col_idx].axis('off')
                    else:
                        axes[3, col_idx].text(
                            0.5, 0.5, 'Dose\nNot Available',
                            ha='center', va='center', fontsize=10
                        )
                        axes[3, col_idx].axis('off')
                    
                    # Row 5: MR + Dose overlay
                    if dose_array is not None and slice_idx < dose_array.shape[0]:
                        axes[4, col_idx].imshow(mr_slice, cmap='gray', origin='lower')
                        dose_slice = dose_array[slice_idx, :, :]
                        # Overlay dose with transparency
                        axes[4, col_idx].imshow(
                            dose_slice, cmap='jet', alpha=0.4, origin='lower',
                            vmin=0, vmax=np.percentile(dose_array, 99)
                        )
                        axes[4, col_idx].set_title('MR + Dose', fontsize=8)
                        axes[4, col_idx].axis('off')
                    else:
                        axes[4, col_idx].text(
                            0.5, 0.5, 'MR+Dose\nOverlay\nNot Available',
                            ha='center', va='center', fontsize=10
                        )
                        axes[4, col_idx].axis('off')
                
                # Add row labels
                row_labels = ['MR Only', 'MR + Points', 'MR + Contours', 'Dose', 'MR + Dose']
                for row_idx, label in enumerate(row_labels):
                    axes[row_idx, 0].text(
                        -0.1, 0.5, label,
                        transform=axes[row_idx, 0].transAxes,
                        fontsize=10, fontweight='bold',
                        ha='right', va='center', rotation=90
                    )
                
                # Overall title
                fig.suptitle(
                    f"{lesion_label} - {timepoint_label}\n{folder_info['mr_folder'][:60]}",
                    fontsize=12, fontweight='bold'
                )
                
                plt.tight_layout()
                plt.show()
                
                print("✓ Visualization complete")
                print()
                
            except Exception as e:
                print(f"❌ Error creating visualization: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                print()
            
        except Exception as e:
            print(f"❌ ERROR processing {timepoint_label}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            print()
    
    print("=" * 100)
    print()

