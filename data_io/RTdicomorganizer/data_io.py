"""
==============================================================================
MODULE NAME: data_io.py
==============================================================================

PURPOSE:
    This module provides functions for reading and writing data files used in
    the brain metastasis lesion tracking workflow.
    
    Key responsibilities:
    - Read CSV and Excel files containing lesion data
    - Write summary tables to both CSV and Excel formats
    - Parse DICOM overview log files to extract QC information
    - Generate comprehensive QC reports combining multiple data sources
    
    This module handles all file I/O operations, providing a clean interface
    between raw data files and the rest of the package.

DEPENDENCIES:
    External packages:
    - pandas: For DataFrame I/O and manipulation
    - pathlib: For cross-platform path handling
    - re: For log file parsing with regular expressions
    
    Internal modules:
    - path_utils: Used implicitly through pathlib operations

FUNCTIONS:
    1. read_table(path: str) -> pd.DataFrame
       Read CSV or Excel files containing lesion data
       
    2. save_outputs(per_lesion: pd.DataFrame, per_pair: pd.DataFrame,
                    out_root: str) -> dict
       Save summary tables to CSV and Excel formats
       
    3. parse_log_file(log_path: str) -> dict
       Parse DICOM overview log file to extract QC information
       
    4. generate_qc_report(csv_input: str, qc_data: dict, output_csv: str,
                          output_xlsx: str) -> pd.DataFrame
       Generate QC report files from parsed log data

USAGE EXAMPLE:
    ```python
    from code.RTdicomorganizer import data_io
    
    # Read input data
    df = data_io.read_table("input/Pat_stat_251128_v3.xlsx")
    
    # Save summary tables
    output_paths = data_io.save_outputs(
        per_lesion_df,
        per_pair_df,
        out_root="output"
    )
    
    # Parse log file and generate QC report
    qc_data = data_io.parse_log_file("output/overview_log.txt")
    qc_df = data_io.generate_qc_report(
        "output/summary_by_lesion.csv",
        qc_data,
        "output/QC_report.csv",
        "output/QC_report.xlsx"
    )
    ```

NOTES:
    - All functions use pathlib for cross-platform compatibility
    - CSV files preserve exact data types (strings stay strings)
    - Excel files provide human-readable formatting
    - QC report generation requires specific log format from overview script

==============================================================================
"""

import re
import pandas as pd
from pathlib import Path


def read_table(path: str) -> pd.DataFrame:
    """
    Read CSV or Excel file containing lesion data.
    
    DETAILED DESCRIPTION:
        This function provides a unified interface for reading lesion data
        from either CSV or Excel format. It automatically detects the file
        type from the extension and applies appropriate parsing.
        
        Special handling:
        - 'lesno' column is read as string to preserve leading zeros
        - Column names are normalized to lowercase with whitespace stripped
        - Supports both .xlsx and legacy .xls formats
    
    Args:
        path (str): Path to CSV or Excel file
            - Absolute or relative path
            - Must have extension .csv, .xlsx, or .xls
            - Example: './input/Pat_stat_251128_v3.xlsx'
    
    Returns:
        pd.DataFrame: DataFrame with lesion data
            - Column names are lowercase
            - 'lesno' column is string type (preserves '0871.01' format)
            - All other columns have inferred types
    
    Raises:
        ValueError: If file extension is not .csv, .xlsx, or .xls
        FileNotFoundError: If file does not exist
    
    INPUT FILE FORMAT:
        Expected columns (case-insensitive):
        - lesno (str): Lesion identifier
          Format: 'PPPP.TT' or numeric (e.g., 1885.09)
          Example: '0871.01', '1885.09', 1885.09
          
        - datepriorsrs (str/date): Initial scan date
          Any common date format (Excel dates, ISO strings, etc.)
          Example: '1997-07-18', 43891
          
        - dategk (str/date): Follow-up scan date
          Any common date format
          Example: '1999-04-02', 43950
    
    OUTPUT FORMAT:
        DataFrame with columns:
        - lesno (str): Preserved as string (e.g., '1885.09', '0871.01')
        - datepriorsrs (varies): Date in original format
        - dategk (varies): Date in original format
        - Other columns: As present in input file
    
    Example:
        >>> # Read Excel file
        >>> df = read_table('./input/patient_data.xlsx')
        >>> print(df.dtypes)
        lesno           object
        datepriorsrs    object
        dategk          object
        
        >>> # Read CSV file
        >>> df = read_table('./input/patient_data.csv')
        >>> print(df['lesno'].iloc[0])
        '0871.01'  # Leading zero preserved
    
    Notes:
        - Column name normalization allows case-insensitive access
        - 'lesno' is read as string to prevent Excel auto-formatting
        - For CSV, encoding is assumed to be UTF-8
        - For Excel, openpyxl engine is used for .xlsx files
    """
    p = Path(path)
    
    # Read based on file extension
    if p.suffix.lower() in [".xlsx", ".xls"]:
        # Read Excel file, preserving lesno as string
        df = pd.read_excel(p, dtype={"lesno": str}, engine="openpyxl")
    elif p.suffix.lower() == ".csv":
        # Read CSV file, preserving lesno as string
        df = pd.read_csv(p, dtype={"lesno": str})
    else:
        raise ValueError(
            f"File must be .csv or .xlsx/.xls format. Got: {p.suffix}\n"
            f"Path: {path}"
        )
    
    # Normalize column names to lowercase for consistent access
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    return df


def save_outputs(per_lesion: pd.DataFrame, per_pair: pd.DataFrame, 
                 out_root: str) -> dict:
    """
    Save summary tables to CSV and Excel formats.
    
    DETAILED DESCRIPTION:
        This function saves two types of summary tables (per-lesion and
        per-pair) in both CSV and Excel formats. It creates the output
        directory if needed and uses standardized file naming conventions.
        
        Output file naming:
        - prep1_step1_summary_by_lesion.csv/xlsx
        - prep1_step1_summary_by_registrationpair.csv/xlsx
        
        CSV files are preferred for programmatic access (preserves types),
        while Excel files are convenient for manual review.
    
    Args:
        per_lesion (pd.DataFrame): Per-lesion summary table
            See OUTPUT FORMAT below for required columns
            
        per_pair (pd.DataFrame): Per-registration-pair summary table
            See OUTPUT FORMAT below for required columns
            
        out_root (str): Output directory path
            - Will be created if it doesn't exist
            - Example: './excel/output'
    
    Returns:
        dict: Dictionary mapping output names to file paths
            Keys: 'per_lesion_csv', 'per_lesion_xlsx',
                  'per_pair_csv', 'per_pair_xlsx'
            Values: Path objects to created files
    
    OUTPUT FORMAT - Per-Lesion CSV/XLSX:
        Columns:
        - patient_id (str): 4-digit patient ID
          Example: '0871', '1885', '3126'
          
        - target (str): 2-digit target ID
          Example: '01', '09', '15'
          
        - lesion_label (str): Combined identifier 'PPPP.TT'
          Example: '0871.01', '1885.09'
          
        - initial_moving (str): Initial scan date (YYYY-MM-DD)
          Example: '1997-07-18', '2004-01-14'
          
        - followup_fixed (str): Follow-up scan date (YYYY-MM-DD)
          Example: '1999-04-02', '2010-10-05'
    
    OUTPUT FORMAT - Per-Pair CSV/XLSX:
        Columns:
        - patient_id (str): 4-digit patient ID
          Example: '0871', '1885'
          
        - initial_moving (str): Initial scan date (YYYY-MM-DD)
          Example: '1997-07-18'
          
        - followup_fixed (str): Follow-up scan date (YYYY-MM-DD)
          Example: '1999-04-02'
          
        - targets (str): Slash-separated target IDs
          Example: '01', '01/09/15', '06/07/08/11/12'
          
        - n_targets (int): Number of lesions in this pair
          Example: 1, 3, 5
    
    Example:
        >>> output_paths = save_outputs(
        ...     per_lesion_df,
        ...     per_pair_df,
        ...     out_root="./excel/output"
        ... )
        >>> print(output_paths['per_lesion_csv'])
        excel/output/prep1_step1_summary_by_lesion.csv
        
        >>> # Verify files were created
        >>> for key, path in output_paths.items():
        ...     print(f"{key}: {path.exists()}")
        per_lesion_csv: True
        per_lesion_xlsx: True
        per_pair_csv: True
        per_pair_xlsx: True
    
    Notes:
        - Creates output directory automatically if missing
        - Overwrites existing files without warning
        - Excel files contain single sheet ('by_lesion' or 'by_pair')
        - CSV files preserve exact data types
        - All string columns maintain zero-padding
    """
    # Create output directory if needed
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Define output file paths with standardized names
    paths = {
        "per_lesion_csv": out_root / "prep1_step1_summary_by_lesion.csv",
        "per_lesion_xlsx": out_root / "prep1_step1_summary_by_lesion.xlsx",
        "per_pair_csv": out_root / "prep1_step1_summary_by_registrationpair.csv",
        "per_pair_xlsx": out_root / "prep1_step1_summary_by_registrationpair.xlsx",
    }
    
    # Save per-lesion summaries
    per_lesion.to_csv(paths["per_lesion_csv"], index=False)
    with pd.ExcelWriter(paths["per_lesion_xlsx"]) as w:
        per_lesion.to_excel(w, index=False, sheet_name="by_lesion")

    # Save per-pair summaries
    per_pair.to_csv(paths["per_pair_csv"], index=False)
    with pd.ExcelWriter(paths["per_pair_xlsx"]) as w:
        per_pair.to_excel(w, index=False, sheet_name="by_pair")

    return paths


def parse_log_file(log_path: str) -> dict:
    """
    Parse DICOM overview log file to extract QC information.
    
    DETAILED DESCRIPTION:
        This function parses the detailed log file generated by the DICOM
        overview script (prep1_step2) to extract quality control information
        for each lesion. It uses regular expressions to:
        
        1. Split log into sections by lesion
        2. Extract lesion identifiers
        3. Count MR series in initial and follow-up scans
        4. Check for presence of required RTStruct models
        5. Count RTDOSE series
        6. Detect missing folders
        
        The parsed information is used to generate QC reports showing data
        completeness and identifying issues.
    
    Args:
        log_path (str): Path to log file from prep1_step2 overview script
            - Generated by process_all_lesions() workflow
            - Contains structured output with section markers
            - Example: './excel/output/prep1_step2_overview_all_pairs.log'
    
    Returns:
        dict: Dictionary mapping lesion_label to QC data
            Keys: lesion labels in 'PPPP.TT' format
            Values: dictionaries with QC metrics
    
    OUTPUT DICTIONARY FORMAT:
        {
            'lesion_label': {
                'mr_init': int or 'N/A',
                'mr_followup': int or 'N/A',
                'rtstruct_init_model': '' or 'MISSING' or 'FOLDER MISSING',
                'rtstruct_followup_model': '' or 'MISSING' or 'FOLDER MISSING',
                'rtdose_init': int or 'N/A',
                'rtdose_followup': int or 'N/A'
            }
        }
    
    QC METRICS:
        - mr_init/mr_followup: Count of MR series
          Example: 1, 3, 5 (or 'N/A' if folder missing)
          
        - rtstruct_init_model: Status of Brain_MS_Init_Model
          Values: '' (present), 'MISSING', 'FOLDER MISSING'
          
        - rtstruct_followup_model: Status of Brain_MS_ReTx_Model
          Values: '' (present), 'MISSING', 'FOLDER MISSING'
          
        - rtdose_init/rtdose_followup: Count of RTDOSE series
          Example: 1, 2, 5 (or 'N/A' if folder missing)
    
    Example:
        >>> qc_data = parse_log_file('./output/overview_log.txt')
        >>> print(qc_data['0871.01'])
        {
            'mr_init': 1,
            'mr_followup': 1,
            'rtstruct_init_model': '',
            'rtstruct_followup_model': '',
            'rtdose_init': 1,
            'rtdose_followup': 1
        }
        
        >>> # Check for missing RTStruct
        >>> print(qc_data['1885.09']['rtstruct_init_model'])
        'MISSING'
    
    Notes:
        - Returns None if log file doesn't exist
        - Expects specific log format from prep1_step2 workflow
        - Patient IDs are zero-padded to 4 digits in output keys
        - Handles cases where folders are missing entirely
        - Counts are based on log file parsing, not direct filesystem access
    """
    if not Path(log_path).exists():
        print(f"❌ Error: Log file not found at: {log_path}")
        return None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary to store results keyed by lesion_label
    qc_data = {}
    
    # Split by lesion sections using regex pattern
    # Pattern matches: LESION N/M: PPPP.TT (Patient PPPP, Target TT)
    lesion_sections = re.split(
        r'={100,}\n.*?LESION (\d+)/(\d+): ([\d.]+) \(Patient (\d+), Target (\d+)\)',
        content
    )
    
    # Process each lesion (skip first empty section)
    # After split, we get: [empty, num, total, label, pid, target, content, ...]
    for i in range(1, len(lesion_sections), 6):
        if i + 4 >= len(lesion_sections):
            break
            
        lesion_num = lesion_sections[i]
        total_lesions = lesion_sections[i + 1]
        lesion_label_raw = lesion_sections[i + 2]  # May be '871.01' or '0871.01'
        patient_id = lesion_sections[i + 3]
        target = lesion_sections[i + 4]
        lesion_content = lesion_sections[i + 5] if i + 5 < len(lesion_sections) else ""
        
        # Reconstruct lesion_label with proper formatting (4-digit patient, 2-digit target)
        patient_id_padded = patient_id.zfill(4)
        target_padded = target.zfill(2)
        lesion_label = f"{patient_id_padded}.{target_padded}"
        
        # Split into initial and followup sections
        initial_section = ""
        followup_section = ""
        
        if "📁 INITIAL (MOVING) SCAN OVERVIEW" in lesion_content:
            parts = lesion_content.split("📁 INITIAL (MOVING) SCAN OVERVIEW")
            if len(parts) > 1:
                initial_part = parts[1]
                if "📁 FOLLOWUP (FIXED) SCAN OVERVIEW" in initial_part:
                    initial_section, followup_section = initial_part.split(
                        "📁 FOLLOWUP (FIXED) SCAN OVERVIEW", 1
                    )
                else:
                    initial_section = initial_part
        
        # Check if folders were missing
        folders_missing = "⏭️  Skipping this lesion due to missing folders" in lesion_content
        
        # Extract QC data
        if folders_missing:
            # Mark as N/A for missing folders
            qc_data[lesion_label] = {
                'mr_init': 'N/A',
                'mr_followup': 'N/A',
                'rtstruct_init_model': 'FOLDER MISSING',
                'rtstruct_followup_model': 'FOLDER MISSING',
                'rtdose_init': 'N/A',
                'rtdose_followup': 'N/A'
            }
        else:
            # Count MR series (pattern: "MR         44     axial  ...")
            mr_init_count = len(re.findall(r'^MR\s+\d+', initial_section, re.MULTILINE))
            mr_followup_count = len(re.findall(r'^MR\s+\d+', followup_section, re.MULTILINE))
            
            # Check RTSTRUCT models
            has_init_model = "Brain_MS_Init_Model" in initial_section
            has_retx_model = "Brain_MS_ReTx_Model" in followup_section
            
            # Count RTDOSE series
            rtdose_init_count = len(re.findall(r'^RTDOSE\s+\d+', initial_section, re.MULTILINE))
            rtdose_followup_count = len(re.findall(r'^RTDOSE\s+\d+', followup_section, re.MULTILINE))
            
            qc_data[lesion_label] = {
                'mr_init': mr_init_count,
                'mr_followup': mr_followup_count,
                'rtstruct_init_model': '' if has_init_model else 'MISSING',
                'rtstruct_followup_model': '' if has_retx_model else 'MISSING',
                'rtdose_init': rtdose_init_count,
                'rtdose_followup': rtdose_followup_count
            }
    
    return qc_data


def generate_qc_report(csv_input: str, qc_data: dict, output_csv: str, 
                      output_xlsx: str) -> pd.DataFrame:
    """
    Generate QC report CSV and XLSX files from parsed log data.
    
    DETAILED DESCRIPTION:
        This function combines the per-lesion summary CSV with QC metrics
        extracted from the overview log file to create a comprehensive
        quality control report. The report includes both the original lesion
        data and additional QC metrics for data completeness verification.
        
        This is the final step in the QC workflow, producing human-readable
        reports that identify data quality issues.
    
    Args:
        csv_input (str): Path to per-lesion summary CSV
            - Output from save_outputs() function
            - Example: './output/prep1_step1_summary_by_lesion.csv'
            
        qc_data (dict): QC metrics dictionary from parse_log_file()
            - Keys: lesion labels ('PPPP.TT')
            - Values: dicts with QC metrics
            
        output_csv (str): Path for output CSV file
            - Example: './output/QC_report.csv'
            
        output_xlsx (str): Path for output Excel file
            - Example: './output/QC_report.xlsx'
    
    Returns:
        pd.DataFrame: Complete QC report DataFrame
            - Combines input CSV with QC metrics
            - See OUTPUT FORMAT below
    
    INPUT CSV FORMAT:
        Required columns:
        - patient_id (str): '0871', '1885'
        - target (str): '01', '09'
        - lesion_label (str): '0871.01', '1885.09'
        - initial_moving (str): '1997-07-18'
        - followup_fixed (str): '1999-04-02'
    
    OUTPUT FORMAT (CSV/XLSX):
        Original columns plus:
        - MR_init (int/'N/A'): Count of MR series in initial scan
        - MR_followup (int/'N/A'): Count of MR series in follow-up scan
        - RTstruct_init_has_Brain_MS_Init_Model (str): '', 'MISSING', or 'FOLDER MISSING'
        - RTstruct_followup_has_Brain_MS_ReTx_Model (str): '', 'MISSING', or 'FOLDER MISSING'
        - RTdose_init (int/'N/A'): Count of RTDOSE series in initial scan
        - RTdose_followup (int/'N/A'): Count of RTDOSE series in follow-up scan
    
    Example:
        >>> qc_data = parse_log_file('./output/overview_log.txt')
        >>> df_qc = generate_qc_report(
        ...     csv_input='./output/summary_by_lesion.csv',
        ...     qc_data=qc_data,
        ...     output_csv='./output/QC_report.csv',
        ...     output_xlsx='./output/QC_report.xlsx'
        ... )
        >>> print(df_qc.columns)
        Index(['patient_id', 'target', 'lesion_label', 'initial_moving',
               'followup_fixed', 'MR_init', 'MR_followup',
               'RTstruct_init_has_Brain_MS_Init_Model',
               'RTstruct_followup_has_Brain_MS_ReTx_Model',
               'RTdose_init', 'RTdose_followup'], dtype='object')
        
        >>> # Check for issues
        >>> issues = df_qc[df_qc['RTstruct_init_has_Brain_MS_Init_Model'] == 'MISSING']
        >>> print(f"Found {len(issues)} lesions with missing initial RTStruct")
    
    Notes:
        - Patient IDs and targets are zero-padded in output
        - Lesion labels are reconstructed to ensure consistent format
        - Missing QC data defaults to 'N/A'
        - Excel file uses 'QC_Report' as sheet name
        - Files are overwritten without warning if they exist
    """
    # Read the input CSV
    df = pd.read_csv(csv_input, dtype={'patient_id': str, 'target': str})
    
    # Ensure proper formatting (4-digit patient_id, 2-digit target)
    df['patient_id'] = df['patient_id'].astype(str).str.zfill(4)
    df['target'] = df['target'].astype(str).str.zfill(2)
    
    # Reconstruct lesion_label with proper formatting to match qc_data keys
    df['lesion_label'] = df['patient_id'] + '.' + df['target']
    
    # Add QC columns by looking up in qc_data dictionary
    df['MR_init'] = df['lesion_label'].apply(
        lambda x: qc_data.get(x, {}).get('mr_init', 'N/A')
    )
    df['MR_followup'] = df['lesion_label'].apply(
        lambda x: qc_data.get(x, {}).get('mr_followup', 'N/A')
    )
    df['RTstruct_init_has_Brain_MS_Init_Model'] = df['lesion_label'].apply(
        lambda x: qc_data.get(x, {}).get('rtstruct_init_model', 'N/A')
    )
    df['RTstruct_followup_has_Brain_MS_ReTx_Model'] = df['lesion_label'].apply(
        lambda x: qc_data.get(x, {}).get('rtstruct_followup_model', 'N/A')
    )
    df['RTdose_init'] = df['lesion_label'].apply(
        lambda x: qc_data.get(x, {}).get('rtdose_init', 'N/A')
    )
    df['RTdose_followup'] = df['lesion_label'].apply(
        lambda x: qc_data.get(x, {}).get('rtdose_followup', 'N/A')
    )
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved CSV to: {output_csv}")
    
    # Save to XLSX
    df.to_excel(output_xlsx, index=False, sheet_name='QC_Report')
    print(f"✓ Saved XLSX to: {output_xlsx}")
    
    return df

