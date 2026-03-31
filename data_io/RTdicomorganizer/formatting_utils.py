"""
==============================================================================
MODULE NAME: formatting_utils.py
==============================================================================

PURPOSE:
    This module provides utilities for console output formatting and logging.
    It helps create well-structured, readable output for data processing
    workflows and QC reports.
    
    Key responsibilities:
    - Print formatted section headers for console output
    - Print formatted subsection headers
    - Generate comprehensive QC summary reports
    - Redirect output to both console and log files simultaneously
    
    This module is purely for presentation and has no dependencies on other
    package modules.

DEPENDENCIES:
    External packages:
    - sys: For stdout manipulation and file writing
    - pandas: For DataFrame operations in QC summary
    
    Internal modules: None (Level 0 module)

FUNCTIONS:
    1. print_section_header(title: str, width: int = 100) -> None
       Print formatted section header with horizontal lines
       
    2. print_subsection_header(title: str, width: int = 100) -> None
       Print formatted subsection header with dashed lines
       
    3. print_qc_summary(df: pd.DataFrame) -> None
       Print comprehensive QC summary with statistics

CLASSES:
    1. TeeOutput
       Redirect output to both stdout and a file simultaneously

USAGE EXAMPLE:
    ```python
    from code.RTdicomorganizer import formatting_utils
    import sys
    
    # Print formatted headers
    formatting_utils.print_section_header("PROCESSING LESIONS")
    print("Processing data...")
    formatting_utils.print_subsection_header("Initial Scan")
    print("Loading initial scan...")
    
    # Redirect output to file and console
    tee = formatting_utils.TeeOutput("output.log")
    sys.stdout = tee
    print("This goes to both console and file")
    tee.close()
    sys.stdout = tee.terminal  # Restore original stdout
    
    # Print QC summary
    formatting_utils.print_qc_summary(qc_dataframe)
    ```

NOTES:
    - All widths default to 100 characters for consistency
    - TeeOutput requires explicit close() to flush file buffer
    - QC summary expects specific DataFrame column names

==============================================================================
"""

import sys
import pandas as pd


def print_section_header(title: str, width: int = 100) -> None:
    """
    Print a formatted section header with horizontal lines.
    
    DETAILED DESCRIPTION:
        Creates a visually prominent section header for console output,
        consisting of:
        1. Blank line for spacing
        2. Horizontal line of '=' characters
        3. Centered title text
        4. Another horizontal line of '=' characters
        
        This is used to clearly separate major sections in long-running
        workflows or reports.
    
    Args:
        title (str): Title text to display
            - Will be centered within the specified width
            - Example: "PROCESSING LESIONS", "QC REPORT"
            
        width (int): Total width of the header line in characters
            - Default: 100
            - Should match console/log file width
            - Example: 80, 100, 120
    
    Returns:
        None: Prints directly to stdout
    
    OUTPUT FORMAT:
        
        ====================================================================================================
                                              PROCESSING LESIONS                                           
        ====================================================================================================
    
    Example:
        >>> from code.RTdicomorganizer import formatting_utils
        
        >>> formatting_utils.print_section_header("LESION PROCESSING")
        
        ====================================================================================================
                                             LESION PROCESSING                                             
        ====================================================================================================
        
        >>> # Custom width
        >>> formatting_utils.print_section_header("SUMMARY", width=50)
        
        ==================================================
                         SUMMARY                         
        ==================================================
    
    Notes:
        - Adds newline before header for spacing
        - Title is centered using Python's .center() method
        - Uses '=' characters for visual prominence
        - Suitable for major section breaks
        - See print_subsection_header() for less prominent headers
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection_header(title: str, width: int = 100) -> None:
    """
    Print a formatted subsection header with dashed lines.
    
    DETAILED DESCRIPTION:
        Creates a subsection header that is less visually prominent than
        print_section_header(), using dashes instead of equal signs:
        1. Blank line for spacing
        2. Horizontal line of '-' characters
        3. Left-aligned title text
        4. Another horizontal line of '-' characters
        
        This is used for subsections within major sections.
    
    Args:
        title (str): Title text to display
            - Left-aligned (not centered)
            - Example: "Initial Scan Overview", "Follow-up Scan Overview"
            
        width (int): Total width of the header line in characters
            - Default: 100
            - Should match console/log file width
            - Example: 80, 100, 120
    
    Returns:
        None: Prints directly to stdout
    
    OUTPUT FORMAT:
        
        ----------------------------------------------------------------------------------------------------
        Initial Scan Overview
        ----------------------------------------------------------------------------------------------------
    
    Example:
        >>> from code.RTdicomorganizer import formatting_utils
        
        >>> formatting_utils.print_subsection_header("MR Series 1")
        
        ----------------------------------------------------------------------------------------------------
        MR Series 1
        ----------------------------------------------------------------------------------------------------
        
        >>> # Within a larger structure
        >>> formatting_utils.print_section_header("LESION 0871.01")
        >>> print("Patient: 0871, Target: 01")
        >>> formatting_utils.print_subsection_header("Initial (Moving) Scan")
        >>> print("Date: 1997-07-18")
    
    Notes:
        - Adds newline before header for spacing
        - Title is left-aligned (not centered)
        - Uses '-' characters for less visual prominence
        - Suitable for subsections within major sections
        - Hierarchically subordinate to print_section_header()
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def print_qc_summary(df: pd.DataFrame) -> None:
    """
    Print a comprehensive QC summary with statistics and issue identification.
    
    DETAILED DESCRIPTION:
        This function analyzes a QC DataFrame and prints a detailed summary
        report including:
        1. Overall statistics (total lesions, processable lesions)
        2. RTStruct validation results
        3. MR series statistics (mean, min, max counts)
        4. RTDOSE series statistics
        5. Detailed lists of lesions with missing required structures
        
        The report is formatted with clear sections and visual separators
        for easy reading.
    
    Args:
        df (pd.DataFrame): QC report DataFrame with required columns
            See INPUT FORMAT below
    
    Returns:
        None: Prints formatted report to stdout
    
    INPUT FORMAT:
        Required columns:
        - patient_id (str): 4-digit patient ID
        - target (str): 2-digit target ID
        - lesion_label (str): Combined 'PPPP.TT' format
        - MR_init (int/'N/A'): Count of MR series in initial scan
        - MR_followup (int/'N/A'): Count of MR series in follow-up scan
        - RTstruct_init_has_Brain_MS_Init_Model (str): '' or 'MISSING' or 'FOLDER MISSING'
        - RTstruct_followup_has_Brain_MS_ReTx_Model (str): '' or 'MISSING' or 'FOLDER MISSING'
        - RTdose_init (int/'N/A'): Count of RTDOSE series in initial scan
        - RTdose_followup (int/'N/A'): Count of RTDOSE series in follow-up scan
    
    OUTPUT FORMAT:
        ====================================================================================================
        QC REPORT SUMMARY
        ====================================================================================================
        
        📊 OVERALL STATISTICS
        ----------------------------------------------------------------------------------------------------
        Total lesions:                       46
        Lesions with missing folders:        0
        Processable lesions:                 46
        
        📋 RTSTRUCT VALIDATION
        ----------------------------------------------------------------------------------------------------
        ✓ Initial with Brain_MS_Init_Model:  46/46 (100.0%)
        ✓ Followup with Brain_MS_ReTx_Model: 46/46 (100.0%)
        ✓ Both models present:               46/46 (100.0%)
        
        ❌ 2 lesion(s) missing 'Brain_MS_Init_Model':
           - 1885.09 (Patient 1885, Target 09)
           - 3126.15 (Patient 3126, Target 15)
        
        📊 MR SERIES STATISTICS (processable lesions only)
        ----------------------------------------------------------------------------------------------------
        Initial MR folders:  mean=1.3, min=1, max=5
        Followup MR folders: mean=1.4, min=1, max=4
        
        📊 RTDOSE SERIES STATISTICS (processable lesions only)
        ----------------------------------------------------------------------------------------------------
        Initial RTDOSE:  mean=1.3, min=1, max=5
        Followup RTDOSE: mean=1.4, min=1, max=4
        
        ====================================================================================================
    
    Example:
        >>> from code.RTdicomorganizer import data_io, formatting_utils
        
        >>> # Generate QC DataFrame
        >>> qc_data = data_io.parse_log_file("output/overview_log.txt")
        >>> df_qc = data_io.generate_qc_report(
        ...     "output/summary_by_lesion.csv",
        ...     qc_data,
        ...     "output/QC_report.csv",
        ...     "output/QC_report.xlsx"
        ... )
        
        >>> # Print summary
        >>> formatting_utils.print_qc_summary(df_qc)
    
    Notes:
        - Automatically filters to processable lesions for statistics
        - Provides percentages for RTStruct validation
        - Lists specific lesions with issues for follow-up
        - Uses emoji icons for visual clarity (📊, 📋, ✓, ❌)
        - Separates concerns: overall stats, RTStruct, MR, RTDOSE
    """
    print("\n" + "=" * 100)
    print("QC REPORT SUMMARY")
    print("=" * 100)
    
    total_lesions = len(df)
    
    # Count issues
    folders_missing = len(df[df['RTstruct_init_has_Brain_MS_Init_Model'] == 'FOLDER MISSING'])
    processable = total_lesions - folders_missing
    
    # RTSTRUCT checks
    init_model_missing = len(df[df['RTstruct_init_has_Brain_MS_Init_Model'] == 'MISSING'])
    retx_model_missing = len(df[df['RTstruct_followup_has_Brain_MS_ReTx_Model'] == 'MISSING'])
    both_models_ok = len(df[
        (df['RTstruct_init_has_Brain_MS_Init_Model'] == '') & 
        (df['RTstruct_followup_has_Brain_MS_ReTx_Model'] == '')
    ])
    
    # MR checks (for processable lesions only)
    processable_df = df[df['RTstruct_init_has_Brain_MS_Init_Model'] != 'FOLDER MISSING']
    
    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 100)
    print(f"Total lesions:                       {total_lesions}")
    print(f"Lesions with missing folders:        {folders_missing}")
    print(f"Processable lesions:                 {processable}")
    
    print(f"\n📋 RTSTRUCT VALIDATION")
    print("-" * 100)
    if processable > 0:
        init_pct = (processable - init_model_missing) / processable * 100
        retx_pct = (processable - retx_model_missing) / processable * 100
        both_pct = both_models_ok / processable * 100
        
        print(f"✓ Initial with Brain_MS_Init_Model:  {processable - init_model_missing}/{processable} ({init_pct:.1f}%)")
        print(f"✓ Followup with Brain_MS_ReTx_Model: {processable - retx_model_missing}/{processable} ({retx_pct:.1f}%)")
        print(f"✓ Both models present:               {both_models_ok}/{processable} ({both_pct:.1f}%)")
        
        if init_model_missing > 0:
            print(f"\n❌ {init_model_missing} lesion(s) missing 'Brain_MS_Init_Model':")
            missing_df = df[df['RTstruct_init_has_Brain_MS_Init_Model'] == 'MISSING']
            for _, row in missing_df.iterrows():
                print(f"   - {row['lesion_label']} (Patient {row['patient_id']}, Target {row['target']})")
        
        if retx_model_missing > 0:
            print(f"\n❌ {retx_model_missing} lesion(s) missing 'Brain_MS_ReTx_Model':")
            missing_df = df[df['RTstruct_followup_has_Brain_MS_ReTx_Model'] == 'MISSING']
            for _, row in missing_df.iterrows():
                print(f"   - {row['lesion_label']} (Patient {row['patient_id']}, Target {row['target']})")
    
    if len(processable_df) > 0:
        print(f"\n📊 MR SERIES STATISTICS (processable lesions only)")
        print("-" * 100)
        print(f"Initial MR folders:  mean={processable_df['MR_init'].mean():.1f}, "
              f"min={processable_df['MR_init'].min()}, max={processable_df['MR_init'].max()}")
        print(f"Followup MR folders: mean={processable_df['MR_followup'].mean():.1f}, "
              f"min={processable_df['MR_followup'].min()}, max={processable_df['MR_followup'].max()}")
        
        print(f"\n📊 RTDOSE SERIES STATISTICS (processable lesions only)")
        print("-" * 100)
        print(f"Initial RTDOSE:  mean={processable_df['RTdose_init'].mean():.1f}, "
              f"min={processable_df['RTdose_init'].min()}, max={processable_df['RTdose_init'].max()}")
        print(f"Followup RTDOSE: mean={processable_df['RTdose_followup'].mean():.1f}, "
              f"min={processable_df['RTdose_followup'].min()}, max={processable_df['RTdose_followup'].max()}")
    
    print("\n" + "=" * 100)


class TeeOutput:
    """
    Redirect output to both stdout and a file simultaneously.
    
    DETAILED DESCRIPTION:
        This class acts as a "T-connector" for output streams, sending
        everything written to it to both the original stdout (console) and
        a log file. This is useful for long-running workflows where you want
        to see real-time output while also saving it to a file.
        
        The class mimics the file-like interface that sys.stdout expects,
        implementing write(), flush(), and close() methods.
        
        TYPICAL USAGE:
        1. Create TeeOutput instance with log file path
        2. Save original stdout
        3. Replace sys.stdout with TeeOutput instance
        4. Print normally (goes to both console and file)
        5. Restore original stdout when done
        6. Close the TeeOutput to flush and close log file
    
    Attributes:
        terminal: Original sys.stdout for console output
        log: Open file handle for log file writing
    
    Methods:
        write(message): Write message to both console and file
        flush(): Flush both console and file buffers
        close(): Close log file and cleanup
    
    Example:
        >>> import sys
        >>> from code.RTdicomorganizer import formatting_utils
        
        >>> # Set up dual output
        >>> tee = formatting_utils.TeeOutput("processing.log")
        >>> original_stdout = sys.stdout
        >>> sys.stdout = tee
        
        >>> # Everything printed goes to both console and file
        >>> print("Processing lesion 1...")
        >>> print("Processing lesion 2...")
        
        >>> # Restore and cleanup
        >>> sys.stdout = original_stdout
        >>> tee.close()
        >>> print("Logging complete (to console only)")
    
    Context Manager Example:
        >>> # Not implemented as context manager, but can be used as:
        >>> tee = formatting_utils.TeeOutput("log.txt")
        >>> try:
        ...     sys.stdout = tee
        ...     print("This goes to both")
        ... finally:
        ...     sys.stdout = tee.terminal
        ...     tee.close()
    
    Notes:
        - Requires explicit close() to flush and close log file
        - Always restore original stdout in finally block to prevent issues
        - Log file is created/overwritten (mode='w')
        - UTF-8 encoding used for log file
        - Does not capture stderr (only stdout)
    """
    
    def __init__(self, filename: str):
        """
        Initialize TeeOutput with log file path.
        
        Args:
            filename (str): Path to log file
                - Will be created or overwritten
                - Example: './output/processing.log'
        """
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message: str) -> None:
        """
        Write message to both terminal and log file.
        
        Args:
            message (str): Text to write
                - Can be any string, including newlines
        """
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self) -> None:
        """
        Flush both terminal and log file buffers.
        
        This ensures all pending output is written immediately.
        """
        self.terminal.flush()
        self.log.flush()
    
    def close(self) -> None:
        """
        Close the log file.
        
        Call this when done logging to ensure all output is written
        and file handle is properly released.
        """
        self.log.close()

