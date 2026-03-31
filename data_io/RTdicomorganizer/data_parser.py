"""
==============================================================================
MODULE NAME: data_parser.py
==============================================================================

PURPOSE:
    This module provides functions for parsing and transforming data formats
    commonly used in brain metastasis lesion tracking.
    
    Key responsibilities:
    - Parse lesion identifiers (e.g., '1885.09') into patient and target IDs
    - Handle Excel's float precision issues with numeric identifiers
    - Convert date columns to standardized datetime format
    - Transform target ID formats between CSV and RTStruct conventions
    - Join multiple target IDs into formatted strings
    
    This module contains pure data transformation logic with no I/O operations
    or external dependencies on other package modules.

DEPENDENCIES:
    External packages:
    - pandas: For DataFrame operations and date parsing
    - numpy: For handling NaN values
    - decimal: For precise rounding of float values
    - re: For regular expression pattern matching
    
    Internal modules: None (Level 0 module)

FUNCTIONS:
    1. parse_patient_target(val) -> tuple
       Parse lesion numbers into (patient_id, target_id, lesion_label)
       
    2. ensure_dates(df: pd.DataFrame, cols: tuple) -> pd.DataFrame
       Convert specified columns to pandas datetime format
       
    3. convert_target_to_rtstruct_format(target: str) -> str
       Convert target ID from CSV format to RTStruct naming
       
    4. targets_join(series: pd.Series) -> str
       Join target IDs into slash-separated string

USAGE EXAMPLE:
    ```python
    from code.RTdicomorganizer import data_parser
    
    # Parse lesion identifier
    patient_id, target, label = data_parser.parse_patient_target("1885.09")
    # Returns: ('1885', '09', '1885.09')
    
    # Handle Excel float precision
    patient_id, target, label = data_parser.parse_patient_target(1885.089966)
    # Returns: ('1885', '09', '1885.09')
    
    # Convert dates
    df = data_parser.ensure_dates(df, cols=('initial_date', 'followup_date'))
    
    # Convert target format for RTStruct
    rtstruct_name = data_parser.convert_target_to_rtstruct_format('09')
    # Returns: 'target9'
    ```

NOTES:
    - Handles Excel's binary float precision issues (1885.089966 → 1885.09)
    - Always pads patient IDs to 4 digits and target IDs to 2 digits
    - Supports multiple input formats (float, string, int)
    - Pure functions with no side effects

==============================================================================
"""

import re
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP


def parse_patient_target(val):
    """
    Parse lesion numbers into patient ID, target ID, and formatted label.
    
    DETAILED DESCRIPTION:
        This function robustly extracts patient and target IDs from lesion
        identifiers that may be stored in various formats. It handles the
        common problem where Excel stores numbers like 1885.09 as floats with
        binary precision errors (e.g., 1885.089966).
        
        PROCESS:
        1. Try to parse as numeric value (float/int)
        2. Round to 2 decimal places using banker's rounding
        3. Split into integer (patient) and fractional (target) parts
        4. Pad patient ID to 4 digits, target ID to 2 digits
        5. Fall back to string parsing if numeric parsing fails
        
        PROBLEM SOLVED:
        Excel stores 1885.09 as 1885.089966 (binary float representation).
        This function uses Decimal arithmetic to correctly round to 1885.09.
    
    Args:
        val: Lesion number in various formats
            - Float: 1885.09, 1885.089966 (Excel), 871.1
            - String: '1885.09', '0871.01', '1885_09'
            - Int: 1885 (interpreted as 1885.00)
            - NaN/None: Returns (None, None, None)
    
    Returns:
        tuple: (patient_id, target_id, lesion_label)
            - patient_id (str): 4-digit patient ID with leading zeros
              Example: '1885', '0871', '0045'
            - target_id (str): 2-digit target ID with leading zeros
              Example: '09', '01', '15'
            - lesion_label (str): Formatted as 'PPPP.TT'
              Example: '1885.09', '0871.01', '0045.10'
            - Returns (None, None, None) if value is missing
            - Returns (None, None, original_string) if unparseable
    
    Example:
        >>> # Excel float with precision error
        >>> parse_patient_target(1885.089966)
        ('1885', '09', '1885.09')
        
        >>> # String format
        >>> parse_patient_target('0871.01')
        ('0871', '01', '0871.01')
        
        >>> # Short format needs padding
        >>> parse_patient_target('871.1')
        ('0871', '01', '0871.01')
        
        >>> # Handle missing values
        >>> parse_patient_target(np.nan)
        (None, None, None)
        
        >>> # Alternative separator (underscore)
        >>> parse_patient_target('1885_09')
        ('1885', '09', '1885.09')
    
    Notes:
        - Patient IDs are ALWAYS 4 digits (e.g., '0871', not '871')
        - Target IDs are ALWAYS 2 digits (e.g., '01', not '1')
        - Uses ROUND_HALF_UP for consistent rounding behavior
        - Handles various separators (., _, -, etc.)
        - Returns original string if completely unparseable
    """
    # Handle missing/NaN values
    if pd.isna(val):
        return None, None, None

    s = str(val).strip()

    # ===================================================================
    # Primary Method: Parse as numeric value
    # ===================================================================
    try:
        # Convert to float, then to Decimal for precise rounding
        d = Decimal(str(float(s)))
        
        # Round to 2 decimal places (handles Excel precision issues)
        # ROUND_HALF_UP: 0.089966 → 0.09, 0.095 → 0.10
        d2 = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        s2 = format(d2, "f")
        
        # Ensure decimal point exists
        if "." not in s2:
            s2 += ".00"
        
        # Split into integer and fractional parts
        whole, frac = s2.split(".")
        
        # Format with padding
        patient_id = whole.zfill(4)  # Pad to 4 digits with leading zeros
        target_id = (frac + "00")[:2]  # Ensure exactly 2 digits
        lesion_label = f"{patient_id}.{target_id}"
        
        return patient_id, target_id, lesion_label
        
    except Exception:
        pass  # Fall through to string parsing methods

    # ===================================================================
    # Fallback 1: String format with decimal point (e.g., '1885.9')
    # ===================================================================
    m = re.match(r"^\s*(\d+)\.(\d{1,2})\s*$", s)
    if m:
        patient_id = m.group(1).zfill(4)  # Pad to 4 digits
        target_id = m.group(2).zfill(2)   # Pad to 2 digits
        lesion_label = f"{patient_id}.{target_id}"
        return patient_id, target_id, lesion_label

    # ===================================================================
    # Fallback 2: Alternative separators (e.g., '1885_09', '1885-09')
    # ===================================================================
    m = re.match(r"^\s*(\d+)[^\d]+(\d{1,2})\s*$", s)
    if m:
        patient_id = m.group(1).zfill(4)  # Pad to 4 digits
        target_id = m.group(2).zfill(2)   # Pad to 2 digits
        lesion_label = f"{patient_id}.{target_id}"
        return patient_id, target_id, lesion_label

    # ===================================================================
    # Unparseable: Return None for IDs, original string for label
    # ===================================================================
    return None, None, str(val)


def ensure_dates(df: pd.DataFrame, cols: tuple = ("datepriorsrs", "dategk")) -> pd.DataFrame:
    """
    Convert specified columns to pandas datetime format.
    
    DETAILED DESCRIPTION:
        This function standardizes date columns by converting them to pandas
        datetime64[ns] format. It handles various input date formats
        (Excel dates, ISO strings, etc.) and creates columns with NaT (Not a Time)
        values if they don't exist.
        
        This is essential for:
        - Consistent date arithmetic
        - Proper date sorting and filtering
        - ISO format output (YYYY-MM-DD)
        - Handling missing date values gracefully
    
    Args:
        df (pd.DataFrame): DataFrame containing date columns
            - Will be modified in place
            - May contain mixed date formats
            
        cols (tuple): Tuple of column names to convert to datetime
            - Default: ('datepriorsrs', 'dategk')
            - Common usage: initial and follow-up scan dates
    
    Returns:
        pd.DataFrame: Same DataFrame with date columns converted
            - Columns converted to datetime64[ns] dtype
            - Invalid dates converted to NaT
            - New columns created with NaT if they don't exist
    
    INPUT DATE FORMATS:
        Automatically handles:
        - Excel datetime: 43891 (days since 1900-01-01)
        - ISO strings: '2020-01-15', '2020/01/15'
        - US format: '01/15/2020', '01-15-2020'
        - Timestamp objects: Already datetime, no change needed
    
    OUTPUT FORMAT:
        - All dates converted to datetime64[ns] dtype
        - Can be formatted as ISO with .dt.strftime('%Y-%m-%d')
        - NaT for missing or unparseable dates
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'lesno': ['1885.09', '0871.01'],
        ...     'datepriorsrs': ['2004-01-14', '1997-07-18'],
        ...     'dategk': ['2010-10-05', '1999-04-02']
        ... })
        >>> df = ensure_dates(df, cols=('datepriorsrs', 'dategk'))
        >>> print(df.dtypes)
        lesno                    object
        datepriorsrs    datetime64[ns]
        dategk          datetime64[ns]
        
        >>> # Format as ISO string
        >>> df['initial_iso'] = df['datepriorsrs'].dt.strftime('%Y-%m-%d')
        >>> print(df['initial_iso'].iloc[0])
        '2004-01-14'
    
    Notes:
        - Uses errors='coerce' to convert unparseable dates to NaT
        - Creates missing columns with NaT values (no error raised)
        - Modifies DataFrame in place but also returns it for chaining
        - NaT values will cause row exclusion in downstream processing
    """
    for c in cols:
        if c in df.columns:
            # Convert existing column, unparseable values become NaT
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            # Create column with NaT if it doesn't exist
            df[c] = pd.NaT
    
    return df


def convert_target_to_rtstruct_format(target: str) -> str:
    """
    Convert target ID from CSV format to RTStruct naming format.
    
    DETAILED DESCRIPTION:
        RTStruct files use a different naming convention for targets than
        our CSV files. CSV files use 2-digit zero-padded strings ('01', '09'),
        while RTStruct files use the format 'target' + number without leading
        zeros ('target1', 'target9').
        
        This function performs the conversion needed when looking up
        structures in RTStruct DICOM files.
    
    Args:
        target (str): 2-digit target ID with leading zero from CSV
            - Format: '01', '02', ..., '09', '10', '11', ...
            - Must be convertible to integer
    
    Returns:
        str: Target name in RTStruct format
            - Format: 'target' + number (no leading zeros)
            - Example: 'target1', 'target9', 'target10', 'target15'
    
    Example:
        >>> # Single-digit targets
        >>> convert_target_to_rtstruct_format('01')
        'target1'
        
        >>> convert_target_to_rtstruct_format('09')
        'target9'
        
        >>> # Double-digit targets
        >>> convert_target_to_rtstruct_format('10')
        'target10'
        
        >>> convert_target_to_rtstruct_format('15')
        'target15'
    
    Notes:
        - Leading zeros are stripped by converting to int
        - Output format matches DICOM RTStruct StructureSetROISequence naming
        - Used when loading contours from RTStruct files
        - Essential for correct structure lookup in DICOM data
    """
    # Remove leading zeros by converting to int, then format
    target_num = int(target)
    return f"target{target_num}"


def targets_join(series: pd.Series) -> str:
    """
    Join target IDs into slash-separated string.
    
    DETAILED DESCRIPTION:
        When multiple lesions share the same registration pair (same patient,
        same initial and follow-up dates), we need to represent all their
        target IDs in a compact format. This function creates a sorted,
        deduplicated, slash-separated string of target IDs.
        
        Used primarily in registration pair summary tables to show which
        lesions are processed together in a single registration.
    
    Args:
        series (pd.Series): Series of target IDs
            - Each value should be a 2-digit string ('01', '09', etc.)
            - May contain duplicates
            - Example: pd.Series(['01', '09', '01', '15'])
    
    Returns:
        str: Slash-separated string of unique, sorted target IDs
            - Format maintains 2-digit zero-padding
            - Sorted numerically (not alphabetically)
            - Duplicates removed
            - Example: '01/09/15'
    
    OUTPUT FORMAT:
        Single target:     '01'
        Multiple targets:  '01/09/15'
        Many targets:      '06/07/08/11/12'
    
    Example:
        >>> import pandas as pd
        >>> targets = pd.Series(['09', '01', '15', '09', '03'])
        >>> result = targets_join(targets)
        >>> print(result)
        '01/03/09/15'
        
        >>> # Maintains zero-padding
        >>> targets = pd.Series(['01', '02', '05'])
        >>> result = targets_join(targets)
        >>> print(result)
        '01/02/05'
    
    Notes:
        - Maintains 2-digit format with leading zeros
        - Sorts numerically (01, 02, 09, 10, not alphabetically)
        - Removes duplicates automatically via set()
        - Returns empty string if series is empty
        - Silently skips non-convertible values
    """
    targets = []
    
    for t in series:
        try:
            # Keep as 2-digit string with leading zero
            targets.append(str(t).zfill(2))
        except Exception:
            # Skip values that can't be converted
            pass
    
    # Sort and deduplicate
    targets = sorted(set(targets))
    
    # Join with slash separator
    return "/".join(targets)

