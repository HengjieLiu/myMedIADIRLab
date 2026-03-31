"""
==============================================================================
MODULE NAME: create_selected_inventory.py
==============================================================================

PURPOSE:
    This module provides functionality to create a filtered version of an
    inventory Excel file, keeping only rows where the 'selected' column
    contains 'x'. The filtered data is saved to both Excel and CSV formats.

DEPENDENCIES:
    External packages:
    - pathlib: For cross-platform path manipulation
    - pandas: For reading and writing Excel/CSV files

FUNCTIONS:
    1. create_selected_inventory(input_excel_path: str) -> dict
       Filter inventory file to keep only selected rows and save to Excel/CSV

AUTHOR:
    Created for filtering study folder inventory files
"""

from pathlib import Path
import pandas as pd
from typing import Dict


def create_selected_inventory(input_excel_path: str) -> Dict[str, Path]:
    """
    Create a filtered version of an inventory Excel file, keeping only rows
    where the 'selected' column contains 'x'.
    
    DETAILED DESCRIPTION:
        This function reads an inventory Excel file, filters rows where the
        'selected' column has 'x' (case-insensitive), and saves the filtered
        data to both Excel and CSV formats. The output files are automatically
        named by adding '_selected' before the file extension.
        
        WORKFLOW:
        1. Load the input Excel file
        2. Filter rows where 'selected' column contains 'x' (case-insensitive)
        3. Generate output filenames by adding '_selected' before extension
        4. Save filtered data to Excel format
        5. Save filtered data to CSV format
        6. Return paths to both output files
    
    Args:
        input_excel_path (str): Path to the input Excel file
            - Can be relative or absolute path
            - Must be a valid Excel file (.xlsx or .xls)
            - Must contain a 'selected' column
            - Example: "./excel/output/inventory.xlsx"
    
    Returns:
        dict: Dictionary containing:
            - 'excel_path' (Path): Path to the output Excel file
            - 'csv_path' (Path): Path to the output CSV file
            - 'input_rows' (int): Number of rows in input file
            - 'output_rows' (int): Number of rows in filtered output
    
    Raises:
        FileNotFoundError: If input Excel file doesn't exist
        KeyError: If 'selected' column doesn't exist in the file
        ValueError: If input file is not a valid Excel file
    
    Example:
        >>> from code.RTdicomorganizer import create_selected_inventory
        
        >>> result = create_selected_inventory(
        ...     "./excel/output/inventory.xlsx"
        ... )
        >>> print(f"Filtered {result['input_rows']} rows to {result['output_rows']} rows")
        >>> print(f"Excel saved to: {result['excel_path']}")
        >>> print(f"CSV saved to: {result['csv_path']}")
    
    Notes:
        - Case-insensitive matching for 'x' in selected column
        - Empty cells, None, and non-'x' values are filtered out
        - Output files are saved in the same directory as input file
        - Preserves all columns from the original file
    """
    # ===================================================================
    # Step 1: Validate input file
    # ===================================================================
    input_path = Path(input_excel_path)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input Excel file not found: {input_excel_path}\n"
            f"Please ensure the file exists and path is correct."
        )
    
    if not input_path.suffix.lower() in ['.xlsx', '.xls']:
        raise ValueError(
            f"Input file must be an Excel file (.xlsx or .xls): {input_excel_path}"
        )
    
    # ===================================================================
    # Step 2: Load Excel file
    # ===================================================================
    print(f"Loading inventory file: {input_path.name}...")
    df = pd.read_excel(input_path)
    
    input_rows = len(df)
    print(f"Loaded {input_rows} rows from input file")
    
    # ===================================================================
    # Step 3: Validate 'selected' column exists
    # ===================================================================
    if 'selected' not in df.columns:
        available_cols = ', '.join(df.columns.tolist())
        raise KeyError(
            f"'selected' column not found in input file.\n"
            f"Available columns: {available_cols}"
        )
    
    # ===================================================================
    # Step 4: Filter rows where 'selected' column has 'x'
    # ===================================================================
    # Convert 'selected' column to string and check for 'x' (case-insensitive)
    # Filter out empty, None, and non-'x' values
    df_selected = df[
        df['selected'].astype(str).str.lower().str.strip() == 'x'
    ].copy()
    
    output_rows = len(df_selected)
    print(f"Filtered to {output_rows} rows with 'x' in 'selected' column")
    
    if output_rows == 0:
        print("WARNING: No rows found with 'x' in 'selected' column!")
        print("Output files will be empty.")
    
    # ===================================================================
    # Step 5: Generate output filenames
    # ===================================================================
    # Add '_selected' before the file extension
    # Example: "inventory.xlsx" -> "inventory_selected.xlsx"
    stem = input_path.stem  # filename without extension
    suffix = input_path.suffix  # extension including dot
    output_dir = input_path.parent  # same directory as input
    
    excel_output_path = output_dir / f"{stem}_selected{suffix}"
    csv_output_path = output_dir / f"{stem}_selected.csv"
    
    # ===================================================================
    # Step 6: Save to Excel format
    # ===================================================================
    print(f"Saving Excel file: {excel_output_path.name}...")
    df_selected.to_excel(excel_output_path, index=False)
    print(f"Excel file saved: {excel_output_path}")
    
    # ===================================================================
    # Step 7: Save to CSV format
    # ===================================================================
    print(f"Saving CSV file: {csv_output_path.name}...")
    df_selected.to_csv(csv_output_path, index=False)
    print(f"CSV file saved: {csv_output_path}")
    
    # ===================================================================
    # Step 8: Return results
    # ===================================================================
    return {
        'excel_path': excel_output_path,
        'csv_path': csv_output_path,
        'input_rows': input_rows,
        'output_rows': output_rows
    }


if __name__ == "__main__":
    """
    Command-line interface for creating selected inventory.
    
    Usage:
        python create_selected_inventory.py <input_excel_path>
    
    Example:
        python create_selected_inventory.py ./excel/output/inventory.xlsx
    """
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python create_selected_inventory.py <input_excel_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    try:
        result = create_selected_inventory(input_path)
        print("\n" + "="*60)
        print("SUCCESS: Selected inventory files created!")
        print("="*60)
        print(f"Input rows:  {result['input_rows']}")
        print(f"Output rows: {result['output_rows']}")
        print(f"\nExcel file: {result['excel_path']}")
        print(f"CSV file:   {result['csv_path']}")
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

