# Study Folder Inventory Feature - Implementation Summary

## Overview
This document summarizes the new study folder inventory generation feature added to the RTdicomorganizer package.

**Date:** December 17, 2024  
**Feature:** Comprehensive DICOM folder inventory with manual selection interface

---

## Purpose

The study folder inventory feature generates a detailed Excel spreadsheet listing all MR, RTStruct, and RTDose folders found in each study folder. This allows manual quality control and selection of which DICOM series to use for subsequent processing steps.

### Key Benefits:
- **Complete visibility** of all available MR series per study
- **Automatic validation** of RTStruct naming patterns
- **Manual selection** interface via Excel
- **Quality control** tracking with file counts and metadata
- **Flexible configuration** with optional metadata fields

---

## New Functions Added

### 1. `path_utils.py` (3 new functions)

#### `find_all_rtstruct_folders(study_folder: Path) -> list`
- Finds all RTStruct folders in a study folder
- Identifies folders by checking DICOM Modality tag == "RTSTRUCT"
- Returns list of Path objects

#### `find_all_rtdose_folders(study_folder: Path) -> list`
- Finds all RTDOSE folders in a study folder
- Identifies folders by checking DICOM Modality tag == "RTDOSE"
- Returns list of Path objects

#### `count_dicom_files(folder_path: Path) -> int`
- Counts number of files in a folder
- Used for quick validation of folder contents
- Returns 0 if folder doesn't exist

### 2. `dicom_utils.py` (1 new function)

#### `read_series_description(folder_path: Path) -> str`
- Reads SeriesDescription tag from first DICOM file in folder
- Returns human-readable series name (e.g., "T1_axial_post_gad")
- Returns "N/A", "Unknown", or "No Description" for error cases

### 3. `data_analysis.py` (1 new function + 1 helper)

#### `generate_study_folder_inventory(csv_path, base_folder, output_path, **options) -> str`
Main function that:
1. Reads registration pair CSV
2. For each unique study folder:
   - Finds all MR folders
   - Finds all RTStruct folders
   - Finds all RTDose folders
3. Extracts metadata for each folder:
   - Folder name
   - Series description
   - File count
   - Optional: image dimensions, orientation, acquisition date
4. Validates RTStruct naming patterns (optional)
5. Exports to Excel with formatting

**Configuration Options:**
- `include_metadata` (bool): Add optional columns (dimensions, orientation, date)
- `validate_rtstruct` (bool): Flag RTStruct files matching expected patterns
- `expected_rtstruct_patterns` (list): Patterns to validate against

#### `_read_optional_metadata(folder_path: Path) -> dict`
Helper function that reads:
- Image dimensions (e.g., "512x512x47")
- Orientation (Axial/Sagittal/Coronal)
- Acquisition date (YYYY-MM-DD)

---

## Output Format

### Excel File Structure

**File:** `./excel/output/prep1_step1_study_folder_inventory.xlsx`

**Required Columns:**
- `patient_id`: 4-digit patient ID
- `study_date`: ISO format date (YYYY-MM-DD)
- `study_type`: "initial_moving" or "followup_fixed"
- `targets`: Slash-separated target IDs
- `n_targets`: Number of lesions
- `modality`: "MR", "RTSTRUCT", or "RTDOSE"
- `folder_name`: Name of DICOM folder
- `series_description`: Series description from DICOM
- `file_count`: Number of files in folder
- `selected`: **Empty - for manual entry of "x"**

**Optional Columns (if include_metadata=True):**
- `image_dimensions`: e.g., "512x512x47"
- `orientation`: "Axial", "Sagittal", "Coronal"
- `acquisition_date`: YYYY-MM-DD

**Optional Columns (if validate_rtstruct=True):**
- `rtstruct_validated`: "PASS" if matches expected pattern

### Example Rows

```
patient_id | study_date  | study_type     | targets | modality  | folder_name                    | series_description    | file_count | selected | rtstruct_validated
-----------|-------------|----------------|---------|-----------|--------------------------------|-----------------------|------------|----------|-------------------
0871       | 1997-07-18  | initial_moving | 01      | MR        | SRS0871_..._axial_n47__00000  | T1_axial_post_gad     | 47         |          |
0871       | 1997-07-18  | initial_moving | 01      | RTSTRUCT  | SRS0871_..._Brain.MS..._n1... | Brain_MS_Init_Model   | 1          |          | PASS
0871       | 1997-07-18  | initial_moving | 01      | RTDOSE    | SRS0871_..._Dose_n1__00003    | Treatment_Dose        | 1          |          |
```

---

## Usage Example

### In Notebook (prep1_step1_summarize_data_cleanup.ipynb)

```python
# Import
from RTdicomorganizer import data_analysis

# Configuration
inventory_path = data_analysis.generate_study_folder_inventory(
    csv_path="./excel/output/prep1_step1_summary_by_registrationpair.csv",
    base_folder="/database/brainmets/dicom",
    output_path="./excel/output/prep1_step1_study_folder_inventory.xlsx",
    include_metadata=False,          # Toggle optional metadata
    validate_rtstruct=True,          # Auto-flag expected RTStruct patterns
    expected_rtstruct_patterns=[     # Patterns to validate
        "Brain_MS_Init_Model",
        "Brain_MS_ReTx_Model"
    ]
)
```

### Workflow

1. **Generate inventory** - Run notebook cell to create Excel file
2. **Review in Excel** - Open file, review all series
3. **Manual selection** - Enter "x" in `selected` column for chosen series
4. **Save Excel** - Save file with selections
5. **Use in next steps** - Subsequent processing reads selections from Excel

---

## Documentation Updates

All changes follow the `coding_guide.md` standards:

### ✅ Module-level documentation
- Updated function lists in file headers
- Added detailed docstrings for all new functions

### ✅ README.md updates
- Updated function inventory counts
- Added new functions to module listings
- Maintained consistency with existing style

### ✅ Type hints and examples
- All functions have complete type hints
- Comprehensive examples in docstrings
- Detailed INPUT/OUTPUT format specifications

### ✅ Error handling
- Graceful handling of missing folders
- Clear error messages with context
- Validation of required inputs

---

## Testing Checklist

Before running in production:

- [ ] Verify `base_dicom_folder` path is correct
- [ ] Ensure registration pair CSV exists
- [ ] Check that study folders follow expected naming convention
- [ ] Test with small subset of patients first
- [ ] Verify Excel output formatting
- [ ] Test manual selection workflow
- [ ] Validate optional metadata extraction (if enabled)
- [ ] Confirm RTStruct validation logic (if enabled)

---

## Next Steps

After generating the inventory:

1. **Manual QC**:
   - Open Excel file
   - Review all MR series (check for multiple acquisitions)
   - Verify RTStruct files are correctly identified
   - Check RTDose files are present

2. **Select Data**:
   - For each study, mark ONE MR series with "x"
   - Mark appropriate RTStruct (validated ones should have "PASS" flag)
   - Mark RTDose if needed

3. **Save and Proceed**:
   - Save Excel file with selections
   - Use this file as input for subsequent processing steps
   - Selections will guide which folders to load/process

---

## Design Decisions

### Why long-format (one row per series)?
- Easier to review in spreadsheet format
- Simple manual selection with "x" marking
- Natural sorting by patient → date → type → modality
- Easy to extend with additional metadata columns

### Why manual selection?
- MR series selection requires human judgment (quality, contrast, artifacts)
- RTStruct validation benefits from visual confirmation
- Allows flexibility for special cases
- Creates documented audit trail of selections

### Why optional metadata?
- Basic workflow doesn't need dimensions/orientation
- Can slow down processing for large datasets
- User can enable when needed for QC or debugging

### Why Excel output?
- Universal format, easy to open and edit
- Supports formatting and column widths
- Familiar interface for manual data entry
- Easy to share with collaborators

---

**Implementation completed:** December 17, 2024  
**Follows:** RTdicomorganizer coding_guide.md v1.0  
**Tested:** Pending user validation

