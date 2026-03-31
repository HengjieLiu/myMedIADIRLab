# RTdicomorganizer

A Python package for organizing, processing, and analyzing brain metastasis DICOM data for stereotactic radiosurgery (SRS) treatment planning and follow-up analysis.

---

## 📁 Folder Structure

```
RTdicomorganizer/
├── coding_guide.md               # Coding standards and documentation requirements
├── README.md                     # This file
├── __init__.py                   # Package initialization with exports
│
├── path_utils.py                 # Path construction and folder discovery
├── data_io.py                    # File I/O operations (CSV, Excel, logs)
├── data_parser.py                # Data parsing and transformation
├── data_analysis.py              # Data analysis and summary generation
├── dicom_utils.py                # DICOM-specific operations
├── visualization.py              # Image visualization and plotting
├── formatting_utils.py           # Console output formatting
└── workflow_orchestration.py    # High-level workflow coordination
```

---

## 🔗 Module Dependencies

### Dependency Graph

```
Level 0 (No dependencies):
  └── path_utils.py
  └── formatting_utils.py

Level 1 (Depends on Level 0):
  └── data_parser.py
  └── data_io.py (uses path_utils)
  └── dicom_utils.py (uses path_utils)

Level 2 (Depends on Level 0-1):
  └── data_analysis.py (uses data_parser)
  └── visualization.py (uses dicom_utils)

Level 3 (Depends on multiple modules):
  └── workflow_orchestration.py (uses all modules)
```

### Detailed Dependencies

| Module | Depends On | External Packages |
|--------|------------|-------------------|
| `path_utils.py` | None | pathlib, datetime |
| `data_parser.py` | None | pandas, numpy, decimal, re |
| `data_io.py` | `path_utils` | pandas, pathlib |
| `data_analysis.py` | `data_parser` | pandas |
| `dicom_utils.py` | `path_utils` | pydicom, SimpleITK, numpy, subprocess |
| `visualization.py` | `dicom_utils` | matplotlib, numpy |
| `formatting_utils.py` | None | sys |
| `workflow_orchestration.py` | All modules | All of above |

---

## 📚 Complete Function and Class Inventory

### `path_utils.py` (7 functions)

1. **`convert_iso_date_to_folder_format(iso_date: str) -> str`**
   - Convert ISO date (YYYY-MM-DD) to folder format (YYYY-MM)

2. **`build_study_folder_path(base_folder: str, patient_id: str, iso_date: str) -> Path`**
   - Build full path to a study folder from base folder, patient ID, and date

3. **`find_rtstruct_folder(study_folder: Path, folder_pattern: str) -> Path`**
   - Find RTStruct folder matching a specific naming pattern

4. **`find_all_mr_folders(study_folder: Path) -> list`**
   - Find all MR image folders in a study folder

5. **`find_all_rtstruct_folders(study_folder: Path) -> list`**
   - Find all RTStruct folders in a study folder

6. **`find_all_rtdose_folders(study_folder: Path) -> list`**
   - Find all RTDOSE folders in a study folder

7. **`count_dicom_files(folder_path: Path) -> int`**
   - Count the number of files in a DICOM folder

---

### `data_io.py` (4 functions)

1. **`read_table(path: str) -> pd.DataFrame`**
   - Read CSV or Excel files containing lesion data

2. **`save_outputs(per_lesion: pd.DataFrame, per_pair: pd.DataFrame, out_root: str) -> dict`**
   - Save summary tables to CSV and Excel formats

3. **`parse_log_file(log_path: str) -> dict`**
   - Parse DICOM overview log file to extract QC information

4. **`generate_qc_report(csv_input: str, qc_data: dict, output_csv: str, output_xlsx: str) -> pd.DataFrame`**
   - Generate QC report CSV and XLSX files from parsed log data

---

### `data_parser.py` (4 functions)

1. **`parse_patient_target(val) -> tuple`**
   - Parse lesion numbers (e.g., '1885.09') into patient ID and target ID

2. **`ensure_dates(df: pd.DataFrame, cols: tuple) -> pd.DataFrame`**
   - Convert date columns to pandas datetime format

3. **`convert_target_to_rtstruct_format(target: str) -> str`**
   - Convert target ID from CSV format (e.g., '01') to RTStruct format (e.g., 'target1')

4. **`targets_join(series: pd.Series) -> str`**
   - Join target IDs into slash-separated string (e.g., '01/09/15')

---

### `data_analysis.py` (4 functions)

1. **`build_summaries(df: pd.DataFrame) -> tuple`**
   - Build per-lesion and per-pair summary tables from raw lesion data
   - Automatically adds regpair counter (01, 02, ...) that resets per patient

2. **`count_modality_in_section(section_text: str, modality: str) -> int`**
   - Count number of DICOM series for a specific modality in log text

3. **`check_series_description_in_section(section_text: str, series_description: str) -> bool`**
   - Check if a specific series description exists in log text section

4. **`generate_study_folder_inventory(csv_path: str, base_folder: str, output_path: str, **options) -> str`**
   - Generate comprehensive inventory of DICOM folders for all study folders with manual selection interface

---

### `dicom_utils.py` (4 functions)

1. **`verify_rtstruct_series_description(rtstruct_folder: Path, expected_description: str) -> bool`**
   - Verify that RTStruct series description matches expected value

2. **`get_center_slice_from_contours(rtstruct_reader, structure_name: str, image_reader) -> int`**
   - Get center slice index based on contour Z-coordinates

3. **`run_overview_script(folder_path: Path) -> tuple`**
   - Run DICOM overview script on a folder and capture output

4. **`read_series_description(folder_path: Path) -> str`**
   - Read series description from the first DICOM file in a folder

---

### `visualization.py` (1 function)

1. **`display_mr_with_contour(image_reader, rtstruct_reader, structure_name: str, title: str) -> None`**
   - Display MR image slice with RTStruct contour overlay using matplotlib

---

### `formatting_utils.py` (3 functions + 1 class)

#### Functions:
1. **`print_section_header(title: str, width: int = 100) -> None`**
   - Print formatted section header with horizontal lines

2. **`print_subsection_header(title: str, width: int = 100) -> None`**
   - Print formatted subsection header with dashed lines

3. **`print_qc_summary(df: pd.DataFrame) -> None`**
   - Print comprehensive QC summary with statistics

#### Classes:
1. **`TeeOutput`**
   - Redirect output to both stdout and a file simultaneously

---

### `workflow_orchestration.py` (4 functions)

1. **`process_all_lesions() -> None`**
   - Main workflow to process all lesions and generate DICOM overviews

2. **`process_and_display_lesion_v1(row: pd.Series) -> None`**
   - Process and display MR images with RTStruct overlays for a single lesion (Version 1)
   - Uses DataFrame row input and auto-discovers DICOM folders

3. **`process_and_display_lesion_v2(lesion_label: str, study_type: str, inventory_excel_path: str) -> None`**
   - Process and display MR/RTStruct/RTDose using pre-validated inventory file (Version 2)
   - Uses manual selection from inventory Excel file
   - Supports multi-row visualization with MR, RTStruct points, RTStruct contours, RTDose

4. **`process_and_display_lesion`** (alias for v1)
   - Backward compatibility alias

---

## 🚀 Quick Start

### Installation Requirements

```bash
pip install numpy pandas openpyxl pydicom SimpleITK matplotlib
```

### Basic Usage

```python
# Import the package
from code.RTdicomorganizer import path_utils, data_io, data_parser

# Read lesion data
df = data_io.read_table("path/to/lesion_data.csv")

# Parse lesion information
patient_id, target, label = data_parser.parse_patient_target("1885.09")

# Build study folder path
study_path = path_utils.build_study_folder_path(
    base_folder="/database/dicom",
    patient_id="1885",
    iso_date="2004-01-14"
)

# Find RTStruct folder
rtstruct_folder = path_utils.find_rtstruct_folder(
    study_path,
    "Brain.MS.Init.Model"
)
```

### Workflow Example

```python
from code.RTdicomorganizer import data_io, data_analysis

# Step 1: Read raw lesion data
df_raw = data_io.read_table("input/patient_data.xlsx")

# Step 2: Build summaries
per_lesion_df, per_pair_df = data_analysis.build_summaries(df_raw)

# Step 3: Save outputs
output_paths = data_io.save_outputs(
    per_lesion_df,
    per_pair_df,
    out_root="output"
)

print(f"Saved to: {output_paths}")
```

---

## 📊 Data Format Specifications

### Input Data Format (Patient Statistics CSV/XLSX)

**Expected Columns:**
- `lesno` or `lesno_1` (str/float): Lesion number in format PPPP.TT
  - Example: 1885.09, 0871.01, 1204.12
  - May be stored as float in Excel (e.g., 1885.089966)
  
- `datepriorsrs` (str/date): Initial scan date
  - Any common date format (Excel dates, YYYY-MM-DD, etc.)
  
- `dategk` (str/date): Follow-up scan date
  - Any common date format

### Output Data Formats

#### 1. Per-Lesion Summary CSV/XLSX
**Columns:**
- `patient_id` (str): 4-digit patient ID with leading zeros (e.g., '0871')
- `target` (str): 2-digit target ID with leading zeros (e.g., '09')
- `lesion_label` (str): Combined format 'PPPP.TT' (e.g., '0871.01')
- `initial_moving` (str): Initial scan date in ISO format (YYYY-MM-DD)
- `followup_fixed` (str): Follow-up scan date in ISO format (YYYY-MM-DD)

#### 2. Per-Registration-Pair Summary CSV/XLSX
**Columns:**
- `patient_id` (str): 4-digit patient ID
- `regpair` (str): 2-digit registration pair counter (resets per patient, e.g., '01', '02')
- `initial_moving` (str): Initial scan date (YYYY-MM-DD)
- `followup_fixed` (str): Follow-up scan date (YYYY-MM-DD)
- `targets` (str): Slash-separated target IDs (e.g., '01/09/15')
- `n_targets` (int): Number of lesions sharing this registration pair

#### 3. QC Report CSV/XLSX
**Columns:**
- All columns from per-lesion summary, plus:
- `MR_init` (int): Number of MR series in initial scan
- `MR_followup` (int): Number of MR series in follow-up scan
- `RTstruct_init_has_Brain_MS_Init_Model` (str): '' if present, 'MISSING' if absent
- `RTstruct_followup_has_Brain_MS_ReTx_Model` (str): '' if present, 'MISSING' if absent
- `RTdose_init` (int): Number of RTDOSE series in initial scan
- `RTdose_followup` (int): Number of RTDOSE series in follow-up scan

---

## 🛠️ Development Guidelines

### Before Editing Code
1. Read `coding_guide.md` thoroughly
2. Understand the module dependencies
3. Check existing function signatures

### After Editing Code
1. Update function docstrings if behavior changed
2. Update file-level documentation if functions added/removed
3. Update this README.md if structure/dependencies changed
4. Update CSV/XLSX format documentation if data formats changed
5. Test with example data

### Adding a New Function
1. Choose appropriate module based on function purpose
2. Add function with complete docstring following `coding_guide.md`
3. Update file-level function list
4. Update this README.md function inventory
5. Update dependency graph if needed

---

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're importing from the correct module
2. **Path errors**: Check that base folder paths are correct and folders exist
3. **Date parsing errors**: Verify date column names and formats match expected
4. **Patient ID formatting**: Ensure patient IDs are zero-padded to 4 digits

### Debug Tips

- Enable DEBUG flag in visualization functions for detailed slice information
- Check log files in `excel/output/` for detailed processing information
- Use `print_qc_summary()` to get overview of data quality issues

---

## 📝 Notes

- **Patient ID Format**: Always stored as 4-digit strings with leading zeros (e.g., '0871', not '871')
- **Target ID Format**: Always stored as 2-digit strings with leading zeros (e.g., '01', not '1')
- **Lesion Label Format**: Always 'PPPP.TT' format (e.g., '0871.01')
- **Date Format**: ISO format YYYY-MM-DD for consistency
- **Folder Format**: Study folders use YYYY-MM format (e.g., '1997-07__Studies')

---

## 📄 License

Internal use for brain metastasis research project.

---

## 👥 Contributors

- RTdicomorganizer Development Team

---

**Last Updated:** 2024-12-17  
**Version:** 1.0.0

