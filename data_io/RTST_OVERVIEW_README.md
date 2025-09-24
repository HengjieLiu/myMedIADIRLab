# RT Structure Overview Script

## Overview

The `overview_all_rtst_folder.py` script analyzes RT Structure (RTSTRUCT) files in DICOM folders and provides a comprehensive overview of all structures contained within them. It can analyze either a single study folder or a patient folder containing multiple studies.

## Features

- **RT Structure File Detection**: Automatically finds all RT Structure files in the specified folder
- **Structure Extraction**: Extracts all structure names from RT Structure files
- **Chronological Organization**: For patient folders, organizes studies chronologically
- **Cross-Reference Table**: Shows which structures exist in which files
- **Summary Statistics**: Provides counts of files and structures
- **JSON Export**: Optional export of results to JSON format

## Installation Requirements

```bash
pip install pydicom
```

## Usage Examples

### Basic Usage

```bash
# Analyze a single study folder
python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126/2009-08__Studies

# Analyze a patient folder with multiple studies
python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126
```

### Advanced Usage

```bash
# Save results to JSON file
python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --output rtst_overview.json

# Enable verbose output for debugging
python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --verbose

# Set maximum width for structure names in table
python overview_all_rtst_folder.py /data/hengjie/brainmets/dicom/Data/SRS3126 --max-width 40
```

## Command Line Arguments

- `folder_path`: Path to the study folder or patient folder containing RT structure files
- `--output, -o`: Output file path for JSON summary (optional)
- `--verbose, -v`: Enable verbose output for debugging
- `--max-width`: Maximum width for structure names in table (default: 30)

## Output Format

The script provides several types of output:

### 1. Summary Statistics
```
RT STRUCTURE FOLDER OVERVIEW
================================================================================
Scan Date: 2024-01-15T10:30:45.123456
Folder: /data/hengjie/brainmets/dicom/Data/SRS3126
Total RT Structure Files: 5
Total Unique Structures: 25
```

### 2. Study Summary (for patient folders)
```
STUDY SUMMARY (Chronological Order):
------------------------------------------------------------
Study Date   RTST Files  Structures  
------------------------------------------------------------
20090801     2           15          
20090815     1           12          
20090901     2           18          
```

### 3. Detailed File Information
```
DETAILED RT STRUCTURE FILES:
--------------------------------------------------------------------------------

Study Date: 20090801
----------------------------------------

File: 20090801-T1_Post_Contrast
  Structures (8): Brain, Brainstem, Cerebellum, Eye_L, Eye_R, Optic_Nerve_L, Optic_Nerve_R, PTV

File: 20090801-T2_FLAIR
  Structures (7): Brain, Brainstem, Cerebellum, Eye_L, Eye_R, Optic_Nerve_L, Optic_Nerve_R
```

### 4. Cross-Reference Table
```
CROSS-REFERENCE TABLE:
================================================================================
Structure presence across RT Structure files
'x' = structure exists, blank = structure missing
================================================================================
Structure                  20090801-T1_Post_Contrast 20090801-T2_FLAIR        20090815-T1_Post_Contrast
--------------------------------------------------------------------------------------------------------
Brain                      x                         x                         x                        
Brainstem                  x                         x                         x                        
Cerebellum                 x                         x                         x                        
Eye_L                      x                         x                         x                        
Eye_R                      x                         x                         x                        
Optic_Nerve_L              x                         x                         x                        
Optic_Nerve_R              x                         x                         x                        
PTV                        x                                                    
--------------------------------------------------------------------------------------------------------
Total structures: 7
Total files: 3
```

## File Identification

Files are identified using the format: `{StudyDate}-{SeriesDescription}`

- **StudyDate**: The study date from the DICOM file
- **SeriesDescription**: The series description from the DICOM file
- If no series description is available, it defaults to "RTSTRUCT"

## JSON Output Format

When using the `--output` option, the script saves detailed information in JSON format:

```json
{
  "folder_path": "/data/hengjie/brainmets/dicom/Data/SRS3126",
  "total_rtst_files": 5,
  "total_structures": 25,
  "studies": {
    "20090801": {
      "study_date": "20090801",
      "rtst_files": [...],
      "total_rtst_files": 2,
      "total_structures": 15
    }
  },
  "all_structures": ["Brain", "Brainstem", "Cerebellum", ...],
  "structure_file_map": {
    "Brain": ["20090801-T1_Post_Contrast", "20090801-T2_FLAIR"],
    "PTV": ["20090801-T1_Post_Contrast"]
  },
  "scan_timestamp": "2024-01-15T10:30:45.123456"
}
```

## Error Handling

The script includes comprehensive error handling:

- **Invalid DICOM files**: Skips non-DICOM files gracefully
- **Corrupted RT Structure files**: Reports errors but continues processing
- **Missing metadata**: Uses default values for missing DICOM tags
- **Path validation**: Checks if input paths exist and are directories

## Troubleshooting

### Common Issues

1. **No RT Structure files found**
   - Verify the folder contains RT Structure (RTSTRUCT) files
   - Check that files are valid DICOM files
   - Ensure the modality tag is set to "RTSTRUCT"

2. **Permission errors**
   - Ensure you have read permissions for the folder and files
   - Check if files are locked by other applications

3. **Memory issues with large datasets**
   - The script loads DICOM headers only (not pixel data)
   - For very large datasets, consider processing subsets

### Debug Mode

Use the `--verbose` flag to see detailed error information:

```bash
python overview_all_rtst_folder.py /path/to/folder --verbose
```

## Integration with Other Scripts

The script can be easily integrated with other DICOM processing workflows:

```python
from overview_all_rtst_folder import RTStructureOverviewScanner

# Use in your own scripts
scanner = RTStructureOverviewScanner(Path("/path/to/folder"))
overview_data = scanner.scan_rtst_files()
scanner.print_overview(overview_data)
```

## Performance Notes

- The script only reads DICOM headers (not pixel data) for fast processing
- Processing time scales linearly with the number of RT Structure files
- Memory usage is minimal as only metadata is stored
- Large datasets with hundreds of files should process in seconds

## Version History

- **v1.0**: Initial release with basic RT Structure analysis
- Features: File detection, structure extraction, cross-reference table, JSON export
