"""
==============================================================================
MODULE NAME: dicom_utils.py
==============================================================================

PURPOSE:
    This module provides DICOM-specific utility functions for working with
    medical imaging data, particularly RTStruct (radiation therapy structure)
    files and image geometry.
    
    Key responsibilities:
    - Verify RTStruct file metadata and series descriptions
    - Calculate center slice indices from contour geometry
    - Execute external DICOM overview scripts
    - Handle DICOM spatial coordinate transformations
    
    This module requires both pydicom for DICOM file reading and SimpleITK
    for geometric transformations between physical and image coordinates.

DEPENDENCIES:
    External packages:
    - pydicom: For reading DICOM file metadata and structures
    - SimpleITK (via DICOMImageReader, RTStructReader): For geometric transformations
    - numpy: For array operations on contour points
    - subprocess: For running external DICOM overview scripts
    - sys: For accessing Python interpreter path
    
    Internal modules:
    - path_utils: For Path objects (implicit dependency)

FUNCTIONS:
    1. verify_rtstruct_series_description(rtstruct_folder: Path,
                                         expected_description: str) -> bool
       Verify that RTStruct series description matches expected value
       
    2. get_center_slice_from_contours(rtstruct_reader, structure_name: str,
                                     image_reader) -> int
       Calculate center slice index from contour Z-coordinates
       
    3. run_overview_script(folder_path: Path) -> tuple
       Execute DICOM overview script and capture output
       
    4. read_series_description(folder_path: Path) -> str
       Read series description from the first DICOM file in a folder

USAGE EXAMPLE:
    ```python
    from code.RTdicomorganizer import dicom_utils
    from pathlib import Path
    
    # Verify RTStruct metadata
    rtstruct_folder = Path('/path/to/rtstruct/folder')
    is_valid = dicom_utils.verify_rtstruct_series_description(
        rtstruct_folder,
        "Brain_MS_Init_Model"
    )
    
    # Get center slice for visualization
    from dicomreader.RTStructReader import RTStructReader
    from dicomreader.DICOMImageReader import DICOMImageReader
    
    rtstruct_reader = RTStructReader(str(rtstruct_folder))
    rtstruct_reader.read()
    
    image_reader = DICOMImageReader(str(mr_folder), modality='MR')
    image_reader.read()
    
    center_slice = dicom_utils.get_center_slice_from_contours(
        rtstruct_reader,
        'target1',
        image_reader
    )
    ```

NOTES:
    - Requires DICOMImageReader and RTStructReader classes (not included here)
    - Uses SimpleITK's coordinate transformation methods
    - Z-coordinate convention follows DICOM standard (patient-based coordinates)
    - Debug output can be enabled via DEBUG global variable

==============================================================================
"""

import sys
import subprocess
from pathlib import Path
import pydicom
import numpy as np


def verify_rtstruct_series_description(rtstruct_folder: Path,
                                      expected_description: str) -> bool:
    """
    Verify that RTStruct series description matches expected value.
    
    DETAILED DESCRIPTION:
        This function reads the DICOM metadata from an RTStruct folder to
        verify that it contains the expected series description. This is
        used for QC to ensure the correct RTStruct file is being used.
        
        PROCESS:
        1. Find first DICOM file in the folder
        2. Read DICOM metadata (without pixel data for efficiency)
        3. Verify Modality is "RTSTRUCT"
        4. Check SeriesDescription tag matches expected value
        
        This validation helps catch cases where folder naming doesn't match
        the actual DICOM content.
    
    Args:
        rtstruct_folder (Path): Path to folder containing RTStruct DICOM file(s)
            - Should contain at least one DICOM file
            - Example: Path('/database/.../SRS0871_..._Brain.MS.Init.Model_...')
            
        expected_description (str): Expected value of SeriesDescription tag
            - Case-sensitive exact match required
            - Example: 'Brain_MS_Init_Model', 'Brain_MS_ReTx_Model'
    
    Returns:
        bool: True if series description matches, False otherwise
            - False if folder empty
            - False if file is not RTSTRUCT
            - False if description doesn't match
            - False if any error occurs during reading
    
    Example:
        >>> from pathlib import Path
        >>> rtstruct_folder = Path('/database/.../Brain.MS.Init.Model_n1__00000')
        
        >>> # Correct description
        >>> is_valid = verify_rtstruct_series_description(
        ...     rtstruct_folder,
        ...     'Brain_MS_Init_Model'
        ... )
        >>> print(is_valid)
        True
        
        >>> # Wrong description
        >>> is_valid = verify_rtstruct_series_description(
        ...     rtstruct_folder,
        ...     'Brain_MS_ReTx_Model'  # Wrong!
        ... )
        >>> print(is_valid)
        False  # Prints warning message
    
    Notes:
        - Checks only the first DICOM file found
        - Reads metadata only (stop_before_pixels=True for efficiency)
        - Prints warning messages for common failure modes
        - Returns False rather than raising exceptions for robustness
        - Case-sensitive matching on SeriesDescription
    """
    # Find the first DICOM file in the folder
    dicom_files = [f for f in rtstruct_folder.iterdir() if f.is_file()]
    
    if not dicom_files:
        print(f"WARNING: No files found in {rtstruct_folder}")
        return False
    
    try:
        # Read the first DICOM file (metadata only, no pixels)
        ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
        
        # Check if it's an RTStruct and verify series description
        if ds.Modality == "RTSTRUCT":
            series_desc = ds.get('SeriesDescription', '')
            if series_desc == expected_description:
                return True
            else:
                print(
                    f"WARNING: Series description mismatch.\n"
                    f"  Expected: '{expected_description}'\n"
                    f"  Got: '{series_desc}'\n"
                    f"  Folder: {rtstruct_folder.name}"
                )
                return False
        else:
            print(
                f"WARNING: File is not an RTSTRUCT (Modality: {ds.Modality})\n"
                f"  File: {dicom_files[0].name}"
            )
            return False
            
    except Exception as e:
        print(f"ERROR reading DICOM file: {e}\n  File: {dicom_files[0]}")
        return False


def get_center_slice_from_contours(rtstruct_reader, structure_name: str,
                                   image_reader, debug: bool = False) -> int:
    """
    Calculate the center slice index based on contour Z-coordinates.
    
    DETAILED DESCRIPTION:
        This function determines which slice to display by analyzing the
        3D geometry of RTStruct contours. It extracts all unique Z-coordinates
        where contours exist, finds the middle Z-coordinate, and transforms
        it to the corresponding slice index in the image volume.
        
        PROCESS:
        1. Look up structure in RTStruct by name
        2. Extract all contour points from all contour sequences
        3. Collect unique Z-coordinates (physical space, mm)
        4. Sort Z-coordinates and select the middle one
        5. Transform physical Z-coordinate to image slice index
        6. Return slice index (or middle slice if transformation fails)
        
        This is more robust than simply using the middle slice of the image,
        as it accounts for the actual spatial extent of the target lesion.
    
    Args:
        rtstruct_reader: RTStructReader instance with loaded RTStruct
            - Must have read() method already called
            - Should contain the specified structure
            - Example: RTStructReader object from dicomreader package
            
        structure_name (str): Name of structure to analyze
            - Case-sensitive
            - Example: 'target1', 'target9', 'target15'
            
        image_reader: DICOMImageReader instance with loaded MR/CT image
            - Must have read() method already called
            - Used for coordinate transformation
            - Example: DICOMImageReader object from dicomreader package
            
        debug (bool): If True, print detailed coordinate information
            - Default: False
            - Useful for troubleshooting slice selection
    
    Returns:
        int: Slice index (0-based) representing center of contours
            - Range: [0, num_slices-1]
            - Falls back to middle slice if any error occurs
            - Example: 23 (for 47-slice volume)
    
    Example:
        >>> from dicomreader.RTStructReader import RTStructReader
        >>> from dicomreader.DICOMImageReader import DICOMImageReader
        >>> from code.RTdicomorganizer import dicom_utils
        
        >>> # Load RTStruct and image
        >>> rtstruct_reader = RTStructReader('/path/to/rtstruct')
        >>> rtstruct_reader.read()
        >>> image_reader = DICOMImageReader('/path/to/mr', modality='MR')
        >>> image_reader.read()
        
        >>> # Get center slice for target1
        >>> center_slice = dicom_utils.get_center_slice_from_contours(
        ...     rtstruct_reader,
        ...     'target1',
        ...     image_reader,
        ...     debug=True
        ... )
        >>> print(f"Display slice {center_slice}")
        Display slice 23
    
    DEBUG OUTPUT (when debug=True):
        - Number of unique Z-coordinates in contours
        - All Z-coordinates in mm
        - Selected center Z-coordinate
        - Image origin, spacing, and size
        - Transformed slice index
        - Verification of transformation accuracy
    
    Notes:
        - Uses SimpleITK's TransformPhysicalPointToIndex for accuracy
        - Handles missing structures gracefully (returns middle slice)
        - Physical coordinates are in DICOM patient coordinate system (mm)
        - Slice index may not match expected if image orientation is unusual
        - Falls back to middle slice on any error rather than crashing
    """
    try:
        # Get structure index from structure name
        structure_index = rtstruct_reader.get_structure_index(structure_name)
        contour_data = rtstruct_reader.roi_contours[structure_index]
        
        # Collect all unique Z-coordinates from contours
        z_coords = set()
        for contour in contour_data.ContourSequence:
            contour_points = np.array(contour.ContourData).reshape(-1, 3)
            # Extract Z-coordinates (3rd column, physical space in mm)
            z_coords.update(contour_points[:, 2])
        
        # Sort Z-coordinates and get the center one
        z_coords_sorted = sorted(z_coords)
        
        if len(z_coords_sorted) == 0:
            # No contours found, return middle slice
            image_array = image_reader.get_image_array()
            fallback_slice = image_array.shape[0] // 2
            if debug:
                print(
                    f"    [DEBUG] No contours found for {structure_name}, "
                    f"using middle slice: {fallback_slice}"
                )
            return fallback_slice
        
        # Get center Z-coordinate
        center_idx = len(z_coords_sorted) // 2
        center_z = z_coords_sorted[center_idx]
        
        if debug:
            print(f"    [DEBUG] Slice Selection for {structure_name}:")
            print(f"    [DEBUG]   - Number of unique Z-coordinates in contours: {len(z_coords_sorted)}")
            print(f"    [DEBUG]   - All Z-coordinates (mm): {[f'{z:.2f}' for z in z_coords_sorted]}")
            print(f"    [DEBUG]   - Selected center Z (index {center_idx}): {center_z:.2f} mm")
        
        # Convert physical Z-coordinate to slice index
        # Use the image to transform physical point to index
        image = image_reader.get_image()
        
        # Get image properties for debugging
        if debug:
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            size = image.GetSize()
            print(f"    [DEBUG]   - Image origin (mm): ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
            print(f"    [DEBUG]   - Image spacing (mm): ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f})")
            print(f"    [DEBUG]   - Image size (voxels): ({size[0]}, {size[1]}, {size[2]})")
        
        # Create a dummy point at the center Z with arbitrary X, Y
        origin = image.GetOrigin()
        dummy_point = (origin[0], origin[1], center_z)
        
        try:
            # Transform physical point to image index
            slice_idx = image.TransformPhysicalPointToIndex(dummy_point)[2]
            
            if debug:
                # Calculate what the physical Z would be for this slice index
                # by transforming the slice index back to physical coordinates
                index_point = (0, 0, slice_idx)
                physical_point = image.TransformIndexToPhysicalPoint(index_point)
                print(f"    [DEBUG]   - Transformed to slice index: {slice_idx}")
                print(f"    [DEBUG]   - Verification: slice {slice_idx} corresponds to Z = {physical_point[2]:.2f} mm")
                
                # Show the Z-range of the image
                first_slice_z = image.TransformIndexToPhysicalPoint((0, 0, 0))[2]
                last_slice_z = image.TransformIndexToPhysicalPoint((0, 0, size[2]-1))[2]
                print(f"    [DEBUG]   - Image Z-range: {first_slice_z:.2f} mm to {last_slice_z:.2f} mm")
            
            return int(slice_idx)
            
        except Exception as transform_error:
            # If transformation fails, return middle slice
            image_array = image_reader.get_image_array()
            fallback_slice = image_array.shape[0] // 2
            if debug:
                print(
                    f"    [DEBUG] Transformation failed ({transform_error}), "
                    f"using middle slice: {fallback_slice}"
                )
            return fallback_slice
            
    except Exception as e:
        print(f"WARNING: Could not determine center slice from contours: {e}")
        # Return middle slice as fallback
        image_array = image_reader.get_image_array()
        fallback_slice = image_array.shape[0] // 2
        if debug:
            print(f"    [DEBUG] Exception occurred, using middle slice: {fallback_slice}")
        return fallback_slice


def run_overview_script(folder_path: Path, script_path: str = None) -> tuple:
    """
    Run external DICOM overview script and capture output.
    
    DETAILED DESCRIPTION:
        This function executes an external Python script that generates
        a comprehensive overview of DICOM folder contents. It captures
        both stdout and stderr, allowing the calling code to process or
        display the results.
        
        The overview script typically analyzes:
        - Patient IDs and study information
        - Series counts by modality (MR, CT, RTSTRUCT, RTDOSE)
        - Series descriptions and metadata
        - File organization and structure
        
        This function provides timeout protection and graceful error handling.
    
    Args:
        folder_path (Path): Path to DICOM folder to analyze
            - Should be a valid directory
            - Example: Path('/database/.../SRS0871/1997-07__Studies')
            
        script_path (str): Optional path to overview script
            - If None, uses default hardcoded path
            - Example: '/homebase/DL_projects/.../overview_all_dicom_folder.py'
    
    Returns:
        tuple: (success, stdout, stderr)
            - success (bool): True if script ran successfully (returncode == 0)
            - stdout (str): Standard output from script
            - stderr (str): Standard error from script
            - On timeout or error: (False, "", error_message)
    
    Example:
        >>> from pathlib import Path
        >>> from code.RTdicomorganizer import dicom_utils
        
        >>> folder = Path('/database/.../SRS0871/1997-07__Studies')
        >>> success, output, errors = dicom_utils.run_overview_script(folder)
        
        >>> if success:
        ...     print("DICOM Overview:")
        ...     print(output)
        ... else:
        ...     print(f"Error: {errors}")
    
    OUTPUT FORMAT (from script):
        Typical output includes:
        ================================================================================
        DICOM FOLDER OVERVIEW
        ================================================================================
        Total Patients: 1
        Total Studies: 1
        
        Patient Summary:
        Patient ID      Studies 
        -------------------------
        SRS0871         1       
        
        Study Details:
        Study Date: 1997-07-18
        
        Series by Modality:
        Modality    Count  Description
        -------------------------
        MR          1      Axial T1 contrast
        RTSTRUCT    1      Brain_MS_Init_Model
        RTDOSE      1      Treatment plan
    
    Notes:
        - Uses subprocess.run() with capture_output=True
        - 60-second timeout to prevent hanging
        - Uses current Python interpreter (sys.executable)
        - Returns empty strings for stdout/stderr on timeout
        - Error messages are descriptive for troubleshooting
        - Script path can be overridden for testing or alternative scripts
    """
    # Default script path if not provided
    if script_path is None:
        script_path = (
            "/homebase/DL_projects/code_sync/myMedIADIRLab/"
            "data_io/dicom_reader/overview_all_dicom_folder.py"
        )
    
    # Check if folder exists
    if not folder_path.exists():
        return False, "", f"Folder does not exist: {folder_path}"
    
    try:
        # Run the overview script
        result = subprocess.run(
            [sys.executable, str(script_path), str(folder_path)],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "", "Script execution timed out (>60 seconds)"
        
    except Exception as e:
        return False, "", f"Error running script: {str(e)}"


def read_series_description(folder_path: Path) -> str:
    """
    Read series description from the first DICOM file in a folder.
    
    DETAILED DESCRIPTION:
        This function reads the SeriesDescription tag from the first DICOM file
        found in the specified folder. The series description is a human-readable
        string that describes the imaging series (e.g., "T1_axial_post_gad",
        "Brain_MS_Init_Model", "FLAIR").
        
        This function is useful for:
        - Building folder inventories with descriptive metadata
        - Identifying specific series types without manual inspection
        - Quality control and data organization
        
        The function reads only the first valid DICOM file found for efficiency.
        If the folder contains mixed modalities or series, only the first file's
        description will be returned.
        
        Common series descriptions in brain metastasis studies:
        - MR: "T1_axial_post_gad", "T2_FLAIR", "T1_pre_contrast"
        - RTStruct: "Brain_MS_Init_Model", "Brain_MS_ReTx_Model"
        - RTDose: "Treatment_Dose", "Plan_Dose"
    
    Args:
        folder_path (Path): Path to folder containing DICOM files
            - Should be a valid directory path
            - Should contain at least one valid DICOM file
            - Example: Path('/.../SRS0871_..._MR_...axial_n47__00000')
    
    Returns:
        str: Series description from DICOM file
            - Returns the value of the SeriesDescription tag (0008,103E)
            - Returns "N/A" if folder doesn't exist
            - Returns "Unknown" if no valid DICOM files found
            - Returns "No Description" if tag is missing or empty
            - Example: "T1_axial_post_gad", "Brain_MS_Init_Model"
    
    Example:
        >>> from pathlib import Path
        >>> mr_folder = Path('/.../SRS0871_..._MR_...axial_n47__00000')
        >>> description = read_series_description(mr_folder)
        >>> print(f"Series: {description}")
        Series: T1_axial_post_gad
        
        >>> rtstruct_folder = Path('/.../SRS0871_..._Brain.MS.Init.Model_n1__00000')
        >>> description = read_series_description(rtstruct_folder)
        >>> print(f"RTStruct: {description}")
        RTStruct: Brain_MS_Init_Model
        
        >>> # Handle missing folder
        >>> description = read_series_description(Path('/nonexistent'))
        >>> print(description)
        N/A
    
    Notes:
        - Only reads the first DICOM file found for efficiency
        - Does not validate that all files in folder have same series description
        - Silently handles missing or malformed DICOM files
        - SeriesDescription tag is optional in DICOM standard
        - Some older DICOM files may not have this tag
        - Returns cleaned string with trailing whitespace removed
    """
    if not folder_path.exists():
        return "N/A"
    
    # Try to find a valid DICOM file
    for dicom_file in folder_path.iterdir():
        if not dicom_file.is_file():
            continue
            
        try:
            # Read DICOM metadata without loading pixel data
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
            
            # Try to get SeriesDescription tag (0008,103E)
            if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription:
                return str(ds.SeriesDescription).strip()
            else:
                return "No Description"
                
        except Exception:
            # If file is not a valid DICOM, try next file
            continue
    
    # No valid DICOM files found
    return "Unknown"

