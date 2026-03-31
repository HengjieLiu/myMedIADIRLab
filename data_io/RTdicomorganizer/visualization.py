"""
==============================================================================
MODULE NAME: visualization.py
==============================================================================

PURPOSE:
    This module provides visualization functions for displaying medical
    imaging data with overlaid contour structures. It creates publication-
    quality figures for quality control, documentation, and analysis.
    
    Key responsibilities:
    - Display MR/CT images with RTStruct contour overlays
    - Handle slice selection based on contour geometry
    - Apply appropriate color coding from RTStruct metadata
    - Create clear, annotated visualizations
    
    This module bridges the gap between raw DICOM data and visual inspection,
    making it easy to verify that contours align correctly with anatomical
    structures.

DEPENDENCIES:
    External packages:
    - matplotlib.pyplot: For creating figures and displaying images
    - numpy: For array operations on images and contours
    
    Internal modules:
    - dicom_utils: Uses get_center_slice_from_contours

FUNCTIONS:
    1. display_mr_with_contour(image_reader, rtstruct_reader,
                               structure_name: str, title: str) -> None
       Display MR image slice with RTStruct contour overlay

USAGE EXAMPLE:
    ```python
    from code.RTdicomorganizer import visualization
    from dicomreader.DICOMImageReader import DICOMImageReader
    from dicomreader.RTStructReader import RTStructReader
    
    # Load image and structure
    mr_reader = DICOMImageReader('/path/to/mr', modality='MR')
    mr_reader.read()
    
    rtstruct_reader = RTStructReader('/path/to/rtstruct')
    rtstruct_reader.read()
    
    # Display with contour overlay
    visualization.display_mr_with_contour(
        mr_reader,
        rtstruct_reader,
        structure_name='target1',
        title='Patient 0871 - Target 01 - Initial Scan'
    )
    ```

NOTES:
    - Displays the center slice based on contour Z-coordinates
    - Contours are drawn as outlines (not filled polygons)
    - Uses structure color from RTStruct if available
    - Falls back to red color if structure color not available

==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from . import dicom_utils


def display_mr_with_contour(image_reader, rtstruct_reader, 
                            structure_name: str, title: str) -> None:
    """
    Display MR image slice with RTStruct contour overlay.
    
    DETAILED DESCRIPTION:
        This function creates a visualization of a medical image (typically MR)
        with overlaid contour structures from an RTStruct file. It performs
        the following steps:
        
        1. Calculate center slice index from contour geometry
        2. Extract image slice at that index
        3. Get contour points in pixel coordinates for that slice
        4. Extract structure color from RTStruct metadata
        5. Create matplotlib figure with image and contour overlay
        6. Add title showing slice number
        7. Display figure
        
        This is essential for quality control to verify:
        - Correct image-contour alignment
        - Accurate structure delineation
        - Proper target lesion identification
    
    Args:
        image_reader: DICOMImageReader instance with loaded image
            - Must have read() method already called
            - Typically MR or CT image
            - Example: DICOMImageReader object from dicomreader package
            
        rtstruct_reader: RTStructReader instance with loaded RTStruct
            - Must have read() method already called
            - Contains structure contours and metadata
            - Example: RTStructReader object from dicomreader package
            
        structure_name (str): Name of structure to display
            - Must exist in RTStruct file
            - Case-sensitive
            - Example: 'target1', 'target9', 'Brain_target1'
            
        title (str): Title for the figure
            - Will be displayed above image
            - Should include identifying information
            - Example: '0871.01 - Initial - MR Series 1'
    
    Returns:
        None: Creates and displays matplotlib figure
            - Figure remains open until user closes window
            - Can be saved manually via matplotlib interface
    
    VISUALIZATION DETAILS:
        - Image displayed in grayscale
        - Contours overlaid as colored lines (linewidth=2)
        - Contour color from RTStruct metadata (or red as fallback)
        - Slice number shown in title
        - Axes hidden for cleaner appearance
        - Warning text if no contours on displayed slice
    
    Example:
        >>> from dicomreader.DICOMImageReader import DICOMImageReader
        >>> from dicomreader.RTStructReader import RTStructReader
        >>> from code.RTdicomorganizer import visualization
        
        >>> # Load data
        >>> mr_reader = DICOMImageReader(
        ...     '/database/.../SRS0871_..._MR_.../',
        ...     modality='MR'
        ... )
        >>> mr_reader.read()
        
        >>> rtstruct_reader = RTStructReader(
        ...     '/database/.../SRS0871_..._Brain.MS.Init.Model_.../'
        ... )
        >>> rtstruct_reader.read()
        
        >>> # Display target1 contour
        >>> visualization.display_mr_with_contour(
        ...     mr_reader,
        ...     rtstruct_reader,
        ...     structure_name='target1',
        ...     title='Patient 0871 - Target 01 - Initial Scan'
        ... )
        
        >>> # Will display figure with:
        >>> # - Grayscale MR image
        >>> # - Colored contour outline
        >>> # - Title: "Patient 0871 - Target 01 - Initial Scan\nSlice: 23/47"
    
    CONTOUR VISUALIZATION:
        - Contours drawn as lines, not filled polygons
        - Multiple contours on same slice are all drawn
        - Line width: 2 pixels
        - Color: From RTStruct metadata (typically unique per structure)
        - If no contours on slice: Yellow warning box displayed
    
    Notes:
        - Uses get_center_slice_from_contours() from dicom_utils
        - Contours in pixel space (not physical coordinates)
        - Slice selection accounts for actual contour extent
        - Figure size: 10x10 inches
        - DPI determined by matplotlib defaults
        - plt.show() blocks until figure closed
    """
    # Get the center slice index based on contour geometry
    center_slice_idx = dicom_utils.get_center_slice_from_contours(
        rtstruct_reader, 
        structure_name, 
        image_reader
    )
    
    # Get image array (shape: [slices, height, width])
    image_array = image_reader.get_image_array()
    
    # Get contour points in pixel space
    try:
        contours_dict = rtstruct_reader.get_structure_contour_points_in_pixel_space(
            structure_name, 
            image_reader
        )
        
        # Get structure color from RTStruct metadata
        try:
            color_rgb = rtstruct_reader.get_structure_color(structure_name)
            # Normalize RGB values from [0, 255] to [0, 1] for matplotlib
            color_normalized = [c / 255.0 for c in color_rgb]
        except:
            # Default to red if color not available
            color_normalized = [1.0, 0.0, 0.0]
        
    except Exception as e:
        print(f"WARNING: Could not load contours for {structure_name}: {e}")
        contours_dict = {}
        color_normalized = [1.0, 0.0, 0.0]  # Red
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display the image slice in grayscale
    ax.imshow(image_array[center_slice_idx, :, :], cmap='gray')
    
    # Add title showing slice number
    ax.set_title(
        f"{title}\nSlice: {center_slice_idx}/{image_array.shape[0]}",
        fontsize=12
    )
    
    # Overlay contours if available for this slice
    if center_slice_idx in contours_dict:
        for contour in contours_dict[center_slice_idx]:
            contour_array = np.array(contour)
            # Plot contour as line (not filled)
            # contour_array columns: [row, col] = [y, x] in image coordinates
            ax.plot(
                contour_array[:, 1],  # x-coordinates (columns)
                contour_array[:, 0],  # y-coordinates (rows)
                color=color_normalized,
                linewidth=2,
                label=structure_name
            )
    else:
        # No contours on this slice - add warning text
        ax.text(
            0.5, 0.95,
            f"No contours on this slice",
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )
    
    # Hide axes for cleaner appearance
    ax.axis('off')
    
    # Adjust layout to prevent title cutoff
    plt.tight_layout()
    
    # Display the figure
    plt.show()

