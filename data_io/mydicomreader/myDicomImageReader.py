"""
DICOM Image Reader for medical imaging data.

This module provides a reader class for DICOM image files and series, supporting both
single files and series of 2D images. It handles metadata extraction, geometry validation,
and image loading using SimpleITK and pydicom.
"""

import os
import numpy as np
import pydicom
import SimpleITK as sitk
from .myDicomHeader import type_map


class myDicomImageReader:
    def __init__(self, dicom_path, modality=None):
        """
        Initialize the DICOM image reader with a path to a DICOM file or folder.
        
        Parameters:
        -----------
        dicom_path : str
            Path to a DICOM file or folder containing DICOM file(s).
            - If a file: reads a single DICOM image
            - If a folder with one file: reads a single DICOM image
            - If a folder with multiple files: reads as a DICOM series
        modality : str
            Expected DICOM modality (e.g., 'CT', 'MR', 'PT'). Must be provided.
            The reader will validate that the DICOM file(s) match this modality.
            
        Raises:
        -------
        AssertionError
            If modality is not provided
        FileNotFoundError
            If the path doesn't exist
        ValueError
            If folder is empty, contains multiple DICOM series, or modality doesn't match
        """
        # Assert that modality is provided
        assert modality is not None, "modality parameter must be provided (e.g., 'CT', 'MR', 'PT')"
        
        self.dicom_path = dicom_path
        self.modality = modality
        self.is_series = False
        self.series_file_names = []
        
        # Get file names from path using the static helper method
        file_names = self._get_file_names_from_path(dicom_path)
        
        # Determine if it's a single file or series based on number of files
        if len(file_names) == 1:
            # Single DICOM file
            self.is_series = False
            self.file_path = file_names[0]
            self.series_file_names = file_names
            if os.path.isfile(dicom_path):
                print(f"[Info] Reading single DICOM file: {os.path.basename(dicom_path)}")
            else:
                print(f"[Info] Reading single DICOM file from folder: {os.path.basename(self.file_path)}")
        else:
            # Multiple files - it's a series
            self.is_series = True
            self.series_file_names = file_names
            self.file_path = file_names[0]  # First slice for metadata
            print(f"[Info] Reading DICOM series from folder: {dicom_path}")
            print(f"[Info] Series contains {len(file_names)} slices")
        
        # Validate that it's a DICOM file with the correct modality
        try:
            ds = pydicom.dcmread(self.file_path, stop_before_pixels=True)
            if not hasattr(ds, 'Modality') or ds.Modality != modality:
                raise ValueError(f"DICOM file modality is '{ds.Modality if hasattr(ds, 'Modality') else 'Unknown'}', expected modality: {modality}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            else:
                raise ValueError(f"Failed to read DICOM file or file is not a valid DICOM: {self.file_path}. Error: {str(e)}")
    
    def print_metadata_pydicom(self):
        """
        Print all DICOM metadata using pydicom in a nicely formatted table.
        For series, prints metadata from the first slice.
        """
        ds = pydicom.dcmread(self.file_path)
        
        print("="*120)
        if self.is_series:
            print(f"DICOM Metadata (using pydicom) - First slice of series: {os.path.basename(self.file_path)}")
        else:
            print(f"DICOM Metadata (using pydicom) - File: {os.path.basename(self.file_path)}")
        print("="*120)
        print(f"{'Keyword':<35} {'Tag':<15} {'VR':<5} {'VM':<5} {'Type':<20} {'Value'}")
        print("-"*120)
        
        for elem in ds:
            # Skip pixel data and large binary data
            if elem.tag == 0x7FE00010:  # Pixel Data
                value_str = "[Pixel Data - not displayed]"
            elif elem.VR in ['OB', 'OW', 'OD', 'OF']:
                value_str = f"[Binary data - {len(elem.value)} bytes]" if hasattr(elem, 'value') else "[Binary data]"
            else:
                try:
                    value = elem.value
                    if isinstance(value, bytes):
                        value_str = value.decode('utf-8', errors='ignore')
                    elif isinstance(value, (list, tuple)) and len(value) > 5:
                        value_str = f"[{value[0]}, {value[1]}, ..., {value[-1]}] (length: {len(value)})"
                    else:
                        value_str = str(value)
                    # Limit value string length
                    if len(value_str) > 80:
                        value_str = value_str[:77] + "..."
                except:
                    value_str = "[Unable to display]"
            
            keyword = elem.keyword if elem.keyword else "(Unknown)"
            tag_str = f"({elem.tag.group:04X},{elem.tag.element:04X})"
            vr = elem.VR if elem.VR else "??"
            vm = str(elem.VM) if hasattr(elem, 'VM') else "?"
            
            elem_type = type_map.get(vr, 'Other')
            
            print(f"{keyword:<35} {tag_str:<15} {vr:<5} {vm:<5} {elem_type:<20} {value_str}")
        
        print("="*120)
    
    def print_metadata_sitk(self):
        """
        Print all DICOM metadata using SimpleITK for values and pydicom for keyword lookup.
        For series, prints metadata from the first slice.
        """
        # Read with SimpleITK to get metadata values
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.file_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # Read with pydicom only for keyword extraction (don't load pixel data)
        ds = pydicom.dcmread(self.file_path, stop_before_pixels=True)
        
        # Create a mapping from tag to pydicom element for quick lookup
        tag_to_elem = {}
        for elem in ds:
            tag_to_elem[elem.tag] = elem
        
        print("="*120)
        if self.is_series:
            print(f"DICOM Metadata (using SimpleITK) - First slice of series: {os.path.basename(self.file_path)}")
        else:
            print(f"DICOM Metadata (using SimpleITK) - File: {os.path.basename(self.file_path)}")
        print("="*120)
        print(f"{'Keyword':<35} {'Tag':<15} {'VR':<5} {'VM':<5} {'Type':<20} {'Value'}")
        print("-"*120)
        
        # Get all metadata keys from SimpleITK
        keys = reader.GetMetaDataKeys()
        
        # Sort keys by tag (group, element)
        def sort_key(key):
            if '|' in key:
                parts = key.split('|')
                if len(parts) >= 2:
                    try:
                        group = int(parts[0], 16)
                        element = int(parts[1], 16)
                        return (group, element)
                    except:
                        return (0, 0)
            return (0, 0)
        
        sorted_keys = sorted(keys, key=sort_key)
        
        for key in sorted_keys:
            try:
                # Get value from SimpleITK
                value = reader.GetMetaData(key)
                
                # Parse SimpleITK key format (e.g., "0008|0005")
                keyword = "(Unknown)"
                tag_str = key
                vr = "??"
                vm = "?"
                elem_type = "String"
                
                if '|' in key:
                    parts = key.split('|')
                    if len(parts) >= 2:
                        try:
                            group = int(parts[0], 16)
                            element = int(parts[1], 16)
                            tag = pydicom.tag.Tag(group, element)
                            tag_str = f"({group:04X},{element:04X})"
                            
                            # Look up keyword and metadata from pydicom
                            if tag in tag_to_elem:
                                elem = tag_to_elem[tag]
                                keyword = elem.keyword if elem.keyword else "(Unknown)"
                                vr = elem.VR if elem.VR else "??"
                                vm = str(elem.VM) if hasattr(elem, 'VM') else "?"
                                
                                elem_type = type_map.get(vr, 'Other')
                        except:
                            pass
                
                # Format value string
                if len(value) > 80:
                    value_str = value[:77] + "..."
                else:
                    value_str = value
                
                print(f"{keyword:<35} {tag_str:<15} {vr:<5} {vm:<5} {elem_type:<20} {value_str}")
                
            except Exception as e:
                print(f"{key:<35} {'':<15} {'??':<5} {'?':<5} {'Unknown':<20} [Unable to read: {str(e)}]")
        
        print("="*120)
        print(f"\nImage Information:")
        print(f"  Dimensions: {reader.GetSize()}")
        print(f"  Spacing: {reader.GetSpacing()}")
        print(f"  Origin: {reader.GetOrigin()}")
        print("="*120)
    
    @staticmethod
    def _get_file_names_from_path(dicom_path):
        """
        Extract file names from a DICOM path (file or directory).
        
        Static helper method to get a list of DICOM file paths from either a single file
        or a directory containing a DICOM series. This is used both during initialization
        and for the geometry override feature.
        
        Parameters:
        -----------
        dicom_path : str
            Path to a DICOM file or folder containing DICOM file(s)
            
        Returns:
        --------
        list of str
            List of DICOM file paths (single element for file, multiple for series)
            
        Raises:
        -------
        FileNotFoundError
            If the path doesn't exist
        ValueError
            If folder is empty or contains multiple DICOM series
        """
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"Path does not exist: {dicom_path}")
        
        if os.path.isfile(dicom_path):
            # Single DICOM file
            return [dicom_path]
            
        elif os.path.isdir(dicom_path):
            # Directory - check for DICOM series
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_path)
            
            if len(series_IDs) == 0:
                raise ValueError(f"No DICOM series found in directory: {dicom_path}")
            elif len(series_IDs) > 1:
                raise ValueError(f"Multiple DICOM series found ({len(series_IDs)} series) in directory: {dicom_path}. "
                               f"Please provide a directory with exactly one series. Series IDs: {series_IDs}")
            
            # Get file names for the single series
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_path, series_IDs[0])
            
            if len(file_names) == 0:
                raise ValueError(f"No DICOM files found in series: {dicom_path}")
            
            return file_names
        else:
            raise ValueError(f"Path is neither a file nor a directory: {dicom_path}")
    
    def _get_geometry_from_dicom(self, file_names):
        """
        Extract geometry information (Origin, Direction, Spacing) directly from DICOM headers.
        
        This provides the ground truth geometry by reading the DICOM metadata fields:
        - ImagePositionPatient for Origin
        - ImageOrientationPatient for Direction
        - PixelSpacing for in-plane spacing
        - SliceThickness or inter-slice distance for Z-spacing
        
        Parameters:
        -----------
        file_names : list of str
            List of DICOM file paths (for series) or single file path (for single image)
            
        Returns:
        --------
        tuple : (origin, direction, spacing)
            origin : tuple of 3 floats
                Image origin in physical space (x, y, z)
            direction : tuple of 9 floats
                Direction cosine matrix flattened (row-major: xx, xy, xz, yx, yy, yz, zx, zy, zz)
            spacing : tuple of 3 floats
                Voxel spacing (x, y, z)
                
        Raises:
        -------
        ValueError
            If required DICOM tags are missing
        """
        # Ensure file_names is a list
        if isinstance(file_names, str):
            file_names = [file_names]
        
        # Read the first DICOM file headers
        first_ds = pydicom.dcmread(file_names[0], stop_before_pixels=True)
        
        # --- 1. Extract Origin ---
        if 'ImagePositionPatient' not in first_ds:
            raise ValueError("ImagePositionPatient not found in DICOM header. Cannot determine origin.")
        origin = tuple([float(x) for x in first_ds.ImagePositionPatient])
        
        # --- 2. Extract Direction ---
        if 'ImageOrientationPatient' not in first_ds:
            raise ValueError("ImageOrientationPatient not found in DICOM header. Cannot determine direction.")
        
        iop = [float(x) for x in first_ds.ImageOrientationPatient]
        row_cosines = np.array(iop[:3])
        col_cosines = np.array(iop[3:])
        
        # Compute Z-axis direction (slice direction) via cross product
        slice_direction = np.cross(row_cosines, col_cosines)
        
        # Flatten the 3 vectors into a 9-component tuple
        direction = tuple(row_cosines) + tuple(col_cosines) + tuple(slice_direction)
        
        # --- 3. Extract Spacing ---
        # Get in-plane spacing
        if 'PixelSpacing' not in first_ds:
            raise ValueError("PixelSpacing not found in DICOM header. Cannot determine spacing.")
        
        spacing_x = float(first_ds.PixelSpacing[0])
        spacing_y = float(first_ds.PixelSpacing[1])
        
        # Calculate Z-spacing
        spacing_z = None
        
        if len(file_names) > 1:
            # For series: calculate from distance between first two slices
            try:
                p1 = np.array(first_ds.ImagePositionPatient, dtype=float)
                ds2 = pydicom.dcmread(file_names[1], stop_before_pixels=True)
                p2 = np.array(ds2.ImagePositionPatient, dtype=float)
                # Euclidean distance between first two slices
                spacing_z = float(np.linalg.norm(p2 - p1))
            except Exception as e:
                # Fallback to SliceThickness if available
                if 'SliceThickness' in first_ds:
                    spacing_z = float(first_ds.SliceThickness)
                    print(f"[Warning] Could not calculate Z-spacing from slice positions: {e}")
                    print(f"[Warning] Using SliceThickness tag instead: {spacing_z}")
                else:
                    raise ValueError(f"Could not determine Z-spacing: {e}")
        else:
            # For single slice: use SliceThickness if available, otherwise default to 1.0
            if 'SliceThickness' in first_ds:
                spacing_z = float(first_ds.SliceThickness)
            else:
                spacing_z = 1.0
                print(f"[Warning] SliceThickness not found for single slice. Using default Z-spacing: {spacing_z}")
        
        spacing = (spacing_x, spacing_y, spacing_z)
        
        return origin, direction, spacing
    
    def _restore_geometry_from_dicom(self, image, file_names):
        """
        Restore geometry (Origin, Direction, Spacing) to a SimpleITK image from DICOM headers.
        
        This method manually sets the geometry metadata on a SimpleITK image object using
        values extracted directly from DICOM headers. This is useful when SimpleITK defaults
        or incorrectly sets geometry during image reading.
        
        Parameters:
        -----------
        image : sitk.Image
            SimpleITK image object to modify
        file_names : list of str or str
            List of DICOM file paths (for series) or single file path
            
        Returns:
        --------
        sitk.Image
            The modified image with restored geometry (same object, modified in-place)
        """
        try:
            # Get ground truth geometry from DICOM headers
            origin, direction, spacing = self._get_geometry_from_dicom(file_names)
            
            # Set geometry on the image
            image.SetOrigin(origin)
            image.SetDirection(direction)
            image.SetSpacing(spacing)
            
            print(f"[Info] Geometry restored from DICOM headers")
            
        except Exception as e:
            print(f"[Error] Failed to restore geometry from DICOM: {e}")
            raise
        
        return image
    
    def _compare_geometries(self, sitk_origin, sitk_direction, sitk_spacing, 
                           dicom_origin, dicom_direction, dicom_spacing):
        """
        Compare two sets of geometry parameters and report any mismatches.
        
        Parameters:
        -----------
        sitk_origin, sitk_direction, sitk_spacing : tuples
            Geometry from SimpleITK image reader
        dicom_origin, dicom_direction, dicom_spacing : tuples
            Geometry extracted from DICOM headers
            
        Returns:
        --------
        bool
            True if geometries match within tolerance, False otherwise
        """
        tolerance = 1e-6
        mismatch_found = False
        
        print("\n" + "="*120)
        print("Geometry Validation: Comparing SimpleITK vs DICOM Headers")
        print("="*120)
        
        # 1. Compare Origin
        origin_diff = np.array(sitk_origin) - np.array(dicom_origin)
        origin_match = np.allclose(sitk_origin, dicom_origin, atol=tolerance)
        
        print(f"\nOrigin:")
        print(f"  SimpleITK: {sitk_origin}")
        print(f"  DICOM:     {dicom_origin}")
        print(f"  Difference: {tuple(origin_diff)}")
        print(f"  Status:    {'✓ MATCH' if origin_match else '✗ MISMATCH'}")
        
        if not origin_match:
            mismatch_found = True

        # 2. Compare Spacing
        spacing_diff = np.array(sitk_spacing) - np.array(dicom_spacing)
        spacing_match = np.allclose(sitk_spacing, dicom_spacing, atol=tolerance)
        
        print(f"\nSpacing:")
        print(f"  SimpleITK: {sitk_spacing}")
        print(f"  DICOM:     {dicom_spacing}")
        print(f"  Difference: {tuple(spacing_diff)}")
        print(f"  Status:    {'✓ MATCH' if spacing_match else '✗ MISMATCH'}")
        
        if not spacing_match:
            mismatch_found = True
        
        # 3. Compare Direction
        direction_diff = np.array(sitk_direction) - np.array(dicom_direction)
        direction_match = np.allclose(sitk_direction, dicom_direction, atol=tolerance)
        
        print(f"\nDirection (3x3 matrix flattened):")
        print(f"  SimpleITK: [{sitk_direction[0]:.6f}, {sitk_direction[1]:.6f}, {sitk_direction[2]:.6f},")
        print(f"              {sitk_direction[3]:.6f}, {sitk_direction[4]:.6f}, {sitk_direction[5]:.6f},")
        print(f"              {sitk_direction[6]:.6f}, {sitk_direction[7]:.6f}, {sitk_direction[8]:.6f}]")
        print(f"  DICOM:     [{dicom_direction[0]:.6f}, {dicom_direction[1]:.6f}, {dicom_direction[2]:.6f},")
        print(f"              {dicom_direction[3]:.6f}, {dicom_direction[4]:.6f}, {dicom_direction[5]:.6f},")
        print(f"              {dicom_direction[6]:.6f}, {dicom_direction[7]:.6f}, {dicom_direction[8]:.6f}]")
        print(f"  Max difference: {np.max(np.abs(direction_diff)):.10f}")
        print(f"  Status:    {'✓ MATCH' if direction_match else '✗ MISMATCH'}")
        
        if not direction_match:
            mismatch_found = True
        
        print("\n" + "="*120)
        if mismatch_found:
            print("⚠ GEOMETRY MISMATCH DETECTED - Geometry will be restored from DICOM headers")
        else:
            print("✓ All geometry parameters match within tolerance")
        print("="*120 + "\n")
        
        return not mismatch_found
    
    def get_image(self, restore_geometry=True, return_file_names=False, override_geometry_from=None):
        """
        Read the DICOM image using SimpleITK.
        
        This method reads either a single DICOM file or a DICOM series, validates the
        geometry by comparing SimpleITK's extracted geometry with values from DICOM headers,
        and optionally restores the geometry if mismatches are detected.
        
        Parameters:
        -----------
        restore_geometry : bool, default=True
            If True, always restore geometry from DICOM headers after reading.
            This ensures the image has correct spatial information regardless of
            SimpleITK's automatic geometry extraction.
        return_file_names : bool, default=False
            If True, returns both the image and the list of file names used.
        override_geometry_from : str, optional
            Path to another DICOM file or series to use for geometry information.
            When provided, the geometry (Origin, Direction, Spacing) will be extracted
            from this external series and applied to the loaded image, overriding the
            image's own geometry. Useful when the image series has incorrect geometry
            metadata but another series has the correct spatial information.
        
        Returns:
        --------
        sitk.Image or tuple
            If return_file_names=False: The DICOM image with validated/restored geometry
            If return_file_names=True: Tuple of (image, file_names)
        """
        if self.is_series:
            # Read DICOM series
            print(f"\n[Info] Reading DICOM series ({len(self.series_file_names)} slices)...")
            
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(self.series_file_names)
            
            # Configure the reader to load all DICOM tags (public + private)
            # This is from example code in SimpleITK documentation (https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html#lbl-dicom-series-read-modify-write)
            # But I'm not sure if it's necessary.
            series_reader.MetaDataDictionaryArrayUpdateOn()
            series_reader.LoadPrivateTagsOn()
            
            image = series_reader.Execute()
            file_names = self.series_file_names
        else:
            # Read single DICOM file
            print(f"\n[Info] Reading single DICOM file...")
            image = sitk.ReadImage(self.file_path)
            file_names = [self.file_path]
        
        # Handle geometry override from external series
        if override_geometry_from is not None:
            print("\n" + "="*120)
            print(f"⚠ MANUAL GEOMETRY OVERRIDE - Loading geometry from external series")
            print(f"  Override source: {override_geometry_from}")
            print("="*120)
            
            # Get file names from the override path
            override_file_names = self._get_file_names_from_path(override_geometry_from)
            
            # Get geometry from original series
            try:
                orig_origin, orig_direction, orig_spacing = self._get_geometry_from_dicom(file_names)
                print(f"\n[Info] Original image geometry:")
                print(f"  Origin:  {orig_origin}")
                print(f"  Spacing: {orig_spacing}")
                # Format direction matrix: within each row use "," to separate, rows separated by ";"
                # Direction is 9 numbers representing 3x3 matrix: [row1_col1, row1_col2, row1_col3, row2_col1, ...]
                rows = []
                for i in range(0, 9, 3):
                    row = ",".join([f"{orig_direction[j]:.4e}" for j in range(i, i+3)])
                    rows.append(row)
                direction_str = ";".join(rows)
                print(f"  Direction: {direction_str}")
            except Exception as e:
                print(f"[Warning] Could not extract geometry from original series: {e}")
                orig_origin, orig_direction, orig_spacing = None, None, None
            
            # Get geometry from override series
            try:
                override_origin, override_direction, override_spacing = self._get_geometry_from_dicom(override_file_names)
                print(f"\n[Info] Override series geometry:")
                print(f"  Origin:  {override_origin}")
                print(f"  Spacing: {override_spacing}")
                # Format direction matrix: within each row use "," to separate, rows separated by ";"
                # Direction is 9 numbers representing 3x3 matrix: [row1_col1, row1_col2, row1_col3, row2_col1, ...]
                rows = []
                for i in range(0, 9, 3):
                    row = ",".join([f"{override_direction[j]:.4e}" for j in range(i, i+3)])
                    rows.append(row)
                direction_str = ";".join(rows)
                print(f"  Direction: {direction_str}")
            except Exception as e:
                raise ValueError(f"Failed to extract geometry from override series '{override_geometry_from}': {e}")
            
            # Apply override geometry
            print(f"\n[Info] Applying geometry from override series...")
            image.SetOrigin(override_origin)
            image.SetDirection(override_direction)
            image.SetSpacing(override_spacing)
            print(f"[Info] ✓ Geometry override complete")
            print("="*120 + "\n")
            
        else:
            # Normal geometry validation and restoration
            # Get geometry from SimpleITK image
            sitk_origin = image.GetOrigin()
            sitk_direction = image.GetDirection()
            sitk_spacing = image.GetSpacing()
            
            # Get geometry from DICOM headers
            try:
                dicom_origin, dicom_direction, dicom_spacing = self._get_geometry_from_dicom(file_names)
            except Exception as e:
                print(f"[Warning] Could not extract geometry from DICOM headers: {e}")
                print(f"[Warning] Using SimpleITK geometry as-is")
                if return_file_names:
                    return image, file_names
                else:
                    return image
            
            # Compare geometries and report
            geometries_match = self._compare_geometries(
                sitk_origin, sitk_direction, sitk_spacing,
                dicom_origin, dicom_direction, dicom_spacing
            )
            
            # Restore geometry if requested or if mismatch detected
            if restore_geometry:
                print(f"[Info] Restoring geometry from DICOM headers (restore_geometry=True)...")
                image = self._restore_geometry_from_dicom(image, file_names)
        
        if return_file_names:
            return image, file_names
        else:
            return image

