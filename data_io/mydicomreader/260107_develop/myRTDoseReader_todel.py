import os
import numpy as np
import pydicom
import SimpleITK as sitk


# Determine type based on VR
# DS in DICOM VR means Decimal String.
# type_map = {
#     'CS': 'String', 'SH': 'String', 'LO': 'String', 'ST': 'String',
#     'LT': 'String', 'UT': 'String', 'PN': 'String', 'UI': 'UID',
#     'DA': 'Date', 'TM': 'Time', 'DT': 'DateTime',
#     'IS': 'String', 'DS': 'Number (as String)',
#     'SS': 'Integer', 'US': 'Integer', 'SL': 'Integer', 'UL': 'Integer',
#     'FL': 'Float', 'FD': 'Float',
#     'OB': 'Binary', 'OW': 'Binary', 'OF': 'Binary', 'OD': 'Binary',
#     'SQ': 'Sequence', 'AT': 'Tag'
# }
type_map = {
    'CS': 'String',             # Code String (controlled vocabulary-ish, typically uppercase)
    'SH': 'String',             # Short String (<= 16 chars)
    'LO': 'String',             # Long String (longer free text, <= 64 chars)
    'ST': 'String',             # Short Text (multi-line text, <= 1024 chars)
    'LT': 'String',             # Long Text (multi-line text, <= 10240 chars)
    'UT': 'String',             # Unlimited Text (very long free text)
    'PN': 'String',             # Person Name (formatted name components)
    'UI': 'UID',                # Unique Identifier (UID string like "1.2.840....")

    'DA': 'Date',               # Date (YYYYMMDD)
    'TM': 'Time',               # Time (HHMMSS.frac)
    'DT': 'DateTime',           # DateTime (YYYYMMDDHHMMSS.frac&timezone)

    'IS': 'Integer (as String)',# Integer String (integer stored as ASCII text)
    'DS': 'Number (as String)', # Decimal String (decimal number stored as ASCII text)

    'SS': 'Integer',            # Signed Short (16-bit integer)
    'US': 'Integer',            # Unsigned Short (16-bit integer)
    'SL': 'Integer',            # Signed Long (32-bit integer)
    'UL': 'Integer',            # Unsigned Long (32-bit integer)

    'FL': 'Float',              # Floating Point Single (32-bit float)
    'FD': 'Float',              # Floating Point Double (64-bit float)

    'OB': 'Binary',             # Other Byte (raw bytes)
    'OW': 'Binary',             # Other Word (16-bit words; often pixel data)
    'OF': 'Binary',             # Other Float (32-bit float array)
    'OD': 'Binary',             # Other Double (64-bit float array)

    'SQ': 'Sequence',           # Sequence of items (nested datasets)
    'AT': 'Tag'                 # Attribute Tag (stores a DICOM tag like (gggg,eeee))
}


class myRTDoseReader:
    def __init__(self, dicom_path):
        """
        Initialize the RTDose reader with a path to a DICOM file or folder.
        
        Parameters:
        -----------
        dicom_path : str
            Path to the RTDOSE DICOM file or folder containing a single RTDOSE DICOM file
            
        Raises:
        -------
        FileNotFoundError
            If the path doesn't exist
        ValueError
            If folder contains 0 or multiple files, or if modality is not RTDOSE
        """
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"Path does not exist: {dicom_path}")
        
        # Check if it's a file or folder
        if os.path.isfile(dicom_path):
            file_path = dicom_path
        elif os.path.isdir(dicom_path):
            # Find all files in the folder
            files = [f for f in os.listdir(dicom_path) if os.path.isfile(os.path.join(dicom_path, f))]
            
            if len(files) == 0:
                raise ValueError(f"Folder contains no files: {dicom_path}")
            elif len(files) > 1:
                raise ValueError(f"Folder contains multiple files ({len(files)} files): {dicom_path}")
            
            file_path = os.path.join(dicom_path, files[0])
        else:
            raise ValueError(f"Path is neither a file nor a directory: {dicom_path}")
        
        # Validate that it's an RTDOSE DICOM file
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            if not hasattr(ds, 'Modality') or ds.Modality != "RTDOSE":
                raise ValueError(f"DICOM file modality is '{ds.Modality if hasattr(ds, 'Modality') else 'Unknown'}', expected 'RTDOSE': {file_path}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            else:
                raise ValueError(f"Failed to read DICOM file or file is not a valid DICOM: {file_path}. Error: {str(e)}")
        
        self.dicom_path = file_path
        
    def print_metadata_pydicom(self):
        """
        Print all DICOM metadata using pydicom in a nicely formatted table.
        """
        ds = pydicom.dcmread(self.dicom_path)
        
        print("="*120)
        print(f"DICOM Metadata (using pydicom) - File: {os.path.basename(self.dicom_path)}")
        print("="*120)
        print(f"{'Keyword':<35} {'Tag':<15} {'VR':<5} {'VM':<5} {'Type':<15} {'Value'}")
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
            
            # # Determine type based on VR
            # type_map = {
            #     'CS': 'String', 'SH': 'String', 'LO': 'String', 'ST': 'String',
            #     'LT': 'String', 'UT': 'String', 'PN': 'String', 'UI': 'UID',
            #     'DA': 'Date', 'TM': 'Time', 'DT': 'DateTime',
            #     'IS': 'String', 'DS': 'Number (as String)',
            #     'SS': 'Integer', 'US': 'Integer', 'SL': 'Integer', 'UL': 'Integer',
            #     'FL': 'Float', 'FD': 'Float',
            #     'OB': 'Binary', 'OW': 'Binary', 'OF': 'Binary', 'OD': 'Binary',
            #     'SQ': 'Sequence', 'AT': 'Tag'
            # }
            elem_type = type_map.get(vr, 'Other')
            
            print(f"{keyword:<35} {tag_str:<15} {vr:<5} {vm:<5} {elem_type:<20} {value_str}")
        
        print("="*120)
        
    def print_metadata_sitk(self):
        """
        Print all DICOM metadata using SimpleITK for values and pydicom for keyword lookup.
        """
        # Read with SimpleITK to get metadata values
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.dicom_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # Read with pydicom only for keyword extraction (don't load pixel data)
        ds = pydicom.dcmread(self.dicom_path, stop_before_pixels=True)
        
        # Create a mapping from tag to pydicom element for quick lookup
        tag_to_elem = {}
        for elem in ds:
            tag_to_elem[elem.tag] = elem
        
        print("="*120)
        print(f"DICOM Metadata (using SimpleITK) - File: {os.path.basename(self.dicom_path)}")
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
                                
                                # # Determine type based on VR
                                # type_map = {
                                #     'CS': 'String', 'SH': 'String', 'LO': 'String', 'ST': 'String',
                                #     'LT': 'String', 'UT': 'String', 'PN': 'String', 'UI': 'UID',
                                #     'DA': 'Date', 'TM': 'Time', 'DT': 'DateTime',
                                #     'IS': 'String', 'DS': 'Number (as String)',
                                #     'SS': 'Integer', 'US': 'Integer', 'SL': 'Integer', 'UL': 'Integer',
                                #     'FL': 'Float', 'FD': 'Float',
                                #     'OB': 'Binary', 'OW': 'Binary', 'OF': 'Binary', 'OD': 'Binary',
                                #     'SQ': 'Sequence', 'AT': 'Tag'
                                # }
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
                print(f"{key:<35} {'':<15} {'??':<5} {'?':<5} {'Unknown':<15} [Unable to read: {str(e)}]")
        
        print("="*120)
        print(f"\nImage Information:")
        print(f"  Dimensions: {reader.GetSize()}")
        print(f"  Spacing: {reader.GetSpacing()}")
        print(f"  Origin: {reader.GetOrigin()}")
        print("="*120)


    def get_image(self, apply_dose_scaling=True):
        """
        Read the RTDOSE image using SimpleITK.
        
        Parameters:
        -----------
        apply_dose_scaling : bool, default=True
            If True, scale the image array by DoseGridScaling factor.
            Validates that DoseUnits is "GY" before applying scaling.
        
        Returns:
        --------
        sitk.Image
            The RTDOSE image, optionally scaled by DoseGridScaling
        """
        # Read image using SimpleITK
        image = sitk.ReadImage(self.dicom_path)
        
        if apply_dose_scaling:
            # Read DICOM metadata to get dose scaling information
            ds = pydicom.dcmread(self.dicom_path, stop_before_pixels=True)
            
            # Check DoseUnits (3004,0002) - try keyword first, then tag
            if hasattr(ds, 'DoseUnits'):
                dose_units = ds.DoseUnits
            else:
                dose_units_tag = pydicom.tag.Tag(0x3004, 0x0002)
                if dose_units_tag in ds:
                    dose_units = ds[dose_units_tag].value
                else:
                    raise ValueError("DoseUnits (3004,0002) not found in DICOM file. Cannot apply dose scaling.")
            
            if dose_units != "GY":
                raise ValueError(f"DoseUnits is '{dose_units}', expected 'GY'. Cannot apply dose scaling.")
            
            # Extract DoseGridScaling (3004,000E) - try keyword first, then tag
            if hasattr(ds, 'DoseGridScaling'):
                dose_grid_scaling = float(ds.DoseGridScaling)
            else:
                dose_grid_scaling_tag = pydicom.tag.Tag(0x3004, 0x000E)
                if dose_grid_scaling_tag in ds:
                    dose_grid_scaling = float(ds[dose_grid_scaling_tag].value)
                else:
                    raise ValueError("DoseGridScaling (3004,000E) not found in DICOM file. Cannot apply dose scaling.")
            
            # Preserve original image metadata
            original_spacing = image.GetSpacing()
            original_origin = image.GetOrigin()
            original_direction = image.GetDirection()
            
            # Convert image to numpy array, scale, and convert back
            image_array = sitk.GetArrayFromImage(image)
            image_array = image_array * dose_grid_scaling
            image = sitk.GetImageFromArray(image_array)
            
            # Copy metadata from original image
            image.SetSpacing(original_spacing)
            image.SetOrigin(original_origin)
            image.SetDirection(original_direction)
        
        return image
