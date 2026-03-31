import os
import numpy as np
import pydicom
import SimpleITK as sitk
from .myDicomHeader import type_map


class myRTDoseReader:
    """
    A class to read RTDOSE DICOM files and folders.
    
    Attributes:
    -----------
    dicom_file : str
        Path to the RTDOSE DICOM file
    dicom_folder : str
        Folder containing a single RTDOSE DICOM file
        
    Methods:
    --------
    print_metadata_pydicom(self)
        Print all DICOM metadata using pydicom in a nicely formatted table.
    print_metadata_sitk(self)
        Print all DICOM metadata using SimpleITK for values and pydicom for keyword lookup.
    get_image(self, apply_dose_scaling=True)
        Read the RTDOSE image using SimpleITK.
    compare_rtdose_headers(cls, dicom_path_a, dicom_path_b, max_value_len=80)
        Compare RTDOSE headers from two paths and print similarities/differences.
    
    """

    def __init__(self, rtdose_file_or_folder):
        """
        Initialize the RTDose reader with a path to a DICOM file or folder.
        
        Parameters:
        -----------
        rtdose_file_or_folder : str
            Path to the RTDOSE DICOM file or folder containing a single RTDOSE DICOM file
            
        Raises:
        -------
        FileNotFoundError
            If the path doesn't exist
        ValueError
            If folder contains 0 or multiple files, or if modality is not RTDOSE
        """
        self.dicom_file, self.dicom_folder = self._resolve_rtdose_path(rtdose_file_or_folder)
    
    @staticmethod
    def _resolve_rtdose_path(rtdose_file_or_folder):
        """
        Resolve a file or folder path to a single RTDOSE DICOM file path and folder.
        """
        if not os.path.exists(rtdose_file_or_folder):
            raise FileNotFoundError(f"Path does not exist: {rtdose_file_or_folder}")
        
        # Check if it's a file or folder
        if os.path.isfile(rtdose_file_or_folder):
            file_path = rtdose_file_or_folder
            folder_path = os.path.dirname(rtdose_file_or_folder)
        elif os.path.isdir(rtdose_file_or_folder):
            # Find all files in the folder
            files = [f for f in os.listdir(rtdose_file_or_folder) if os.path.isfile(os.path.join(rtdose_file_or_folder, f))]
            
            if len(files) == 0:
                raise ValueError(f"Folder contains no files: {rtdose_file_or_folder}")
            elif len(files) > 1:
                raise ValueError(f"Folder contains multiple files ({len(files)} files): {rtdose_file_or_folder}")
            
            file_path = os.path.join(rtdose_file_or_folder, files[0])
            folder_path = rtdose_file_or_folder
        else:
            raise ValueError(f"Path is neither a file nor a directory: {rtdose_file_or_folder}")
        
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
        
        return file_path, folder_path

    @staticmethod
    def _format_elem_value(elem, max_len=80):
        """
        Format a pydicom DataElement value for display.
        """
        # Skip pixel data and large binary data
        if elem.tag == 0x7FE00010:  # Pixel Data
            value_str = "[Pixel Data - not displayed]"
        elif elem.VR == "SQ":
            try:
                value_str = f"[Sequence: {len(elem.value)} item(s)]"
            except:
                value_str = "[Sequence]"
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
                if len(value_str) > max_len:
                    value_str = value_str[:max_len - 3] + "..."
            except:
                value_str = "[Unable to display]"
        
        return value_str

    
    @classmethod
    def _collect_elements(cls, ds, prefix=""):
        """
        Collect elements into a flattened mapping, including nested sequences.
        """
        elements = {}
        for elem in ds:
            tag = elem.tag
            tag_str = f"({tag.group:04X},{tag.element:04X})"
            keyword = elem.keyword if elem.keyword else "(Unknown)"
            path = f"{prefix}.{keyword}" if prefix else keyword
            vr = elem.VR if elem.VR else "??"
            
            key = (path, tag)
            elements[key] = {
                "elem": elem,
                "path": path,
                "tag": tag,
                "tag_str": tag_str,
                "vr": vr,
            }
            
            if vr == "SQ":
                try:
                    seq_items = elem.value or []
                except:
                    seq_items = []
                
                for idx, item in enumerate(seq_items):
                    item_prefix = f"{path}[{idx}]"
                    elements.update(cls._collect_elements(item, prefix=item_prefix))
        
        return elements
    
        
    def print_metadata_pydicom(self):
        """
        Print all DICOM metadata using pydicom in a nicely formatted table.
        """
        ds = pydicom.dcmread(self.dicom_file)
        
        print("="*120)
        print(f"DICOM Metadata (using pydicom) - File: {os.path.basename(self.dicom_file)}")
        print("="*120)
        print(f"{'Keyword':<35} {'Tag':<15} {'VR':<5} {'VM':<5} {'Type':<15} {'Value'}")
        print("-"*120)
        
        for elem in ds:
            value_str = self._format_elem_value(elem, max_len=80)
            
            keyword = elem.keyword if elem.keyword else "(Unknown)"
            tag_str = f"({elem.tag.group:04X},{elem.tag.element:04X})"
            vr = elem.VR if elem.VR else "??"
            vm = str(elem.VM) if hasattr(elem, 'VM') else "?"
            
            elem_type = type_map.get(vr, 'Other')
            
            print(f"{keyword:<35} {tag_str:<15} {vr:<5} {vm:<5} {elem_type:<20} {value_str}")
        
        print("="*120)
        
    @classmethod
    def compare_rtdose_headers(cls, dicom_path_a, dicom_path_b, max_value_len=80):
        """
        Compare RTDOSE headers from two paths and print similarities/differences,
        including nested sequences.
        """
        file_a, _ = cls._resolve_rtdose_path(dicom_path_a)
        file_b, _ = cls._resolve_rtdose_path(dicom_path_b)
        
        ds_a = pydicom.dcmread(file_a, stop_before_pixels=True)
        ds_b = pydicom.dcmread(file_b, stop_before_pixels=True)
        
        elems_a = cls._collect_elements(ds_a)
        elems_b = cls._collect_elements(ds_b)
        
        keys_a = set(elems_a.keys())
        keys_b = set(elems_b.keys())
        
        common_keys = keys_a & keys_b
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        
        same = []
        diff = []
        
        for key in common_keys:
            elem_a = elems_a[key]["elem"]
            elem_b = elems_b[key]["elem"]
            if elem_a.VR == "SQ":
                try:
                    same_seq = len(elem_a.value) == len(elem_b.value)
                except:
                    same_seq = False
                if same_seq:
                    same.append(key)
                else:
                    diff.append(key)
            elif elem_a.value == elem_b.value:
                same.append(key)
            else:
                diff.append(key)
        
        def sort_key(key):
            tag = key[1]
            path = key[0]
            return (tag.group, tag.element, path)
        
        same = sorted(same, key=sort_key)
        diff = sorted(diff, key=sort_key)
        only_a = sorted(only_a, key=sort_key)
        only_b = sorted(only_b, key=sort_key)
        
        print("="*180)
        print("RTDOSE Header Comparison")
        print("="*180)
        print(f"File A: {os.path.basename(file_a)}")
        print(f"File B: {os.path.basename(file_b)}")
        print("-"*180)
        
        def print_row(keyword, tag_str, vr, value_a, value_b):
            print(f"{keyword:<60} {tag_str:<15} {vr:<5} {value_a:<45} {value_b}")
        
        print(f"{'Keyword':<60} {'Tag':<15} {'VR':<5} {'Value A':<45} {'Value B'}")
        print("-"*180)
        
        if same:
            print("SAME")
            print("-"*180)
            for key in same:
                info = elems_a[key]
                elem = info["elem"]
                keyword = info["path"]
                tag_str = info["tag_str"]
                vr = info["vr"]
                value_str = cls._format_elem_value(elem, max_len=max_value_len)
                print_row(keyword, tag_str, vr, value_str, value_str)
            print("-"*180)
        
        if diff:
            print("DIFFERENT")
            print("-"*180)
            for key in diff:
                info_a = elems_a[key]
                info_b = elems_b[key]
                elem_a = info_a["elem"]
                elem_b = info_b["elem"]
                keyword = info_a["path"]
                tag_str = info_a["tag_str"]
                vr = info_a["vr"]
                value_a = cls._format_elem_value(elem_a, max_len=max_value_len)
                value_b = cls._format_elem_value(elem_b, max_len=max_value_len)
                print_row(keyword, tag_str, vr, value_a, value_b)
            print("-"*180)
        
        if only_a:
            print("ONLY IN A")
            print("-"*180)
            for key in only_a:
                info = elems_a[key]
                elem = info["elem"]
                keyword = info["path"]
                tag_str = info["tag_str"]
                vr = info["vr"]
                value_str = cls._format_elem_value(elem, max_len=max_value_len)
                print_row(keyword, tag_str, vr, value_str, "[Missing]")
            print("-"*180)
        
        if only_b:
            print("ONLY IN B")
            print("-"*180)
            for key in only_b:
                info = elems_b[key]
                elem = info["elem"]
                keyword = info["path"]
                tag_str = info["tag_str"]
                vr = info["vr"]
                value_str = cls._format_elem_value(elem, max_len=max_value_len)
                print_row(keyword, tag_str, vr, "[Missing]", value_str)
            print("-"*180)

    def print_metadata_sitk(self):
        """
        Print all DICOM metadata using SimpleITK for values and pydicom for keyword lookup.
        """
        # Read with SimpleITK to get metadata values
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.dicom_file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # Read with pydicom only for keyword extraction (don't load pixel data)
        ds = pydicom.dcmread(self.dicom_file, stop_before_pixels=True)
        
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
        image = sitk.ReadImage(self.dicom_file)
        
        if apply_dose_scaling:
            # Read DICOM metadata to get dose scaling information
            ds = pydicom.dcmread(self.dicom_file, stop_before_pixels=True)
            
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

    def get_reference_rtplan(self, info=True, debug=False):
        """
        Find and return the referenced RTPLAN file and folder.
        """
        from .myRTPlanReader import myRTPlanReader
        
        ds = pydicom.dcmread(self.dicom_file, stop_before_pixels=True)
        seq_tag = pydicom.tag.Tag(0x300C, 0x0002)
        rtplan_class_uid = "1.2.840.10008.5.1.4.1.1.481.5"
        
        if seq_tag not in ds:
            if info:
                print("INFO: ReferencedRTPlanSequence (300C,0002) not found in RTDOSE.")
            return None, None
        
        try:
            sequence = ds[seq_tag].value
        except Exception as e:
            if info:
                print(f"INFO: Unable to read ReferencedRTPlanSequence: {str(e)}")
            return None, None
        
        referenced_uids = []
        for i, item in enumerate(sequence):
            sop_class_uid = getattr(item, "ReferencedSOPClassUID", None)
            sop_instance_uid = getattr(item, "ReferencedSOPInstanceUID", None)
            
            if debug:
                print(f"DEBUG: Sequence item {i} SOPClassUID={sop_class_uid} SOPInstanceUID={sop_instance_uid}")
            
            if sop_class_uid != rtplan_class_uid:
                if debug:
                    print(f"DEBUG: Skipping item {i}: SOPClassUID not RTPLAN.")
                continue
            
            if sop_instance_uid:
                referenced_uids.append(sop_instance_uid)
        
        if not referenced_uids:
            if info:
                print("INFO: No referenced RTPLAN SOPInstanceUID found in RTDOSE.")
            return None, None
        
        parent_folder = os.path.dirname(self.dicom_folder)
        if not os.path.isdir(parent_folder):
            if info:
                print(f"INFO: Parent folder not found: {parent_folder}")
            return None, None
        
        candidate_folders = []
        for name in os.listdir(parent_folder):
            folder_path = os.path.join(parent_folder, name)
            if os.path.isdir(folder_path) and "RTPLAN" in name.upper():
                candidate_folders.append(folder_path)
        
        if debug:
            print(f"DEBUG: RTPLAN candidate folders: {candidate_folders}")
        
        for folder in candidate_folders:
            try:
                rtplan_file, rtplan_folder = myRTPlanReader._resolve_rtplan_path(folder)
                sop_uid = myRTPlanReader.get_sop_instance_uid(rtplan_file)
                if debug:
                    print(f"DEBUG: RTPLAN file {rtplan_file} SOPInstanceUID={sop_uid}")
                if sop_uid in referenced_uids:
                    if info:
                        print(f"INFO: Found referenced RTPLAN: {rtplan_file}")
                    return rtplan_file, rtplan_folder
            except Exception as e:
                if debug:
                    print(f"DEBUG: Skipping folder {folder}: {str(e)}")
                continue
        
        if info:
            print("INFO: Referenced RTPLAN not found under parent folder.")
        return None, None

    def get_prescription_dose_from_reference_rtplan(self, info=True, debug=False):
        """
        Get prescription dose info from the referenced RTPLAN.
        """
        from .myRTPlanReader import myRTPlanReader
        
        rtplan_file, _ = self.get_reference_rtplan(info=info, debug=debug)
        if not rtplan_file:
            return None
        
        return myRTPlanReader.get_prescription_dose(rtplan_file)
