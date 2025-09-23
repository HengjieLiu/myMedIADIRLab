import os
import pydicom
import numpy as np
import SimpleITK as sitk
from pydicom import dcmread
from skimage.draw import polygon2mask
from concurrent.futures import ThreadPoolExecutor
from .DICOMReader import DICOMReader
from pydicom.errors import InvalidDicomError


class RTStructReader(DICOMReader):
    """
    A class for reading DICOM RTSTRUCT files and generating structure masks.

    This class reads RTSTRUCT files, extracts ROI contours, and creates binary masks
    for each structure on a corresponding CT image.

    Attributes:
        rtstruct (pydicom.Dataset or None): The RTSTRUCT dataset read from the file or
        provided dataset.
        roi_contours (list or None): The list of ROI contours extracted from the RTSTRUCT.
        structure_masks (dict): A cache storing generated binary masks for structures.
    """

    def __init__(self, dicom_path_or_dataset):
        """
        Initializes the RTStructReader with the path to the RTSTRUCT file or a pydicom.Dataset.

        Args:
            dicom_path_or_dataset (str or pydicom.Dataset): The path to the RTSTRUCT DICOM file,
            directory, or a pydicom Dataset representing an RTSTRUCT.
        """
        super().__init__(dicom_path_or_dataset if isinstance(dicom_path_or_dataset, str) else None)
        self.rtstruct = None
        self.roi_contours = None
        self.structure_masks = {}  # Cache for storing generated masks
        self.datasets = (
            dicom_path_or_dataset if isinstance(dicom_path_or_dataset, pydicom.Dataset) else None
        )
        self.structure_contours = {}  # Cache for storing converted contour points

    def read(self):
        """
        Reads the RTSTRUCT file or dataset and extracts ROI contours.

        This method loads the RTSTRUCT DICOM file or uses the provided dataset,
        extracts the ROI contours, and clears any previously cached structure masks.

        Raises:
            IOError: If the RTSTRUCT file cannot be read or if no RTSTRUCT is found in a directory.
        """
        if self.datasets:
            self._read_from_dataset(self.datasets)
        else:
            if os.path.isdir(self.dicom_path):
                rtstruct_file = self._find_rtstruct_in_directory(self.dicom_path)
                if not rtstruct_file:
                    raise IOError(f"No RTSTRUCT file found in directory: {self.dicom_path}")
                self._read_from_file(rtstruct_file)
            else:
                self._read_from_file(self.dicom_path)

    def _read_from_file(self, file_path):
        """
        Reads the RTSTRUCT file from the specified path and extracts ROI contours.

        Args:
            file_path (str): The path to the RTSTRUCT file.

        Raises:
            IOError: If the RTSTRUCT file cannot be read.
        """
        try:
            self.rtstruct = dcmread(file_path)
            if self.rtstruct.Modality != "RTSTRUCT":
                raise InvalidDicomError(f"File at {file_path} is not an RTSTRUCT DICOM file.")
            self.roi_contours = self.rtstruct.ROIContourSequence
            self.structure_masks.clear()
        except InvalidDicomError as e:
            raise IOError(f"Invalid DICOM file: {e}")
        except Exception as e:
            raise IOError(f"Error reading RTSTRUCT file: {e}")

    def _read_from_dataset(self, dataset):
        """
        Reads the RTSTRUCT data from a pydicom.Dataset and extracts ROI contours.

        Args:
            dataset (pydicom.Dataset): The RTSTRUCT DICOM dataset.

        Raises:
            IOError: If the dataset is not an RTSTRUCT.
        """
        if dataset.Modality != "RTSTRUCT":
            raise IOError("Provided dataset is not an RTSTRUCT.")
        self.rtstruct = dataset
        self.roi_contours = self.rtstruct.ROIContourSequence
        self.structure_masks.clear()

    def _find_rtstruct_in_directory(self, directory_path):
        """
        Iterates through all files in a directory to find an RTSTRUCT file.

        Args:
            directory_path (str): The path to the directory to search in.

        Returns:
            str: The path to the RTSTRUCT file if found, otherwise None.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = dcmread(file_path, stop_before_pixels=True)
                    if ds.Modality == "RTSTRUCT":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None

    def get_structure_names(self):
        """
        Returns a list of all structure names defined in the RTSTRUCT file.

        Returns:
            list of str: A list of structure names.

        Raises:
            ValueError: If the RTSTRUCT file has not been read. Call `read` method first.
        """
        return [structure.ROIName for structure in self.rtstruct.StructureSetROISequence]

    def get_structure_color(self, structure_name):
        """
        Returns the display color of a specific structure.

        Args:
            structure_name (str): The name of the structure to get the color for.

        Returns:
            list of int: A list of three integers representing the RGB color of the structure.

        Raises:
            ValueError: If the structure name is not found in the RTSTRUCT.
        """
        structure_index = self.get_structure_index(structure_name)
        color = self.rtstruct.ROIContourSequence[structure_index].ROIDisplayColor
        return color

    def get_structure_index(self, structure_name):
        """
        Returns the index of a specific structure based on its name.

        Args:
            structure_name (str): The name of the structure to find.

        Returns:
            int: The index of the structure in the ROIContourSequence.

        Raises:
            ValueError: If the structure name is not found in the RTSTRUCT.
        """
        return self.get_structure_names().index(structure_name)

    def get_structure_mask(self, structure_name, image_reader):
        """
        Generates a binary mask for a specific structure on a corresponding CT image.

        This method creates a binary mask for the given structure by processing its
        contour data against the specified CT image. The mask is cached for faster
        subsequent access.

        Args:
            structure_name (str): The name of the structure to generate the mask for.
            image_reader (DICOMImageReader): An instance of DICOMImageReader containing
            the CT image.

        Returns:
            numpy.ndarray: A 3D numpy array representing the binary mask of the structure.

        Raises:
            ValueError: If the structure name is not found in the RTSTRUCT or if the
                        contour points do not lie on a single slice.
        """
        if structure_name in self.structure_masks:
            return self.structure_masks[structure_name]

        structure_index = None
        for i, structure in enumerate(self.rtstruct.StructureSetROISequence):
            if structure.ROIName == structure_name:
                structure_index = i
                break

        if structure_index is None:
            raise ValueError(f"Structure {structure_name} not found in RTSTRUCT.")

        contour_data = self.roi_contours[structure_index]
        mask = self._create_mask_from_contours(image_reader.image, contour_data)
        self.structure_masks[structure_name] = mask  # Cache the generated mask

        return mask

    def _create_mask_from_contours(self, image, contour_data):
        """
        Creates a binary mask from the contour data for a given structure.

        This method processes each contour in the contour data and generates a binary
        mask for the entire structure on the corresponding CT image.

        Args:
            image (SimpleITK.Image): The CT image on which to generate the mask.
            contour_data (pydicom.Sequence): The contour data sequence for the structure.

        Returns:
            numpy.ndarray: A 3D numpy array representing the binary mask of the structure.
        """
        ct_array_shape = sitk.GetArrayFromImage(image).shape
        mask = np.zeros(ct_array_shape, dtype=np.uint8)

        # Precompute slice indices and use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            futures = []
            for contour in contour_data.ContourSequence:
                futures.append(executor.submit(self._process_contour, contour, image, mask))

            # Ensure all contours are processed
            for future in futures:
                future.result()

        return mask

    def _process_contour(self, contour, image, mask):
        """
        Processes a single contour and updates the mask with its binary representation.

        Args:
            contour (pydicom.dataset.Dataset): The contour dataset containing ContourData.
            image (SimpleITK.Image): The CT image on which to apply the contour mask.
            mask (numpy.ndarray): The binary mask array to update with the contour.

        Raises:
            ValueError: If contour points do not lie on a single slice.
        """
        contour_points = np.array(contour.ContourData).reshape(-1, 3)
        indices = [image.TransformPhysicalPointToIndex(tuple(pt)) for pt in contour_points]

        indices = np.array(indices)

        # Make sure to swap x and y so that it corresponds to column and row
        indices[:, [1, 0]] = indices[:, [0, 1]]

        slice_index = np.unique(indices[:, 2])
        if len(slice_index) != 1:
            raise ValueError("Contour points do not lie on a single slice")
        slice_index = int(slice_index[0])

        # Convert indices to 2D for the slice
        points_2d = indices[:, :2]

        # Create a boolean mask for the contour
        slice_shape = mask[slice_index].shape
        slice_mask = polygon2mask(slice_shape, points_2d)

        # Update the mask array with this slice mask
        mask[slice_index] = np.maximum(mask[slice_index], slice_mask.astype(np.uint8))

    def get_structure_contour_points_in_pixel_space(self, structure_name, image_reader):
        """
        Get the contour points for a structure in pixel space and cache them.

        Args:
            structure_name (str): The name of the structure.
            image_reader (DICOMImageReader): An instance of DICOMImageReader containing the
            CT image.

        Returns:
            dict: A dictionary with slice indices as keys and a list of contour points
            (in pixel space) as values.
        """
        # Check if the contours for this structure are already cached
        if structure_name in self.structure_contours:
            return self.structure_contours[structure_name]

        structure_index = None
        for i, structure in enumerate(self.rtstruct.StructureSetROISequence):
            if structure.ROIName == structure_name:
                structure_index = i
                break

        if structure_index is None:
            raise ValueError(f"Structure {structure_name} not found in RTSTRUCT.")

        contour_data = self.roi_contours[structure_index]
        image = image_reader.image  # Get SimpleITK image object from the DICOMImageReader
        contours_in_pixel_space = {}

        # Iterate through the contour sequences
        for contour in contour_data.ContourSequence:
            contour_points = np.array(contour.ContourData).reshape(-1, 3)

            # Convert each contour point from physical space to pixel space using SimpleITK
            pixel_coords = []
            for point in contour_points:
                # Transform the physical point to index (pixel space)
                pixel_index = image.TransformPhysicalPointToIndex(point)

                # Flip vertically by adjusting the Y-axis (row) value
                flipped_pixel_y = image.GetHeight() - pixel_index[1] - 1  # Flip vertically

                # Append the (flipped row, col) pixel coordinates (Y, X) for 2D representation
                pixel_coords.append((flipped_pixel_y, pixel_index[0]))

            # Add the pixel coordinates to the appropriate slice index
            slice_index = image.TransformPhysicalPointToIndex(contour_points[0])[2]
            if slice_index not in contours_in_pixel_space:
                contours_in_pixel_space[slice_index] = []
            contours_in_pixel_space[slice_index].append(pixel_coords)

        # Cache the converted contour points for this structure
        self.structure_contours[structure_name] = contours_in_pixel_space

        return contours_in_pixel_space
