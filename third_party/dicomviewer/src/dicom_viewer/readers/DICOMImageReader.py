import os
import numpy as np
import SimpleITK as sitk
from pydicom import dcmread
from collections import defaultdict


class DICOMImageReader:
    """
    A class for reading DICOM image files, specifically for a given modality (e.g., CT, MR, PET).

    This class reads and processes DICOM image files or a list of pydicom.Dataset objects,
    allowing for selection based on modality and SeriesInstanceUID, and provides methods
    for sorting, transforming, and retrieving images.

    Attributes:
        image (SimpleITK.Image or None): The DICOM image read from the file.
        modality (str): The modality of the DICOM images to read (e.g., 'CT', 'MR', 'PET').
        series_instance_uid (str or None): The SeriesInstanceUID to filter images by.
        datasets (list of pydicom.Dataset or None): A list of pydicom Dataset objects, if provided.
    """

    def __init__(self, dicom_path_or_datasets, modality="CT", series_instance_uid=None):
        """
        Initializes the DICOMImageReader with the path to the DICOM files or a list of DICOM
        datasets, modality, and optional SeriesInstanceUID.

        Args:
            dicom_path_or_datasets (str or list of pydicom.Dataset): The path to the directory
            containing DICOM files or a list of pydicom Dataset objects.
            modality (str): The modality of the DICOM images to read (e.g., 'CT', 'MR', 'PET').
            series_instance_uid (str, optional): The SeriesInstanceUID to filter images by.
        """

        self.image = None
        self.modality = modality
        self.series_instance_uid = series_instance_uid
        self.input_type = None
        if isinstance(dicom_path_or_datasets, list):
            if isinstance(dicom_path_or_datasets[0], str):
                self.datasets = None
                self.input_type = "files"
                self.file_names = dicom_path_or_datasets
            else:
                self.datasets = dicom_path_or_datasets
                self.input_type = "datasets"
        else:
            self.datasets = None
            self.input_type = "dir"
            self.dicom_path = dicom_path_or_datasets

    def read(self):
        """
        Reads DICOM image files or datasets and loads them as a SimpleITK image.

        This method filters the files or datasets based on the specified modality and
        SeriesInstanceUID, sorts them according to their position along the imaging axis,
        and reads them into a SimpleITK image.

        Returns:
            SimpleITK.Image: The DICOM image loaded from the files or datasets.

        Raises:
            ValueError: If no series matching the specified modality and SeriesInstanceUID
            is found.
        """
        series_file_names = []

        if self.input_type == "datasets":
            # Process the list of pydicom datasets
            series_file_names = self._process_datasets()
        elif self.input_type == "dir":
            # Process the directory of DICOM files
            series_file_names = self._process_dicom_directory()
        elif self.input_type == "files":
            series_file_names = self.file_names

        if not series_file_names:
            raise ValueError(
                f"No {self.modality} series found with the specified SeriesInstanceUID."
            )

        # Sort the file names or datasets based on ImagePositionPatient along the imaging axis
        series_file_names_sorted = self.sort_by_image_position_patient(series_file_names)

        # Set filenames manually for SimpleITK reader
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_file_names_sorted)
        self.image = reader.Execute()
        return self.image

    def _process_dicom_directory(self):
        """
        Processes a directory of DICOM files to filter by modality and SeriesInstanceUID.

        Returns:
            list of str: The filtered and sorted list of DICOM file paths.
        """
        all_files = [
            os.path.join(self.dicom_path, f)
            for f in os.listdir(self.dicom_path)
            if os.path.isfile(os.path.join(self.dicom_path, f))
        ]

        files_by_series = defaultdict(list)

        for file in all_files:
            try:
                ds = dcmread(file, stop_before_pixels=True)
                if ds.Modality == self.modality:
                    files_by_series[ds.SeriesInstanceUID].append(file)
            except Exception as e:
                print(f"Error reading file {file}: {e}")

        if not files_by_series:
            raise ValueError(f"No {self.modality} series found in the specified directory.")

        # Use specific SeriesInstanceUID or the first available series
        if self.series_instance_uid:
            series_file_names = files_by_series.get(self.series_instance_uid, [])
        else:
            series_file_names = next(iter(files_by_series.values()), [])

        return series_file_names

    def _process_datasets(self):
        """
        Processes a list of pydicom.Dataset objects to filter by modality and SeriesInstanceUID.

        Returns:
            list of str: The filtered list of filenames extracted from datasets.
        """
        datasets_by_series = defaultdict(list)

        for ds in self.datasets:
            if ds.Modality == self.modality:
                datasets_by_series[ds.SeriesInstanceUID].append(ds.filename)

        if not datasets_by_series:
            raise ValueError(f"No {self.modality} series found in the provided datasets.")

        if self.series_instance_uid:
            series_file_names = datasets_by_series.get(self.series_instance_uid, [])
        else:
            series_file_names = next(iter(datasets_by_series.values()), [])

        return series_file_names

    def sort_by_image_position_patient(self, file_names_or_datasets):
        """
        Sorts DICOM image files or datasets based on their position along the imaging axis.

        This method computes the imaging axis using the ImageOrientationPatient tag
        and sorts the files based on the ImagePositionPatient tag along that axis.

        Args:
            file_names_or_datasets (list of str or list of pydicom.Dataset):
                The list of DICOM file paths or datasets to sort.

        Returns:
            list of str or list of pydicom.Dataset: The sorted list of DICOM file paths or
            datasets.
        """

        def get_image_position_along_imaging_axis(ds):
            try:
                if isinstance(ds, str):
                    ds = dcmread(ds, stop_before_pixels=True)

                image_position_patient = np.array(ds.ImagePositionPatient, dtype=float)
                image_orientation_patient = np.array(ds.ImageOrientationPatient, dtype=float)
                row_cosines = image_orientation_patient[:3]
                col_cosines = image_orientation_patient[3:]
                imaging_axis = np.cross(row_cosines, col_cosines)
                return np.dot(image_position_patient, imaging_axis)
            except Exception as e:
                print(f"Could not read dataset: {e}")
                return float("inf")

        sorted_items = sorted(file_names_or_datasets, key=get_image_position_along_imaging_axis)
        return sorted_items

    def get_image_array(self):
        """
        Returns the image data as a numpy array.

        This method converts the SimpleITK image to a numpy array for easier manipulation.

        Returns:
            numpy.ndarray: The image data as a numpy array.

        Raises:
            ValueError: If the image has not been loaded. Call `read` method first.
        """
        if self.image is not None:
            return sitk.GetArrayFromImage(self.image)
        else:
            raise ValueError(f"{self.modality} image not loaded. Call `read` method first.")

    def get_image(self):
        """
        Returns the SimpleITK image object.

        This method provides access to the underlying SimpleITK image object for
        advanced processing.

        Returns:
            SimpleITK.Image: The SimpleITK image object.

        Raises:
            ValueError: If the image has not been loaded. Call `read` method first.
        """
        if self.image is not None:
            return self.image
        else:
            raise ValueError(f"{self.modality} image not loaded. Call `read` method first.")

    def get_transformed_image_array(self, transformation_matrix=np.eye(4)):
        """
        Applies a transformation matrix to the image and returns the transformed image data
        as a numpy array.

        This method uses an affine transformation to transform the image based on the provided
        matrix.

        Args:
            transformation_matrix (numpy.ndarray): A 4x4 affine transformation matrix.

        Returns:
            numpy.ndarray: The transformed image data as a numpy array.

        Raises:
            ValueError: If the image has not been loaded. Call `read` method first.
        """
        if self.image is None:
            raise ValueError(f"{self.modality} image not loaded. Call `read` method first.")

        # Get image size, spacing, and origin
        size = np.array(self.image.GetSize())
        spacing = np.array(self.image.GetSpacing())
        origin = np.array(self.image.GetOrigin())

        # Compute the center of the image in physical space
        center_physical = origin + spacing * (size / 2.0)

        # Convert transformation matrix from numpy to SimpleITK format
        transform = sitk.AffineTransform(3)

        # SimpleITK expects the matrix as a flat list
        matrix_flat_list = transformation_matrix[:3, :3].flatten().tolist()
        transform.SetMatrix(matrix_flat_list)

        # Set the translation part of the transform to correctly center the transformation
        translation_vector = transformation_matrix[:3, 3]

        # Compute the correct translation
        adjusted_translation = (
            translation_vector
            - np.dot(transformation_matrix[:3, :3], center_physical)
            + center_physical
        )
        transform.SetTranslation(adjusted_translation.tolist())

        # Resample the image using the transform
        resampled_image = sitk.Resample(
            self.image,
            self.image,  # Reference image to keep original properties
            transform,
            sitk.sitkLinear,  # Interpolation method
            0,  # Default pixel value for areas outside the original image
            self.image.GetPixelID(),
        )

        # Convert the resampled image back to a numpy array
        transformed_image_array = sitk.GetArrayFromImage(resampled_image)
        return transformed_image_array

    def transform_to_standard_orientation(self):
        """
        Transforms the image to standard orientation using SimpleITK.

        Returns:
            numpy.ndarray: The transformed image array in standard orientation.
        """
        if self.image is None:
            raise ValueError(f"{self.modality} image not loaded. Call `read` method first.")

        # Get the current orientation matrix (direction cosines)
        direction_cosines = np.array(self.image.GetDirection()).reshape(3, 3)

        # Define the standard orientation matrix (identity)
        standard_orientation = np.eye(3)

        # Compute the rotation matrix to align the current orientation to the standard
        rotation_matrix = np.dot(standard_orientation, np.linalg.inv(direction_cosines))

        # Create a SimpleITK affine transform from the rotation matrix
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(rotation_matrix.flatten())

        # Center the rotation on the physical center of the image
        center_physical = np.array(
            self.image.TransformContinuousIndexToPhysicalPoint(
                np.array(self.image.GetSize()) / 2.0
            )
        )
        transform.SetCenter(center_physical.tolist())

        # Adjust translation to prevent shifting
        transform.SetTranslation([0, 0, 0])  # Set translation to zero

        # Resample the image to the new orientation
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.image)
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampled_image = resampler.Execute(self.image)

        # Convert the resampled image back to a numpy array
        transformed_image_array = sitk.GetArrayFromImage(resampled_image)
        return transformed_image_array
