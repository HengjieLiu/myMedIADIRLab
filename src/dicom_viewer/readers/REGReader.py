import os
import pydicom
import numpy as np
from pydicom.errors import InvalidDicomError


class REGReader:
    """
    A class for reading DICOM REG files and extracting transformation matrices and metadata.

    This class reads DICOM REG files, extracts transformation matrices and metadata for both
    fixed and moving images, and retrieves referenced series information.

    Attributes:
        reg_file_path (str or None): The path to the REG DICOM file, if provided.
        reg_dataset (pydicom.Dataset or None): The DICOM dataset object representing the REG file.
        fixed_image_info (dict): A dictionary containing transformation and metadata for the
        fixed image.
        moving_image_info (dict): A dictionary containing transformation and metadata for the
        moving image.
        referenced_series_info (dict): A dictionary containing referenced series information.
    """

    def __init__(self, reg_input):
        """
        Initializes the REGReader with either a path to a REG file or a pydicom.Dataset object.

        Args:
            reg_input (str or pydicom.Dataset): Path to the REG file or a pydicom.Dataset object.

        Raises:
            ValueError: If reg_input is neither a file path (str) nor a pydicom.Dataset object.
        """
        self.reg_file_path = None
        self.reg_dataset = None
        self.fixed_image_info = {}
        self.moving_image_info = {}
        self.referenced_series_info = {}

        if isinstance(reg_input, str):
            # reg_input is a path to a REG file
            self.reg_file_path = reg_input
        elif isinstance(reg_input, pydicom.Dataset):
            # reg_input is already a pydicom.Dataset
            self.reg_dataset = reg_input
        else:
            raise ValueError(
                "reg_input must be either a file path (str) or a pydicom.Dataset object."
            )

    def read(self):
        """
        Reads the REG file or dataset and extracts transformation matrices, metadata,
        and referenced series information for both fixed and moving images.

        Raises:
            ValueError: If neither a file path nor a dataset is provided.
            Exception: If there is an error reading the REG file or dataset.
        """
        try:
            if self.reg_file_path:
                if os.path.isdir(self.reg_file_path):
                    # If the path is a directory, find the REG file within it
                    reg_file = self._find_reg_in_directory(self.reg_file_path)
                    if not reg_file:
                        raise IOError(f"No REG file found in directory: {self.reg_file_path}")
                    self.reg_dataset = pydicom.dcmread(reg_file)

                else:
                    # If the path is a file, read it directly
                    self.reg_dataset = pydicom.dcmread(self.reg_file_path)
                ds = self.reg_dataset
            elif self.reg_dataset:
                # Use the provided pydicom.Dataset
                ds = self.reg_dataset
            else:
                raise ValueError("No REG file path or dataset provided.")

            # Extract transformation matrices and metadata
            self.extract_transformation_matrices_and_metadata(ds)

            # Extract referenced series information
            self.extract_referenced_series_info(ds)

            # Check for other references if needed
            self.check_other_references(ds)

        except Exception as e:
            print(f"Error reading REG file {self.reg_file_path}: {e}")
            raise

    def _find_reg_in_directory(self, directory_path):
        """
        Iterates through all files in a directory to find a REG file.

        Args:
            directory_path (str): The path to the directory to search in.

        Returns:
            str: The path to the REG file if found, otherwise None.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if ds.Modality == "REG":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None

    def extract_transformation_matrices_and_metadata(self, ds):
        """
        Extracts the transformation matrices and metadata from the DICOM dataset for both fixed
        and moving images.

        Args:
            ds (pydicom.Dataset): The DICOM dataset object representing the REG file.

        Raises:
            ValueError: If the RegistrationSequence is not found or does not contain exactly
            two items.
        """
        # Check for the RegistrationSequence tag
        if "RegistrationSequence" in ds:
            reg_sequence = ds.RegistrationSequence

            # Ensure there are two items in the RegistrationSequence
            if len(reg_sequence) == 2:
                # Extract data for the fixed image
                self.fixed_image_info = self.extract_image_info(reg_sequence[0])

                # Extract data for the moving image
                self.moving_image_info = self.extract_image_info(reg_sequence[1])
            else:
                raise ValueError(
                    "Expected exactly two items in RegistrationSequence for fixed "
                    "and moving images."
                )
        else:
            raise ValueError("RegistrationSequence not found in the REG file.")

    def extract_image_info(self, reg_item):
        """
        Extracts transformation matrix and metadata from a single item in the RegistrationSequence.

        Args:
            reg_item (pydicom.dataset.Dataset): An item from the RegistrationSequence.

        Returns:
            dict: A dictionary containing the transformation matrix, transformation type,
                  SOPClassUID, and referenced SOP Instance UIDs.

        Raises:
            ValueError: If the MatrixRegistrationSequence or MatrixSequence is not found.
        """
        image_info = {}

        # Extract the transformation matrix from the MatrixRegistrationSequence
        matrix_registration_seq = reg_item.MatrixRegistrationSequence
        if len(matrix_registration_seq) > 0:
            matrix_seq = matrix_registration_seq[0].MatrixSequence
            if len(matrix_seq) > 0:
                # Extract the transformation matrix
                transformation_matrix = np.array(
                    matrix_seq[0].FrameOfReferenceTransformationMatrix
                ).reshape(4, 4)
                image_info["transformation_matrix"] = transformation_matrix
                image_info["transformation_type"] = matrix_seq[
                    0
                ].FrameOfReferenceTransformationMatrixType
            else:
                raise ValueError("MatrixSequence not found in MatrixRegistrationSequence.")
        else:
            raise ValueError("MatrixRegistrationSequence not found in RegistrationSequence.")

        # Extract Referenced SOP Class UID and SOP Instance UIDs
        if "ReferencedImageSequence" in reg_item:
            ref_image_seq = reg_item.ReferencedImageSequence
            image_info["referenced_images"] = [
                ref_item.ReferencedSOPInstanceUID for ref_item in ref_image_seq
            ]
            image_info["SOPClassUID"] = ref_image_seq[0].ReferencedSOPClassUID

        return image_info

    def extract_referenced_series_info(self, ds):
        """
        Extracts referenced series information from the DICOM dataset and matches it with fixed
        and moving images.

        Args:
            ds (pydicom.Dataset): The DICOM dataset object representing the REG file.

        Raises:
            ValueError: If the ReferencedSeriesSequence is not found in the REG file.
        """
        # Check for the ReferencedSeriesSequence tag
        if "ReferencedSeriesSequence" in ds:
            referenced_series_seq = ds.ReferencedSeriesSequence

            # Extract series information for each referenced series
            for series_item in referenced_series_seq:
                series_info = {
                    "SeriesInstanceUID": series_item.SeriesInstanceUID,
                    "ReferencedInstances": [],
                }
                if "ReferencedInstanceSequence" in series_item:
                    ref_instance_seq = series_item.ReferencedInstanceSequence
                    series_info["ReferencedInstances"] = [
                        instance.ReferencedSOPInstanceUID for instance in ref_instance_seq
                    ]

                # Determine if the series matches the fixed or moving image based on SOPInstanceUID
                if self.match_series_with_image(
                    series_info["ReferencedInstances"], self.fixed_image_info["referenced_images"]
                ):
                    self.fixed_image_info["SeriesInstanceUID"] = series_info["SeriesInstanceUID"]
                elif self.match_series_with_image(
                    series_info["ReferencedInstances"], self.moving_image_info["referenced_images"]
                ):
                    self.moving_image_info["SeriesInstanceUID"] = series_info["SeriesInstanceUID"]

                # Add the series info to the dictionary with the SeriesInstanceUID as the key
                self.referenced_series_info[series_info["SeriesInstanceUID"]] = series_info
        else:
            raise ValueError("ReferencedSeriesSequence not found in the REG file.")

    def check_other_references(self, ds):
        """
        Checks for additional references in StudiesContainingOtherReferencedInstancesSequence.

        Args:
            ds (pydicom.Dataset): The DICOM dataset object representing the REG file.
        """
        if "StudiesContainingOtherReferencedInstancesSequence" in ds:
            other_refs_seq = ds.StudiesContainingOtherReferencedInstancesSequence

            for study in other_refs_seq:
                if "ReferencedSeriesSequence" in study:
                    for series_item in study.ReferencedSeriesSequence:
                        if "ReferencedInstanceSequence" in series_item:
                            ref_instance_seq = series_item.ReferencedInstanceSequence
                            other_referenced_instances = [
                                instance.ReferencedSOPInstanceUID for instance in ref_instance_seq
                            ]

                            # Check if these instances reference the fixed or moving image
                            if self.match_series_with_image(
                                other_referenced_instances,
                                self.fixed_image_info["referenced_images"],
                            ):
                                self.fixed_image_info["SeriesInstanceUID"] = (
                                    series_item.SeriesInstanceUID
                                )
                            elif self.match_series_with_image(
                                other_referenced_instances,
                                self.moving_image_info["referenced_images"],
                            ):
                                self.moving_image_info["SeriesInstanceUID"] = (
                                    series_item.SeriesInstanceUID
                                )

    def match_series_with_image(self, series_instances, image_instances):
        """
        Matches a series with an image based on SOPInstanceUIDs.

        Args:
            series_instances (list of str): SOPInstanceUIDs from the series.
            image_instances (list of str): SOPInstanceUIDs from the image.

        Returns:
            bool: True if there is a match, False otherwise.
        """
        # Check if any SOPInstanceUID in the series matches the SOPInstanceUIDs in the image info
        return any(instance_uid in image_instances for instance_uid in series_instances)

    def get_fixed_image_info(self):
        """
        Returns the transformation matrix and metadata for the fixed image.

        Returns:
            dict: A dictionary containing the transformation matrix and metadata for
            the fixed image.

        Raises:
            ValueError: If the fixed image information has not been loaded.
            Call `read` method first.
        """
        if not self.fixed_image_info:
            raise ValueError("Fixed image information not loaded. Call `read` method first.")
        return self.fixed_image_info

    def get_moving_image_info(self):
        """
        Returns the transformation matrix and metadata for the moving image.

        Returns:
            dict: A dictionary containing the transformation matrix and metadata for
            the moving image.

        Raises:
            ValueError: If the moving image information has not been loaded.
            Call `read` method first.
        """
        if not self.moving_image_info:
            raise ValueError("Moving image information not loaded. Call `read` method first.")
        return self.moving_image_info

    def get_referenced_series_info(self):
        """
        Returns the referenced series information extracted from the REG file.

        Returns:
            dict: A dictionary containing the referenced series information.

        Raises:
            ValueError: If the referenced series information has not been loaded.
            Call `read` method first.
        """
        if not self.referenced_series_info:
            raise ValueError("Referenced series information not loaded. Call `read` method first.")
        return self.referenced_series_info
