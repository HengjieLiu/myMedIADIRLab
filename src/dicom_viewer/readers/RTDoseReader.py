import os
import pydicom
import numpy as np
import SimpleITK as sitk
from pydicom import dcmread
from pydicom.errors import InvalidDicomError


class RTDoseReader:
    """
    A class for reading DICOM RTDOSE files and resampling dose data onto a reference image grid.

    This class reads RTDOSE files, extracts dose data, applies dose grid scaling,
    and resamples the dose data onto a specified image grid (e.g., CT, MR, PT).

    Attributes:
        dose_file_path (str): The path to the RTDOSE DICOM file.
        rtdose_image (SimpleITK.Image or None): The RTDOSE image read from the file.
        dose_array (numpy.ndarray or None): The dose data extracted from the RTDOSE image.
        dose_grid_scaling (float or None): The scaling factor applied to the dose data.
        dataset (pydicom.Dataset or None): A pydicom Dataset object representing the RTDOSE,
        if provided.
    """

    def __init__(self, dose_path_or_dataset):
        """
        Initializes the RTDoseReader with the path to the RTDOSE file or a pydicom.Dataset.

        Args:
            dose_path_or_dataset (str or pydicom.Dataset): The path to the RTDOSE DICOM file or
            directory, or a pydicom Dataset representing an RTDOSE.
        """
        self.dose_file_path = (
            dose_path_or_dataset if isinstance(dose_path_or_dataset, str) else None
        )
        self.rtdose_image = None
        self.dose_array = None
        self.dose_grid_scaling = None
        self.dataset = (
            dose_path_or_dataset if isinstance(dose_path_or_dataset, pydicom.Dataset) else None
        )

    def read(self):
        """
        Reads the RTDOSE file or dataset, extracts dose data, and applies the dose grid
        scaling factor.

        This method reads the RTDOSE DICOM file using SimpleITK or uses the provided dataset,
        converts the dose image to a NumPy array, retrieves the DoseGridScaling from the DICOM
        metadata, and applies the scaling factor to the dose data.

        Raises:
            RuntimeError: If the DoseGridScaling metadata is not found in the RTDOSE file or
            dataset.
            ValueError: If the RTDOSE image cannot be converted to a NumPy array.
        """
        if self.dataset:
            self._read_from_dataset(self.dataset)
        else:
            if os.path.isdir(self.dose_file_path):
                dose_file = self._find_rtdose_in_directory(self.dose_file_path)
                if not dose_file:
                    raise IOError(f"No RTDOSE file found in directory: {self.dose_file_path}")
                self._read_from_file(dose_file)
            else:
                self._read_from_file(self.dose_file_path)

    def _read_from_file(self, file_path):
        """
        Reads the RTDOSE file from the specified path and extracts dose data.

        Args:
            file_path (str): The path to the RTDOSE file.

        Raises:
            RuntimeError: If the DoseGridScaling metadata is not found in the RTDOSE file.
        """
        try:
            self.rtdose_image = sitk.ReadImage(file_path)
            self.dose_array = sitk.GetArrayFromImage(self.rtdose_image)
            if self.rtdose_image.HasMetaDataKey("3004|000e"):
                self.dose_grid_scaling = float(self.rtdose_image.GetMetaData("3004|000e"))
            else:
                raise RuntimeError("DoseGridScaling metadata not found in the RTDOSE file.")
            # self._extract_dose_grid_scaling()
            self.dose_array = self.dose_array.astype(np.float64) * self.dose_grid_scaling

        except Exception as e:
            raise IOError(f"Error reading RTDOSE file: {e}")

    def _read_from_dataset(self, dataset):
        """
        Reads the RTDOSE data from a pydicom.Dataset and extracts dose data.

        Args:
            dataset (pydicom.Dataset): The RTDOSE DICOM dataset.

        Raises:
            RuntimeError: If the DoseGridScaling metadata is not found in the dataset.
        """
        if dataset.Modality != "RTDOSE":
            raise IOError("Provided dataset is not an RTDOSE.")
        self.dataset = dataset
        # Convert the dataset to a SimpleITK image using the pixel data
        dose_grid = dataset.pixel_array

        # # Get the grid size
        # grid_size = (dataset.Rows, dataset.Columns, dataset.NumberOfFrames)

        # # Reshape the data
        # dose_grid = dose_grid.reshape(grid_size)

        # Voxel spacing (Pixel Spacing + Slice Thickness)
        pixel_spacing = dataset.PixelSpacing  # [spacing_x, spacing_y]
        slice_thickness = dataset.GridFrameOffsetVector[1] - dataset.GridFrameOffsetVector[0]
        voxel_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness))

        # Origin (Image Position Patient)
        origin = tuple(map(float, dataset.ImagePositionPatient))

        # Orentation (Image Orientation Patient)
        orientation = dataset.ImageOrientationPatient
        # First and second direction cosines from ImageOrientationPatient
        row_direction = np.array(orientation[0:3], dtype=float)  # direction of the first row
        column_direction = np.array(orientation[3:6], dtype=float)  # direction of the first column

        # Compute the third direction cosine using the cross product
        slice_direction = np.cross(row_direction, column_direction)
        # Flatten the direction cosines to match SimpleITK's expected format
        direction_cosines = np.concatenate(
            (row_direction, column_direction, slice_direction)
        ).tolist()

        self.rtdose_image = sitk.GetImageFromArray(dose_grid)
        self.rtdose_image.SetSpacing(voxel_spacing)
        self.rtdose_image.SetOrigin(origin)
        self.rtdose_image.SetDirection(direction_cosines)

        self.dose_array = sitk.GetArrayFromImage(self.rtdose_image)
        # self.dose_array = sitk.GetArrayFromImage(self.rtdose_image)
        # self.dose_array = dataset.pixel_array
        # Apply metadata if necessary
        self._extract_dose_grid_scaling()

    def _extract_dose_grid_scaling(self):
        """
        Extracts the DoseGridScaling from the RTDOSE image metadata or dataset.

        Raises:
            RuntimeError: If the DoseGridScaling metadata is not found.
        """
        if self.rtdose_image and self.rtdose_image.HasMetaDataKey("3004|000e"):
            self.dose_grid_scaling = float(self.rtdose_image.GetMetaData("3004|000e"))
        elif self.dataset and "DoseGridScaling" in self.dataset:
            self.dose_grid_scaling = float(self.dataset.DoseGridScaling)
        else:
            raise RuntimeError("DoseGridScaling metadata not found in the RTDOSE data.")
        # Apply the scaling factor to the dose array
        self.dose_array = self.dose_array.astype(np.float64) * self.dose_grid_scaling

    def _find_rtdose_in_directory(self, directory_path):
        """
        Iterates through all files in a directory to find an RTDOSE file.

        Args:
            directory_path (str): The path to the directory to search in.

        Returns:
            str: The path to the RTDOSE file if found, otherwise None.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = dcmread(file_path, stop_before_pixels=True)
                    if ds.Modality == "RTDOSE":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None

    def get_dose_data(self):
        """
        Returns the dose data extracted from the RTDOSE file.

        This method provides access to the dose data as a NumPy array after it has been
        read and scaled.

        Returns:
            numpy.ndarray: The dose data as a NumPy array.

        Raises:
            ValueError: If the dose data has not been loaded. Call `read` method first.
        """
        if self.dose_array is None:
            raise ValueError("RTDOSE not loaded. Call `read` method first.")
        return self.dose_array

    def resample_dose_to_image_grid(self, image_reader):
        """
        Resamples the RTDOSE data onto the grid of a specified reference image.

        This method resamples the dose data to match the grid of a reference image
        (e.g., CT, MR, PT) using linear interpolation.

        Args:
            image_reader (DICOMImageReader): An instance of DICOMImageReader
            containing the reference image.

        Returns:
            numpy.ndarray: The resampled dose data as a NumPy array.

        Raises:
            ValueError: If the RTDOSE data has not been loaded. Call `read` method first.
            TypeError: If the reference image is not a SimpleITK Image object.
        """
        # Extract the reference image from the ImageReader (can be CT, MR, PT, etc.)
        reference_image = image_reader.image

        if not isinstance(reference_image, sitk.Image):
            raise TypeError(
                "The image extracted from ImageReader must be a SimpleITK Image object."
            )

        if self.rtdose_image is None:
            raise ValueError("RTDOSE not loaded. Call `read` method first.")

        # Step 5: Set up resampling parameters to match the reference image grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(
            reference_image
        )  # Use reference image (CT, MR, PT, etc.) as reference
        resampler.SetInterpolator(sitk.sitkLinear)  # Linear interpolation for resampling
        resampler.SetOutputSpacing(reference_image.GetSpacing())
        resampler.SetOutputOrigin(reference_image.GetOrigin())
        resampler.SetOutputDirection(reference_image.GetDirection())
        resampler.SetSize(reference_image.GetSize())
        resampler.SetDefaultPixelValue(
            0
        )  # Set default value for regions outside the original dose grid

        # Step 6: Apply the resampling filter to the RTDOSE image
        resampled_dose_image = resampler.Execute(self.rtdose_image)

        # Step 7: Convert the resampled dose image back to a NumPy array
        resampled_dose_array = (
            sitk.GetArrayFromImage(resampled_dose_image) * self.dose_grid_scaling
        )

        return resampled_dose_array
