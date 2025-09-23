import pydicom
from pydicom import dcmread, Dataset


class DICOMRawReader:
    """
    A class for reading DICOM RAW files and extracting embedded datasets from
    the MIMSoftwareSessionMetaSeq tag.

    This class reads DICOM RAW files or accepts a pydicom.Dataset directly and
    extracts all embedded datasets that are contained within the
    MIMSoftwareSessionMetaSeq (0013, 2050) tag, which could include full datasets
    of other modalities like REG or RTSTRUCT.

    Attributes:
        raw_file_path (str or None): The path to the RAW DICOM file, if provided.
        dataset (pydicom.Dataset or None): The DICOM dataset object read from the RAW file.
        embedded_datasets (list of pydicom.Dataset): A list of extracted embedded datasets.
    """

    def __init__(self, raw_input):
        """
        Initializes the DICOMRawReader with either a path to the RAW DICOM file or a
        pydicom.Dataset object.

        Args:
            raw_input (str or pydicom.Dataset): Path to the RAW DICOM file or a
            pydicom.Dataset object.

        Raises:
            ValueError: If raw_input is neither a file path (str) nor a pydicom.Dataset object.
        """
        self.raw_file_path = None
        self.dataset = None
        self.embedded_datasets = []
        self.referenced_series_uid = None
        if isinstance(raw_input, str):
            # raw_input is a path to a RAW DICOM file
            self.raw_file_path = raw_input
        elif isinstance(raw_input, pydicom.Dataset):
            # raw_input is already a pydicom.Dataset
            self.dataset = raw_input
        else:
            raise ValueError(
                "raw_input must be either a file path (str) or a pydicom.Dataset object."
            )

    def read(self):
        """
        Reads the RAW DICOM file or uses the provided dataset and extracts embedded datasets
        from the MIMSoftwareSessionMetaSeq tag.

        This method loads the RAW DICOM file (if a path is provided), or uses the provided
        pydicom.Dataset, reads the dataset, and calls the method to extract any embedded datasets.

        Raises:
            IOError: If the RAW DICOM file cannot be read.
            ValueError: If the MIMSoftwareSessionMetaSeq tag is not found.
        """
        try:
            if self.raw_file_path:
                # Read the RAW DICOM file using pydicom
                self.dataset = dcmread(self.raw_file_path)
            elif self.dataset is not None:
                # Use the provided dataset
                pass
            else:
                raise ValueError("No RAW file path or dataset provided.")

            # Extract embedded datasets
            self.extract_embedded_datasets()
            self._get_referenced_series_uid()

        except Exception as e:
            raise IOError(f"Failed to read RAW DICOM file or dataset: {e}")

    def extract_embedded_datasets(self):
        """
        Extracts all embedded datasets from the MIMSoftwareSessionMetaSeq tag.

        This method searches for the MIMSoftwareSessionMetaSeq (0013, 2050) tag in the
        RAW DICOM file and extracts each item as an individual dataset, storing them in
        the embedded_datasets list.

        Raises:
            ValueError: If the MIMSoftwareSessionMetaSeq tag is not found in the RAW file.
        """
        if self.dataset is None:
            raise ValueError("RAW DICOM file not loaded. Call `read` method first.")

        if (0x0013, 0x2050) in self.dataset:
            # mim_seq = self.dataset["0013,2050"]
            mim_seq = self.dataset[(0x0013, 0x2050)]

            # Iterate over each item in MIMSoftwareSessionMetaSeq
            for item in mim_seq:
                if isinstance(item, Dataset):
                    self.embedded_datasets.append(item)
        else:
            raise ValueError(
                "MIMSoftwareSessionMetaSeq (0013, 2050) tag not found in the RAW DICOM file."
            )

    def get_embedded_datasets(self):
        """
        Returns the list of extracted embedded datasets.

        Returns:
            list of pydicom.Dataset: A list of embedded datasets extracted from the RAW file.

        Raises:
            ValueError: If no embedded datasets have been extracted. Call `read` method first.
        """
        if not self.embedded_datasets:
            if not self.dataset:
                raise ValueError("No embedded datasets extracted. Call `read` method first.")
        return self.embedded_datasets

    def _get_referenced_series_uid(self):
        try:
            self.referenced_series_uid = getattr(
                getattr(self.dataset, "ReferencedSeriesSequence")[0], "SeriesInstanceUID"
            )
        except Exception as e:
            print(f"Couldn't extract ReferencedSeriesUID: {e}")
