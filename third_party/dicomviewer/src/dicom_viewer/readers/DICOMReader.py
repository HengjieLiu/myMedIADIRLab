class DICOMReader:
    """
    A base class for reading DICOM files.

    This class provides a template for reading DICOM files and is intended to be
    subclassed to implement specific DICOM file readers, such as for CT, MR, or PET images.

    Attributes:
        dicom_path (str): The path to the directory containing DICOM files.
        dataset (pydicom.Dataset or None): The DICOM dataset object read from the file.
    """

    def __init__(self, dicom_path):
        """
        Initializes the DICOMReader with the path to the DICOM files.

        Args:
            dicom_path (str): The path to the directory containing DICOM files.
        """
        self.dicom_path = dicom_path
        self.dataset = None

    def read(self):
        """
        Reads the DICOM files from the specified path.

        This method must be implemented by subclasses to provide specific reading functionality.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement this method")
