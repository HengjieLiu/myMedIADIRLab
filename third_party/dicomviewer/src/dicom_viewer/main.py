import os
import sys
from PySide6.QtWidgets import QApplication
from dicom_viewer.dicom_viewer import DICOMViewer  # Updated import path


from dicom_viewer.config.logging_config import logger
from dicom_viewer.styles.dark_theme import dark_theme

import importlib.resources as pkg_resources


def resource_path(relative_path):
    """Get absolute path to resource, works for package data."""
    try:
        with pkg_resources.path("dicom_viewer.icons", relative_path) as p:
            return str(p)
    except FileNotFoundError:
        return os.path.join(os.path.dirname(__file__), relative_path)


def main():
    logger.info("Starting DICOM Viewer Application with SQLAlchemy logging.")
    app = QApplication(sys.argv)

    viewer = DICOMViewer()

    # Apply the dark theme styesheet
    app.setStyleSheet(dark_theme)

    viewer.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
