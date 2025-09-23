# about_tab.py

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt
from dicom_viewer import __version__


class AboutTab(QWidget):
    def __init__(self):
        super().__init__()

        # Set up layout with adjusted spacing
        layout = QVBoxLayout(self)
        layout.setSpacing(5)  # Reduced spacing between elements
        layout.setContentsMargins(15, 15, 15, 15)  # Reduced margins around the layout

        # Add application description
        description = QLabel(
            "DICOM Viewer Application\n\n"
            "This application allows you to load, view, and analyze DICOM files for "
            "various modalities including CT, MRI, and RT objects. It also provides tools "
            "for adjusting image settings and viewing DICOM metadata.\n"
        )
        description.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        description.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(description)

        # Display version information
        version_info = QLabel(f"Version: {__version__}")
        version_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        version_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(version_info)

        # Add link to GitLab page
        gitlab_link = QLabel(
            '<a href="https://gitlab.com/YAAF/dicomviewer">Project GitLab Page</a>'
        )
        gitlab_link.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        gitlab_link.setOpenExternalLinks(True)
        gitlab_link.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(gitlab_link)

        # Add link to LICENSE information
        license_info = QLabel(
            '<a href="https://www.gnu.org/licenses/lgpl-3.0.html">Licensed under LGPL 3.0</a>'
        )
        license_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        license_info.setOpenExternalLinks(True)
        license_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(license_info)

        # Add copyright information
        copyright_info = QLabel("© 2024 Yasin Abdulkadir. All rights reserved.")
        copyright_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        copyright_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(copyright_info)

        # Add a spacer to prevent elements from expanding
        layout.addStretch()
