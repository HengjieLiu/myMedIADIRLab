from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QSlider,
)
from PySide6.QtCore import Qt

from dicom_viewer.config.logging_config import logger
from dicom_viewer.load_dicom_tab import LoadDICOMTab
from dicom_viewer.view_dicom_tab import ViewDICOMTab
from dicom_viewer.dicom_info_tab import DICOMInfoTab
from dicom_viewer.about_tab import AboutTab


class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        logger.info("Initializing DICOMViewer")

        # Set up the main window
        self.setWindowTitle("DICOM Viewer")
        # self.setWindowState(Qt.WindowMaximized)
        self.setGeometry(100, 100, 1200, 800)

        # Initialize the tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("outerTabWidget")
        self.setCentralWidget(self.tab_widget)

        # Set tab variables to None
        self.load_dicom_tab = None
        self.view_dicom_tab = None
        self.dicom_info_tab = None
        self.about_tab = None

        # Add tabs
        self.add_load_dicom_tab()
        self.add_view_dicom_tab()
        self.add_dicom_info_tab()
        self.add_about_tab()

        # Connect signals
        self.connect_signals()

        # Maximize the window on startup
        # self.showMaximized()

    def add_load_dicom_tab(self):
        """Add 'Load DICOMs' tab."""
        logger.debug("Adding Load DICOMs tab")
        self.load_dicom_tab = LoadDICOMTab()
        self.tab_widget.addTab(self.load_dicom_tab, "Load DICOMs")

    def add_view_dicom_tab(self):
        """Add 'View DICOMs' tab."""
        logger.debug("Adding View DICOMs tab")
        self.view_dicom_tab = ViewDICOMTab()
        self.tab_widget.addTab(self.view_dicom_tab, "View DICOMs")

    def add_dicom_info_tab(self):
        """Add 'DICOM Info' tab."""
        logger.debug("Adding DICOM Info tab")  # Log tab addition
        self.dicom_info_tab = DICOMInfoTab()
        self.tab_widget.addTab(self.dicom_info_tab, "DICOM Info")

    def add_about_tab(self):
        """Add 'About' tab."""
        logger.debug("Adding About tab")
        self.about_tab = AboutTab()
        self.tab_widget.addTab(self.about_tab, "About")

    def connect_signals(self):
        """Connect all necessary signals between tabs."""
        logger.debug("Connecting signals between tabs")  # Log signal connections

        # Connect the dicoms_loaded signal to a slot
        self.load_dicom_tab.dicoms_loaded.connect(self.on_dicoms_loaded)

        # Connect the series selected signal to a slot
        self.load_dicom_tab.series_selected.connect(self.on_series_selected)

        # Connect the open image series signal to a slot
        self.load_dicom_tab.open_series_signal.connect(self.view_dicom_tab.open_series)

        # Connect the show_info_signal to the update_info_tab slot
        self.load_dicom_tab.show_info_signal.connect(self.dicom_info_tab.update_info_tab)

        self.load_dicom_tab.switch_to_view_tab_signal.connect(self.switch_to_view_tab)
        self.load_dicom_tab.show_info_signal.connect(self.show_info_tab)

    def switch_to_view_tab(self):
        """Switch to the View DICOM tab."""
        self.tab_widget.setCurrentWidget(self.view_dicom_tab)

    def show_info_tab(self, series_datasets):
        """Slot to handle displaying the DICOM Info tab."""
        self.dicom_info_tab.update_info_tab(series_datasets)
        self.tab_widget.setCurrentWidget(self.dicom_info_tab)  # Switch to the DICOM Info tab

    def create_sidebar(self):
        """Create the sidebar with 'Contours', 'Dose', and 'Settings' tabs."""
        logger.debug("Creating sidebar for View DICOMs tab")
        sidebar = QTabWidget()

        # Contours tab
        contours_tab = QWidget()
        contours_layout = QVBoxLayout()
        contours_list = QListWidget()
        contours_list.addItem("ROI 1")
        contours_list.addItem("ROI 2")
        contours_layout.addWidget(contours_list)
        contours_tab.setLayout(contours_layout)
        sidebar.addTab(contours_tab, "Contours")

        # Dose tab
        dose_tab = QWidget()
        dose_layout = QVBoxLayout()
        dose_layout.addWidget(QLabel("Dose controls go here."))
        dose_tab.setLayout(dose_layout)
        sidebar.addTab(dose_tab, "Dose")

        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()

        # Add controls for windowing and leveling
        window_slider = self.create_slider("Window", 0, 4095)
        level_slider = self.create_slider("Level", 0, 4095)

        settings_layout.addWidget(QLabel("Adjust Window/Level:"))
        settings_layout.addWidget(window_slider)
        settings_layout.addWidget(level_slider)
        settings_tab.setLayout(settings_layout)

        sidebar.addTab(settings_tab, "Settings")

        return sidebar

    def create_slider(self, label_text, min_val, max_val):
        """Helper function to create a slider with a label."""
        logger.debug(f"Creating slider for {label_text}")
        container = QWidget()
        layout = QVBoxLayout()

        label = QLabel(f"{label_text} ({min_val}-{max_val})")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue((min_val + max_val) // 2)

        # Connect slider to functionality (to be implemented)
        slider.valueChanged.connect(lambda value: self.on_slider_change(label_text, value))

        layout.addWidget(label)
        layout.addWidget(slider)
        container.setLayout(layout)

        return container

    def on_slider_change(self, control, value):
        """Handle slider changes for windowing and leveling."""
        logger.info(f"{control} slider changed to {value}")
        # Implement the logic to adjust window/level settings using VTK or image processing
        # libraries here.

    def create_view_area(self):
        """Create the main viewing area divided into four quadrants."""
        logger.debug("Creating main viewing area for View DICOMs tab")
        view_area = QWidget()
        view_layout = QVBoxLayout()
        view_layout.addWidget(QLabel("Main viewing area (axial, sagittal, coronal, info)."))
        view_area.setLayout(view_layout)
        return view_area

    def on_dicoms_loaded(self, dicom_files):
        """Slot to handle DICOMs loaded event."""
        logger.info("DICOM files loaded successfully")
        # Implement any logic to update other tabs or widgets based on loaded DICOM data

    def on_series_selected(self, selected_series):
        """Handle the display of selected series in the 'View DICOMs' tab."""
        logger.info(f"Series selected for viewing: {selected_series}")
        # Implement the logic to update the "View DICOMs" tab with the selected series

    def closeEvent(self, event):
        """Clean up resources when the main window is closed."""
        # Call the VTK cleanup method for ViewDICOMTab
        if self.view_dicom_tab:
            self.view_dicom_tab.cleanup_all_vtk_widgets()
        event.accept()


# Main application execution
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec())
