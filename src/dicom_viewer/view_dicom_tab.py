from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSplitter,
    QTabWidget,
    QLabel,
    QHBoxLayout,
    QSlider,
    QCheckBox,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QIcon
from pydicom import dcmread
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from dicom_viewer.readers.DICOMImageReader import DICOMImageReader
from dicom_viewer.readers.RTDoseReader import RTDoseReader
from dicom_viewer.readers.RTStructReader import RTStructReader
from dicom_viewer.viewer import Viewer

# from dicom_viewer.config.logging_config import logger


class ViewDICOMTab(QWidget):
    def __init__(self):
        super().__init__()
        self.series_uid_map = {}  # Dictionary to map tab indices to Series UID
        self.init_ui()
        self.add_default_view_tab()
        self.init_signals()

    def init_ui(self):
        """Initializes the user interface for the View DICOMs tab."""
        self.layout = QVBoxLayout(self)
        self.view_tab_widget = QTabWidget()
        self.view_tab_widget.setObjectName("innerTabWidget")
        self.view_tab_widget.setTabsClosable(True)
        self.view_tab_widget.tabCloseRequested.connect(self.close_view_tab)
        self.layout.addWidget(self.view_tab_widget)

    def add_default_view_tab(self):
        """Adds a default view tab when there are no other tabs open."""
        default_tab = QWidget()
        default_tab.setObjectName("viewTabWidget")
        layout = QHBoxLayout(default_tab)

        # Create the main splitter to divide the sidebar and viewing area
        main_splitter = QSplitter(Qt.Horizontal, default_tab)

        layout.addWidget(main_splitter)

        # Create the sidebar and add it to the main splitter
        sidebar, self.contours_list_widget = (
            self.create_sidebar()
        )  # Ensure contours_list_widget is properly initialized
        main_splitter.addWidget(sidebar)

        # Create the viewing area and add it to the main splitter
        view_area = self.create_view_area()
        main_splitter.addWidget(view_area)

        main_splitter.setSizes([200, 900])  # Adjusted width for sidebar to accommodate new layout

        self.view_tab_widget.addTab(default_tab, "")

    def create_sidebar(self):
        """Creates a combined sidebar for the viewing tab with Contours, Dose, and Settings."""
        sidebar = QWidget()
        sidebar.setObjectName("sidebarWidget")
        sidebar_layout = QVBoxLayout(sidebar)

        # Contours section
        contours_widget = QWidget()
        contours_widget.setObjectName("contoursWidget")
        contours_layout = QVBoxLayout(contours_widget)

        contours_label = QLabel("Contours")
        contours_label.setObjectName("contoursLabel")
        contours_label.setAlignment(Qt.AlignCenter)

        # Initialize contours_list_widget as QListWidget
        contours_list_widget = QListWidget()
        contours_list_widget.setObjectName("contourListWidget")
        contours_list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        contours_layout.addWidget(contours_label)
        contours_layout.addWidget(contours_list_widget)

        contours_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sidebar_layout.addWidget(contours_widget, 9)

        # Dose section
        dose_widget = QWidget()
        dose_widget.setObjectName("doseWidget")
        dose_layout = QVBoxLayout(dose_widget)

        dose_label = QLabel("Dose")
        dose_label.setObjectName("doseLabel")
        dose_label.setAlignment(Qt.AlignCenter)
        self.dose_checkbox = QCheckBox("Overlay Dose")
        self.dose_checkbox.stateChanged.connect(self.on_toggle_dose)

        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setRange(0, 100)
        threshold_slider.valueChanged.connect(self.set_dose_threshold)

        dose_layout.addWidget(dose_label)
        dose_layout.addWidget(self.dose_checkbox)
        dose_layout.addWidget(threshold_slider)
        dose_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        sidebar_layout.addWidget(dose_widget, 1)

        sidebar.setLayout(sidebar_layout)

        return (
            sidebar,
            contours_list_widget,
        )  # Return both the sidebar and the contours list widget

    def create_view_area(self):
        """Creates the viewing area for displaying DICOM images (Axial View)."""
        view_area = QWidget()
        view_area.setObjectName("viewAreaWidget")
        view_area_layout = QVBoxLayout(view_area)

        # Create an instance of your Viewer class
        self.viewer_widget = Viewer()
        view_area_layout.addWidget(self.viewer_widget)

        return view_area

    def open_series(self, image_files, rt_files):
        """Open an image series and/or overlay RT series in the viewer."""
        if image_files:
            self._open_image_series(image_files)

            # Overlay RT if present
            if rt_files:
                for rt_file in rt_files:
                    ds = dcmread(rt_file[0])
                    if ds.Modality == "RTSTRUCT":
                        self._overlay_rtstruct_on_image(image_files[0], rt_file[0])
                    elif ds.Modality == "RTDOSE":
                        self._overlay_rtdose_on_image(image_files[0], rt_file[0])

        elif rt_files:
            for rt_file in rt_files:
                ds = dcmread(rt_file[0])
                if ds.Modality == "RTDOSE":
                    self._open_rtdose_as_image(rt_file[0])

    def _open_image_series(self, image_files):
        """Open the image series in a new tab or activate the existing tab."""
        series_uid = dcmread(image_files[0]).SeriesInstanceUID
        tab_index = self._find_tab_by_series_uid(series_uid)
        if tab_index is not None:
            # Image is already open; activate the tab
            self.view_tab_widget.setCurrentIndex(tab_index)
        else:
            # Determine if a new tab is needed or use the default tab
            if self.view_tab_widget.count() == 1 and self.view_tab_widget.tabText(0).strip() == "":
                # Use the default tab if no other images are open
                tab_index = 0
            else:
                # Create a new tab for the new image series
                new_tab = self.add_new_view_tab()
                tab_index = self.view_tab_widget.indexOf(new_tab)
                self.view_tab_widget.setCurrentIndex(tab_index)

            # Set tab text with Series Description and Series UID
            image_ds = dcmread(image_files[0])
            modality = image_ds.Modality
            series_description = image_ds.SeriesDescription

            tab_text = f"{modality} - {series_description}"
            self.view_tab_widget.setTabText(tab_index, tab_text)

            # Store the Series Instance UID in the tab data
            self.series_uid_map[tab_index] = series_uid

            # Read image and display
            image_reader = DICOMImageReader(image_files, modality)
            image_reader.read()
            image_array = image_reader.get_image_array()

            # Set the viewer widget's image reader and display the image
            viewer_widget = self.view_tab_widget.widget(tab_index).findChild(Viewer)
            viewer_widget.image_reader = image_reader
            viewer_widget.display_image(image_array)

    def _overlay_rtstruct_on_image(self, image_file, rtstruct_file):
        """Overlay RTSTRUCT on the existing image series."""
        series_uid = dcmread(image_file).SeriesInstanceUID
        tab_index = self._find_tab_by_series_uid(series_uid)
        if tab_index is not None:
            viewer_widget = self.get_viewer_in_tab(tab_index)
            if viewer_widget:
                rtstruct_reader = RTStructReader(rtstruct_file)
                rtstruct_reader.read()
                viewer_widget.rtstruct_reader = rtstruct_reader
                self.populate_contours_list(rtstruct_reader, tab_index)

    def populate_contours_list(self, rtstruct_reader, tab_index):
        """Populate the contours list with structure names and set up toggling."""
        viewer_widget = self.get_viewer_in_tab(tab_index)
        if not viewer_widget:
            return

        # Correctly access the contours_list_widget associated with this tab
        contours_list_widget = self.view_tab_widget.widget(tab_index).findChild(QListWidget)
        if not contours_list_widget:
            print("Contours list widget not found.")
            return

        contours_list_widget.clear()

        for structure_name in rtstruct_reader.get_structure_names():
            color = rtstruct_reader.get_structure_color(structure_name)

            # Create a pixmap with the structure color
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor(*color))

            # Create the item for the QListWidget
            item = QListWidgetItem()
            item.setText(structure_name)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            item.setIcon(QIcon(pixmap))  # Set the pixmap as an icon to represent the color

            contours_list_widget.addItem(item)

        contours_list_widget.itemChanged.connect(self.on_structure_toggled)

    def _open_rtdose_as_image(self, dose_file):
        # TODO: implement this method
        pass

    def on_contour_item_changed(self, item):
        """Handle changes to contour visibility based on user input."""
        structure_name = item.text()
        visible = item.checkState() == Qt.Checked

        # Find the tab in which this contour is being modified
        for i in range(self.view_tab_widget.count()):
            tab = self.view_tab_widget.widget(i)
            if tab.contours_list_widget == item.listWidget():
                viewer_widget = self.get_viewer_in_tab(i)
                if viewer_widget and viewer_widget.rtstruct_reader:
                    viewer_widget.toggle_structures(structure_name, visible)
                break

    def _overlay_rtdose_on_image(self, image_file, rtdose_file):
        """Overlay RTDOSE on the existing image series."""
        series_uid = dcmread(image_file).SeriesInstanceUID
        tab_index = self._find_tab_by_series_uid(series_uid)
        if tab_index is not None:
            viewer_widget = self.get_viewer_in_tab(tab_index)
            if viewer_widget:
                rtdose_reader = RTDoseReader(rtdose_file)
                rtdose_reader.read()
                # Get the image_reader from the current tab
                image_reader = viewer_widget.image_reader
                dose_array = rtdose_reader.resample_dose_to_image_grid(image_reader)
                viewer_widget.overlay_dose(dose_array)

    def _find_tab_by_series_uid(self, series_uid):
        """Find the tab index by the series UID."""
        for tab_index, tab_uid in self.series_uid_map.items():
            if tab_uid == series_uid:
                return tab_index
        return None

    def get_viewer_in_tab(self, index):
        """Helper function to get the Viewer widget in a given tab."""
        tab_content = self.view_tab_widget.widget(index)
        if tab_content:
            return tab_content.findChild(Viewer)
        return None

    def add_new_view_tab(self):
        """Adds a new view tab and returns the tab widget."""
        new_tab = QWidget()
        new_tab.setObjectName("viewTabWidget")
        layout = QHBoxLayout(new_tab)

        main_splitter = QSplitter(Qt.Horizontal, new_tab)
        layout.addWidget(main_splitter)

        # Create the sidebar and add it to the main splitter
        sidebar, self.contours_list_widget = (
            self.create_sidebar()
        )  # Ensure contours_list_widget is properly initialized
        main_splitter.addWidget(sidebar)

        # Create the viewing area and add it to the main splitter
        view_area = self.create_view_area()
        main_splitter.addWidget(view_area)

        main_splitter.setSizes([200, 900])

        self.view_tab_widget.addTab(new_tab, "")
        return new_tab

    def set_dose_threshold(self, value):
        """Set the dose threshold based on slider value."""
        tab_index = self.view_tab_widget.currentIndex()
        viewer_widget = self.get_viewer_in_tab(tab_index)
        if viewer_widget and viewer_widget.dose_overlay is not None:
            dose_threshold = value / 100.0 * viewer_widget.dose_overlay.max()
            viewer_widget.set_dose_threshold(dose_threshold)

    def on_toggle_dose(self):
        checked = self.dose_checkbox.isChecked()
        self.viewer_widget.toggle_dose_overlay(checked)

    def on_structure_toggled(self, item):
        """Update the structure visibility based on the toggle state."""
        structure_name = item.text()  # Get the structure name from the item
        is_visible = item.checkState() == Qt.Checked  # Check if the item is checked

        # Find the Viewer widget from the sender context
        tab_index = self.view_tab_widget.currentIndex()
        viewer_widget = self.get_viewer_in_tab(tab_index)
        if viewer_widget:
            viewer_widget.toggle_structures(structure_name, is_visible)

    def close_view_tab(self, index):
        """Handles the closing of a single tab."""
        # if self.view_tab_widget.count() > 1:
        widget = self.view_tab_widget.widget(index)
        if widget is not None:
            # Clean up VTK resources specific to this tab
            self.cleanup_vtk_widget(widget)
            self.view_tab_widget.removeTab(index)

            # Remove the closed tab's UID from the map
            if index in self.series_uid_map:
                del self.series_uid_map[index]

        # Restore a default tab when no tabs are open
        if self.view_tab_widget.count() == 0:
            self.add_default_view_tab()

    def cleanup_vtk_widget(self, widget):
        """Properly clean up VTK-related resources to prevent errors."""
        for child in widget.findChildren(QVTKRenderWindowInteractor):
            render_window = child.GetRenderWindow()
            if render_window:
                render_window.Finalize()
                render_window.RemoveRenderer(self.viewer_widget.vtk_renderer)
                render_window.SetInteractor(None)
            child.deleteLater()  # Mark the widget for deletion
        widget.deleteLater()  # Clean up the widget itself

    def init_signals(self):
        """Initializes the signals for the tab."""
        pass

    def closeEvent(self, event):
        """Ensure proper cleanup when the entire tab or application is closed."""
        self.cleanup_all_vtk_widgets()
        event.accept()
        # super().closeEvent(event)

    def cleanup_all_vtk_widgets(self):
        """Clean up all VTK widgets in all tabs."""
        for i in range(self.view_tab_widget.count()):
            widget = self.view_tab_widget.widget(i)
            if widget is not None:
                viewer_widget = widget.findChild(Viewer)
                if viewer_widget:
                    viewer_widget.cleanup_vtk()
                # self.cleanup_vtk_widget(widget)
            self.cleanup_vtk_widget(widget)

    def is_series_open(self, series_uid):
        """Check if a series is already open in any of the tabs."""
        for i in range(self.view_tab_widget.count()):
            tab_text = self.view_tab_widget.tabText(i)
            if series_uid in tab_text:  # Check if the Series UID is in the tab text
                return True, i
        return False, -1
