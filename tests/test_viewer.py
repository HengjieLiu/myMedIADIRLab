import pytest
from PySide6.QtWidgets import QApplication
from dicom_viewer.dicom_viewer import DICOMViewer


@pytest.fixture(scope="module")
def app():
    """Fixture for QApplication."""
    return QApplication([])


@pytest.fixture
def viewer(app):
    """Fixture for DICOMViewer instance."""
    return DICOMViewer()


def test_viewer_initialization(viewer):
    """Test initialization of DICOMViewer."""
    assert viewer.windowTitle() == "DICOM Viewer"
    assert viewer.geometry().width() == 1200
    assert viewer.geometry().height() == 800


def test_tabs_exist(viewer):
    """Test if all tabs are initialized in DICOMViewer."""
    tab_widget = viewer.tab_widget
    assert tab_widget.count() == 4
    assert tab_widget.tabText(0) == "Load DICOMs"
    assert tab_widget.tabText(1) == "View DICOMs"
    assert tab_widget.tabText(2) == "DICOM Info"
    assert tab_widget.tabText(3) == "About"
