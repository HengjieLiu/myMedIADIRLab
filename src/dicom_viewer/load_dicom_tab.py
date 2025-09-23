import os
from pydicom import dcmread
from PySide6.QtCore import Signal, Qt, QThread, QObject, QSize
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QProgressBar,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
    QMessageBox,
)
from dicom_viewer.readers.REGReader import REGReader
from dicom_viewer.readers.DICOMRawReader import DICOMRawReader


class DICOMLoaderWorker(QObject):
    progress = Signal(int)
    finished = Signal(dict)

    def __init__(self, selected_path):
        super().__init__()
        self.selected_path = selected_path
        self.dicom_files = {}
        self.load_successful = False

    def load(self):
        try:
            if os.path.isdir(self.selected_path):
                dicom_files = self._load_from_directory(self.selected_path)

            else:
                dicom_files = [self.selected_path]

            total_files = len(dicom_files)
            for i, filepath in enumerate(dicom_files):
                try:
                    ds = dcmread(filepath, stop_before_pixels=True)
                    modality = ds.Modality

                    if modality in ["CT", "MR", "PT", "RTSTRUCT", "RTPLAN", "RTDOSE"]:
                        self._process_standard_dicom(ds, filepath)
                    elif modality == "REG":
                        self._process_reg_file(filepath)
                    elif modality == "RAW":
                        self._process_raw_file(filepath)
                    else:
                        # print(f"Unsupported modality: {modality} in file: {filepath}")
                        pass

                except Exception as e:
                    print(str(e))
                    print(f"Error reading DICOM file {filepath}: {e}")

                self.progress.emit(int((i + 1) / total_files * 100))

            self.load_successful = True
        except Exception as e:
            print(f"Error loading DICOM files: {e}")
        finally:
            self.finished.emit(self.dicom_files)

    def _load_from_directory(self, directory_path):
        dicom_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                dicom_files.append(os.path.join(root, file))
        return dicom_files

    def _process_standard_dicom(self, ds, filepath):
        patient_id = ds.PatientID
        series_uid = ds.SeriesInstanceUID
        modality = ds.Modality
        series_description = getattr(ds, "SeriesDescription", "")

        if patient_id not in self.dicom_files:
            self.dicom_files[patient_id] = {"images": {}, "rt": {}, "reg": {}, "raw": {}}

        if modality in ["CT", "MR", "PT"]:
            # Handle images
            if series_uid not in self.dicom_files[patient_id]["images"]:
                self.dicom_files[patient_id]["images"][series_uid] = {
                    "metadata": {
                        "Modality": modality,
                        "SeriesDescription": series_description,
                        "NumberOfInstances": 0,
                        "FrameOfReferenceUID": getattr(ds, "FrameOfReferenceUID", None),
                    },
                    "files": [],
                }
            self.dicom_files[patient_id]["images"][series_uid]["files"].append(filepath)
            self.dicom_files[patient_id]["images"][series_uid]["metadata"][
                "NumberOfInstances"
            ] += 1

        elif modality in ["RTSTRUCT", "RTPLAN", "RTDOSE"]:
            # Handle RT objects
            if series_uid not in self.dicom_files[patient_id]["rt"]:
                self.dicom_files[patient_id]["rt"][series_uid] = {
                    "metadata": {
                        "Modality": modality,
                        "SeriesDescription": series_description,
                        "ReferencedSOPInstanceUIDs": self._get_referenced_sop_instance_uids(ds),
                        "FrameOfReferenceUID": getattr(ds, "FrameOfReferenceUID", None),
                        "NumberOfInstances": 0,
                    },
                    "files": [],
                }
            self.dicom_files[patient_id]["rt"][series_uid]["files"].append(filepath)
            self.dicom_files[patient_id]["rt"][series_uid]["metadata"]["NumberOfInstances"] += 1

    def _process_reg_file(self, filepath):
        # Initialize REGReader and read the REG file
        reg_reader = REGReader(filepath)
        reg_reader.read()

        ds = reg_reader.reg_dataset

        patient_id = ds.PatientID
        series_uid = ds.SeriesInstanceUID
        modality = ds.Modality
        fixed_image_info = (reg_reader.get_fixed_image_info(),)
        moving_image_info = (reg_reader.get_moving_image_info(),)
        referenced_series_info = reg_reader.get_referenced_series_info()
        series_description = getattr(ds, "SeriesDescription", "")
        if patient_id not in self.dicom_files:
            self.dicom_files[patient_id] = {"images": {}, "rt": {}, "reg": {}, "raw": {}}

        if series_uid not in self.dicom_files[patient_id]["reg"]:
            self.dicom_files[patient_id]["reg"][series_uid] = {
                "metadata": {
                    "Modality": modality,
                    "SeriesDescription": series_description,
                    "ReferencedSeriesInfo": referenced_series_info,
                    "FixedImageInfo": fixed_image_info,
                    "MovingImageInfo": moving_image_info,
                    "NumberOfInstances": 0,
                },
                "files": [],
            }
        self.dicom_files[patient_id]["raw"][series_uid]["files"].append(filepath)
        self.dicom_files[patient_id]["raw"][series_uid]["metadata"]["NumberOfInstances"] += 1

    def _process_raw_file(self, filepath):
        # Initialize RAWReader and read the RAW file
        raw_reader = DICOMRawReader(filepath)
        raw_reader.read()

        ds = raw_reader.dataset

        patient_id = ds.PatientID
        series_uid = ds.SeriesInstanceUID
        modality = ds.Modality
        series_description = getattr(ds, "SeriesDescription", "")
        referenced_series_uid = raw_reader.referenced_series_uid

        if patient_id not in self.dicom_files:
            self.dicom_files[patient_id] = {"images": {}, "rt": {}, "reg": {}, "raw": {}}

        if series_uid not in self.dicom_files[patient_id]["raw"]:
            self.dicom_files[patient_id]["raw"][series_uid] = {
                "metadata": {
                    "Modality": modality,
                    "SeriesDescription": series_description,
                    "ReferencedSeriesUID": referenced_series_uid,
                    "NumberOfInstances": 0,
                },
                "files": [],
            }
        self.dicom_files[patient_id]["raw"][series_uid]["files"].append(filepath)
        self.dicom_files[patient_id]["raw"][series_uid]["metadata"]["NumberOfInstances"] += 1

    def _get_referenced_sop_instance_uids(self, ds):
        """Helper method to extract referenced SOPInstanceUIDs from RTSTRUCT objects."""
        referenced_uids = []
        if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
            for item in ds.ReferencedFrameOfReferenceSequence:
                if hasattr(item, "RTReferencedStudySequence"):
                    for study_item in item.RTReferencedStudySequence:
                        if hasattr(study_item, "RTReferencedSeriesSequence"):
                            for series_item in study_item.RTReferencedSeriesSequence:
                                if hasattr(series_item, "ContourImageSequence"):
                                    for contour_item in series_item.ContourImageSequence:
                                        referenced_uids.append(
                                            contour_item.ReferencedSOPInstanceUID
                                        )
        if hasattr(ds, "ROIContourSequence"):
            for roi_item in ds.ROIContourSequence:
                if hasattr(roi_item, "ContourSequence"):
                    for contour_seq in roi_item.ContourSequence:
                        if hasattr(contour_seq, "ContourImageSequence"):
                            for image_seq in contour_seq.ContourImageSequence:
                                referenced_uids.append(image_seq.ReferencedSOPInstanceUID)
        return referenced_uids


class LoadDICOMTab(QWidget):
    # Signal to emit when DICOM files are loaded
    dicoms_loaded = Signal(dict)

    # Signal to emit the selected series to be viewed
    series_selected = Signal(dict)

    # Signal to emit DICOM metadata for display
    show_info_signal = Signal(dict)

    # Signal to emit DICOM datasets for viewing
    open_series_signal = Signal(list, list)  # list of file paths, list of list of file paths

    # signal to overlay RT series on an image in ViewDICOMTab
    overlay_rt_signal = Signal(str, str)  # patient_id, series_uid

    # signal to switch to the opened tab
    switch_to_view_tab_signal = Signal()

    def __init__(self):
        super().__init__()
        self.dicom_files = {}  # Store loaded DICOM data

        # Setup layout and widgets
        self.layout = QVBoxLayout(self)

        self.select_button = QPushButton("Select DICOM Directory or File")
        self.select_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.select_button)

        self.path_label = QLineEdit()
        self.path_label.setPlaceholderText("No directory or file selected")
        self.path_label.setReadOnly(False)
        self.layout.addWidget(self.path_label)

        self.load_button = QPushButton("Load DICOMs")
        self.load_button.clicked.connect(self.load_dicoms)
        self.layout.addWidget(self.load_button)

        self.status_label = QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        self.dicom_tree = QTreeWidget()
        self.dicom_tree.setHeaderLabels(
            ["Patient ID", "Modality", "Instances", "Series Description"]
        )
        self.dicom_tree.setIconSize(QSize(8, 8))
        self.dicom_tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.dicom_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Disable focus rectangle
        self.dicom_tree.setFocusPolicy(Qt.NoFocus)
        self.layout.addWidget(self.dicom_tree)

        # Connect double-click signal to open selected series
        self.dicom_tree.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Add a button to open selected series in View DICOMs tab
        self.open_button = QPushButton("Open Selected Series")
        self.open_button.clicked.connect(self.open_selected_series)
        button_layout.addWidget(self.open_button)

        # Add a button to show info for the selected series
        self.info_button = QPushButton("Show Info")
        self.info_button.clicked.connect(self.show_info)
        button_layout.addWidget(self.info_button)

        # Add the button layout to the main layout
        self.layout.addLayout(button_layout)

    def open_file_dialog(self):
        # Open a file dialog to select a directory or a single file
        file_dialog = QFileDialog(self)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        if file_dialog.exec_():
            selected_path = file_dialog.selectedFiles()[0]
            self.path_label.setText(selected_path)

    def load_dicoms(self):
        selected_path = self.path_label.text()
        if not selected_path:
            self.status_label.setText("Error: No directory or file selected")
            return

        self.status_label.setText("Loading...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Set up the thread and worker
        self.thread = QThread()
        self.worker = DICOMLoaderWorker(selected_path)
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.load)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_dicoms_loaded)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start the thread
        self.thread.start()

    def on_dicoms_loaded(self, dicom_files):
        self.dicom_files = dicom_files
        self.progress_bar.setVisible(False)
        self.status_label.setText("Loading completed")
        self.populate_dicom_tree(dicom_files)
        self.dicoms_loaded.emit(dicom_files)

    def populate_dicom_tree(self, dicom_files):
        """Populate the tree view with loaded DICOM data."""
        self.dicom_tree.clear()

        for patient_id, patient_data in dicom_files.items():
            patient_item = self._create_patient_item(patient_id)

            # Create a sub-item for images under each patient
            images_item = QTreeWidgetItem(["Images"])
            patient_item.addChild(images_item)

            # Track RT, REG, and RAW objects associated with any image
            associated_rt_uids = set()
            associated_reg_uids = set()
            associated_raw_uids = set()

            # Add imaging series and associated objects
            for series_uid, series_info in patient_data["images"].items():
                image_series_item = self._add_image_series(images_item, series_uid, series_info)
                sop_instance_uid_cache = {}

                associated_rt_uids.update(
                    self._add_associated_rt(
                        patient_data,
                        series_uid,
                        series_info,
                        image_series_item,
                        associated_rt_uids,
                        sop_instance_uid_cache,
                        patient_id,
                    )
                )

                associated_reg_uids.update(
                    self._add_associated_reg(
                        patient_data, series_uid, image_series_item, associated_reg_uids
                    )
                )

                associated_raw_uids.update(
                    self._add_associated_raw(
                        patient_data, series_uid, image_series_item, associated_raw_uids
                    )
                )

            # Add unassociated objects directly under the patient
            self._add_unassociated_rt(patient_item, patient_data, associated_rt_uids)
            self._add_unassociated_reg(patient_item, patient_data, associated_reg_uids)
            self._add_unassociated_raw(patient_item, patient_data, associated_raw_uids)

        # Expand all items after population
        self._expand_all_items()

    def _create_patient_item(self, patient_id):
        """Create a top-level patient item in the tree."""
        patient_item = QTreeWidgetItem([patient_id])
        self.dicom_tree.addTopLevelItem(patient_item)
        return patient_item

    def _add_image_series(self, parent_item, series_uid, series_info):
        """Add image series under the parent item."""
        modality = series_info["metadata"]["Modality"]
        series_description = series_info["metadata"]["SeriesDescription"]
        num_instances = series_info["metadata"]["NumberOfInstances"]

        image_series_item = self._create_tree_item(
            parent_item, "", modality, str(num_instances), series_description
        )
        image_series_item.setData(0, Qt.UserRole, series_uid)
        return image_series_item

    def _add_associated_rt(
        self,
        patient_data,
        series_uid,
        series_info,
        parent_item,
        associated_rt_uids,
        sop_instance_uid_cache,
        patient_id,
    ):
        """Add associated RT objects under an image series."""
        associated = set()
        for rt_series_uid, rt_info in patient_data["rt"].items():
            if rt_series_uid in associated_rt_uids:
                continue

            rt_item = self._create_associated_item_if_needed(
                "rt", series_uid, series_info, rt_info, sop_instance_uid_cache, patient_id
            )
            if rt_item:
                rt_item.setData(0, Qt.UserRole, rt_series_uid)
                parent_item.addChild(rt_item)
                associated.add(rt_series_uid)

        return associated

    def _add_associated_reg(self, patient_data, series_uid, parent_item, associated_reg_uids):
        """Add associated REG objects under an image series."""
        associated = set()
        for reg_series_uid, reg_info in patient_data["reg"].items():
            if reg_series_uid in associated_reg_uids:
                continue

            ref_series_uid = reg_info["metadata"]["ReferencedSeriesUID"]
            if ref_series_uid == series_uid:
                reg_item = self._create_tree_item(
                    parent_item,
                    "",
                    f"    {reg_info['metadata']['Modality']}",
                    reg_info["metadata"]["NumberOfInstances"],
                    reg_info["metadata"]["SeriesDescription"],
                )
                reg_item.setData(0, Qt.UserRole, reg_series_uid)
                associated.add(reg_series_uid)

        return associated

    def _add_associated_raw(self, patient_data, series_uid, parent_item, associated_raw_uids):
        """Add associated RAW files under an image series."""
        associated = set()
        for raw_series_uid, raw_info in patient_data["raw"].items():
            if raw_series_uid in associated_raw_uids:
                continue

            if series_uid == raw_info["metadata"]["ReferencedSeriesUID"]:
                raw_item = self._create_tree_item(
                    parent_item,
                    "",
                    f"    {raw_info['metadata']['Modality']}",
                    raw_info["metadata"]["NumberOfInstances"],
                    raw_info["metadata"]["SeriesDescription"],
                )
                raw_item.setData(0, Qt.UserRole, raw_series_uid)
                associated.add(raw_series_uid)

        return associated

    def _create_associated_item_if_needed(
        self,
        series_type,
        series_uid,
        series_info,
        associated_info,
        sop_instance_uid_cache,
        patient_id,
    ):
        """Create an associated RT/REG/RAW item only if needed."""
        referenced_sop_uids = associated_info["metadata"].get("ReferencedSOPInstanceUIDs", [])
        frame_of_reference_uid = associated_info["metadata"].get("FrameOfReferenceUID", None)

        if series_uid not in sop_instance_uid_cache:
            sop_instance_uid_cache[series_uid] = self.get_image_sop_instance_uids(
                patient_id, series_uid
            )
        cached_sop_uids = sop_instance_uid_cache[series_uid]

        # Check if the associated object references the current image series
        if any(referenced_uid in cached_sop_uids for referenced_uid in referenced_sop_uids) or (
            frame_of_reference_uid
            and frame_of_reference_uid == series_info["metadata"].get("FrameOfReferenceUID", None)
        ):

            return self._create_tree_item(
                None,
                "",
                f"    {associated_info['metadata']['Modality']}",
                associated_info["metadata"].get("NumberOfInstances", "1"),
                associated_info["metadata"]["SeriesDescription"],
            )
        return None

    def _add_unassociated_rt(self, parent_item, patient_data, associated_rt_uids):
        """Add unassociated RT objects directly under the patient."""
        self._add_unassociated_items("rt", parent_item, patient_data, associated_rt_uids)

    def _add_unassociated_reg(self, parent_item, patient_data, associated_reg_uids):
        """Add unassociated REG objects directly under the patient."""
        self._add_unassociated_items("reg", parent_item, patient_data, associated_reg_uids)

    def _add_unassociated_raw(self, parent_item, patient_data, associated_raw_uids):
        """Add unassociated RAW objects directly under the patient."""
        self._add_unassociated_items("raw", parent_item, patient_data, associated_raw_uids)

    def _add_unassociated_items(self, series_type, parent_item, patient_data, associated_uids):
        """Add unassociated RT/REG/RAW objects under the parent item."""
        unassociated_uids = set(patient_data[series_type].keys()) - associated_uids
        for series_uid in unassociated_uids:
            series_info = patient_data[series_type][series_uid]
            item = self._create_tree_item(
                parent_item,
                "",
                series_info["metadata"]["Modality"],
                series_info["metadata"]["NumberOfInstances"],
                series_info["metadata"]["SeriesDescription"],
            )
            item.setData(0, Qt.UserRole, series_uid)

    def _create_tree_item(self, parent_item, *columns):
        """Helper method to create a QTreeWidgetItem with specified columns."""
        item = QTreeWidgetItem([col for col in columns])
        if parent_item:
            parent_item.addChild(item)
        return item

    def _expand_all_items(self):
        """Expand all items in the tree."""
        for i in range(self.dicom_tree.topLevelItemCount()):
            self.expand_all_items(self.dicom_tree.topLevelItem(i))

    def expand_all_items(self, item):
        item.setExpanded(True)
        for i in range(item.childCount()):
            self.expand_all_items(item.child(i))

    def get_image_sop_instance_uids(self, patient_id, series_instance_uid):
        """Helper method to get all SOPInstanceUIDs for a given image series."""
        sop_instance_uids = set()
        image_series = self.dicom_files[patient_id]["images"].get(series_instance_uid)
        if image_series:
            for filepath in image_series["files"]:
                dataset = dcmread(filepath)
                sop_instance_uids.add(dataset.SOPInstanceUID)
        return sop_instance_uids

    def show_info(self):
        """Slot to show info for the selected series using datasets."""
        selected_items = self.dicom_tree.selectedItems()
        if not selected_items:
            self.status_label.setText("No series selected for info display")
            return

        # Collect datasets for the selected series
        series_datasets = {}
        for item in selected_items:
            # Determine if the item is a top-level item (Patient) or a child item (Images, RT, REG)
            if item.parent() and item.parent().text(0) == "Images":
                # This is an image or RT/REG not associated with an image
                patient_item = item.parent().parent()  # Go up to the patient level
                patient_id = patient_item.text(0)
            elif item.parent() and item.parent().text(0) == "":
                # This is an RT associated with an image
                patient_id = item.parent().parent().parent().text(0)
            elif item.parent():
                # This is an unassociated RT or REG directly under the patient
                patient_item = item.parent()
                patient_id = patient_item.text(0)
            else:
                # This is a top-level item (Patient) which should not be directly selectable
                continue

            series_uid = item.data(0, Qt.UserRole)

            if patient_id in self.dicom_files:
                datasets = self.get_datasets_for_series(patient_id, series_uid)
                if datasets:
                    series_datasets[series_uid] = datasets

        # Emit signal with the datasets to display in the DICOM Info tab
        self.show_info_signal.emit(series_datasets)

    def open_selected_series(self):
        """Handle the opening of the selected series in the View DICOMs tab."""
        selected_items = self.dicom_tree.selectedItems()
        if not selected_items:
            self.status_label.setText("Error: No series selected")
            return

        # Organize selected series into subgroups
        selected_series = self.organize_selected_series(selected_items)

        # Determine the number of subgroups selected
        subgroup_count = len(selected_series)

        if subgroup_count == 1:
            # Handle single subgroup selection
            self.handle_single_subgroup(selected_series[0])
        else:
            # Handle multiple subgroups selection
            self.handle_multiple_subgroups(selected_series)

    def organize_selected_series(self, selected_items):
        """Organize selected series into subgroups based on relationships."""
        selected_series = {}

        # To keep track of images that have already been associated with an RT
        associated_images = set()

        for item in selected_items:
            # Determine if the item is a top-level item (Patient) or a child item (Images, RT, REG)
            if item.parent() and item.parent().text(0) == "Images":
                # This is an image or RT/REG not associated with an image
                patient_item = item.parent().parent()  # Go up to the patient level
                patient_id = patient_item.text(0)
            elif item.parent() and item.parent().text(0) == "":
                # This is an RT associated with an image
                patient_id = item.parent().parent().parent().text(0)
            elif item.parent():
                # This is an unassociated RT or REG directly under the patient
                patient_item = item.parent()
                patient_id = patient_item.text(0)
            else:
                # This is a top-level item (Patient) which should not be directly selectable
                continue

            modality = item.text(1).strip()
            series_uid = item.data(0, Qt.UserRole)

            # Validate patient_id and series_uid before proceeding
            if not patient_id or not series_uid:
                print(f"Invalid Patient ID or Series UID: {patient_id}, {series_uid}")
                continue

            if modality in ["CT", "MR", "PT"]:
                # Check if this image has already been associated with an RT
                if (patient_id, series_uid) in associated_images:
                    continue

                # Create a new subgroup for this image series
                selected_series[(patient_id, series_uid)] = {
                    "images": [(patient_id, series_uid)],
                    "rt": [],
                    "reg": [],
                    "raw": [],
                }
            elif modality in ["RTSTRUCT", "RTPLAN", "RTDOSE", "REG", "RAW"]:
                # Find the associated image series for this RT or REG object
                associated_image_series = self.find_associated_image_series(patient_id, series_uid)
                if associated_image_series:
                    key = (patient_id, associated_image_series)
                    if key not in selected_series:
                        selected_series[key] = {
                            "images": [(patient_id, associated_image_series)],
                            "rt": [],
                            "reg": [],
                            "raw": [],
                        }
                    match modality:
                        case "REG":
                            selected_series[key]["reg"].append((patient_id, series_uid))
                        case "RAW":
                            selected_series[key]["raw"].append((patient_id, series_uid))
                        case _:
                            selected_series[key]["rt"].append((patient_id, series_uid))
                    # Mark the associated image as processed
                    associated_images.add((patient_id, associated_image_series))
                else:
                    # No associated image series, treat as standalone RT/REG/RAW object
                    selected_series[(patient_id, series_uid)] = {
                        "images": [],
                        "rt": [],
                        "reg": [],
                        "raw": [],
                    }
                    match modality:
                        case "REG":
                            selected_series[(patient_id, series_uid)]["reg"].append(
                                (patient_id, series_uid)
                            )
                        case "RAW":
                            selected_series[(patient_id, series_uid)]["raw"].append(
                                (patient_id, series_uid)
                            )
                        case _:
                            selected_series[(patient_id, series_uid)]["rt"].append(
                                (patient_id, series_uid)
                            )

        return list(selected_series.values())

    def find_associated_image_series(self, patient_id, series_uid):
        """Find the associated image series for an RT or REG object."""
        rt_metadata = self.dicom_files[patient_id]["rt"][series_uid]["metadata"]
        frame_of_reference_uid = rt_metadata.get("FrameOfReferenceUID")
        referenced_sop_uids = rt_metadata.get("ReferencedSOPInstanceUIDs", [])

        for image_series_uid, image_info in self.dicom_files[patient_id]["images"].items():
            image_frame_of_reference_uid = image_info["metadata"].get("FrameOfReferenceUID")
            image_sop_uids = self.get_image_sop_instance_uids(patient_id, image_series_uid)

            if frame_of_reference_uid == image_frame_of_reference_uid or any(
                uid in image_sop_uids for uid in referenced_sop_uids
            ):
                return image_series_uid

        return None  # No associated image series found

    def handle_single_subgroup(self, subgroup):
        """Handle single subgroup selection for opening."""
        image_series = subgroup["images"][0] if subgroup["images"] else None
        rt_series = subgroup["rt"]
        # print(f"{subgroup=}")

        if image_series and rt_series:
            # Get file paths instead of datasets
            image_files = self.get_file_paths_for_series(image_series[0], image_series[1])
            rt_files = []

            for rt in rt_series:
                rt_file = self.get_file_paths_for_series(rt[0], rt[1])
                rt_files.append(rt_file)

            # Emit signal to open the image series in ViewDICOMTab
            self.open_series_signal.emit(image_files, rt_files)
            self.switch_to_view_tab_signal.emit()

        elif image_series:
            # Get file paths for the image series
            image_files = self.get_file_paths_for_series(image_series[0], image_series[1])

            # Emit signal to open the image series in ViewDICOMTab
            self.open_series_signal.emit(image_files, [])
            self.switch_to_view_tab_signal.emit()

        elif rt_series:
            rt_files = []
            for rt in rt_series:
                rt_dataset = self.get_datasets_for_series(rt[0], rt[1])
                if rt_dataset[0].Modality == "RTDOSE":
                    rt_file = self.get_file_paths_for_series(rt[0], rt[1])
                    rt_files.append(rt_file)

            if rt_files:
                # Emit signal to open the RTDOSE('s) as image('s)
                self.open_series_signal.emit([], rt_files)
                self.switch_to_view_tab_signal.emit()
            else:
                # TODO:
                # I haven't decided what should happen here. May be just show DICOM Info
                # for each RT
                pass

    def ask_user_opening_preference(self):
        """Prompt user to choose how to open multiple subgroups."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Open Multiple Series")
        msg_box.setText(
            "Do you want to open the selected series in separate tabs or in the same tab?"
        )
        separate_button = msg_box.addButton("Separate Tabs", QMessageBox.AcceptRole)
        same_button = msg_box.addButton("Same Tab", QMessageBox.RejectRole)  # noqa
        msg_box.exec_()

        if msg_box.clickedButton() == separate_button:
            return "Separate Tabs"
        else:
            return "Same Tab"

    def handle_multiple_subgroups(self, subgroups):
        """Handle multiple subgroups selection for opening."""

        for subgroup in subgroups:
            self.handle_single_subgroup(subgroup)

        # TODO: implement to handle multiple groups in a single tab
        # choice = self.ask_user_opening_preference()

        # if choice == "Separate Tabs":
        #     for subgroup in subgroups:
        #         self.handle_single_subgroup(subgroup)
        # elif choice == "Same Tab":
        #     # TODO: implement this
        #     self.view_dicom_tab.open_multiple_series_in_one_tab(subgroups)
        #     print("Opening multiple images in one tab")

    def open_image_series(self, patient_id, series_uid):
        """Open the selected image series in the DICOM Viewer tab."""
        datasets = self.get_datasets_for_series(patient_id, series_uid)
        # Logic to display these datasets in the viewer tab
        self.view_dicom_tab.open_image_series(patient_id, series_uid, datasets)

    def overlay_rt_series(self, patient_id, rt_series_uid, image_series_uid):
        """Overlay RT series data on an existing image series in the viewer."""
        rt_datasets = self.get_datasets_for_series(patient_id, rt_series_uid)
        self.view_dicom_tab.overlay_rt_series(rt_datasets, image_series_uid, rt_datasets)

    def open_rt_series(self, patient_id, series_uid):
        """Handle opening an RT series by finding a matching image series."""
        if not patient_id or not series_uid:
            print(f"No data found for PatientID: {patient_id} and SeriesInstanceUID: {series_uid}")
            return

        rt_metadata = (
            self.dicom_files.get(patient_id, {}).get("rt", {}).get(series_uid, {}).get("metadata")
        )
        if not rt_metadata:
            print(f"No data found for PatientID: {patient_id} and SeriesInstanceUID: {series_uid}")
            return

        rt_metadata = self.dicom_files[patient_id]["rt"][series_uid]["metadata"]
        referenced_sop_instance_uids = rt_metadata.get("ReferencedSOPInstanceUIDs", [])
        frame_of_reference_uid = rt_metadata.get("FrameOfReferenceUID", None)

        # Start by trying to find a directly referenced image series
        matching_image_series_uid = self.find_matching_image_series(
            patient_id, referenced_sop_instance_uids
        )

        # If no direct match found, use recursive search for referenced RT objects
        if not matching_image_series_uid:
            matching_image_series_uid = self.recursive_find_referenced_image(
                patient_id, series_uid
            )

        # If no exact match is found using recursive search, check FrameOfReferenceUID
        if not matching_image_series_uid and frame_of_reference_uid:
            for image_series_uid, image_info in self.dicom_files[patient_id]["images"].items():
                if frame_of_reference_uid == image_info["metadata"].get("FrameOfReferenceUID"):
                    matching_image_series_uid = image_series_uid
                    break

        if matching_image_series_uid:
            # Open both image and RT series together in the viewer
            self.open_image_series(patient_id, matching_image_series_uid)
            # Logic to overlay the RT series on the image
            self.overlay_rt_series(patient_id, series_uid, matching_image_series_uid)
        else:
            self.show_no_image_match_message()

    def find_matching_image_series(self, patient_id, referenced_sop_instance_uids):
        """Find a matching image series by SOPInstanceUID."""
        for image_series_uid, image_info in self.dicom_files[patient_id]["images"].items():
            for image_file in image_info["files"]:
                dataset = dcmread(image_file)
                if dataset.SOPInstanceUID in referenced_sop_instance_uids:
                    return image_series_uid
        return None

    def recursive_find_referenced_image(self, patient_id, rt_series_uid, visited=set()):
        """Recursively find a referenced image series from RT objects."""
        if rt_series_uid in visited:
            return None  # Avoid infinite loops if there's a circular reference
        visited.add(rt_series_uid)

        rt_metadata = self.dicom_files[patient_id]["rt"][rt_series_uid]["metadata"]
        referenced_sop_instance_uids = rt_metadata.get("ReferencedSOPInstanceUIDs", [])

        # Check if any referenced UID matches an image series
        matching_image_series_uid = self.find_matching_image_series(
            patient_id, referenced_sop_instance_uids
        )
        if matching_image_series_uid:
            return matching_image_series_uid

        # Recursively check if referenced UIDs point to another RT object
        for rt_series_uid, rt_info in self.dicom_files[patient_id]["rt"].items():
            if rt_info["metadata"].get("SOPInstanceUID") in referenced_sop_instance_uids:
                found_image_uid = self.recursive_find_referenced_image(
                    patient_id, rt_series_uid, visited
                )
                if found_image_uid:
                    return found_image_uid

        return None

    def show_no_image_match_message(self):
        """Show a message when no matching image series is found for the RT object."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText("No matching image series found for the selected RT object.")
        msg_box.setWindowTitle("Warning")
        msg_box.exec_()

    def on_item_double_clicked(self, item):
        """Handle double-clicking on a single row to open the series."""
        selected_items = self.dicom_tree.selectedItems()
        if len(selected_items) == 1:
            self.open_selected_series()

    def get_datasets_for_series(self, patient_id, series_instance_uid):
        """
        Retrieve a list of pydicom.Dataset objects for a given patient_id and series_instance_uid.
        """
        datasets = []
        try:
            # series = (
            #     self.dicom_files[patient_id]["images"].get(series_instance_uid)
            #     or self.dicom_files[patient_id]["rt"].get(series_instance_uid)
            #     or self.dicom_files[patient_id]["reg"].get(series_instance_uid)
            # )
            series = None
            for key in self.dicom_files[patient_id].keys():
                series = self.dicom_files[patient_id][key].get(series_instance_uid)
                if series:
                    break
            if series:
                for filepath in series["files"]:
                    datasets.append(dcmread(filepath))
        except KeyError:
            print(
                f"No data found for PatientID: {patient_id} "
                f"and SeriesInstanceUID: {series_instance_uid}"
            )
        return datasets

    def get_file_paths_for_series(self, patient_id, series_instance_uid):
        """
        Retrieve a list of file paths for a given patient_id and series_instance_uid.
        """
        file_paths = []
        try:
            series = self.dicom_files[patient_id]["images"].get(
                series_instance_uid
            ) or self.dicom_files[patient_id]["rt"].get(series_instance_uid)
            if series:
                for filepath in series["files"]:
                    file_paths.append(filepath)
        except KeyError:
            print(
                f"No data found for PatientID: {patient_id} "
                f"and SeriesInstanceUID: {series_instance_uid}"
            )
        return file_paths
