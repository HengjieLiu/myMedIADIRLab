from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QTabWidget,
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QApplication,
    QAbstractItemView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QBrush, QColor
from dicom_viewer.config.logging_config import logger


class DICOMInfoTab(QWidget):
    def __init__(self):
        super().__init__()
        # Set up layout
        self.info_layout = QVBoxLayout(self)

        # Initialize the QTabWidget for displaying DICOM information
        self.info_tab_widget = QTabWidget()
        self.info_tab_widget.setObjectName("innerTabWidget")
        self.info_tab_widget.setTabsClosable(True)
        self.info_tab_widget.tabCloseRequested.connect(self.close_info_tab)
        self.info_layout.addWidget(self.info_tab_widget)

        # Dictionary to map tab indices to Series UID
        self.series_uid_map = {}

        # Add the default tab
        self.default_tab_index = -1
        self.add_default_info_tab()

    def add_default_info_tab(self):
        """Add a default closeable tab to the DICOM Info tab."""
        default_tab = QWidget()
        default_tab.setObjectName("infoTabWidget")
        default_layout = QVBoxLayout()
        no_dicom_label = QLabel("No DICOM series selected to show.")
        no_dicom_label.setObjectName("infoLabel")
        no_dicom_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        default_layout.addWidget(no_dicom_label)
        default_tab.setLayout(default_layout)

        # Add default tab to the tab widget
        self.default_tab_index = self.info_tab_widget.addTab(default_tab, "")

    def close_info_tab(self, index):
        """Handle closing of info tabs."""
        # Remove the tab at the given index
        self.info_tab_widget.removeTab(index)

        # Remove the tab's UID from the map if it was not the default tab
        if index in self.series_uid_map:
            del self.series_uid_map[index]

        # Reset the default tab index if the default tab was closed
        if index == self.default_tab_index:
            self.default_tab_index = -1

        # If no tabs are left, add the default tab back
        if self.info_tab_widget.count() == 0:
            self.add_default_info_tab()

    def update_info_tab(self, series_datasets):
        """Update the DICOM Info tab with datasets from the selected series."""
        logger.info("Updating DICOM Info tab with datasets")

        # Check if the default tab is open and close it
        if self.default_tab_index != -1:
            self.info_tab_widget.removeTab(self.default_tab_index)
            self.default_tab_index = -1  # Reset the default tab index

        for series_uid, datasets in series_datasets.items():
            # Check if a tab with this series UID already exists
            existing_tab_index = self._find_tab_by_series_uid(series_uid)
            if existing_tab_index is not None:
                # If it exists, make it the active tab
                self.info_tab_widget.setCurrentIndex(existing_tab_index)
            else:
                # If it does not exist, add a new tab for this series
                series_tab = QWidget()
                series_tab.setObjectName("infoWidget")
                series_layout = QVBoxLayout(series_tab)

                # Initialize QTreeWidget with updated header labels
                tree_widget = QTreeWidget()
                tree_widget.setObjectName("infoTreeWidget")
                tree_widget.setHeaderLabels(
                    ["Instance", "Tag (Group, Element)", "Attribute", "VR", "VM", "Value"]
                )
                tree_widget.setFocusPolicy(Qt.NoFocus)
                tree_widget.setColumnWidth(0, 100)  # Adjust the width for the instance column
                tree_widget.setColumnWidth(5, 300)  # Set the width for the value column

                # Enable text selection and full-row selection
                tree_widget.setSelectionMode(QAbstractItemView.SingleSelection)
                tree_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
                tree_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
                tree_widget.itemDoubleClicked.connect(self.copy_selected_text)

                # Connect the itemExpanded signal to adjust column widths
                tree_widget.itemExpanded.connect(self.resize_columns)

                # Determine if there is only one instance
                instance_count = len(datasets)

                # Populate tree widget with DICOM information
                for idx, dataset in enumerate(datasets):
                    # Set instance number to 1 if there's only one dataset; otherwise,
                    # use the actual instance number
                    instance_number = (
                        "1"
                        if instance_count == 1
                        else (str(dataset.InstanceNumber) if "InstanceNumber" in dataset else idx)
                    )

                    instance_item = QTreeWidgetItem([instance_number])
                    tree_widget.addTopLevelItem(instance_item)

                    self.populate_tree_with_dataset(instance_item, dataset)

                for i in range(tree_widget.topLevelItemCount()):
                    top_level_item = tree_widget.topLevelItem(i)
                    top_level_item.setExpanded(True)

                series_layout.addWidget(tree_widget)
                new_tab_index = self.info_tab_widget.addTab(
                    series_tab, f"{datasets[0].Modality} - {datasets[0].PatientName}"
                )
                # Store the series UID in the dictionary
                self.series_uid_map[new_tab_index] = series_uid

    def _find_tab_by_series_uid(self, series_uid):
        """Find the tab index by the series UID."""
        for index, uid in self.series_uid_map.items():
            if uid == series_uid:
                return index
        return None

    def populate_tree_with_dataset(self, parent_item, dataset):
        """Populate the QTreeWidget with DICOM dataset information."""
        for element in dataset:
            tag_name = element.name
            vr = element.VR
            vm = element.VM
            value = self.get_element_value(element)

            # Format the tag as a string (e.g., "(0010, 0010)")
            tag_string = f"({element.tag.group:04X}, {element.tag.element:04X})"

            if value:
                if value == "Sequence":
                    num_items = (
                        len(element.value) if hasattr(element, "value") and element.value else 0
                    )
                    attr_item = QTreeWidgetItem(
                        ["", tag_string, tag_name, vr, str(vm), f"{value} ({num_items} Items)"]
                    )
                else:
                    attr_item = QTreeWidgetItem(["", tag_string, tag_name, vr, str(vm), value])
            else:
                font = QFont()
                font.setItalic(True)
                gray_brush = QBrush(QColor("gray"))

                attr_item = QTreeWidgetItem(["", tag_string, tag_name, vr, str(vm), "NULL"])
                attr_item.setFont(5, font)
                attr_item.setForeground(5, gray_brush)

            parent_item.addChild(attr_item)

            # Indent the tag column for child items based on depth
            self.indent_tag_column(attr_item)

            # If the element is a sequence, recursively populate with item numbers
            if vr == "SQ":
                for seq_index, seq_item in enumerate(element):
                    # Add a child item labeled "Item {item number}" for each sequence item
                    item_number = seq_index + 1
                    item_label = f"Item {item_number}"

                    # Create the sequence item
                    sequence_item = QTreeWidgetItem(["", item_label])

                    # Add the sequence item under the attribute item (sequence header)
                    attr_item.addChild(sequence_item)

                    # Indent the "Item {item_number}" to match the tree depth
                    self.indent_tag_column(sequence_item)

                    # Recursively populate with the sequance's dataset
                    self.populate_tree_with_dataset(sequence_item, seq_item)

    def resize_columns(self):
        """Resize the columns of the QTreeWidget to fit contents when an item is expanded."""
        current_tree_widget = self.sender()  # Get the QTreeWidget that emitted the signal
        for col in range(current_tree_widget.columnCount()):
            current_tree_widget.resizeColumnToContents(col)

    def indent_tag_column(self, item):
        """Indent the DICOM tag column based on the item's depth in the tree."""
        depth = 0
        current_item = item
        while current_item.parent() is not None:
            depth += 1
            current_item = current_item.parent()

        # Calculate the indentation based on the depth
        indent = "    " * depth  # Four spaces per level of depth

        # Prepend the indentation to the tag string in the "DICOM Tag" column
        tag_text = item.text(1)  # Get current tag text from column 1 (DICOM Tag)

        # If the item is a sequence label(e.g., "Item {item_number}"), handle indentation as well
        if tag_text.startswith("Item"):
            item.setText(1, f"{indent}{tag_text}")  # Set indented item text
        else:
            item.setText(1, f"{indent}{tag_text}")  # Set indented tag text

    def get_element_value(self, element, size_threshold=1024):
        """
        Get the value of a DICOM element with special handling for large binary data types.

        Args:
            element: The DICOM element to retrieve the value from.
            size_threshold: The maximum size (in bytes) for returning the actual value.
                            If the size exceeds this threshold, only the size will be returned.

        Returns:
            str: The value of the element or information about the size if it's a large binary
            element.
        """

        # VR types that may contain large binary data
        large_binary_vrs = {"OB", "OW", "OF", "OD", "UN", "OL", "UR"}

        if element.VR == "SQ":
            return "Sequence"

        # Check if the element is Pixel Data or large binary data (based on VR)
        elif element.VR in large_binary_vrs:
            if element.is_undefined_length:
                return "Not Loaded"
            else:
                data_size = len(element.value) if element.value else 0

                # Return the value if size is smaller than the threshold
                if data_size <= size_threshold:
                    return str(element.value)
                else:
                    return f"{data_size} bytes, Not Loaded"

        # For other elements, return their value normally
        else:
            return str(element.value) if element.value else None

    def enable_copy(self):
        """Enable copying the selected item's text to the clipboard."""
        current_tree_widget = self.sender()
        current_tree_widget.itemDoubleClicked.connect(self.copy_selected_text)

    def copy_selected_text(self, item, column):
        """Copy the selected item's text to the clipboard."""
        text_to_copy = item.text(column)
        clipboard = QApplication.clipboard()
        clipboard.setText(text_to_copy)
