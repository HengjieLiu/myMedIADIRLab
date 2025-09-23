from dicom_viewer import resources_rc

dark_theme = """
QWidget {
    background-color: #2b2b2b;  /* General background for main widget */
    color: #ffffff;
    font-size: 14px;
}

/* Outer Tab Widget */
#outerTabWidget::pane {
    background-color: #2b2b2b; /* Matches main content area */
    border: none;
    padding: 0px;
}

#outerTabWidget::tab-bar {
    alignment: left;
    min-width: 80px;
    background-color: #1e1e1e; /* Background color for the outer tab bar */
    border-bottom: 1px solid #3c3c3c;
}

#outerTabWidget QTabBar::tab {
    background: #1e1e1e; /* Background color for inactive outer tabs */
    color: #ffffff;
    padding: 8px 12px;
    border: none;
    margin-right: 1px;
}

#outerTabWidget QTabBar::tab:selected {
    background: #2b2b2b; /* Background for active outer tab */
    color: #ffffff;
    border-bottom: none; /* No border for a seamless blend with the content area */
}

#outerTabWidget QTabBar::tab:hover {
    background: #454545; /* Hover effect for outer tabs */
}

/* Inner Tab Widget */
#innerTabWidget::pane {
    background-color: #252525; /* Background color for inner tab content area */
    border: none;
    padding: 0px;
}

#innerTabWidget::tab-bar {
    alignment: left;
    background-color: #252525; /* Background color for inner tab bar */
    border-bottom: 1px solid #4c4c4c;
}

#innerTabWidget QTabBar::tab {
    background: #252525; /* Background for inactive inner tabs */
    color: #ffffff;
    min-width: 250px;
    padding: 6px 10px;
    border: none;
    margin-right: 1px;
}

#innerTabWidget QTabBar::tab:selected {
    background: #1e1e1e; /* Dark Slate Gray for active inner tab */
    color: #ffffff;
    border-bottom: none; /* No bottom border to blend seamlessly */
}

#innerTabWidget QTabBar::tab:hover {
    background: #555555; /* Hover effect for inner tabs */
}

/* Match 'infoWidget' appearance to the active inner tab */
#infoTabWidget {
    background-color: #1e1e1e; /* Same background as the active inner tab */
    color: #ffffff;
    border: none; /* No border to match the seamless style of the tabs */
    padding: 10px; /* Ensure consistent padding with tab contents */
}

#infoLabel {
    background-color: #1e1e1e
}

#infoWidget {
    background-color: #1e1e1e
}

#viewTabWidget {
    background-color: #1e1e1e; /* Same background as the active inner tab */
    color: #ffffff;
    border: none; /* No border to match the seamless style of the tabs */
    padding: 10px; /* Ensure consistent padding with tab contents */
}

#sidebarWidget {
    background-color: #1e1e1e
}

#viewAreaWidget {
    background-color: #1e1e1e
}

#layout {
    background-color: #1e1e1e
}

#contoursLabel {
    background-color: #3a3a3a;
    color: #ffffff;
    padding: 5px;
}

#doseLabel {
    background-color: #3a3a3a;
    color: #ffffff;
    padding: 5px;
}

#contourListWidget {
    background-color: #1e1e1e;
    border: none;
}

QFileDialog QTreeView::focus, QFileDialog QListView::focus {
    outline: none;
}

/* General tab styles */
QTabBar::tab {
    margin: 0px;
    min-height: 25px;
    border-bottom: 1px solid #3c3c3c;
}

QTabWidget::pane {
    border-top: none; /* Remove top border for clean transition */
}

QMainWindow {
    background-color: #1e1e1e;
}

/* Additional widget styles as needed */
QLineEdit, QTreeWidget, QProgressBar {
    background-color: #3c3f41;
    border: 1px solid #4a4e52;
    color: #ffffff;
}

/* Ensure header visibility */
QHeaderView::section {
    background-color: #3c3f41;
    color: #ffffff;
    padding: 4px;
    border: none;
    border-bottom: 1px solid #4a4e52;
}
QSplitter::handle {
    background-color: #0f0f0f; /* A very dark gray to match the outer tab bar */
    width: 2px; /* Slightly thicker to keep it draggable but subtle */
}

/* Tree widget item styling */
QTreeView::item {
    background-color: #2b2b2b;
    color: #ffffff;
    margin: 0px;
    padding: 2px;
}

QTreeView::item:selected, QTreeView::item:hover {
    background-color: #363636;
    color: #ffffff;
    border: 1px solid #000000;
}

QTreeView::item {
    outline: none;
}

QPushButton {
    background-color: #3c3f41;
    border: 1px solid #4a4e52;
    padding: 5px;
    color: #ffffff;
}

QPushButton:hover {
    background-color: #4a4e52;
}

QSlider::groove:horizontal {
    height: 4px;
    background-color: #3c3f41; /* Matches the darker background */
    border: 1px solid #2a2a2a; /* Add a subtle border to define the groove */
    border-radius: 2px; /* Optional: Smooth edges */
}

QSlider::handle:horizontal {
    background-color: #787878; /* Use a medium gray for contrast */
    border: 1px solid #4a4e52; /* Keep the border consistent */
    width: 15px;
    height: 15px; /* Ensure the handle remains a good size */
    border-radius: 7px; /* Make it round for a modern look */
}

/* Base style for QTreeView */
QTreeView {
    background-color: #2b2b2b;
    border: none;
    gridline-color: transparent;
}

QTreeWidget::item, QTreeView::item {
    background-color: #2b2b2b;
    color: #ffffff;
    margin: 0px;
    padding: 2px;
    border: none;
}

QTreeView::item, QTreeView::item:selected, QTreeView::item:hover {
    border: 1px solid #000000;
    border-left: none;
    border-right: none;
}

QTreeView::item:alternate {
    background-color: #323232;
}

QTreeView::item:selected {
    background-color: #3b3b3b;
    color: #ffffff;
    margin: 0px;
    border-left: none;
    border-right: none;
}

QTreeView::item:selected:active {
    background-color: #353535;
    color: #ffffff;
    border-left: none;
    border-right: none;
}

QTreeView::item:selected:!active {
    background-color: #3b3b3b;
    color: #ffffff;
    border-left: none;
    border-right: none;
}

QTreeView::item:hover {
    background-color: #363636;
    color: #ffffff;
    border-left: none;
    border-right: none;
}

QTreeView::item {
    outline: none;
}

/* Branch arrow customization */
QTreeView::branch:closed:has-children {
    border-image: none;
    image: url(:/icons/branch-closed-10.png); /* Set a custom right arrow icon */
}

QTreeView::branch:open:has-children {
    border-image: none;
    image: url(:/icons/branch-open-10.png); /* Set a custom down arrow icon */
}

/* General scrollbar style (vertical and horizontal) */
QScrollBar:vertical, QScrollBar:horizontal {
    background: #2b2b2b;  /* Background of the scrollbar */
    width: 14px;          /* Width of the vertical scrollbar */
    height: 14px;         /* Height of the horizontal scrollbar */
}

/* Page scroll areas: keep them transparent */
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

/* The handle: lighter color with rounded edges */
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #3c3f41;  /* Slightly lighter than the groove #4a4e52 */
    border: 1px solid #3c3f41;
    border-radius: 6px;   /* Rounded edges for the handle */
    min-height: 14px;     /* Minimum size for usability (vertical) */
    min-width: 14px;      /* Minimum size for usability (horizontal) */
}

/* The groove: the track where the handle moves */
QScrollBar::groove:vertical, QScrollBar::groove:horizontal {
    background: #1e1e1e;  /* Darker background for the groove (track) */
    border: 1px solid #3c3f41;
    border-radius: 6px;   /* Rounded edges to match the handle */
    margin: 14px 0;       /* Space for the arrow buttons (vertical) */
}

/* Arrow buttons at the ends of the scrollbar */
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    background: #2b2b2b;
    border: 1px solid #3c3f41;
    height: 14px;          /* Match the width of the scrollbar */
    subcontrol-origin: margin;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    background: #2b2b2b;
    border: 1px solid #3c3f41;
    width: 14px;          /* Match the height of the scrollbar */
    subcontrol-origin: margin;
}

"""
