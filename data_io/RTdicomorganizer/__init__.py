"""
==============================================================================
RTdicomorganizer Package
==============================================================================

A comprehensive package for organizing, processing, and analyzing brain 
metastasis DICOM data for stereotactic radiosurgery (SRS) treatment planning 
and longitudinal follow-up analysis.

MODULES:
    - path_utils: Path construction and folder discovery utilities
    - data_io: File I/O operations for CSV, Excel, and log files
    - data_parser: Data parsing and transformation utilities
    - data_analysis: Data analysis and summary generation
    - dicom_utils: DICOM-specific operations and utilities
    - visualization: Image visualization and plotting
    - formatting_utils: Console output formatting utilities
    - workflow_orchestration: High-level workflow coordination

USAGE:
    from code.RTdicomorganizer import path_utils, data_io, data_analysis
    
    # Read and process data
    df = data_io.read_table("input.csv")
    per_lesion, per_pair = data_analysis.build_summaries(df)
    
==============================================================================
"""

__version__ = "1.0.0"
__author__ = "RTdicomorganizer Team"

# Import all modules for convenient access
from . import path_utils
from . import data_io
from . import data_parser
from . import data_analysis
from . import dicom_utils
from . import visualization
from . import formatting_utils
from . import workflow_orchestration

# Define what's exported with "from RTdicomorganizer import *"
__all__ = [
    "path_utils",
    "data_io",
    "data_parser",
    "data_analysis",
    "dicom_utils",
    "visualization",
    "formatting_utils",
    "workflow_orchestration",
]

