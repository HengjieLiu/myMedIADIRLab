"""Public API for canonical medical image IO, orientation, and display.

Purpose:
This package provides a small medical-image utility layer centered on a
canonical ``MedicalImage`` object. It standardizes 3D images to LPS+
orientation, supports NIfTI and SimpleITK conversions, and offers basic 2D and
orthogonal-view display helpers.

Variables:
- __all__: Public names re-exported by the package.

Functions:
- medical_image_to_ras_array
- medical_image_from_ras_array_like
- read_nifti_plain
- read_nifti
- write_nifti_plain
- write_nifti_to_object
- write_nifti_to_path
- read_sitk
- write_sitk_to_object
- write_sitk_to_path
- to_lps
- crop_image_world
- extract_slice
- show_image_2d
- show_slice
- show_three_views

Classes:
- MedicalImage
"""

from .image_base import MedicalImage
from .image_display import extract_slice, show_image_2d, show_slice, show_three_views
from .image_io import (
    medical_image_from_ras_array_like,
    medical_image_to_ras_array,
    read_nifti,
    read_nifti_plain,
    read_sitk,
    write_nifti_plain,
    write_nifti_to_object,
    write_nifti_to_path,
    write_sitk_to_object,
    write_sitk_to_path,
)
from .image_orientLPS import crop_image_world, to_lps

__all__ = [
    "MedicalImage",
    "crop_image_world",
    "extract_slice",
    "medical_image_from_ras_array_like",
    "medical_image_to_ras_array",
    "read_nifti",
    "read_nifti_plain",
    "read_sitk",
    "show_image_2d",
    "show_slice",
    "show_three_views",
    "to_lps",
    "write_nifti_plain",
    "write_nifti_to_object",
    "write_nifti_to_path",
    "write_sitk_to_object",
    "write_sitk_to_path",
]
