"""Read and write medical images using canonical MedicalImage objects.

Purpose:
This module converts NIfTI files and SimpleITK images into the shared
``MedicalImage`` representation and writes canonical images back out to object
or file forms. Three-dimensional images are standardized to canonical LPS+
orientation during import.

Variables:
- None.

Functions:
- nifti_to_medical_image: Convert a nibabel NIfTI object to MedicalImage.
- medical_image_to_nifti: Convert MedicalImage to a nibabel NIfTI object.
- read_nifti: Load a NIfTI file from disk into MedicalImage.
- write_nifti_to_object: Export MedicalImage to a nibabel NIfTI object.
- write_nifti_to_path: Save MedicalImage to a NIfTI file path.
- sitk_to_medical_image: Convert a SimpleITK image to MedicalImage.
- medical_image_to_sitk: Convert MedicalImage to a SimpleITK image object.
- read_sitk: Load or wrap a SimpleITK image into MedicalImage.
- write_sitk_to_object: Export MedicalImage to a SimpleITK image object.
- write_sitk_to_path: Save MedicalImage to a file using SimpleITK.

Classes:
- None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .image_base import MedicalImage, infer_axis_labels
from .image_orientLPS import (
    affine_lps_to_ras,
    affine_lps_to_sitk_metadata,
    affine_ras_to_lps,
    reorient_spatial_array_to_lps,
    sitk_metadata_to_affine_lps,
)


def _infer_spatial_ndim_from_array(array: np.ndarray) -> int:
    """Infer the number of spatial dimensions from an image array.

    Parameters
    ----------
    array : np.ndarray
        Input image data whose leading axes are spatial.

    Returns
    -------
    int
        ``2`` for two-dimensional arrays and ``3`` for arrays with three or
        more dimensions.
    """

    if array.ndim < 2:
        raise ValueError(f"Expected at least 2 array dimensions, got {array.ndim}.")
    return 2 if array.ndim == 2 else 3


def _transpose_sitk_array_to_spatial_first(array: np.ndarray) -> np.ndarray:
    """Convert a SimpleITK NumPy array into spatial-first axis order.

    Parameters
    ----------
    array : np.ndarray
        NumPy array returned by ``SimpleITK.GetArrayFromImage``. Its axes are
        in reversed image-index order, such as ``(z, y, x)`` or ``(t, z, y, x)``.

    Returns
    -------
    np.ndarray
        Spatial-first array, such as ``(x, y, z)`` or ``(x, y, z, t)``.
    """

    return np.transpose(array, axes=tuple(reversed(range(array.ndim))))


def _transpose_spatial_first_to_sitk(array: np.ndarray) -> np.ndarray:
    """Convert a spatial-first array into SimpleITK NumPy axis order.

    Parameters
    ----------
    array : np.ndarray
        Spatial-first array in canonical layout, such as ``(x, y, z)`` or
        ``(x, y, z, t)``.

    Returns
    -------
    np.ndarray
        Array reordered for ``SimpleITK.GetImageFromArray``, such as
        ``(z, y, x)`` or ``(t, z, y, x)``.
    """

    return np.transpose(array, axes=tuple(reversed(range(array.ndim))))


def nifti_to_medical_image(nii: nib.spatialimages.SpatialImage) -> MedicalImage:
    """Convert a nibabel NIfTI image object into canonical MedicalImage.

    Parameters
    ----------
    nii : nib.spatialimages.SpatialImage
        NIfTI-like image whose affine is expressed in the NIfTI RAS world
        convention.

    Returns
    -------
    MedicalImage
        Canonical image object whose 3D spatial axes are reoriented to LPS+
        and whose affine is stored in LPS world coordinates.
    """

    array_native = np.asarray(nii.dataobj)
    spatial_ndim = _infer_spatial_ndim_from_array(array_native)
    affine_ras = np.asarray(nii.affine, dtype=np.float64)
    affine_lps_native = affine_ras_to_lps(affine_ras)

    array_lps, affine_lps, transform_info = reorient_spatial_array_to_lps(
        array=array_native,
        affine_lps=affine_lps_native,
        spatial_ndim=spatial_ndim,
    )

    metadata: dict[str, Any] = {
        "nifti_header": nii.header.copy(),
        "nifti_affine_ras_original": affine_ras.copy(),
        "nifti_shape_original": tuple(int(v) for v in array_native.shape),
        "nifti_to_lps": transform_info,
    }

    return MedicalImage(
        array=array_lps,
        affine_lps=affine_lps,
        spatial_ndim=spatial_ndim,
        axis_labels=infer_axis_labels(spatial_ndim, array_lps.ndim),
        source_type="nifti",
        metadata=metadata,
    )


def medical_image_to_nifti(image: MedicalImage) -> nib.Nifti1Image:
    """Convert a canonical MedicalImage into a nibabel NIfTI image object.

    Parameters
    ----------
    image : MedicalImage
        Canonical image whose affine is expressed in LPS world coordinates.

    Returns
    -------
    nib.Nifti1Image
        NIfTI object whose affine is expressed in RAS world coordinates and
        whose array remains spatial-first.
    """

    affine_ras = affine_lps_to_ras(image.affine_lps)
    header = None
    if "nifti_header" in image.metadata:
        header = image.metadata["nifti_header"].copy()
        header.set_data_shape(image.array.shape)

        spatial_zooms = tuple(float(v) for v in image.spacing.tolist())
        trailing_dim = image.array.ndim - image.spatial_ndim
        trailing_zooms = tuple(1.0 for _ in range(trailing_dim))
        if "nifti_header" in image.metadata and image.array.ndim > image.spatial_ndim:
            original_zooms = tuple(float(v) for v in image.metadata["nifti_header"].get_zooms())
            if len(original_zooms) > image.spatial_ndim:
                trailing_zooms = original_zooms[image.spatial_ndim : image.array.ndim]
        header.set_zooms(spatial_zooms + trailing_zooms)
        header.set_data_dtype(image.array.dtype)

    nii = nib.Nifti1Image(np.asarray(image.array), affine_ras, header=header)
    nii.set_qform(affine_ras, code=1)
    nii.set_sform(affine_ras, code=1)
    return nii


def read_nifti(path: str | Path) -> MedicalImage:
    """Load a NIfTI file from disk into a canonical MedicalImage object.

    Parameters
    ----------
    path : str | Path
        Filesystem path to a NIfTI image file such as ``.nii`` or ``.nii.gz``.

    Returns
    -------
    MedicalImage
        Canonical image loaded from the requested file path.
    """

    path = Path(path)
    image = nifti_to_medical_image(nib.load(str(path)))
    image.metadata["source_path"] = str(path)
    return image


def write_nifti_to_object(image: MedicalImage) -> nib.Nifti1Image:
    """Export a canonical image to an in-memory nibabel NIfTI object.

    Parameters
    ----------
    image : MedicalImage
        Canonical image to export.

    Returns
    -------
    nib.Nifti1Image
        In-memory NIfTI object representing ``image``.
    """

    return medical_image_to_nifti(image)


def write_nifti_to_path(image: MedicalImage, path: str | Path) -> None:
    """Save a canonical image to a NIfTI file path.

    Parameters
    ----------
    image : MedicalImage
        Canonical image to export.
    path : str | Path
        Destination path for the NIfTI file.

    Returns
    -------
    None
        The image is written to disk.
    """

    nib.save(write_nifti_to_object(image), str(Path(path)))


def _extract_sitk_spatial_geometry(image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract spatial geometry and trailing-axis metadata from a SimpleITK image.

    Parameters
    ----------
    image : sitk.Image
        Input SimpleITK image, potentially 2D, 3D, or 4D scalar.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
        Tuple containing origin, spacing, direction matrix, and trailing-axis
        metadata for later round-trip export.
    """

    dimension = int(image.GetDimension())
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(dimension, dimension)

    extra_metadata: dict[str, Any] = {}
    if dimension == 4:
        if not np.allclose(direction[:3, 3], 0.0) or not np.allclose(direction[3, :3], 0.0):
            raise NotImplementedError(
                "4D SimpleITK images with spatial-time direction coupling are not supported."
            )
        extra_metadata = {
            "extra_axis_origin": [float(origin[3])],
            "extra_axis_spacing": [float(spacing[3])],
            "sitk_dimension": 4,
        }

    spatial_ndim = 2 if dimension == 2 else 3
    return (
        origin[:spatial_ndim].copy(),
        spacing[:spatial_ndim].copy(),
        direction[:spatial_ndim, :spatial_ndim].copy(),
        extra_metadata,
    )


def sitk_to_medical_image(image: sitk.Image) -> MedicalImage:
    """Convert a SimpleITK image object into canonical MedicalImage.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image whose physical coordinates already follow LPS world
        conventions.

    Returns
    -------
    MedicalImage
        Canonical image object whose 3D spatial axes are reoriented to LPS+
        and whose affine is stored in LPS world coordinates.
    """

    array_sitk = sitk.GetArrayFromImage(image)
    array_native = _transpose_sitk_array_to_spatial_first(array_sitk)

    spatial_ndim = 2 if image.GetDimension() == 2 else 3
    origin, spacing, direction, extra_metadata = _extract_sitk_spatial_geometry(image)
    affine_lps_native = sitk_metadata_to_affine_lps(
        origin=origin,
        spacing=spacing,
        direction=direction.reshape(-1),
        spatial_ndim=spatial_ndim,
    )

    array_lps, affine_lps, transform_info = reorient_spatial_array_to_lps(
        array=array_native,
        affine_lps=affine_lps_native,
        spatial_ndim=spatial_ndim,
    )

    metadata: dict[str, Any] = {
        "sitk_origin_original": list(image.GetOrigin()),
        "sitk_spacing_original": list(image.GetSpacing()),
        "sitk_direction_original": list(image.GetDirection()),
        "sitk_dimension_original": int(image.GetDimension()),
        "sitk_to_lps": transform_info,
    }
    metadata.update(extra_metadata)

    return MedicalImage(
        array=array_lps,
        affine_lps=affine_lps,
        spatial_ndim=spatial_ndim,
        axis_labels=infer_axis_labels(spatial_ndim, array_lps.ndim),
        source_type="sitk",
        metadata=metadata,
    )


def medical_image_to_sitk(image: MedicalImage) -> sitk.Image:
    """Convert a canonical MedicalImage into an in-memory SimpleITK image.

    Parameters
    ----------
    image : MedicalImage
        Canonical image to export. For 4D arrays this function assumes the
        trailing axis is time and creates a 4D scalar SimpleITK image.

    Returns
    -------
    sitk.Image
        In-memory SimpleITK image object with LPS physical metadata.
    """

    sitk_array = _transpose_spatial_first_to_sitk(np.asarray(image.array))
    is_vector = False if image.array.ndim > image.spatial_ndim else False
    sitk_image = sitk.GetImageFromArray(sitk_array, isVector=is_vector)

    extra_spacing = list(image.metadata.get("extra_axis_spacing", []))
    extra_origin = list(image.metadata.get("extra_axis_origin", []))
    if image.array.ndim > image.spatial_ndim and len(extra_spacing) == 0:
        extra_spacing = [1.0 for _ in range(image.array.ndim - image.spatial_ndim)]
    if image.array.ndim > image.spatial_ndim and len(extra_origin) == 0:
        extra_origin = [0.0 for _ in range(image.array.ndim - image.spatial_ndim)]

    origin, spacing, direction = affine_lps_to_sitk_metadata(
        image.affine_lps,
        image.spatial_ndim,
        extra_spacing=extra_spacing if image.array.ndim > image.spatial_ndim else None,
        extra_origin=extra_origin if image.array.ndim > image.spatial_ndim else None,
    )
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetDirection(direction)
    return sitk_image


def read_sitk(image_or_path: sitk.Image | str | Path) -> MedicalImage:
    """Load or wrap a SimpleITK image as a canonical MedicalImage object.

    Parameters
    ----------
    image_or_path : sitk.Image | str | Path
        Existing SimpleITK image object or filesystem path readable by
        ``SimpleITK.ReadImage``.

    Returns
    -------
    MedicalImage
        Canonical image converted from the provided SimpleITK input.
    """

    if isinstance(image_or_path, sitk.Image):
        image = image_or_path
        source_path: str | None = None
    else:
        source_path = str(Path(image_or_path))
        image = sitk.ReadImage(source_path)

    output = sitk_to_medical_image(image)
    if source_path is not None:
        output.metadata["source_path"] = source_path
    return output


def write_sitk_to_object(image: MedicalImage) -> sitk.Image:
    """Export a canonical image to an in-memory SimpleITK image object.

    Parameters
    ----------
    image : MedicalImage
        Canonical image to export.

    Returns
    -------
    sitk.Image
        In-memory SimpleITK image object representing ``image``.
    """

    return medical_image_to_sitk(image)


def write_sitk_to_path(image: MedicalImage, path: str | Path) -> None:
    """Save a canonical image to disk using SimpleITK.

    Parameters
    ----------
    image : MedicalImage
        Canonical image to export.
    path : str | Path
        Destination path written by ``SimpleITK.WriteImage``.

    Returns
    -------
    None
        The image is written to disk.
    """

    sitk.WriteImage(write_sitk_to_object(image), str(Path(path)))
