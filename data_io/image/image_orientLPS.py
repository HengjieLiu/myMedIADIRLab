"""Orientation and affine utilities for canonical LPS+ image handling.

Purpose:
This module centralizes image geometry logic. It converts between NIfTI RAS
and canonical LPS, reorients 3D arrays into LPS+, derives SimpleITK geometry
from affine matrices, and applies anatomical-direction cropping.

Variables:
- LPS_AXCODES: Target canonical axis codes for 3D spatial orientation.
- RAS_TO_LPS_4X4: Homogeneous transform from RAS world coordinates to LPS.
- LPS_TO_RAS_4X4: Homogeneous transform from LPS world coordinates to RAS.

Functions:
- affine_ras_to_lps: Convert a NIfTI-style RAS affine into LPS coordinates.
- affine_lps_to_ras: Convert a canonical LPS affine into NIfTI-style RAS.
- get_spacing_from_affine: Compute voxel sizes from affine basis vectors.
- get_axis_codes_from_affine: Report anatomical axis codes for the affine.
- is_lps: Check whether a MedicalImage already uses canonical 3D LPS+.
- reorient_spatial_array_to_lps: Reorder and flip a spatial-first array to LPS+.
- to_lps: Convert a MedicalImage into canonical LPS+ orientation.
- sitk_metadata_to_affine_lps: Build an LPS affine from SimpleITK metadata.
- affine_lps_to_sitk_metadata: Recover SimpleITK metadata from an LPS affine.
- crop_image_world: Crop a 3D image using anatomical L/R/A/P/I/S directions.

Classes:
- None.
"""

from __future__ import annotations

from typing import Any, Sequence

import nibabel as nib
import numpy as np

from .image_base import MedicalImage

LPS_AXCODES: tuple[str, str, str] = ("L", "P", "S")
RAS_TO_LPS_4X4: np.ndarray = np.diag([-1.0, -1.0, 1.0, 1.0])
LPS_TO_RAS_4X4: np.ndarray = np.diag([-1.0, -1.0, 1.0, 1.0])
_NIB_LPS_LABELS: tuple[tuple[str, str], ...] = (("R", "L"), ("A", "P"), ("I", "S"))


def affine_ras_to_lps(affine_ras: np.ndarray) -> np.ndarray:
    """Convert a NIfTI RAS affine matrix into canonical LPS world coordinates.

    Parameters
    ----------
    affine_ras : np.ndarray
        Homogeneous 4x4 voxel-to-world affine whose world axes follow the
        NIfTI RAS convention.

    Returns
    -------
    np.ndarray
        Homogeneous 4x4 affine expressed in LPS world coordinates.
    """

    affine_ras = np.asarray(affine_ras, dtype=np.float64)
    if affine_ras.shape != (4, 4):
        raise ValueError(f"affine_ras must have shape (4, 4), got {affine_ras.shape}.")
    return RAS_TO_LPS_4X4 @ affine_ras


def affine_lps_to_ras(affine_lps: np.ndarray) -> np.ndarray:
    """Convert a canonical LPS affine matrix into NIfTI RAS world coordinates.

    Parameters
    ----------
    affine_lps : np.ndarray
        Homogeneous 4x4 voxel-to-world affine in LPS world coordinates.

    Returns
    -------
    np.ndarray
        Homogeneous 4x4 affine expressed in RAS world coordinates.
    """

    affine_lps = np.asarray(affine_lps, dtype=np.float64)
    if affine_lps.shape != (4, 4):
        raise ValueError(f"affine_lps must have shape (4, 4), got {affine_lps.shape}.")
    return LPS_TO_RAS_4X4 @ affine_lps


def get_spacing_from_affine(affine_lps: np.ndarray, spatial_ndim: int) -> np.ndarray:
    """Compute voxel spacing from the spatial affine basis vectors.

    Parameters
    ----------
    affine_lps : np.ndarray
        Homogeneous 4x4 affine in LPS world coordinates.
    spatial_ndim : int
        Number of spatial axes to inspect. Supported values are 2 and 3.

    Returns
    -------
    np.ndarray
        One-dimensional array containing the Euclidean norm of each spatial
        affine basis vector.
    """

    affine_lps = np.asarray(affine_lps, dtype=np.float64)
    if affine_lps.shape != (4, 4):
        raise ValueError(f"affine_lps must have shape (4, 4), got {affine_lps.shape}.")
    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}.")

    basis = affine_lps[:3, :spatial_ndim]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 0.0):
        raise ValueError(f"Affine contains non-positive voxel spacing: {spacing}.")
    return spacing


def get_axis_codes_from_affine(affine_lps: np.ndarray, spatial_ndim: int) -> tuple[str, ...]:
    """Return anatomical axis codes for the leading spatial affine columns.

    Parameters
    ----------
    affine_lps : np.ndarray
        Homogeneous 4x4 affine in LPS world coordinates.
    spatial_ndim : int
        Number of spatial axes represented by the image. Supported values are
        2 and 3.

    Returns
    -------
    tuple[str, ...]
        Anatomical axis codes for the first ``spatial_ndim`` voxel axes. For a
        canonical 3D image this is ``("L", "P", "S")``.
    """

    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}.")

    full_codes = nib.orientations.aff2axcodes(
        np.asarray(affine_lps, dtype=np.float64),
        labels=_NIB_LPS_LABELS,
    )
    return tuple(str(code) for code in full_codes[:spatial_ndim])


def is_lps(image: MedicalImage) -> bool:
    """Check whether a canonical image already has LPS+ spatial orientation.

    Parameters
    ----------
    image : MedicalImage
        Canonical image object to inspect.

    Returns
    -------
    bool
        ``True`` when the image spatial axes already match the canonical LPS+
        orientation. For 2D images this checks the leading two axes against
        ``("L", "P")`` when meaningful.
    """

    target_codes = LPS_AXCODES[: image.spatial_ndim]
    return get_axis_codes_from_affine(image.affine_lps, image.spatial_ndim) == target_codes


def reorient_spatial_array_to_lps(
    array: np.ndarray,
    affine_lps: np.ndarray,
    spatial_ndim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reorder and flip a spatial-first array into canonical 3D LPS+.

    Parameters
    ----------
    array : np.ndarray
        Spatial-first voxel data. Extra trailing axes such as time are allowed
        and are preserved without reordering.
    affine_lps : np.ndarray
        Homogeneous 4x4 affine in LPS world coordinates aligned with ``array``.
    spatial_ndim : int
        Number of spatial axes in ``array``. Only ``3`` triggers full
        reorientation; ``2`` returns the input unchanged because 2D toy images
        are not forced into an anatomical canonical orientation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, Any]]
        A tuple containing the reoriented array, the updated 4x4 LPS affine,
        and a metadata dictionary describing the applied orientation transform.
    """

    array = np.asarray(array)
    affine_lps = np.asarray(affine_lps, dtype=np.float64)

    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}.")
    if array.ndim < spatial_ndim:
        raise ValueError(
            f"array.ndim={array.ndim} must be at least spatial_ndim={spatial_ndim}."
        )
    if affine_lps.shape != (4, 4):
        raise ValueError(f"affine_lps must have shape (4, 4), got {affine_lps.shape}.")

    if spatial_ndim == 2:
        metadata: dict[str, Any] = {
            "applied": False,
            "reason": "2d_images_are_not_reoriented_to_anatomical_lps",
            "source_axcodes_lps": list(get_axis_codes_from_affine(affine_lps, 2)),
            "target_axcodes_lps": ["L", "P"],
        }
        return array.copy(), affine_lps.copy(), metadata

    current_ornt = nib.orientations.io_orientation(affine_lps)
    target_ornt = nib.orientations.axcodes2ornt(LPS_AXCODES, labels=_NIB_LPS_LABELS)
    transform_ornt = nib.orientations.ornt_transform(current_ornt, target_ornt)

    reoriented_array = nib.orientations.apply_orientation(array, transform_ornt)
    reoriented_affine = affine_lps @ nib.orientations.inv_ornt_aff(
        transform_ornt,
        array.shape[:3],
    )

    metadata = {
        "applied": True,
        "source_axcodes_lps": list(
            nib.orientations.ornt2axcodes(current_ornt, labels=_NIB_LPS_LABELS)
        ),
        "target_axcodes_lps": list(LPS_AXCODES),
        "transform_ornt": transform_ornt.tolist(),
    }
    return reoriented_array, reoriented_affine, metadata


def to_lps(image: MedicalImage) -> MedicalImage:
    """Convert a canonical image object into canonical LPS+ spatial orientation.

    Parameters
    ----------
    image : MedicalImage
        Image object whose array and affine should be expressed in canonical
        LPS+ orientation.

    Returns
    -------
    MedicalImage
        A new image object with spatial axes reoriented to canonical LPS+ when
        ``image.spatial_ndim == 3``. Two-dimensional images are returned with
        their affine preserved and metadata documenting that no reorientation
        was applied.
    """

    array_lps, affine_out, transform_info = reorient_spatial_array_to_lps(
        array=image.array,
        affine_lps=image.affine_lps,
        spatial_ndim=image.spatial_ndim,
    )

    metadata = dict(image.metadata)
    metadata["to_lps"] = transform_info
    return MedicalImage(
        array=array_lps,
        affine_lps=affine_out,
        spatial_ndim=image.spatial_ndim,
        axis_labels=image.axis_labels,
        source_type=image.source_type,
        metadata=metadata,
    )


def sitk_metadata_to_affine_lps(
    origin: Sequence[float],
    spacing: Sequence[float],
    direction: Sequence[float],
    spatial_ndim: int,
) -> np.ndarray:
    """Build a 4x4 LPS affine from SimpleITK geometry metadata.

    Parameters
    ----------
    origin : Sequence[float]
        SimpleITK image origin in LPS world coordinates. The first
        ``spatial_ndim`` values define the spatial translation.
    spacing : Sequence[float]
        SimpleITK voxel spacing. The first ``spatial_ndim`` values define the
        spatial voxel sizes.
    direction : Sequence[float]
        Flattened SimpleITK direction cosine matrix in row-major order. Only
        the leading spatial block is used.
    spatial_ndim : int
        Number of spatial axes represented by the image. Supported values are 2
        and 3.

    Returns
    -------
    np.ndarray
        Homogeneous 4x4 voxel-to-world affine in canonical LPS coordinates.
    """

    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}.")

    origin_arr = np.asarray(origin, dtype=np.float64)
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    direction_arr = np.asarray(direction, dtype=np.float64)

    if origin_arr.size < spatial_ndim:
        raise ValueError(
            f"origin must provide at least {spatial_ndim} values, got {origin_arr.size}."
        )
    if spacing_arr.size < spatial_ndim:
        raise ValueError(
            f"spacing must provide at least {spatial_ndim} values, got {spacing_arr.size}."
        )
    direction_matrix = direction_arr.reshape(int(np.sqrt(direction_arr.size)), -1)
    if direction_matrix.shape[0] < spatial_ndim or direction_matrix.shape[1] < spatial_ndim:
        raise ValueError(
            "direction does not contain a large enough spatial direction matrix."
        )

    affine = np.eye(4, dtype=np.float64)
    spatial_direction = direction_matrix[:spatial_ndim, :spatial_ndim]
    affine[:spatial_ndim, :spatial_ndim] = spatial_direction @ np.diag(
        spacing_arr[:spatial_ndim]
    )
    affine[:spatial_ndim, 3] = origin_arr[:spatial_ndim]
    return affine


def affine_lps_to_sitk_metadata(
    affine_lps: np.ndarray,
    spatial_ndim: int,
    *,
    extra_spacing: Sequence[float] | None = None,
    extra_origin: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Convert a canonical LPS affine into SimpleITK origin/spacing/direction.

    Parameters
    ----------
    affine_lps : np.ndarray
        Homogeneous 4x4 voxel-to-world affine in canonical LPS coordinates.
    spatial_ndim : int
        Number of spatial axes to convert. Supported values are 2 and 3.
    extra_spacing : Sequence[float] | None, optional
        Optional trailing spacing values used when constructing metadata for a
        higher-dimensional SimpleITK image such as 3D+t.
    extra_origin : Sequence[float] | None, optional
        Optional trailing origin values used when constructing metadata for a
        higher-dimensional SimpleITK image such as 3D+t.

    Returns
    -------
    tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
        The ``(origin, spacing, direction)`` triplet ready for
        ``SimpleITK.Image.SetOrigin``, ``SetSpacing``, and ``SetDirection``.
    """

    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}.")

    affine_lps = np.asarray(affine_lps, dtype=np.float64)
    spacing = get_spacing_from_affine(affine_lps, spatial_ndim)
    direction = affine_lps[:3, :spatial_ndim] / spacing

    origin_values = list(affine_lps[:spatial_ndim, 3].tolist())
    spacing_values = list(spacing.tolist())

    full_direction = np.eye(spatial_ndim + (0 if extra_spacing is None else len(extra_spacing)))
    full_direction[:3, :spatial_ndim] = 0.0
    full_direction[:spatial_ndim, :spatial_ndim] = direction[:spatial_ndim, :spatial_ndim]

    if extra_spacing is not None:
        extra_spacing_list = [float(v) for v in extra_spacing]
        spacing_values.extend(extra_spacing_list)
    if extra_origin is not None:
        extra_origin_list = [float(v) for v in extra_origin]
        origin_values.extend(extra_origin_list)

    if extra_spacing is not None:
        full_dim = spatial_ndim + len(extra_spacing)
        full_direction = np.eye(full_dim, dtype=np.float64)
        full_direction[:spatial_ndim, :spatial_ndim] = direction[:spatial_ndim, :spatial_ndim]

    return (
        tuple(float(v) for v in origin_values),
        tuple(float(v) for v in spacing_values),
        tuple(float(v) for v in full_direction.reshape(-1).tolist()),
    )


def crop_image_world(
    image: MedicalImage,
    *,
    crop_L: int | None = None,
    crop_R: int | None = None,
    crop_A: int | None = None,
    crop_P: int | None = None,
    crop_I: int | None = None,
    crop_S: int | None = None,
) -> MedicalImage:
    """Crop a canonical 3D image using anatomical L/R/A/P/I/S directions.

    Parameters
    ----------
    image : MedicalImage
        Canonical 3D image in LPS+ orientation. Extra trailing axes such as
        time are preserved during cropping.
    crop_L, crop_R, crop_A, crop_P, crop_I, crop_S : int | None, optional
        Non-negative voxel counts removed from the named anatomical sides. In
        canonical LPS+ layout this means: ``R`` and ``A`` and ``I`` crop from
        the low index side of axes ``i``, ``j``, and ``k`` respectively, while
        ``L``, ``P``, and ``S`` crop from the high index side.

    Returns
    -------
    MedicalImage
        A new 3D image with cropped voxel data, updated affine translation, and
        copied metadata augmented with crop details.
    """

    if image.spatial_ndim != 3:
        raise ValueError("crop_image_world only supports spatial_ndim == 3.")
    if not is_lps(image):
        raise ValueError("crop_image_world expects the input image to already be in LPS+.")

    crop_values = {
        "L": 0 if crop_L is None else int(crop_L),
        "R": 0 if crop_R is None else int(crop_R),
        "A": 0 if crop_A is None else int(crop_A),
        "P": 0 if crop_P is None else int(crop_P),
        "I": 0 if crop_I is None else int(crop_I),
        "S": 0 if crop_S is None else int(crop_S),
    }
    for name, value in crop_values.items():
        if value < 0:
            raise ValueError(f"crop_{name} must be non-negative, got {value}.")

    size_i, size_j, size_k = image.spatial_shape
    start_i = crop_values["R"]
    stop_i = size_i - crop_values["L"]
    start_j = crop_values["A"]
    stop_j = size_j - crop_values["P"]
    start_k = crop_values["I"]
    stop_k = size_k - crop_values["S"]

    if start_i >= stop_i or start_j >= stop_j or start_k >= stop_k:
        raise ValueError(
            "Requested crop removes an entire axis or produces an empty image."
        )

    slices: list[slice] = [
        slice(start_i, stop_i),
        slice(start_j, stop_j),
        slice(start_k, stop_k),
    ]
    slices.extend(slice(None) for _ in image.extra_shape)
    cropped_array = image.array[tuple(slices)]

    voxel_shift = np.eye(4, dtype=np.float64)
    voxel_shift[:3, 3] = np.array([start_i, start_j, start_k], dtype=np.float64)
    cropped_affine = image.affine_lps @ voxel_shift

    metadata = dict(image.metadata)
    metadata["crop_image_world"] = {
        "crop_values": crop_values,
        "shape_before": list(image.spatial_shape),
        "shape_after": list(cropped_array.shape[:3]),
    }

    return MedicalImage(
        array=cropped_array,
        affine_lps=cropped_affine,
        spatial_ndim=image.spatial_ndim,
        axis_labels=image.axis_labels,
        source_type=image.source_type,
        metadata=metadata,
    )
