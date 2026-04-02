"""Read and write displacement fields in plain and canonical forms.

Purpose:
This module provides two warp-IO layers:
    1. plain NIfTI IO that reads and writes raw arrays plus affines without any
       orientation changes, and
    2. canonical NIfTI IO that converts 3D voxel-unit displacement fields into
       canonical LPS+ orientation with channel-last storage.

    The canonical reorientation assumes the displacement vectors are expressed
    in voxel units along image-array axes. Under that assumption the warp
    lattice is reoriented like the image lattice, and the vector components are
    transformed by the same signed axis permutation.

Variable / function / class list:
    Variables:
        None

    Functions:
        read_warp_nifti_plain
        write_warp_nifti_plain
        read_warp_nifti
        write_warp_nifti

    Classes:
        None
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Tuple

import nibabel as nib
import numpy as np

from ..image.image_orientLPS import (
    LPS_AXCODES,
    _NIB_LPS_LABELS,
    affine_lps_to_ras,
    affine_ras_to_lps,
)
from .warp_base import MedicalWarp

VectorLayout = Literal["channel_first", "channel_last"]


def _infer_spatial_ndim_from_warp_array(array: np.ndarray) -> int:
    """Infer the spatial dimensionality of a warp array from its shape."""
    if array.ndim not in (3, 4):
        raise ValueError(
            f"Warp array must have ndim 3 or 4, got shape {array.shape}."
        )

    candidate_dims: list[int] = []
    if array.shape[0] == array.ndim - 1:
        candidate_dims.append(array.ndim - 1)
    if array.shape[-1] == array.ndim - 1:
        candidate_dims.append(array.ndim - 1)

    if not candidate_dims:
        raise ValueError(
            "Unable to infer warp dimensionality from array shape "
            f"{array.shape}. Expected channel-first or channel-last vector axis."
        )

    return candidate_dims[0]


def _infer_input_vector_layout(array: np.ndarray) -> VectorLayout:
    """Infer whether a raw warp array is channel-first or channel-last."""
    spatial_ndim = _infer_spatial_ndim_from_warp_array(array)
    matches_first = array.shape[0] == spatial_ndim
    matches_last = array.shape[-1] == spatial_ndim

    if matches_first and matches_last:
        raise ValueError(
            "Ambiguous warp layout: both the first and last axis could be the "
            f"vector axis for shape {array.shape}. Please provide a non-ambiguous file."
        )
    if matches_first:
        return "channel_first"
    if matches_last:
        return "channel_last"
    raise ValueError(f"Unable to infer vector layout from shape {array.shape}.")


def _convert_warp_vector_layout(
    array: np.ndarray,
    src_layout: VectorLayout,
    dst_layout: VectorLayout,
) -> np.ndarray:
    """Convert a warp array between channel-first and channel-last storage."""
    if src_layout == dst_layout:
        return np.asarray(array)
    if src_layout == "channel_first" and dst_layout == "channel_last":
        axes = tuple(range(1, array.ndim)) + (0,)
        return np.transpose(array, axes=axes)
    if src_layout == "channel_last" and dst_layout == "channel_first":
        axes = (array.ndim - 1,) + tuple(range(array.ndim - 1))
        return np.transpose(array, axes=axes)
    raise ValueError(f"Unsupported vector-layout conversion: {src_layout} -> {dst_layout}.")


def _reorient_voxel_warp_to_lps(
    array_channel_last: np.ndarray,
    affine_lps: np.ndarray,
    spatial_ndim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reorient a voxel-unit channel-last warp into canonical LPS+ orientation."""
    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}.")

    if spatial_ndim == 2:
        metadata: dict[str, Any] = {
            "applied": False,
            "reason": "2d_warps_are_not_reoriented_to_anatomical_lps",
        }
        return array_channel_last.copy(), affine_lps.copy(), metadata

    current_ornt = nib.orientations.io_orientation(affine_lps)
    target_ornt = nib.orientations.axcodes2ornt(LPS_AXCODES, labels=_NIB_LPS_LABELS)
    transform_ornt = nib.orientations.ornt_transform(current_ornt, target_ornt)

    reoriented_array = nib.orientations.apply_orientation(array_channel_last, transform_ornt)
    reoriented_affine = affine_lps @ nib.orientations.inv_ornt_aff(
        transform_ornt,
        array_channel_last.shape[:3],
    )

    vector_out = np.empty_like(reoriented_array)
    for new_axis in range(spatial_ndim):
        old_axis = int(transform_ornt[new_axis, 0])
        flip_sign = float(transform_ornt[new_axis, 1])
        vector_out[..., new_axis] = reoriented_array[..., old_axis] * flip_sign

    metadata = {
        "applied": True,
        "source_axcodes_lps": list(
            nib.orientations.ornt2axcodes(current_ornt, labels=_NIB_LPS_LABELS)
        ),
        "target_axcodes_lps": list(LPS_AXCODES),
        "transform_ornt": transform_ornt.tolist(),
    }
    return vector_out, reoriented_affine, metadata


def read_warp_nifti_plain(
    path: str | Path,
    vector_layout: VectorLayout = "channel_last",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a warp NIfTI file as a raw array plus raw affine with no reorientation.

    Parameters
    ----------
    path : str | Path
        Filesystem path to a displacement-field NIfTI file such as `.nii` or
        `.nii.gz`.
    vector_layout : Literal["channel_first", "channel_last"], optional
        Desired vector-axis layout for the returned array. The stored layout is
        inferred from the on-disk shape and then converted if needed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - the displacement array in the requested vector layout, and
        - the raw NIfTI affine in its on-disk RAS convention.
    """
    nii = nib.load(str(Path(path)))
    raw_array = np.asarray(nii.dataobj)
    input_layout = _infer_input_vector_layout(raw_array)
    output_array = _convert_warp_vector_layout(raw_array, input_layout, vector_layout)
    return output_array, np.asarray(nii.affine, dtype=np.float64)


def write_warp_nifti_plain(
    array: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
    vector_layout: VectorLayout = "channel_last",
) -> None:
    """Write a raw warp array and raw affine to a NIfTI file without reorientation.

    Parameters
    ----------
    array : np.ndarray
        Warp array to store. The semantic shape is expected to be either
        channel-last or channel-first, depending on `vector_layout`.
    affine : np.ndarray
        Raw 4x4 NIfTI affine to store exactly as provided.
    path : str | Path
        Destination path for the output NIfTI file.
    vector_layout : Literal["channel_first", "channel_last"], optional
        Layout of the provided `array`. The file is written using that same
        layout.

    Returns
    -------
    None
        The warp file is written to disk.
    """
    array = np.asarray(array)
    affine = np.asarray(affine, dtype=np.float64)
    if vector_layout not in ("channel_first", "channel_last"):
        raise ValueError(
            f"vector_layout must be 'channel_first' or 'channel_last', got {vector_layout!r}."
        )
    _ = _infer_spatial_ndim_from_warp_array(array)
    if affine.shape != (4, 4):
        raise ValueError(f"affine must have shape (4, 4), got {affine.shape}.")
    nib.save(nib.Nifti1Image(array, affine), str(Path(path)))


def read_warp_nifti(
    path: str | Path,
    *,
    affine_override: np.ndarray | None = None,
    units: Literal["voxel", "world_mm"] = "voxel",
    vector_layout: VectorLayout = "channel_last",
) -> MedicalWarp:
    """Load a warp NIfTI file into a canonical `MedicalWarp` object.

    Parameters
    ----------
    path : str | Path
        Filesystem path to a displacement-field NIfTI file such as `.nii` or
        `.nii.gz`.
    affine_override : np.ndarray | None, optional
        Optional 4x4 affine in NIfTI/RAS convention to use instead of the file
        affine when canonicalizing the warp. This is useful when the warp file
        should be interpreted on a specific image lattice.
    units : Literal["voxel", "world_mm"], optional
        Units of the displacement vectors. Canonical reorientation currently
        supports only `units="voxel"`.
    vector_layout : Literal["channel_first", "channel_last"], optional
        Requested output layout before canonicalization. The canonical
        `MedicalWarp` returned by this function always stores the array in
        channel-last format, so this parameter mainly controls the initial raw
        loading path and validation.

    Returns
    -------
    MedicalWarp
        Canonical warp object whose 3D spatial axes are reoriented to LPS+ and
        whose affine is stored in LPS world coordinates.
    """
    raw_array, file_affine_ras = read_warp_nifti_plain(path=path, vector_layout=vector_layout)
    input_layout = vector_layout
    array_channel_last = _convert_warp_vector_layout(raw_array, input_layout, "channel_last")
    spatial_ndim = _infer_spatial_ndim_from_warp_array(array_channel_last)

    effective_affine_ras = np.asarray(
        file_affine_ras if affine_override is None else affine_override,
        dtype=np.float64,
    )
    if effective_affine_ras.shape != (4, 4):
        raise ValueError(
            f"Effective warp affine must have shape (4, 4), got {effective_affine_ras.shape}."
        )

    affine_lps = affine_ras_to_lps(effective_affine_ras)

    if units == "voxel":
        array_lps, affine_lps_out, transform_info = _reorient_voxel_warp_to_lps(
            array_channel_last=array_channel_last,
            affine_lps=affine_lps,
            spatial_ndim=spatial_ndim,
        )
    else:
        raise NotImplementedError(
            "Canonical warp reorientation currently supports only voxel-unit "
            "displacement fields defined in image-array coordinates. "
            "World-coordinate warp handling is left for future work."
        )

    metadata: dict[str, Any] = {
        "source_path": str(Path(path)),
        "nifti_affine_ras_original": file_affine_ras.copy(),
        "nifti_affine_ras_effective": effective_affine_ras.copy(),
        "nifti_shape_original": tuple(int(v) for v in raw_array.shape),
        "stored_vector_layout": input_layout,
        "to_lps": transform_info,
    }

    return MedicalWarp(
        array=array_lps,
        affine_lps=affine_lps_out,
        spatial_ndim=spatial_ndim,
        units=units,
        source_type="nifti",
        metadata=metadata,
    )


def write_warp_nifti(
    warp: MedicalWarp,
    path: str | Path,
    *,
    vector_layout: VectorLayout = "channel_last",
) -> None:
    """Write a canonical `MedicalWarp` to a NIfTI file.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical warp object to export. The warp array is assumed to already
        be in canonical channel-last LPS+ orientation.
    path : str | Path
        Destination path for the output NIfTI file.
    vector_layout : Literal["channel_first", "channel_last"], optional
        Vector-axis layout to store in the output NIfTI file. The default is
        channel-last.

    Returns
    -------
    None
        The warp is written to disk with its affine expressed in NIfTI/RAS
        world coordinates.
    """
    if vector_layout not in ("channel_first", "channel_last"):
        raise ValueError(
            f"vector_layout must be 'channel_first' or 'channel_last', got {vector_layout!r}."
        )

    array_out = _convert_warp_vector_layout(warp.array, "channel_last", vector_layout)
    affine_ras = affine_lps_to_ras(warp.affine_lps)
    nib.save(nib.Nifti1Image(np.asarray(array_out), affine_ras), str(Path(path)))
