"""Read and write displacement fields in plain and canonical forms.

Purpose:
This module provides two warp-IO layers:
    1. plain NIfTI IO that reads and writes raw arrays plus affines without any
       orientation changes, and
    2. canonical NIfTI IO that converts 3D voxel-unit displacement fields into
       canonical LPS+ orientation with channel-last storage.

    It also provides array-only conversion helpers for temporary
    interoperability with code that expects canonical RAS+ displacement arrays
    while keeping ``MedicalWarp`` objects strictly canonical LPS+.

    The canonical reorientation assumes the displacement vectors are expressed
    in voxel units along image-array axes. Under that assumption the warp
    lattice is reoriented like the image lattice, and the vector components are
    transformed by the same signed axis permutation.

Variable / function / class list:
    Variables:
        None

    Functions:
        convert_warp_nifti_to_channel_last
        medical_warp_to_ras_array
        medical_warp_from_ras_array_like
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
    get_axis_codes_from_affine,
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


def _reorient_voxel_warp_between_canonical_orientations(
    array_channel_last: np.ndarray,
    spatial_ndim: int,
    source_orientation: str,
    target_orientation: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a canonical voxel-unit warp array between LPS+ and RAS+.

    Parameters
    ----------
    array_channel_last : np.ndarray
        Canonical channel-last warp array whose vector components follow the
        same source canonical orientation as the spatial lattice.
    spatial_ndim : int
        Number of spatial axes represented by the warp. Supported values are
        ``2`` and ``3``.
    source_orientation : str
        Source canonical anatomical orientation. Supported values are
        ``"LPS"`` and ``"RAS"``.
    target_orientation : str
        Target canonical anatomical orientation. Supported values are
        ``"LPS"`` and ``"RAS"``.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        A tuple containing:
        - the reoriented channel-last warp array, and
        - metadata describing the applied signed axis permutation.
    """

    array_channel_last = np.asarray(array_channel_last)
    source_name = str(source_orientation).upper()
    target_name = str(target_orientation).upper()

    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}.")
    if array_channel_last.ndim != spatial_ndim + 1:
        raise ValueError(
            f"array_channel_last must have ndim={spatial_ndim + 1}, got {array_channel_last.ndim}."
        )
    if array_channel_last.shape[-1] != spatial_ndim:
        raise ValueError(
            "The final vector axis must match spatial_ndim, got "
            f"{array_channel_last.shape[-1]} and {spatial_ndim}."
        )
    if source_name not in ("LPS", "RAS"):
        raise ValueError(
            f"source_orientation must be 'LPS' or 'RAS', got {source_orientation!r}."
        )
    if target_name not in ("LPS", "RAS"):
        raise ValueError(
            f"target_orientation must be 'LPS' or 'RAS', got {target_orientation!r}."
        )

    if spatial_ndim == 2:
        metadata_2d: dict[str, Any] = {
            "applied": False,
            "reason": "2d_warps_are_not_reoriented_between_canonical_orientations",
            "source_orientation": source_name,
            "target_orientation": target_name,
        }
        return array_channel_last.copy(), metadata_2d

    if source_name == target_name:
        metadata_same: dict[str, Any] = {
            "applied": False,
            "reason": "source_and_target_orientations_match",
            "source_orientation": source_name,
            "target_orientation": target_name,
        }
        return array_channel_last.copy(), metadata_same

    source_axcodes = ("R", "A", "S") if source_name == "RAS" else LPS_AXCODES
    target_axcodes = ("R", "A", "S") if target_name == "RAS" else LPS_AXCODES
    source_ornt = nib.orientations.axcodes2ornt(source_axcodes)
    target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
    transform_ornt = nib.orientations.ornt_transform(source_ornt, target_ornt)

    reoriented_array = nib.orientations.apply_orientation(array_channel_last, transform_ornt)
    vector_out = np.empty_like(reoriented_array)
    for new_axis in range(spatial_ndim):
        old_axis = int(transform_ornt[new_axis, 0])
        flip_sign = float(transform_ornt[new_axis, 1])
        vector_out[..., new_axis] = reoriented_array[..., old_axis] * flip_sign

    metadata = {
        "applied": True,
        "source_orientation": source_name,
        "target_orientation": target_name,
        "source_axcodes": list(source_axcodes),
        "target_axcodes": list(target_axcodes),
        "transform_ornt": transform_ornt.tolist(),
    }
    return vector_out, metadata


def medical_warp_to_ras_array(warp: MedicalWarp) -> np.ndarray:
    """Convert a canonical LPS+ MedicalWarp into a temporary RAS+ array.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement-field object whose lattice and vector
        components are stored in LPS+ orientation.

    Returns
    -------
    np.ndarray
        Plain channel-last NumPy array reoriented into canonical RAS+ order.
        No affine or other geometry metadata is returned with this array.
    """

    if warp.spatial_ndim == 3:
        current_codes = get_axis_codes_from_affine(warp.affine_lps, warp.spatial_ndim)
        if current_codes != LPS_AXCODES:
            raise ValueError(
                "medical_warp_to_ras_array expects a canonical LPS+ MedicalWarp."
            )

    array_ras, _ = _reorient_voxel_warp_between_canonical_orientations(
        array_channel_last=warp.array,
        spatial_ndim=warp.spatial_ndim,
        source_orientation="LPS",
        target_orientation="RAS",
    )
    return array_ras


def medical_warp_from_ras_array_like(
    array_ras: np.ndarray,
    reference_warp: MedicalWarp,
) -> MedicalWarp:
    """Create a canonical LPS+ warp from a RAS+ array and LPS+ reference.

    Parameters
    ----------
    array_ras : np.ndarray
        Plain channel-last displacement array in canonical RAS+ order. Its
        full shape must match the reference warp because this helper reuses the
        reference geometry unchanged.
    reference_warp : MedicalWarp
        Canonical LPS+ warp that provides the target affine, units, axis
        labels, source type, and metadata template for the returned object.

    Returns
    -------
    MedicalWarp
        New canonical LPS+ warp whose array is obtained by converting
        ``array_ras`` back into LPS+ order while preserving the reference warp
        geometry.
    """

    if reference_warp.spatial_ndim == 3:
        current_codes = get_axis_codes_from_affine(
            reference_warp.affine_lps,
            reference_warp.spatial_ndim,
        )
        if current_codes != LPS_AXCODES:
            raise ValueError(
                "medical_warp_from_ras_array_like expects a canonical LPS+ reference warp."
            )

    array_lps, transform_info = _reorient_voxel_warp_between_canonical_orientations(
        array_channel_last=array_ras,
        spatial_ndim=reference_warp.spatial_ndim,
        source_orientation="RAS",
        target_orientation="LPS",
    )
    if tuple(int(v) for v in array_lps.shape) != reference_warp.shape:
        raise ValueError(
            "array_ras converted back to LPS+ must match the reference warp "
            f"shape {reference_warp.shape}, got {tuple(int(v) for v in array_lps.shape)}."
        )

    metadata = dict(reference_warp.metadata)
    metadata["medical_warp_from_ras_array_like"] = transform_info
    return MedicalWarp(
        array=array_lps,
        affine_lps=reference_warp.affine_lps.copy(),
        spatial_ndim=reference_warp.spatial_ndim,
        axis_labels=reference_warp.axis_labels,
        units=reference_warp.units,
        source_type=reference_warp.source_type,
        metadata=metadata,
    )


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


def convert_warp_nifti_to_channel_last(
    src_path: str | Path,
    dst_path: str | Path,
) -> None:
    """Convert a raw warp NIfTI file to channel-last storage.

    This helper changes only the array storage layout. The affine matrix,
    qform, sform, and copied header metadata are preserved from the source
    file. No anatomical reorientation or unit conversion is applied.

    Parameters
    ----------
    src_path : str | Path
        Input warp NIfTI path. The stored vector layout is inferred from the
        on-disk array shape.
    dst_path : str | Path
        Output path for the channel-last NIfTI file.

    Returns
    -------
    None
        The converted NIfTI file is written to disk.
    """
    src_nii = nib.load(str(Path(src_path)))
    src_array = np.asarray(src_nii.dataobj)
    src_layout = _infer_input_vector_layout(src_array)
    dst_array = _convert_warp_vector_layout(src_array, src_layout, "channel_last")

    header = src_nii.header.copy()
    header.set_data_dtype(dst_array.dtype)
    header.set_data_shape(dst_array.shape)

    dst_nii = nib.Nifti1Image(
        dst_array,
        np.asarray(src_nii.affine, dtype=np.float64),
        header=header,
    )

    qform, qform_code = src_nii.get_qform(coded=True)
    sform, sform_code = src_nii.get_sform(coded=True)
    dst_nii.set_qform(qform, int(qform_code))
    dst_nii.set_sform(sform, int(sform_code))

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(dst_nii, str(dst_path))


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
