"""Warp-derived scalar and vector feature computations.

Purpose:
This module computes scalar and vector features from canonical displacement
fields stored as `MedicalWarp` objects. The computations assume the warp is in
canonical channel-last layout and, for 3D, canonical LPS+ spatial orientation.

    The current feature set includes:
        1. displacement magnitude,
        2. individual displacement components,
        3. Jacobian determinant of the deformation,
        4. Jacobian fold mask,
        5. 2D curl,
        6. 3D curl vector, and
        7. curl magnitude.

    Jacobian determinant and curl are computed on an interior region using
    centered finite differences. The result can then either be returned in the
    cropped interior shape or padded back to the original lattice size using
    border padding.

Variables:
- PaddingMode: Supported padding strategies for derivative-based outputs.
- ComponentLike: Supported displacement or curl component selectors.

Functions:
- compute_warp_magnitude: Compute displacement magnitude.
- compute_warp_component: Extract one displacement component.
- compute_jacobian_determinant: Compute the deformation Jacobian determinant.
- compute_jacobian_fold_mask: Compute a boolean fold mask where Jdet <= 0.
- compute_curl_2d: Compute the scalar curl of a 2D displacement field.
- compute_curl_3d: Compute the vector curl of a 3D displacement field.
- compute_curl_component: Extract one curl component.
- compute_curl_magnitude: Compute curl magnitude.

Classes:
- None.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .warp_base import MedicalWarp

PaddingMode = Literal["border", "crop"]
ComponentLike = int | Literal["x", "y", "z"]


def _validate_padding_mode(padding_mode: PaddingMode) -> None:
    """Validate the supported padding mode used by derivative-based features.

    Parameters
    ----------
    padding_mode : Literal["border", "crop"]
        Requested output strategy. `"crop"` returns only the interior region
        where centered differences are valid. `"border"` pads the interior
        result back to the original lattice size by replicating edge values.

    Returns
    -------
    None
        Validation succeeds silently.
    """

    if padding_mode not in ("border", "crop"):
        raise ValueError(
            f"padding_mode must be 'border' or 'crop', got {padding_mode!r}."
        )


def _normalize_component_index(
    component: ComponentLike,
    spatial_ndim: int,
) -> int:
    """Normalize a component selector into an integer axis index.

    Parameters
    ----------
    component : int | Literal["x", "y", "z"]
        Requested component. Integer indices are zero-based. String selectors
        use anatomical-style axis names aligned with the canonical array axes.
    spatial_ndim : int
        Number of spatial axes in the warp. Supported values are `2` and `3`.

    Returns
    -------
    int
        Zero-based component index.
    """

    if isinstance(component, int):
        if component < 0 or component >= spatial_ndim:
            raise ValueError(
                f"Component index must be in [0, {spatial_ndim - 1}], got {component}."
            )
        return component

    component_map = {"x": 0, "y": 1, "z": 2}
    if component not in component_map:
        raise ValueError(
            f"component must be one of 0..{spatial_ndim - 1} or 'x'/'y'/'z', got {component!r}."
        )

    component_index = component_map[component]
    if component_index >= spatial_ndim:
        raise ValueError(
            f"Component {component!r} is not valid for spatial_ndim={spatial_ndim}."
        )
    return component_index


def _centered_difference_interior(field: np.ndarray, axis: int) -> np.ndarray:
    """Compute a centered finite difference on the interior lattice.

    Parameters
    ----------
    field : np.ndarray
        Scalar field with shape `(X, Y)` or `(X, Y, Z)`.
    axis : int
        Spatial axis along which the derivative is computed.

    Returns
    -------
    np.ndarray
        Centered finite-difference derivative on the cropped interior lattice.
        For 2D the output shape is `(X - 2, Y - 2)`. For 3D the output shape is
        `(X - 2, Y - 2, Z - 2)`.
    """

    if field.ndim not in (2, 3):
        raise ValueError(f"field must be 2D or 3D, got shape {field.shape}.")
    if axis < 0 or axis >= field.ndim:
        raise ValueError(f"axis must be in [0, {field.ndim - 1}], got {axis}.")
    if any(size < 3 for size in field.shape):
        raise ValueError(
            f"All spatial dimensions must be at least 3 for centered differences, got {field.shape}."
        )

    center_slices = [slice(1, -1) for _ in range(field.ndim)]
    plus_slices = center_slices.copy()
    minus_slices = center_slices.copy()
    plus_slices[axis] = slice(2, None)
    minus_slices[axis] = slice(None, -2)

    return 0.5 * (
        field[tuple(plus_slices)] - field[tuple(minus_slices)]
    )


def _pad_spatial_edges(
    array: np.ndarray,
    spatial_ndim: int,
    padding_mode: PaddingMode,
) -> np.ndarray:
    """Pad a cropped derivative-based result back to full spatial size.

    Parameters
    ----------
    array : np.ndarray
        Cropped derivative-based result. The leading `spatial_ndim` axes are
        the cropped spatial axes, and any trailing axis such as a vector axis
        is preserved without padding.
    spatial_ndim : int
        Number of spatial axes to pad.
    padding_mode : Literal["border", "crop"]
        Output strategy. `"crop"` returns the input unchanged. `"border"`
        pads one voxel on both ends of each spatial axis using edge
        replication.

    Returns
    -------
    np.ndarray
        Either the cropped input or a full-size edge-padded array.
    """

    _validate_padding_mode(padding_mode)
    if padding_mode == "crop":
        return array

    pad_width = [(1, 1) for _ in range(spatial_ndim)]
    pad_width.extend([(0, 0) for _ in range(array.ndim - spatial_ndim)])
    return np.pad(array, pad_width=pad_width, mode="edge")


def compute_warp_magnitude(warp: MedicalWarp) -> np.ndarray:
    """Compute the displacement magnitude of a canonical warp field.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field in channel-last layout with shape
        `(X, Y, 2)` for 2D or `(X, Y, Z, 3)` for 3D.

    Returns
    -------
    np.ndarray
        Scalar magnitude field with the same spatial shape as `warp`.
    """

    return np.linalg.norm(np.asarray(warp.array, dtype=np.float32), axis=-1)


def compute_warp_component(
    warp: MedicalWarp,
    component: ComponentLike,
) -> np.ndarray:
    """Extract one displacement component from a canonical warp field.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field in channel-last layout.
    component : int | Literal["x", "y", "z"]
        Requested displacement component.

    Returns
    -------
    np.ndarray
        Scalar component field with the same spatial shape as `warp`.
    """

    component_index = _normalize_component_index(component, warp.spatial_ndim)
    return np.asarray(warp.array[..., component_index], dtype=np.float32)


def compute_jacobian_determinant(
    warp: MedicalWarp,
    padding_mode: PaddingMode = "border",
) -> np.ndarray:
    """Compute the deformation Jacobian determinant from a displacement field.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field in voxel units and channel-last layout.
    padding_mode : Literal["border", "crop"], optional
        Output strategy for the derivative-based interior result. `"crop"`
        returns the interior lattice only. `"border"` pads the result back to
        the original spatial size by replicating the interior edge values.

    Returns
    -------
    np.ndarray
        Jacobian determinant field. The shape is either cropped by one voxel on
        every side or restored to the original warp lattice size, depending on
        `padding_mode`.
    """

    disp = np.asarray(warp.array, dtype=np.float32)

    if warp.spatial_ndim == 2:
        disp_x = disp[..., 0]
        disp_y = disp[..., 1]

        d_ux_dx = _centered_difference_interior(disp_x, axis=0)
        d_ux_dy = _centered_difference_interior(disp_x, axis=1)
        d_uy_dx = _centered_difference_interior(disp_y, axis=0)
        d_uy_dy = _centered_difference_interior(disp_y, axis=1)

        jacdet = (1.0 + d_ux_dx) * (1.0 + d_uy_dy) - d_ux_dy * d_uy_dx
        return _pad_spatial_edges(jacdet, spatial_ndim=2, padding_mode=padding_mode)

    if warp.spatial_ndim == 3:
        disp_x = disp[..., 0]
        disp_y = disp[..., 1]
        disp_z = disp[..., 2]

        d_ux_dx = _centered_difference_interior(disp_x, axis=0)
        d_ux_dy = _centered_difference_interior(disp_x, axis=1)
        d_ux_dz = _centered_difference_interior(disp_x, axis=2)

        d_uy_dx = _centered_difference_interior(disp_y, axis=0)
        d_uy_dy = _centered_difference_interior(disp_y, axis=1)
        d_uy_dz = _centered_difference_interior(disp_y, axis=2)

        d_uz_dx = _centered_difference_interior(disp_z, axis=0)
        d_uz_dy = _centered_difference_interior(disp_z, axis=1)
        d_uz_dz = _centered_difference_interior(disp_z, axis=2)

        j00 = 1.0 + d_ux_dx
        j01 = d_ux_dy
        j02 = d_ux_dz
        j10 = d_uy_dx
        j11 = 1.0 + d_uy_dy
        j12 = d_uy_dz
        j20 = d_uz_dx
        j21 = d_uz_dy
        j22 = 1.0 + d_uz_dz

        jacdet = (
            j00 * (j11 * j22 - j12 * j21)
            - j10 * (j01 * j22 - j02 * j21)
            + j20 * (j01 * j12 - j02 * j11)
        )
        return _pad_spatial_edges(jacdet, spatial_ndim=3, padding_mode=padding_mode)

    raise ValueError(f"Unsupported spatial_ndim={warp.spatial_ndim}.")


def compute_jacobian_fold_mask(
    warp: MedicalWarp,
    padding_mode: PaddingMode = "border",
) -> np.ndarray:
    """Compute a boolean mask for regions where the Jacobian determinant is non-positive.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field in voxel units and channel-last layout.
    padding_mode : Literal["border", "crop"], optional
        Output strategy used by the underlying Jacobian determinant computation.

    Returns
    -------
    np.ndarray
        Boolean array where `True` marks voxels or pixels with
        `Jacobian determinant <= 0`.
    """

    jacdet = compute_jacobian_determinant(warp=warp, padding_mode=padding_mode)
    return jacdet <= 0.0


def compute_curl_2d(
    warp: MedicalWarp,
    padding_mode: PaddingMode = "border",
) -> np.ndarray:
    """Compute the scalar curl of a 2D displacement field.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 2D displacement field in channel-last layout `(X, Y, 2)`.
    padding_mode : Literal["border", "crop"], optional
        Output strategy for the derivative-based interior result.

    Returns
    -------
    np.ndarray
        Scalar curl field defined as `dUy/dX - dUx/dY`.
    """

    if warp.spatial_ndim != 2:
        raise ValueError(
            f"compute_curl_2d expects a 2D warp, got spatial_ndim={warp.spatial_ndim}."
        )

    disp = np.asarray(warp.array, dtype=np.float32)
    disp_x = disp[..., 0]
    disp_y = disp[..., 1]

    d_uy_dx = _centered_difference_interior(disp_y, axis=0)
    d_ux_dy = _centered_difference_interior(disp_x, axis=1)
    curl = d_uy_dx - d_ux_dy
    return _pad_spatial_edges(curl, spatial_ndim=2, padding_mode=padding_mode)


def compute_curl_3d(
    warp: MedicalWarp,
    padding_mode: PaddingMode = "border",
) -> np.ndarray:
    """Compute the vector curl of a 3D displacement field.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 3D displacement field in channel-last layout `(X, Y, Z, 3)`.
    padding_mode : Literal["border", "crop"], optional
        Output strategy for the derivative-based interior result.

    Returns
    -------
    np.ndarray
        Channel-last curl vector field with shape `(X, Y, Z, 3)` when
        `padding_mode="border"` or `(X - 2, Y - 2, Z - 2, 3)` when
        `padding_mode="crop"`.
    """

    if warp.spatial_ndim != 3:
        raise ValueError(
            f"compute_curl_3d expects a 3D warp, got spatial_ndim={warp.spatial_ndim}."
        )

    disp = np.asarray(warp.array, dtype=np.float32)
    disp_x = disp[..., 0]
    disp_y = disp[..., 1]
    disp_z = disp[..., 2]

    d_ux_dz = _centered_difference_interior(disp_x, axis=2)
    d_uy_dx = _centered_difference_interior(disp_y, axis=0)
    d_uy_dz = _centered_difference_interior(disp_y, axis=2)
    d_uz_dx = _centered_difference_interior(disp_z, axis=0)
    d_uz_dy = _centered_difference_interior(disp_z, axis=1)
    d_ux_dy = _centered_difference_interior(disp_x, axis=1)

    curl_x = d_uz_dy - d_uy_dz
    curl_y = d_ux_dz - d_uz_dx
    curl_z = d_uy_dx - d_ux_dy

    curl = np.stack((curl_x, curl_y, curl_z), axis=-1)
    return _pad_spatial_edges(curl, spatial_ndim=3, padding_mode=padding_mode)


def compute_curl_component(
    warp: MedicalWarp,
    component: ComponentLike,
    padding_mode: PaddingMode = "border",
) -> np.ndarray:
    """Extract one component of the curl field.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field. For 2D, use `compute_curl_2d` instead of
        requesting a curl component.
    component : int | Literal["x", "y", "z"]
        Requested curl component.
    padding_mode : Literal["border", "crop"], optional
        Output strategy for the derivative-based interior result.

    Returns
    -------
    np.ndarray
        Scalar component of the 3D curl vector field.
    """

    if warp.spatial_ndim != 3:
        raise ValueError(
            f"compute_curl_component expects a 3D warp, got spatial_ndim={warp.spatial_ndim}."
        )

    component_index = _normalize_component_index(component, spatial_ndim=3)
    curl = compute_curl_3d(warp=warp, padding_mode=padding_mode)
    return curl[..., component_index]


def compute_curl_magnitude(
    warp: MedicalWarp,
    padding_mode: PaddingMode = "border",
) -> np.ndarray:
    """Compute the magnitude of the curl field.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field. For 2D, the returned value is the
        absolute value of the scalar curl. For 3D, the returned value is the
        Euclidean norm of the vector curl.
    padding_mode : Literal["border", "crop"], optional
        Output strategy for the derivative-based interior result.

    Returns
    -------
    np.ndarray
        Scalar curl-magnitude field.
    """

    if warp.spatial_ndim == 2:
        return np.abs(compute_curl_2d(warp=warp, padding_mode=padding_mode))

    if warp.spatial_ndim == 3:
        curl = compute_curl_3d(warp=warp, padding_mode=padding_mode)
        return np.linalg.norm(curl, axis=-1)

    raise ValueError(f"Unsupported spatial_ndim={warp.spatial_ndim}.")
