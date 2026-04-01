"""
General description:
    Unified numpy-based image warping utilities for 2D and 3D images using either
    SciPy `map_coordinates` or PyTorch `grid_sample`.

    This module provides a single public entry point that:
        1. accepts numpy image arrays in 2D or 3D,
        2. accepts displacement fields in channel-last or channel-first format,
        3. validates the displacement convention against the image shape,
        4. dispatches to either a SciPy or PyTorch implementation, and
        5. returns the warped image as a numpy array.

    The displacement convention follows:
        2D: (X, Y, 2) or (2, X, Y)
        3D: (X, Y, Z, 3) or (3, X, Y, Z)

    The mathematical convention is backward / pull warping:
        warped[x, y]     = moving[x + dX, y + dY]
        warped[x, y, z]  = moving[x + dX, y + dY, z + dZ]

Variable / function / class list:
    Variables:
        __all__

    Functions:
        _infer_spatial_ndim
        _convert_disp_to_channel_last
        _prepare_numeric_arrays
        _build_identity_grid_numpy
        _resolve_torch_device
        _warp_scipy
        _warp_torch
        warp_image_unified
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates


__all__ = ["warp_image_unified"]


def _infer_spatial_ndim(image: np.ndarray) -> int:
    """
    Infer whether the input image is 2D or 3D.

    Parameters
    ----------
    image : np.ndarray
        Input image array. The expected semantic meaning is a scalar-valued image
        stored as `(X, Y)` for 2D or `(X, Y, Z)` for 3D.

    Returns
    -------
    int
        Spatial dimensionality of the image. The return value is either `2` or `3`.

    Raises
    ------
    TypeError
        If `image` is not a numpy array.
    ValueError
        If `image` is not 2D or 3D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"`image` must be a numpy array, got {type(image)!r}.")

    if image.ndim not in (2, 3):
        raise ValueError(
            f"`image` must be 2D or 3D. Got shape {image.shape} with ndim={image.ndim}."
        )

    return image.ndim


def _convert_disp_to_channel_last(
    disp: np.ndarray,
    image_shape: Tuple[int, ...],
) -> Tuple[np.ndarray, str]:
    """
    Convert a displacement field to channel-last format and validate its shape.

    Parameters
    ----------
    disp : np.ndarray
        Displacement field as a numpy array. Supported semantic formats are:
        `(X, Y, 2)` or `(2, X, Y)` for 2D, and
        `(X, Y, Z, 3)` or `(3, X, Y, Z)` for 3D.
    image_shape : Tuple[int, ...]
        Spatial shape of the image being warped. This is used to validate that the
        displacement field is defined on the same output lattice as the image.

    Returns
    -------
    Tuple[np.ndarray, str]
        A tuple containing:
        - the displacement field in channel-last format, and
        - a string describing the detected input convention: `"channel_last"` or
          `"channel_first"`.

    Raises
    ------
    TypeError
        If `disp` is not a numpy array.
    ValueError
        If `disp` does not match a supported 2D or 3D displacement shape, if the
        vector channel count is incorrect, or if the displacement spatial shape does
        not match `image_shape`.
    """
    if not isinstance(disp, np.ndarray):
        raise TypeError(f"`disp` must be a numpy array, got {type(disp)!r}.")

    spatial_ndim = len(image_shape)
    expected_channel_count = spatial_ndim
    expected_channel_last_shape = image_shape + (expected_channel_count,)
    expected_channel_first_shape = (expected_channel_count,) + image_shape

    is_channel_last = disp.shape == expected_channel_last_shape
    is_channel_first = disp.shape == expected_channel_first_shape

    if is_channel_last and is_channel_first:
        raise ValueError(
            "Ambiguous displacement layout: the displacement matches both channel-last "
            f"and channel-first conventions for image shape {image_shape}. "
            f"Received disp shape {disp.shape}."
        )

    if is_channel_last:
        return disp, "channel_last"

    if is_channel_first:
        axes = tuple(range(1, spatial_ndim + 1)) + (0,)
        return np.transpose(disp, axes=axes), "channel_first"

    raise ValueError(
        "Unsupported displacement shape. "
        f"For a {spatial_ndim}D image with shape {image_shape}, expected either "
        f"{expected_channel_last_shape} (channel-last) or "
        f"{expected_channel_first_shape} (channel-first), but got {disp.shape}."
    )


def _prepare_numeric_arrays(
    image: np.ndarray,
    disp_channel_last: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert image and displacement arrays to a consistent numeric dtype for warping.

    Parameters
    ----------
    image : np.ndarray
        Input image array in 2D or 3D scalar-valued format.
    disp_channel_last : np.ndarray
        Displacement field in validated channel-last format:
        `(X, Y, 2)` for 2D or `(X, Y, Z, 3)` for 3D.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - the image cast to `np.float32`, and
        - the displacement field cast to `np.float32`.

    Raises
    ------
    TypeError
        If either input is not numeric.
    """
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"`image` must have a numeric dtype, got {image.dtype}.")
    if not np.issubdtype(disp_channel_last.dtype, np.number):
        raise TypeError(f"`disp` must have a numeric dtype, got {disp_channel_last.dtype}.")

    return image.astype(np.float32, copy=False), disp_channel_last.astype(np.float32, copy=False)


def _build_identity_grid_numpy(image_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Build an identity coordinate grid in array-axis order.

    Parameters
    ----------
    image_shape : Tuple[int, ...]
        Spatial shape of the image lattice. The semantic meaning is `(X, Y)` for 2D
        or `(X, Y, Z)` for 3D.

    Returns
    -------
    np.ndarray
        Identity coordinates in SciPy-compatible array-axis order with shape
        `(ndim, *image_shape)`.
    """
    axes = tuple(np.arange(size, dtype=np.float32) for size in image_shape)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=0)


def _resolve_torch_device(device: Optional[Union[str, int]]) -> torch.device:
    """
    Resolve a user-specified torch device into a `torch.device`.

    Parameters
    ----------
    device : Optional[Union[str, int]]
        Device specification for PyTorch warping. Semantic options are:
        - `None`: use CPU
        - `int`: use `cuda:{device}`
        - `str`: pass a torch-style device string such as `"cpu"` or `"cuda:1"`

    Returns
    -------
    torch.device
        The resolved torch device object.

    Raises
    ------
    TypeError
        If `device` is not `None`, `int`, or `str`.
    ValueError
        If an integer device index is negative.
    RuntimeError
        If a CUDA device is requested but CUDA is not available.
    """
    if device is None:
        return torch.device("cpu")

    if isinstance(device, int):
        if device < 0:
            raise ValueError(f"`device` must be non-negative when given as an int, got {device}.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False.")
        return torch.device(f"cuda:{device}")

    if isinstance(device, str):
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False.")
        return resolved_device

    raise TypeError(f"`device` must be None, int, or str, got {type(device)!r}.")


def _warp_scipy(
    image: np.ndarray,
    disp_channel_last: np.ndarray,
    order: Literal[0, 1],
    padding_mode: Literal["border", "constant"],
    cval: float,
) -> np.ndarray:
    """
    Warp a 2D or 3D numpy image using SciPy `map_coordinates`.

    Parameters
    ----------
    image : np.ndarray
        Input image array in `np.float32`, stored as `(X, Y)` for 2D or `(X, Y, Z)`
        for 3D.
    disp_channel_last : np.ndarray
        Displacement field in `np.float32`, stored as `(X, Y, 2)` for 2D or
        `(X, Y, Z, 3)` for 3D. The semantic meaning is backward / pull sampling
        displacement in voxel units.
    order : Literal[0, 1]
        Interpolation order for SciPy. `0` means nearest-neighbor interpolation and
        `1` means linear interpolation.
    padding_mode : Literal["border", "constant"]
        Unified padding mode. `"border"` replicates the nearest border value, while
        `"constant"` samples a constant value outside the image.
    cval : float
        Constant value used when `padding_mode == "constant"`.

    Returns
    -------
    np.ndarray
        Warped image as a numpy array with the same spatial shape as `image`.
    """
    scipy_mode = "nearest" if padding_mode == "border" else "constant"
    identity = _build_identity_grid_numpy(image.shape)
    disp_cf = np.moveaxis(disp_channel_last, -1, 0)
    coords = identity + disp_cf

    return map_coordinates(
        image,
        coords,
        order=order,
        mode=scipy_mode,
        cval=float(cval),
    )


def _warp_torch(
    image: np.ndarray,
    disp_channel_last: np.ndarray,
    order: Literal[0, 1],
    padding_mode: Literal["border", "constant"],
    cval: float,
    device: Optional[Union[str, int]],
) -> np.ndarray:
    """
    Warp a 2D or 3D numpy image using PyTorch `grid_sample`.

    Parameters
    ----------
    image : np.ndarray
        Input image array in `np.float32`, stored as `(X, Y)` for 2D or `(X, Y, Z)`
        for 3D.
    disp_channel_last : np.ndarray
        Displacement field in `np.float32`, stored as `(X, Y, 2)` for 2D or
        `(X, Y, Z, 3)` for 3D. The semantic meaning is backward / pull sampling
        displacement in voxel units.
    order : Literal[0, 1]
        Unified interpolation order. `0` maps to PyTorch `"nearest"` and `1` maps to
        PyTorch `"bilinear"`.
    padding_mode : Literal["border", "constant"]
        Unified padding mode. `"border"` maps to PyTorch `"border"`. `"constant"`
        maps to PyTorch `"zeros"`.
    cval : float
        Constant padding value requested by the caller. At present, PyTorch only
        supports zero-valued constant padding through `grid_sample`. Non-zero constant
        padding is therefore not implemented yet and is left as a future improvement.
    device : Optional[Union[str, int]]
        Torch device selector. `None` uses CPU. An integer selects `cuda:{index}`.
        A string may be used for explicit device names such as `"cpu"` or `"cuda:0"`.

    Returns
    -------
    np.ndarray
        Warped image as a numpy array with the same spatial shape as `image`.

    Raises
    ------
    NotImplementedError
        If constant padding is requested with `cval != 0.0`, because `grid_sample`
        only supports zero-padding directly.
    """
    if padding_mode == "constant" and float(cval) != 0.0:
        raise NotImplementedError(
            "PyTorch grid_sample currently supports only zero-valued constant padding "
            "through padding_mode='zeros'. Non-zero constant padding is left as a "
            "future improvement."
        )

    torch_device = _resolve_torch_device(device)
    torch_interp_mode = "nearest" if order == 0 else "bilinear"
    torch_padding_mode = "border" if padding_mode == "border" else "zeros"

    src = torch.from_numpy(image).to(device=torch_device, dtype=torch.float32)
    flow = torch.from_numpy(np.moveaxis(disp_channel_last, -1, 0)).to(
        device=torch_device,
        dtype=torch.float32,
    )

    src = src.unsqueeze(0).unsqueeze(0)
    flow = flow.unsqueeze(0)

    spatial_shape = tuple(int(size) for size in image.shape)
    grid_axes = tuple(
        torch.arange(size, device=torch_device, dtype=torch.float32) for size in spatial_shape
    )
    grid = torch.stack(torch.meshgrid(*grid_axes, indexing="ij"), dim=0).unsqueeze(0)

    new_locs = grid + flow

    for axis_index, axis_size in enumerate(spatial_shape):
        new_locs[:, axis_index, ...] = 2.0 * (new_locs[:, axis_index, ...] / (axis_size - 1) - 0.5)

    if len(spatial_shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    else:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

    warped = F.grid_sample(
        src,
        new_locs,
        mode=torch_interp_mode,
        padding_mode=torch_padding_mode,
        align_corners=True,
    )

    return warped.squeeze(0).squeeze(0).detach().cpu().numpy()


def warp_image_unified(
    image: np.ndarray,
    disp: np.ndarray,
    backend: Literal["scipy", "torch"] = "scipy",
    order: Literal[0, 1] = 1,
    padding_mode: Literal["border", "constant"] = "constant",
    cval: float = 0.0,
    device: Optional[Union[str, int]] = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Warp a 2D or 3D numpy image using a displacement field and a selectable backend.

    Parameters
    ----------
    image : np.ndarray
        Input scalar-valued image stored as a numpy array. Supported semantic shapes
        are `(X, Y)` for 2D and `(X, Y, Z)` for 3D.
    disp : np.ndarray
        Displacement field stored as a numpy array in voxel units. Supported semantic
        shapes are:
        - 2D: `(X, Y, 2)` or `(2, X, Y)`
        - 3D: `(X, Y, Z, 3)` or `(3, X, Y, Z)`
        The displacement follows the backward / pull warping convention described in
        `WARP_CONVENTION.md`.
    backend : Literal["scipy", "torch"], optional
        Warping backend. `"scipy"` uses `scipy.ndimage.map_coordinates` on CPU.
        `"torch"` uses `torch.nn.functional.grid_sample` on CPU or GPU.
    order : Literal[0, 1], optional
        Unified interpolation order. `0` means nearest-neighbor interpolation.
        `1` means linear interpolation. For the torch backend, `1` maps to
        `mode="bilinear"`; in 3D this results in trilinear interpolation.
    padding_mode : Literal["border", "constant"], optional
        Unified padding mode. `"border"` means replicate the nearest border value.
        `"constant"` means sample a constant value outside the image.
    cval : float, optional
        Constant padding value used when `padding_mode == "constant"`.
        For the SciPy backend, any numeric value is supported.
        For the torch backend, only `cval == 0.0` is currently supported through
        `grid_sample`. Support for arbitrary non-zero constant padding is a future
        improvement.
    device : Optional[Union[str, int]], optional
        Device selector used only when `backend == "torch"`.
        - `None`: use CPU
        - `int`: use `cuda:{index}`
        - `str`: use an explicit torch device string such as `"cpu"` or `"cuda:1"`
    debug : bool, optional
        If `True`, print internal shape, dtype, backend, interpolation, padding, and
        device information to help verify the convention and execution path.

    Returns
    -------
    np.ndarray
        Warped image as a numpy array with the same spatial shape as `image`. The
        computation is performed in `float32`, so the returned array is typically
        `np.float32`.

    Raises
    ------
    ValueError
        If the image dimensionality, displacement shape, interpolation order, padding
        mode, or backend is invalid.
    TypeError
        If the input arrays are not numpy arrays or do not have numeric dtypes.
    NotImplementedError
        If `backend == "torch"` and non-zero constant padding is requested.
    RuntimeError
        If a CUDA device is requested for torch but CUDA is unavailable.
    """
    spatial_ndim = _infer_spatial_ndim(image)

    if order not in (0, 1):
        raise ValueError(f"`order` must be 0 or 1, got {order}.")

    if padding_mode not in ("border", "constant"):
        raise ValueError(
            f"`padding_mode` must be 'border' or 'constant', got {padding_mode!r}."
        )

    if backend not in ("scipy", "torch"):
        raise ValueError(f"`backend` must be 'scipy' or 'torch', got {backend!r}.")

    disp_channel_last, detected_disp_layout = _convert_disp_to_channel_last(disp, image.shape)
    image_float32, disp_float32 = _prepare_numeric_arrays(image, disp_channel_last)

    debug_info: Dict[str, Any] = {
        "backend": backend,
        "spatial_ndim": spatial_ndim,
        "image_shape": tuple(image.shape),
        "image_dtype": str(image.dtype),
        "disp_original_shape": tuple(disp.shape),
        "disp_original_dtype": str(disp.dtype),
        "disp_detected_layout": detected_disp_layout,
        "disp_channel_last_shape": tuple(disp_float32.shape),
        "compute_dtype": "float32",
        "order": order,
        "padding_mode": padding_mode,
        "cval": float(cval),
    }

    if backend == "scipy":
        debug_info["scipy_order"] = order
        debug_info["scipy_mode"] = "nearest" if padding_mode == "border" else "constant"
        result = _warp_scipy(
            image=image_float32,
            disp_channel_last=disp_float32,
            order=order,
            padding_mode=padding_mode,
            cval=cval,
        )
    else:
        resolved_device = _resolve_torch_device(device)
        debug_info["torch_device"] = str(resolved_device)
        debug_info["torch_mode"] = "nearest" if order == 0 else "bilinear"
        debug_info["torch_padding_mode"] = "border" if padding_mode == "border" else "zeros"
        debug_info["align_corners"] = True
        result = _warp_torch(
            image=image_float32,
            disp_channel_last=disp_float32,
            order=order,
            padding_mode=padding_mode,
            cval=cval,
            device=device,
        )

    if debug:
        for key, value in debug_info.items():
            print(f"[warp_image_unified] {key}: {value}")

    return result
