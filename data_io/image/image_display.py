"""Display helpers for canonical 2D, 3D, and 4D medical images.

Purpose:
This module renders canonical ``MedicalImage`` objects using matplotlib. It
extracts view-specific slices from LPS-standardized volumes, applies
radiological display conventions, computes physically plausible aspect ratios
from the affine, and supports configurable grayscale or pseudocolor display.

Variables:
- None.

Functions:
- resolve_intensity_window: Compute vmin/vmax from explicit values or percentiles.
- extract_slice: Extract and orient one 2D slice for radiological display.
- show_image_2d: Display a 2D image or one frame of a 2D+t image.
- show_slice: Display a single sagittal, coronal, or axial slice from a 3D image.
- show_three_views: Display sagittal, coronal, and axial slices side by side.

Classes:
- None.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .image_base import MedicalImage


def resolve_intensity_window(
    array: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float | None = None,
    vmax_percentile: float | None = None,
) -> tuple[float | None, float | None]:
    """Resolve display intensity bounds from explicit values or percentiles.

    Parameters
    ----------
    array : np.ndarray
        Image values used to derive percentile-based bounds when explicit
        ``vmin`` or ``vmax`` are not provided.
    vmin : float | None, optional
        Explicit lower display bound. If provided, percentile settings are
        ignored for the lower bound.
    vmax : float | None, optional
        Explicit upper display bound. If provided, percentile settings are
        ignored for the upper bound.
    vmin_percentile : float | None, optional
        Lower percentile used when ``vmin`` is ``None``.
    vmax_percentile : float | None, optional
        Upper percentile used when ``vmax`` is ``None``.

    Returns
    -------
    tuple[float | None, float | None]
        Resolved ``(vmin, vmax)`` pair suitable for ``matplotlib.imshow``.
    """

    values = np.asarray(array)
    lower = float(vmin) if vmin is not None else None
    upper = float(vmax) if vmax is not None else None

    if lower is None and vmin_percentile is not None:
        lower = float(np.percentile(values, vmin_percentile))
    if upper is None and vmax_percentile is not None:
        upper = float(np.percentile(values, vmax_percentile))
    return lower, upper


def _select_frame(image: MedicalImage, time_index: int | None) -> np.ndarray:
    """Select one spatial frame from a canonical image with optional extra axes.

    Parameters
    ----------
    image : MedicalImage
        Canonical image whose leading axes are spatial.
    time_index : int | None
        Index along the first trailing axis. If ``None`` and a trailing axis is
        present, frame ``0`` is selected.

    Returns
    -------
    np.ndarray
        Spatial-only array containing either a 2D image or a 3D volume.
    """

    if image.array.ndim == image.spatial_ndim:
        return np.asarray(image.array)

    if image.array.ndim != image.spatial_ndim + 1:
        raise NotImplementedError(
            "Display helpers currently support at most one trailing axis such as time."
        )

    frame_count = int(image.array.shape[image.spatial_ndim])
    frame_idx = 0 if time_index is None else int(time_index)
    if frame_idx < 0 or frame_idx >= frame_count:
        raise IndexError(
            f"time_index={frame_idx} is out of bounds for frame_count={frame_count}."
        )
    slicer = [slice(None) for _ in range(image.spatial_ndim)] + [frame_idx]
    return np.asarray(image.array[tuple(slicer)])


def _annotate_orientation_labels(
    ax: Axes,
    labels: dict[str, str],
    fontsize: float,
    color: str,
) -> None:
    """Draw orientation labels around an axes in normalized coordinates.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes that should receive the text annotations.
    labels : dict[str, str]
        Mapping containing ``left``, ``right``, ``top``, and ``bottom`` labels.
    fontsize : float
        Font size used for the orientation text.
    color : str
        Matplotlib-compatible text color used for the orientation labels.

    Returns
    -------
    None
        The labels are drawn on the provided axes.
    """

    ax.text(
        -0.12,
        0.5,
        labels["left"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        clip_on=False,
    )
    ax.text(
        1.12,
        0.5,
        labels["right"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        clip_on=False,
    )
    ax.text(
        0.5,
        1.12,
        labels["top"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        clip_on=False,
    )
    ax.text(
        0.5,
        -0.12,
        labels["bottom"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        clip_on=False,
    )


def extract_slice(
    image: MedicalImage,
    view: str,
    *,
    slice_index: int | None = None,
    time_index: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract and orient a 2D slice from a canonical 3D image for display.

    Parameters
    ----------
    image : MedicalImage
        Canonical 3D image or 3D+t image whose spatial axes are already in LPS+
        orientation.
    view : str
        Slice plane to extract. Supported values are ``"sagittal"``,
        ``"coronal"``, and ``"axial"``.
    slice_index : int | None, optional
        Requested one-based index along the view axis. If ``None``, the central
        slice is used.
    time_index : int | None, optional
        Frame index selected from the trailing time axis when displaying a
        3D+t image. If omitted, frame ``0`` is used.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Display-ready 2D slice array and a metadata dictionary containing the
        used one-based slice index, total slice count, display aspect ratio,
        and orientation labels.
    """

    if image.spatial_ndim != 3:
        raise ValueError("extract_slice only supports images with spatial_ndim == 3.")

    volume = _select_frame(image, time_index)
    spacing = image.spacing

    if view == "sagittal":
        axis = 0
        slice_count = int(volume.shape[axis])
        zero_based_index = (
            slice_count // 2
            if slice_index is None
            else int(np.clip(int(slice_index) - 1, 0, slice_count - 1))
        )
        slice_2d = volume[zero_based_index, :, :].T
        aspect = float(spacing[2] / spacing[1])
        labels = {"left": "A", "right": "P", "top": "S", "bottom": "I"}
    elif view == "coronal":
        axis = 1
        slice_count = int(volume.shape[axis])
        zero_based_index = (
            slice_count // 2
            if slice_index is None
            else int(np.clip(int(slice_index) - 1, 0, slice_count - 1))
        )
        slice_2d = volume[:, zero_based_index, :].T
        aspect = float(spacing[2] / spacing[0])
        labels = {"left": "R", "right": "L", "top": "S", "bottom": "I"}
    elif view == "axial":
        axis = 2
        slice_count = int(volume.shape[axis])
        zero_based_index = (
            slice_count // 2
            if slice_index is None
            else int(np.clip(int(slice_index) - 1, 0, slice_count - 1))
        )
        slice_2d = np.flipud(volume[:, :, zero_based_index].T)
        aspect = float(spacing[1] / spacing[0])
        labels = {"left": "R", "right": "L", "top": "A", "bottom": "P"}
    else:
        raise ValueError(f"Unsupported view={view!r}. Expected sagittal, coronal, or axial.")

    info = {
        "view": view,
        "slice_index": zero_based_index + 1,
        "slice_count": slice_count,
        "aspect": aspect,
        "orientation_labels": labels,
        "time_index": 0 if time_index is None and image.array.ndim > image.spatial_ndim else time_index,
    }
    return slice_2d, info


def show_image_2d(
    image: MedicalImage,
    *,
    time_index: int | None = None,
    cmap: str = "gray",
    figsize: tuple[float, float] = (6.0, 6.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float | None = None,
    vmax_percentile: float | None = None,
    colorbar: bool = False,
    title: str | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
    axis_on: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display a 2D image or a single frame from a 2D+t image.

    Parameters
    ----------
    image : MedicalImage
        Canonical image with ``spatial_ndim == 2``.
    time_index : int | None, optional
        Frame index for a 2D+t image. If omitted, frame ``0`` is used.
    cmap : str, optional
        Matplotlib colormap name applied to the image.
    figsize : tuple[float, float], optional
        Figure size in inches when a new figure is created.
    vmin, vmax : float | None, optional
        Explicit lower and upper intensity bounds.
    vmin_percentile, vmax_percentile : float | None, optional
        Percentile-based lower and upper intensity bounds used when the
        explicit values are omitted.
    colorbar : bool, optional
        Whether to draw a colorbar beside the image.
    title : str | None, optional
        Optional axes title.
    title_fontsize : float, optional
        Font size used for the title.
    label_fontsize : float, optional
        Font size used for axis tick labels.
    axis_on : bool, optional
        Whether to keep axes visible. If ``False``, ticks and frame are hidden.
    ax : Axes | None, optional
        Existing axes to draw into. If omitted, a new figure and axes are
        created.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the rendered image.
    """

    if image.spatial_ndim != 2:
        raise ValueError("show_image_2d only supports images with spatial_ndim == 2.")

    panel = _select_frame(image, time_index).T
    spacing = image.spacing
    aspect = float(spacing[1] / spacing[0])
    vmin_use, vmax_use = resolve_intensity_window(
        panel,
        vmin=vmin,
        vmax=vmax,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(panel, cmap=cmap, origin="lower", aspect=aspect, vmin=vmin_use, vmax=vmax_use)
    ax.tick_params(labelsize=label_fontsize)
    if not axis_on:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if colorbar:
        fig.colorbar(im, ax=ax)
    return fig, ax


def show_slice(
    image: MedicalImage,
    view: str,
    *,
    slice_index: int | None = None,
    time_index: int | None = None,
    cmap: str = "gray",
    figsize: tuple[float, float] = (6.0, 6.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float | None = None,
    vmax_percentile: float | None = None,
    colorbar: bool = False,
    title: str | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
    axis_on: bool = True,
    show_orientation_labels: bool = True,
    orientation_label_color: str = "orange",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display a single sagittal, coronal, or axial slice from a canonical 3D image.

    Parameters
    ----------
    image : MedicalImage
        Canonical 3D image or 3D+t image in LPS+ orientation.
    view : str
        Slice plane to display: ``"sagittal"``, ``"coronal"``, or ``"axial"``.
    slice_index : int | None, optional
        Requested one-based slice index along the selected view axis. If
        omitted, the central slice is used.
    time_index : int | None, optional
        Frame index selected from a trailing time axis. If omitted, frame ``0``
        is used when present.
    cmap : str, optional
        Matplotlib colormap name applied to the image.
    figsize : tuple[float, float], optional
        Figure size in inches when a new figure is created.
    vmin, vmax : float | None, optional
        Explicit lower and upper intensity bounds.
    vmin_percentile, vmax_percentile : float | None, optional
        Percentile-based lower and upper intensity bounds used when explicit
        values are omitted.
    colorbar : bool, optional
        Whether to draw a colorbar beside the image.
    title : str | None, optional
        Optional axes title.
    title_fontsize : float, optional
        Font size used for the title.
    label_fontsize : float, optional
        Font size used for ticks and orientation labels.
    axis_on : bool, optional
        Whether to keep axes visible. If ``False``, ticks and frame are hidden.
    show_orientation_labels : bool, optional
        Whether to draw anatomical orientation labels around the slice.
    orientation_label_color : str, optional
        Matplotlib-compatible text color used for the orientation annotations.
    ax : Axes | None, optional
        Existing axes to draw into. If omitted, a new figure and axes are
        created.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the rendered slice.
    """

    slice_2d, info = extract_slice(
        image,
        view,
        slice_index=slice_index,
        time_index=time_index,
    )
    vmin_use, vmax_use = resolve_intensity_window(
        slice_2d,
        vmin=vmin,
        vmax=vmax,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        slice_2d,
        cmap=cmap,
        origin="lower",
        aspect=float(info["aspect"]),
        vmin=vmin_use,
        vmax=vmax_use,
    )
    ax.tick_params(labelsize=label_fontsize)
    if not axis_on:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    elif title is None:
        ax.set_title(
            f"{view.capitalize()} {info['slice_index']}/{info['slice_count']}",
            fontsize=title_fontsize,
        )

    if show_orientation_labels:
        _annotate_orientation_labels(
            ax,
            labels=info["orientation_labels"],
            fontsize=label_fontsize,
            color=orientation_label_color,
        )
    if colorbar:
        fig.colorbar(im, ax=ax)
    return fig, ax


def show_three_views(
    image: MedicalImage,
    *,
    sagittal_index: int | None = None,
    coronal_index: int | None = None,
    axial_index: int | None = None,
    time_index: int | None = None,
    cmap: str = "gray",
    figsize: tuple[float, float] = (15.0, 5.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float | None = None,
    vmax_percentile: float | None = None,
    colorbar: bool = False,
    title: str | None = None,
    title_fontsize: float = 14.0,
    label_fontsize: float = 12.0,
    axis_on: bool = True,
    show_orientation_labels: bool = True,
    orientation_label_color: str = "orange",
) -> tuple[Figure, np.ndarray]:
    """Display sagittal, coronal, and axial slices from a canonical 3D image.

    Parameters
    ----------
    image : MedicalImage
        Canonical 3D image or 3D+t image in LPS+ orientation.
    sagittal_index, coronal_index, axial_index : int | None, optional
        Optional one-based slice indices for the three orthogonal views. Each
        omitted value defaults to the corresponding central slice.
    time_index : int | None, optional
        Frame index selected from a trailing time axis. If omitted, frame ``0``
        is used when present.
    cmap : str, optional
        Matplotlib colormap name applied to all panels.
    figsize : tuple[float, float], optional
        Figure size in inches.
    vmin, vmax : float | None, optional
        Explicit lower and upper intensity bounds shared across all panels.
    vmin_percentile, vmax_percentile : float | None, optional
        Percentile-based lower and upper intensity bounds computed from the
        selected 3D frame when explicit values are omitted.
    colorbar : bool, optional
        Whether to add one colorbar beside each displayed view.
    title : str | None, optional
        Optional figure-level title.
    title_fontsize : float, optional
        Font size used for subplot titles and the optional figure title.
    label_fontsize : float, optional
        Font size used for ticks and orientation labels.
    axis_on : bool, optional
        Whether to keep axes visible. If ``False``, ticks and frame are hidden.
    show_orientation_labels : bool, optional
        Whether to draw anatomical orientation labels around each panel.
    orientation_label_color : str, optional
        Matplotlib-compatible text color used for the orientation annotations.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and a one-dimensional array of three subplot axes.
    """

    if image.spatial_ndim != 3:
        raise ValueError("show_three_views only supports images with spatial_ndim == 3.")

    volume = _select_frame(image, time_index)
    vmin_use, vmax_use = resolve_intensity_window(
        volume,
        vmin=vmin,
        vmax=vmax,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    view_specs = [
        ("sagittal", sagittal_index),
        ("coronal", coronal_index),
        ("axial", axial_index),
    ]

    last_im = None
    for ax, (view, index) in zip(axes, view_specs):
        slice_2d, info = extract_slice(
            image,
            view,
            slice_index=index,
            time_index=time_index,
        )
        last_im = ax.imshow(
            slice_2d,
            cmap=cmap,
            origin="lower",
            aspect=float(info["aspect"]),
            vmin=vmin_use,
            vmax=vmax_use,
        )
        ax.set_title(
            f"{view.capitalize()} {info['slice_index']}/{info['slice_count']}",
            fontsize=title_fontsize,
        )
        ax.tick_params(labelsize=label_fontsize)
        if not axis_on:
            ax.set_axis_off()
        if show_orientation_labels:
            _annotate_orientation_labels(
                ax,
                labels=info["orientation_labels"],
                fontsize=label_fontsize,
                color=orientation_label_color,
            )
        if colorbar and last_im is not None:
            fig.colorbar(last_im, ax=ax)
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    if title is not None and show_orientation_labels:
        fig.subplots_adjust(top=0.8, wspace=0.5)
    elif title is not None:
        fig.subplots_adjust(top=0.86, wspace=0.35)
    elif show_orientation_labels:
        fig.subplots_adjust(top=0.88, wspace=0.5)
    else:
        fig.subplots_adjust(top=0.92, wspace=0.35)
    return fig, np.asarray(axes)
