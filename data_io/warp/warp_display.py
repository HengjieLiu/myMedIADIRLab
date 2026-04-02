"""Display helpers for canonical displacement fields.

Purpose:
This module renders canonical `MedicalWarp` objects in ways that mirror the
existing image display utilities while remaining specific to displacement
fields. The current implementation focuses on:
    1. scalar field display from a warp, such as magnitude, Jacobian
       determinant, curl magnitude, curl components, and displacement
       components,
    2. backward-grid visualization by warping synthetic 3D grid volumes, and
    3. forward-grid and quiver visualization by projecting in-plane vector
       components into canonical display coordinates.

    All 3D displays assume canonical LPS+ input and use the same slice-view
    conventions as `image_display.py`.

Variables:
- ScalarFieldName: Supported scalar-field selectors derived from a warp.
- GridMode: Supported grid-visualization modes.

Functions:
- show_warp_scalar_slice
- show_warp_scalar_three_views
- show_warp_quiver_slice
- show_warp_quiver_three_views
- show_warp_grid_slice
- show_warp_grid_three_views

Classes:
- None
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from ..image.image_base import MedicalImage
from ..image.image_display import extract_slice, resolve_intensity_window, show_image_2d
from .warp_base import MedicalWarp
from .warp_features import (
    ComponentLike,
    PaddingMode,
    compute_curl_2d,
    compute_curl_component,
    compute_curl_magnitude,
    compute_jacobian_determinant,
    compute_jacobian_fold_mask,
    compute_warp_component,
    compute_warp_magnitude,
)
from .warp_v1_unified import warp_image_unified

ScalarFieldName = Literal[
    "magnitude",
    "component",
    "jacobian",
    "curl_magnitude",
    "curl_component",
    "curl",
]
GridMode = Literal["backward", "forward"]


def _normalize_view_name(view: str) -> Literal["sagittal", "coronal", "axial"]:
    """Normalize supported view aliases to canonical view names.

    Parameters
    ----------
    view : str
        Requested view name. Supported values are `"sagittal"`, `"coronal"`,
        `"axial"` and the short aliases `"sag"`, `"cor"`, `"axi"`.

    Returns
    -------
    Literal["sagittal", "coronal", "axial"]
        Canonical view name.
    """

    aliases = {
        "sag": "sagittal",
        "sagittal": "sagittal",
        "cor": "coronal",
        "coronal": "coronal",
        "axi": "axial",
        "axial": "axial",
    }
    if view not in aliases:
        raise ValueError(
            f"Unsupported view={view!r}. Expected sagittal/coronal/axial or sag/cor/axi."
        )
    return aliases[view]


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
        Matplotlib axes that should receive the orientation annotations.
    labels : dict[str, str]
        Mapping containing `left`, `right`, `top`, and `bottom` labels.
    fontsize : float
        Font size used for the text labels.
    color : str
        Matplotlib-compatible label color.

    Returns
    -------
    None
        The annotations are drawn on the provided axes.
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


def _format_slice_title(
    base_title: str,
    info: dict[str, Any] | None,
    *,
    show_slice_number: bool,
) -> str:
    """Append one-based slice numbering to a default per-view title."""
    if not show_slice_number or info is None:
        return base_title
    return f"{base_title} {info['slice_index']}/{info['slice_count']}"


def _wrap_scalar_array_as_image(array: np.ndarray, warp: MedicalWarp) -> MedicalImage:
    """Wrap a scalar array derived from a warp as a canonical `MedicalImage`.

    Parameters
    ----------
    array : np.ndarray
        Scalar array whose leading spatial axes are aligned with the warp
        lattice.
    warp : MedicalWarp
        Reference canonical warp that provides the affine and spatial
        dimensionality.

    Returns
    -------
    MedicalImage
        Canonical scalar image object that can reuse the image display helpers.
    """

    return MedicalImage(
        array=np.asarray(array, dtype=np.float32),
        affine_lps=warp.affine_lps.copy(),
        spatial_ndim=warp.spatial_ndim,
        source_type="warp_scalar",
        metadata=dict(warp.metadata),
    )


def _make_single_color_cmap(color: str) -> mcolors.ListedColormap:
    """Create a single-color colormap useful for masked grid overlays.

    Parameters
    ----------
    color : str
        Matplotlib-compatible color string.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Single-color colormap.
    """

    return mcolors.ListedColormap([color])


def _make_continuous_color_overlay(
    values: np.ndarray,
    *,
    color: str,
    alpha: float,
    background_value: float,
    peak_value: float,
) -> np.ndarray:
    """Create an RGBA overlay whose opacity follows grid intensity.

    Parameters
    ----------
    values : np.ndarray
        Two-dimensional scalar image whose values encode warped-grid intensity.
    color : str
        Matplotlib-compatible color string used for the grid overlay.
    alpha : float
        Maximum alpha applied to locations whose intensity reaches
        `peak_value`.
    background_value : float
        Scalar value representing the undeformed grid background.
    peak_value : float
        Intensity treated as the fully opaque grid-line value.

    Returns
    -------
    np.ndarray
        RGBA image with shape `(H, W, 4)` and `float32` dtype.
    """

    rgba = np.zeros(values.shape + (4,), dtype=np.float32)
    scale = max(float(peak_value) - float(background_value), np.finfo(np.float32).eps)
    normalized = np.clip(
        (values.astype(np.float32, copy=False) - float(background_value)) / scale,
        0.0,
        1.0,
    )
    rgba[..., :3] = np.asarray(mcolors.to_rgb(color), dtype=np.float32)
    rgba[..., 3] = normalized * float(alpha)
    return rgba


def _apply_axis_style(
    ax: Axes,
    *,
    axis_on: bool,
    show_orientation_labels: bool,
    orientation_labels: dict[str, str] | None,
    label_fontsize: float,
    orientation_label_color: str,
) -> None:
    """Apply common axis and orientation-label styling.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to style.
    axis_on : bool
        Whether axis ticks and frame should remain visible.
    show_orientation_labels : bool
        Whether to draw anatomical orientation labels.
    orientation_labels : dict[str, str] | None
        Orientation-label mapping from the slice metadata.
    label_fontsize : float
        Font size used for tick labels and orientation labels.
    orientation_label_color : str
        Matplotlib-compatible text color for the orientation labels.

    Returns
    -------
    None
        Styling is applied to the provided axes.
    """

    ax.tick_params(labelsize=label_fontsize)
    if not axis_on:
        ax.set_axis_off()
    if show_orientation_labels and orientation_labels is not None:
        _annotate_orientation_labels(
            ax,
            labels=orientation_labels,
            fontsize=label_fontsize,
            color=orientation_label_color,
        )


def _add_axis_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: Any,
    label: str | None = None,
) -> None:
    """Add a colorbar associated with a single axes.

    Parameters
    ----------
    fig : Figure
        Figure that owns the axes.
    ax : Axes
        Axes beside which the colorbar is drawn.
    mappable : Any
        Matplotlib image or quiver object used to generate the colorbar.
    label : str | None, optional
        Optional colorbar label.

    Returns
    -------
    None
        The colorbar is added to the figure.
    """

    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    if label is not None:
        colorbar.set_label(label)


def _extract_component_slice(
    warp: MedicalWarp,
    component_index: int,
    view: str,
    slice_index: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a display-ready slice of one warp component using image conventions.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field.
    component_index : int
        Zero-based displacement component index.
    view : str
        Requested slice view.
    slice_index : int | None, optional
        One-based slice index. If omitted, the central slice is used.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Display-ready 2D panel and slice metadata from `extract_slice`.
    """

    component_image = _wrap_scalar_array_as_image(
        array=warp.array[..., component_index],
        warp=warp,
    )
    return extract_slice(component_image, _normalize_view_name(view), slice_index=slice_index)


def _extract_display_ready_vector_slice(
    warp: MedicalWarp,
    view: str,
    slice_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract in-plane warp components in display coordinates for one 3D view.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 3D displacement field in channel-last layout.
    view : str
        Requested view name or supported alias.
    slice_index : int | None, optional
        One-based slice index. If omitted, the central slice is used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, Any]]
        A tuple containing:
        - horizontal display component,
        - vertical display component, and
        - slice metadata from `extract_slice`.
    """

    if warp.spatial_ndim != 3:
        raise ValueError(
            f"Vector slice extraction currently supports only 3D warps, got spatial_ndim={warp.spatial_ndim}."
        )

    view_name = _normalize_view_name(view)
    comp_x, info = _extract_component_slice(warp, 0, view_name, slice_index=slice_index)
    comp_y, _ = _extract_component_slice(warp, 1, view_name, slice_index=slice_index)
    comp_z, _ = _extract_component_slice(warp, 2, view_name, slice_index=slice_index)

    if view_name == "sagittal":
        u_display = comp_y
        v_display = comp_z
    elif view_name == "coronal":
        u_display = comp_x
        v_display = comp_z
    else:
        u_display = comp_x
        v_display = -comp_y

    return u_display, v_display, info


def _sample_indices(length: int, step: int) -> np.ndarray:
    """Generate regularly spaced indices while ensuring the last index is included.

    Parameters
    ----------
    length : int
        Number of samples along the axis.
    step : int
        Sampling step size. Must be positive.

    Returns
    -------
    np.ndarray
        One-dimensional array of sampled integer indices.
    """

    if step <= 0:
        raise ValueError(f"step must be positive, got {step}.")
    indices = np.arange(0, length, step, dtype=int)
    if indices.size == 0 or indices[-1] != length - 1:
        indices = np.concatenate([indices, np.asarray([length - 1], dtype=int)])
    return np.unique(indices)


def _resolve_scalar_field(
    warp: MedicalWarp,
    field: ScalarFieldName,
    component: ComponentLike | None,
    padding_mode: PaddingMode,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resolve one scalar field derived from a warp and an optional fold mask.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field.
    field : ScalarFieldName
        Requested scalar-field selector.
    component : ComponentLike | None
        Component selector used by `field="component"` or
        `field="curl_component"`.
    padding_mode : Literal["border", "crop"]
        Padding strategy for derivative-based features.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Scalar field and optional Jacobian fold mask.
    """

    if field == "magnitude":
        return compute_warp_magnitude(warp), None
    if field == "component":
        if component is None:
            raise ValueError("component must be provided when field='component'.")
        return compute_warp_component(warp, component), None
    if field == "jacobian":
        return (
            compute_jacobian_determinant(warp, padding_mode=padding_mode),
            compute_jacobian_fold_mask(warp, padding_mode=padding_mode),
        )
    if field == "curl_magnitude":
        return compute_curl_magnitude(warp, padding_mode=padding_mode), None
    if field == "curl_component":
        if component is None:
            raise ValueError("component must be provided when field='curl_component'.")
        if warp.spatial_ndim == 2:
            raise ValueError(
                "curl_component is only defined for 3D warps. Use field='curl' for 2D scalar curl."
            )
        return compute_curl_component(warp, component, padding_mode=padding_mode), None
    if field == "curl":
        if warp.spatial_ndim == 2:
            return compute_curl_2d(warp, padding_mode=padding_mode), None
        raise ValueError(
            "field='curl' is only valid for 2D scalar curl. Use 'curl_component' or "
            "'curl_magnitude' for 3D."
        )

    raise ValueError(f"Unsupported scalar field selector {field!r}.")


def _resolve_scalar_display_params(
    scalar_array: np.ndarray,
    *,
    field: ScalarFieldName,
    fold_mask: np.ndarray | None,
    log_jacobian: bool,
    log_epsilon: float,
    cmap: str | None,
    vmin: float | None,
    vmax: float | None,
    vmin_percentile: float | None,
    vmax_percentile: float | None,
) -> tuple[np.ndarray, str, mcolors.Normalize | None, float | None, float | None]:
    """Resolve display array, colormap, normalization, and value bounds.

    Parameters
    ----------
    scalar_array : np.ndarray
        Scalar field to visualize.
    field : ScalarFieldName
        Requested scalar-field selector.
    fold_mask : np.ndarray | None
        Optional fold mask used only for Jacobian determinant displays.
    log_jacobian : bool
        Whether Jacobian determinant should be log-transformed for display.
    log_epsilon : float
        Positive epsilon used when log-transforming the Jacobian determinant.
    cmap : str | None
        Optional user-specified colormap name.
    vmin, vmax : float | None
        Explicit lower and upper display bounds.
    vmin_percentile, vmax_percentile : float | None
        Percentile-based lower and upper display bounds.

    Returns
    -------
    tuple[np.ndarray, str, matplotlib.colors.Normalize | None, float | None, float | None]
        Display array, colormap name, normalization object, and the resolved
        `vmin` and `vmax`.
    """

    display_array = np.asarray(scalar_array, dtype=np.float32)
    resolved_cmap = cmap or "magma"
    norm: mcolors.Normalize | None = None
    center_value: float | None = None

    if field == "jacobian":
        if log_jacobian:
            display_array = np.log(np.clip(display_array, log_epsilon, None))
            resolved_cmap = cmap or "RdBu_r"
            center_value = 0.0
            if fold_mask is not None:
                valid_values = display_array[~fold_mask]
            else:
                valid_values = display_array
        else:
            resolved_cmap = cmap or "RdBu_r"
            center_value = 1.0
            valid_values = display_array
    elif field in ("component", "curl_component", "curl"):
        resolved_cmap = cmap or "coolwarm"
        center_value = 0.0
        valid_values = display_array
    else:
        valid_values = display_array

    if vmin is None and vmin_percentile is None:
        vmin_percentile = 1.0
    if vmax is None and vmax_percentile is None:
        vmax_percentile = 99.0

    if valid_values.size == 0:
        valid_values = display_array

    resolved_vmin, resolved_vmax = resolve_intensity_window(
        valid_values,
        vmin=vmin,
        vmax=vmax,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
    )

    if (
        center_value is not None
        and resolved_vmin is not None
        and resolved_vmax is not None
        and resolved_vmin < center_value < resolved_vmax
    ):
        norm = mcolors.TwoSlopeNorm(
            vmin=resolved_vmin,
            vcenter=center_value,
            vmax=resolved_vmax,
        )

    return display_array, resolved_cmap, norm, resolved_vmin, resolved_vmax


def _auto_scalar_title(
    field: ScalarFieldName,
    component: ComponentLike | None,
    log_jacobian: bool,
) -> str:
    """Build a readable default title for a scalar display.

    Parameters
    ----------
    field : ScalarFieldName
        Requested scalar-field selector.
    component : ComponentLike | None
        Optional component selector.
    log_jacobian : bool
        Whether log-Jacobian mode is enabled.

    Returns
    -------
    str
        Default title for the display.
    """

    if field == "magnitude":
        return "Warp Magnitude"
    if field == "component":
        return f"Displacement Component {component}"
    if field == "jacobian":
        return "Log Jacobian Determinant" if log_jacobian else "Jacobian Determinant"
    if field == "curl_magnitude":
        return "Curl Magnitude"
    if field == "curl_component":
        return f"Curl Component {component}"
    if field == "curl":
        return "Curl"
    return "Warp Scalar Field"


def _build_planar_grid_volume(
    spatial_shape: tuple[int, int, int],
    view: str,
    grid_step: int,
    line_thickness: int = 1,
    line_value: float = 1.0,
    background_value: float = 0.0,
) -> np.ndarray:
    """Create a 3D volume containing a repeated planar grid pattern for one view.

    Parameters
    ----------
    spatial_shape : tuple[int, int, int]
        Spatial shape `(X, Y, Z)` of the canonical warp lattice.
    view : str
        Requested view name or alias. The grid pattern is created in the plane
        visible in that view and repeated along the slicing axis.
    grid_step : int
        Grid spacing in voxels.
    line_thickness : int, optional
        Grid line thickness in voxels.
    line_value : float, optional
        Intensity value assigned to grid lines.
    background_value : float, optional
        Intensity value assigned to the background.

    Returns
    -------
    np.ndarray
        3D grid volume ready to be deformed with backward warping.
    """

    view_name = _normalize_view_name(view)
    grid_volume = np.full(spatial_shape, fill_value=background_value, dtype=np.float32)

    if grid_step <= 0:
        raise ValueError(f"grid_step must be positive, got {grid_step}.")
    if line_thickness <= 0:
        raise ValueError(f"line_thickness must be positive, got {line_thickness}.")

    if view_name == "sagittal":
        in_plane_axes = (1, 2)
    elif view_name == "coronal":
        in_plane_axes = (0, 2)
    else:
        in_plane_axes = (0, 1)

    for axis in in_plane_axes:
        for start in range(0, spatial_shape[axis], grid_step):
            end = min(start + line_thickness, spatial_shape[axis])
            slicer = [slice(None), slice(None), slice(None)]
            slicer[axis] = slice(start, end)
            grid_volume[tuple(slicer)] = line_value

    return grid_volume


def _render_background_slice(
    ax: Axes,
    background_image: MedicalImage | None,
    view: str,
    slice_index: int | None,
    cmap: str,
    alpha: float,
    vmin: float | None,
    vmax: float | None,
    vmin_percentile: float | None,
    vmax_percentile: float | None,
) -> dict[str, Any] | None:
    """Render an optional background slice and return its slice metadata.

    Parameters
    ----------
    ax : Axes
        Target axes used for background rendering.
    background_image : MedicalImage | None
        Optional canonical image used as a grayscale background.
    view : str
        Requested view name or alias.
    slice_index : int | None
        One-based slice index. If omitted, the central slice is used.
    cmap : str
        Colormap used for the background image.
    alpha : float
        Alpha used for the background image.
    vmin, vmax : float | None
        Explicit display bounds for the background image.
    vmin_percentile, vmax_percentile : float | None
        Percentile-based bounds for the background image.

    Returns
    -------
    dict[str, Any] | None
        Slice metadata from `extract_slice` if a background image was drawn,
        otherwise `None`.
    """

    if background_image is None:
        return None

    panel, info = extract_slice(
        background_image,
        _normalize_view_name(view),
        slice_index=slice_index,
    )
    vmin_use, vmax_use = resolve_intensity_window(
        background_image.array,
        vmin=vmin,
        vmax=vmax,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
    )
    ax.imshow(
        panel,
        cmap=cmap,
        origin="lower",
        aspect=float(info["aspect"]),
        vmin=vmin_use,
        vmax=vmax_use,
        alpha=alpha,
    )
    return info


def show_warp_scalar_slice(
    warp: MedicalWarp,
    field: ScalarFieldName,
    *,
    view: str | None = None,
    slice_index: int | None = None,
    component: ComponentLike | None = None,
    padding_mode: PaddingMode = "border",
    log_jacobian: bool = False,
    log_epsilon: float = 1e-6,
    cmap: str | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float | None = None,
    vmax_percentile: float | None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    title: str | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
    axis_on: bool = True,
    show_slice_number: bool = True,
    show_orientation_labels: bool = True,
    orientation_label_color: str = "orange",
    fold_contours: bool = False,
    fold_contour_color: str = "black",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display one scalar field derived from a canonical warp on a single slice.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical displacement field to visualize.
    field : ScalarFieldName
        Requested scalar field selector.
    view : str | None, optional
        Requested view for 3D warps. Supported values are sagittal/coronal/axial
        and sag/cor/axi. For 2D warps this parameter is ignored.
    slice_index : int | None, optional
        One-based slice index for 3D warps. If omitted, the central slice is
        used.
    component : ComponentLike | None, optional
        Component selector used by `field="component"` or
        `field="curl_component"`.
    padding_mode : Literal["border", "crop"], optional
        Padding strategy for derivative-based fields.
    log_jacobian : bool, optional
        Whether to display the Jacobian determinant in log space. This applies
        only when `field="jacobian"`.
    log_epsilon : float, optional
        Positive epsilon used when log-transforming the Jacobian determinant.
    cmap : str | None, optional
        Optional matplotlib colormap name.
    figsize : tuple[float, float], optional
        Figure size used when a new figure is created.
    vmin, vmax : float | None, optional
        Explicit display bounds.
    vmin_percentile, vmax_percentile : float | None, optional
        Percentile-based display bounds used when explicit bounds are omitted.
    colorbar : bool, optional
        Whether to draw a colorbar beside the slice.
    colorbar_label : str | None, optional
        Optional label used for the colorbar. If omitted, the displayed title is
        reused.
    title : str | None, optional
        Optional axes title.
    title_fontsize : float, optional
        Font size used for the title.
    label_fontsize : float, optional
        Font size used for ticks and orientation labels.
    axis_on : bool, optional
        Whether to keep axes visible.
    show_slice_number : bool, optional
        Whether to append `i/N` slice numbering to the default title for 3D
        displays. Custom titles are left unchanged.
    show_orientation_labels : bool, optional
        Whether to draw anatomical orientation labels for 3D displays.
    orientation_label_color : str, optional
        Matplotlib-compatible color for orientation labels.
    fold_contours : bool, optional
        Whether to overlay fold contours (`Jacobian determinant <= 0`) for
        Jacobian displays.
    fold_contour_color : str, optional
        Contour color used for the fold overlay.
    ax : Axes | None, optional
        Optional target axes. If omitted, a new figure and axes are created.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the rendered scalar field.
    """

    scalar_array, fold_mask = _resolve_scalar_field(
        warp=warp,
        field=field,
        component=component,
        padding_mode=padding_mode,
    )
    display_array, cmap_use, norm, resolved_vmin, resolved_vmax = _resolve_scalar_display_params(
        scalar_array,
        field=field,
        fold_mask=fold_mask,
        log_jacobian=log_jacobian,
        log_epsilon=log_epsilon,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
    )

    scalar_image = _wrap_scalar_array_as_image(display_array, warp)
    title_use = title or _auto_scalar_title(field, component, log_jacobian)

    if warp.spatial_ndim == 2:
        fig, ax_out = show_image_2d(
            scalar_image,
            cmap=cmap_use,
            figsize=figsize,
            vmin=resolved_vmin,
            vmax=resolved_vmax,
            colorbar=colorbar,
            title=title_use,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            axis_on=axis_on,
            ax=ax,
        )
        return fig, ax_out

    if view is None:
        raise ValueError("view must be provided for 3D warp scalar displays.")

    panel, info = extract_slice(
        scalar_image,
        _normalize_view_name(view),
        slice_index=slice_index,
    )

    fold_panel: np.ndarray | None = None
    if fold_mask is not None and field == "jacobian" and fold_contours:
        fold_image = _wrap_scalar_array_as_image(fold_mask.astype(np.float32), warp)
        fold_panel, _ = extract_slice(
            fold_image,
            _normalize_view_name(view),
            slice_index=slice_index,
        )

    if ax is None:
        fig, ax_out = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
        ax_out = ax

    image_artist = ax_out.imshow(
        panel,
        cmap=cmap_use,
        origin="lower",
        aspect=float(info["aspect"]),
        vmin=None if norm is not None else resolved_vmin,
        vmax=None if norm is not None else resolved_vmax,
        norm=norm,
    )
    ax_out.set_title(
        title_use
        if title is not None
        else _format_slice_title(title_use, info, show_slice_number=show_slice_number),
        fontsize=title_fontsize,
    )
    _apply_axis_style(
        ax_out,
        axis_on=axis_on,
        show_orientation_labels=show_orientation_labels,
        orientation_labels=info["orientation_labels"],
        label_fontsize=label_fontsize,
        orientation_label_color=orientation_label_color,
    )

    if fold_panel is not None and np.any(fold_panel > 0.5):
        ax_out.contour(
            fold_panel.astype(np.float32),
            levels=[0.5],
            colors=fold_contour_color,
            linewidths=0.75,
            origin="lower",
        )

    if colorbar:
        _add_axis_colorbar(
            fig,
            ax_out,
            image_artist,
            label=colorbar_label or title_use,
        )

    return fig, ax_out


def show_warp_scalar_three_views(
    warp: MedicalWarp,
    field: ScalarFieldName,
    *,
    sagittal_index: int | None = None,
    coronal_index: int | None = None,
    axial_index: int | None = None,
    component: ComponentLike | None = None,
    padding_mode: PaddingMode = "border",
    log_jacobian: bool = False,
    log_epsilon: float = 1e-6,
    cmap: str | None = None,
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
    show_slice_number: bool = True,
    show_orientation_labels: bool = True,
    orientation_label_color: str = "orange",
    fold_contours: bool = False,
    fold_contour_color: str = "black",
) -> tuple[Figure, np.ndarray]:
    """Display a scalar warp-derived field in sagittal, coronal, and axial views.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 3D displacement field to visualize.
    field : ScalarFieldName
        Requested scalar field selector.
    sagittal_index, coronal_index, axial_index : int | None, optional
        One-based slice indices for the three views. If omitted, the central
        slice is used for each view.
    component : ComponentLike | None, optional
        Component selector used by `field="component"` or
        `field="curl_component"`.
    padding_mode : Literal["border", "crop"], optional
        Padding strategy for derivative-based fields.
    log_jacobian : bool, optional
        Whether to display the Jacobian determinant in log space.
    log_epsilon : float, optional
        Positive epsilon used when log-transforming the Jacobian determinant.
    cmap : str | None, optional
        Optional matplotlib colormap name.
    figsize : tuple[float, float], optional
        Figure size in inches.
    vmin, vmax : float | None, optional
        Explicit display bounds shared across all views.
    vmin_percentile, vmax_percentile : float | None, optional
        Percentile-based display bounds used when explicit bounds are omitted.
    colorbar : bool, optional
        Whether to draw one colorbar per axes.
    title : str | None, optional
        Optional figure-level title.
    title_fontsize : float, optional
        Font size used for axes titles and the optional figure title.
    label_fontsize : float, optional
        Font size used for ticks and orientation labels.
    axis_on : bool, optional
        Whether to keep axes visible.
    show_slice_number : bool, optional
        Whether to append `i/N` slice numbering to the default per-view titles.
    show_orientation_labels : bool, optional
        Whether to draw anatomical orientation labels.
    orientation_label_color : str, optional
        Matplotlib-compatible text color for orientation labels.
    fold_contours : bool, optional
        Whether to overlay fold contours for Jacobian displays.
    fold_contour_color : str, optional
        Contour color used for the fold overlay.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and a one-dimensional array of three axes.
    """

    if warp.spatial_ndim != 3:
        raise ValueError("show_warp_scalar_three_views only supports 3D warps.")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colorbar_label = _auto_scalar_title(field, component, log_jacobian)
    view_specs = [
        ("sagittal", sagittal_index),
        ("coronal", coronal_index),
        ("axial", axial_index),
    ]

    for ax, (view_name, view_index) in zip(axes, view_specs):
        show_warp_scalar_slice(
            warp=warp,
            field=field,
            view=view_name,
            slice_index=view_index,
            component=component,
            padding_mode=padding_mode,
            log_jacobian=log_jacobian,
            log_epsilon=log_epsilon,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            vmin_percentile=vmin_percentile,
            vmax_percentile=vmax_percentile,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            title=None,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            axis_on=axis_on,
            show_slice_number=show_slice_number,
            show_orientation_labels=show_orientation_labels,
            orientation_label_color=orientation_label_color,
            fold_contours=fold_contours,
            fold_contour_color=fold_contour_color,
            ax=ax,
        )

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    if title is not None and show_orientation_labels:
        fig.subplots_adjust(top=0.82, wspace=0.5)
    elif title is not None:
        fig.subplots_adjust(top=0.88, wspace=0.35)
    elif show_orientation_labels:
        fig.subplots_adjust(top=0.92, wspace=0.5)
    else:
        fig.subplots_adjust(top=0.96, wspace=0.35)

    return fig, np.asarray(axes)


def show_warp_quiver_slice(
    warp: MedicalWarp,
    *,
    view: str,
    slice_index: int | None = None,
    background_image: MedicalImage | None = None,
    background_cmap: str = "gray",
    background_alpha: float = 1.0,
    background_vmin: float | None = None,
    background_vmax: float | None = None,
    background_vmin_percentile: float | None = 1.0,
    background_vmax_percentile: float | None = 99.0,
    step: int = 8,
    scale: float | None = None,
    width: float | None = None,
    color: str = "red",
    color_by_magnitude: bool = False,
    cmap: str = "RdYlGn_r",
    colorbar: bool = False,
    figsize: tuple[float, float] = (6.0, 6.0),
    title: str | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
    axis_on: bool = True,
    show_slice_number: bool = True,
    show_orientation_labels: bool = True,
    orientation_label_color: str = "orange",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display an in-plane quiver plot for one slice of a canonical 3D warp.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 3D displacement field to visualize.
    view : str
        Requested view name or alias.
    slice_index : int | None, optional
        One-based slice index. If omitted, the central slice is used.
    background_image : MedicalImage | None, optional
        Optional canonical image used as a background underlay.
    background_cmap : str, optional
        Colormap used for the optional background image.
    background_alpha : float, optional
        Alpha used for the background image.
    background_vmin, background_vmax : float | None, optional
        Explicit display bounds for the background image.
    background_vmin_percentile, background_vmax_percentile : float | None, optional
        Percentile-based display bounds for the background image.
    step : int, optional
        Spatial downsampling step applied before plotting quiver arrows.
    scale : float | None, optional
        Matplotlib quiver scale parameter.
    width : float | None, optional
        Optional quiver shaft width.
    color : str, optional
        Arrow color when `color_by_magnitude` is `False`.
    color_by_magnitude : bool, optional
        Whether to color-code arrows by in-plane magnitude.
    cmap : str, optional
        Colormap used when `color_by_magnitude` is enabled.
    colorbar : bool, optional
        Whether to draw a colorbar for magnitude-coded quiver arrows.
    figsize : tuple[float, float], optional
        Figure size used when a new figure is created.
    title : str | None, optional
        Optional axes title.
    title_fontsize : float, optional
        Font size used for the title.
    label_fontsize : float, optional
        Font size used for ticks and orientation labels.
    axis_on : bool, optional
        Whether to keep axes visible.
    show_slice_number : bool, optional
        Whether to append `i/N` slice numbering to the default title. Custom
        titles are left unchanged.
    show_orientation_labels : bool, optional
        Whether to draw anatomical orientation labels.
    orientation_label_color : str, optional
        Matplotlib-compatible text color for orientation labels.
    ax : Axes | None, optional
        Optional target axes. If omitted, a new figure and axes are created.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the quiver plot.
    """

    if warp.spatial_ndim != 3:
        raise ValueError("show_warp_quiver_slice currently supports only 3D warps.")

    view_name = _normalize_view_name(view)
    u_display, v_display, info = _extract_display_ready_vector_slice(
        warp=warp,
        view=view_name,
        slice_index=slice_index,
    )
    rows, cols = u_display.shape
    row_indices = _sample_indices(rows, step)
    col_indices = _sample_indices(cols, step)

    y_grid, x_grid = np.meshgrid(row_indices, col_indices, indexing="ij")
    u_sub = u_display[np.ix_(row_indices, col_indices)]
    v_sub = v_display[np.ix_(row_indices, col_indices)]

    if ax is None:
        fig, ax_out = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
        ax_out = ax

    _render_background_slice(
        ax=ax_out,
        background_image=background_image,
        view=view_name,
        slice_index=slice_index,
        cmap=background_cmap,
        alpha=background_alpha,
        vmin=background_vmin,
        vmax=background_vmax,
        vmin_percentile=background_vmin_percentile,
        vmax_percentile=background_vmax_percentile,
    )

    quiver_kwargs: dict[str, Any] = {
        "angles": "xy",
        "scale_units": "xy",
        "scale": scale,
    }
    if width is not None:
        quiver_kwargs["width"] = width

    if color_by_magnitude:
        magnitude = np.sqrt(u_sub ** 2 + v_sub ** 2)
        quiver_artist = ax_out.quiver(
            x_grid,
            y_grid,
            u_sub,
            v_sub,
            magnitude,
            cmap=cmap,
            **quiver_kwargs,
        )
        if colorbar:
            _add_axis_colorbar(fig, ax_out, quiver_artist, label="In-plane Magnitude")
    else:
        quiver_artist = ax_out.quiver(
            x_grid,
            y_grid,
            u_sub,
            v_sub,
            color=color,
            **quiver_kwargs,
        )

    ax_out.set_aspect(float(info["aspect"]))
    default_title = _format_slice_title(
        f"Quiver | {view_name.capitalize()}",
        info,
        show_slice_number=show_slice_number,
    )
    ax_out.set_title(title or default_title, fontsize=title_fontsize)
    _apply_axis_style(
        ax_out,
        axis_on=axis_on,
        show_orientation_labels=show_orientation_labels,
        orientation_labels=info["orientation_labels"],
        label_fontsize=label_fontsize,
        orientation_label_color=orientation_label_color,
    )

    return fig, ax_out


def show_warp_quiver_three_views(
    warp: MedicalWarp,
    *,
    sagittal_index: int | None = None,
    coronal_index: int | None = None,
    axial_index: int | None = None,
    background_image: MedicalImage | None = None,
    background_cmap: str = "gray",
    background_alpha: float = 1.0,
    background_vmin: float | None = None,
    background_vmax: float | None = None,
    background_vmin_percentile: float | None = 1.0,
    background_vmax_percentile: float | None = 99.0,
    step: int = 8,
    scale: float | None = None,
    width: float | None = None,
    color: str = "red",
    color_by_magnitude: bool = False,
    cmap: str = "RdYlGn_r",
    colorbar: bool = False,
    figsize: tuple[float, float] = (15.0, 5.0),
    title: str | None = None,
    title_fontsize: float = 14.0,
    label_fontsize: float = 12.0,
    axis_on: bool = True,
    show_slice_number: bool = True,
    show_orientation_labels: bool = True,
    orientation_label_color: str = "orange",
) -> tuple[Figure, np.ndarray]:
    """Display quiver plots for sagittal, coronal, and axial warp slices.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 3D displacement field to visualize.
    sagittal_index, coronal_index, axial_index : int | None, optional
        One-based slice indices for the three views.
    background_image : MedicalImage | None, optional
        Optional canonical image used as a background underlay.
    background_cmap : str, optional
        Colormap used for the background image.
    background_alpha : float, optional
        Alpha used for the background image.
    background_vmin, background_vmax : float | None, optional
        Explicit display bounds for the background image.
    background_vmin_percentile, background_vmax_percentile : float | None, optional
        Percentile-based display bounds for the background image.
    step : int, optional
        Spatial downsampling step used for quiver arrows.
    scale : float | None, optional
        Matplotlib quiver scale parameter.
    width : float | None, optional
        Optional quiver shaft width.
    color : str, optional
        Arrow color when `color_by_magnitude` is `False`.
    color_by_magnitude : bool, optional
        Whether to color-code arrows by in-plane magnitude.
    cmap : str, optional
        Colormap used when `color_by_magnitude` is enabled.
    colorbar : bool, optional
        Whether to draw one colorbar per axes for magnitude-coded arrows.
    figsize : tuple[float, float], optional
        Figure size in inches.
    title : str | None, optional
        Optional figure-level title.
    title_fontsize : float, optional
        Font size used for axes titles and the optional figure title.
    label_fontsize : float, optional
        Font size used for ticks and orientation labels.
    axis_on : bool, optional
        Whether to keep axes visible.
    show_slice_number : bool, optional
        Whether to append `i/N` slice numbering to the default per-view titles.
    show_orientation_labels : bool, optional
        Whether to draw anatomical orientation labels.
    orientation_label_color : str, optional
        Matplotlib-compatible text color for orientation labels.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and a one-dimensional array of three axes.
    """

    if warp.spatial_ndim != 3:
        raise ValueError("show_warp_quiver_three_views only supports 3D warps.")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (view_name, view_index) in zip(
        axes,
        [("sagittal", sagittal_index), ("coronal", coronal_index), ("axial", axial_index)],
    ):
        show_warp_quiver_slice(
            warp=warp,
            view=view_name,
            slice_index=view_index,
            background_image=background_image,
            background_cmap=background_cmap,
            background_alpha=background_alpha,
            background_vmin=background_vmin,
            background_vmax=background_vmax,
            background_vmin_percentile=background_vmin_percentile,
            background_vmax_percentile=background_vmax_percentile,
            step=step,
            scale=scale,
            width=width,
            color=color,
            color_by_magnitude=color_by_magnitude,
            cmap=cmap,
            colorbar=colorbar,
            title=None,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            axis_on=axis_on,
            show_slice_number=show_slice_number,
            show_orientation_labels=show_orientation_labels,
            orientation_label_color=orientation_label_color,
            ax=ax,
        )

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    if title is not None and show_orientation_labels:
        fig.subplots_adjust(top=0.82, wspace=0.5)
    elif title is not None:
        fig.subplots_adjust(top=0.88, wspace=0.35)
    elif show_orientation_labels:
        fig.subplots_adjust(top=0.92, wspace=0.5)
    else:
        fig.subplots_adjust(top=0.96, wspace=0.35)

    return fig, np.asarray(axes)


def _draw_forward_grid(
    ax: Axes,
    u_display: np.ndarray,
    v_display: np.ndarray,
    *,
    aspect: float,
    grid_step: int,
    line_width: float,
    line_color: str,
    line_alpha: float,
) -> None:
    """Draw a forward-mapped grid from display-ready in-plane displacements.

    Parameters
    ----------
    ax : Axes
        Target matplotlib axes.
    u_display : np.ndarray
        Horizontal in-plane displacement component in display coordinates.
    v_display : np.ndarray
        Vertical in-plane displacement component in display coordinates.
    aspect : float
        Display aspect ratio for the current view.
    grid_step : int
        Grid spacing in displayed pixels.
    line_width : float
        Line width used for the grid.
    line_color : str
        Grid line color.
    line_alpha : float
        Grid line alpha.

    Returns
    -------
    None
        The line collections are drawn on the provided axes.
    """

    rows, cols = u_display.shape
    row_indices = _sample_indices(rows, grid_step)
    col_indices = _sample_indices(cols, grid_step)

    y_grid, x_grid = np.meshgrid(row_indices, col_indices, indexing="ij")
    u_sub = u_display[np.ix_(row_indices, col_indices)]
    v_sub = v_display[np.ix_(row_indices, col_indices)]

    x_deformed = x_grid + u_sub
    y_deformed = y_grid + v_sub

    points = np.stack((x_deformed, y_deformed), axis=-1)
    horiz_lines = points
    vert_lines = points.transpose(1, 0, 2)

    ax.add_collection(
        LineCollection(horiz_lines, colors=line_color, linewidths=line_width, alpha=line_alpha)
    )
    ax.add_collection(
        LineCollection(vert_lines, colors=line_color, linewidths=line_width, alpha=line_alpha)
    )
    ax.set_xlim(0, cols - 1)
    ax.set_ylim(0, rows - 1)
    ax.set_aspect(aspect)


def _render_backward_grid_slice(
    warp: MedicalWarp,
    *,
    view: str,
    slice_index: int | None,
    grid_step: int,
    line_thickness: int,
    line_value: float,
    background_value: float,
    device: str | int | None,
    background_image: MedicalImage | None,
    background_cmap: str,
    background_alpha: float,
    background_vmin: float | None,
    background_vmax: float | None,
    background_vmin_percentile: float | None,
    background_vmax_percentile: float | None,
    grid_cmap: str | None,
    grid_color: str | None,
    grid_alpha: float,
    grid_threshold: float | None,
    ax: Axes,
) -> dict[str, Any]:
    """Render one backward-grid slice and return its slice metadata.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 3D displacement field.
    view : str
        Requested view name or alias.
    slice_index : int | None
        One-based slice index.
    grid_step : int
        Grid spacing in voxels.
    line_thickness : int
        Grid line thickness in voxels.
    line_value : float
        Grid line intensity before deformation.
    background_value : float
        Background intensity before deformation.
    device : str | int | None
        Device selector passed to `warp_image_unified`.
    background_image : MedicalImage | None
        Optional background image.
    background_cmap : str
        Background colormap.
    background_alpha : float
        Background alpha.
    background_vmin, background_vmax : float | None
        Explicit background intensity bounds.
    background_vmin_percentile, background_vmax_percentile : float | None
        Percentile-based background bounds.
    grid_cmap : str | None
        Colormap used for the grid image when `grid_color` is not provided.
    grid_color : str | None
        Optional grid color. When `grid_threshold is None`, the color is used
        with intensity-modulated opacity. When `grid_threshold` is a float, the
        color is used for the masked threshold overlay.
    grid_alpha : float
        Grid alpha.
    grid_threshold : float | None
        If `None`, render the continuous warped-grid intensities without
        thresholding. If a float, values at or below the threshold are masked
        before display.
    ax : Axes
        Target axes.

    Returns
    -------
    dict[str, Any]
        Slice metadata from `extract_slice`.
    """

    view_name = _normalize_view_name(view)
    grid_volume = _build_planar_grid_volume(
        spatial_shape=warp.spatial_shape,  # type: ignore[arg-type]
        view=view_name,
        grid_step=grid_step,
        line_thickness=line_thickness,
        line_value=line_value,
        background_value=background_value,
    )
    warped_grid = warp_image_unified(
        image=grid_volume,
        disp=warp.array.astype(np.float32, copy=False),
        backend="torch",
        order=1,
        padding_mode="constant",
        cval=background_value,
        device=device,
        debug=False,
    )

    warped_grid_image = _wrap_scalar_array_as_image(warped_grid, warp)
    grid_panel, info = extract_slice(
        warped_grid_image,
        view_name,
        slice_index=slice_index,
    )

    _render_background_slice(
        ax=ax,
        background_image=background_image,
        view=view_name,
        slice_index=slice_index,
        cmap=background_cmap,
        alpha=background_alpha,
        vmin=background_vmin,
        vmax=background_vmax,
        vmin_percentile=background_vmin_percentile,
        vmax_percentile=background_vmax_percentile,
    )

    if background_image is None and grid_threshold is None and grid_color is None:
        ax.imshow(
            grid_panel,
            cmap=grid_cmap or "gray",
            origin="lower",
            aspect=float(info["aspect"]),
            vmin=background_value,
            vmax=line_value,
            alpha=grid_alpha,
            interpolation="nearest",
        )
    else:
        if grid_threshold is None:
            if grid_color is None:
                masked_grid = np.ma.masked_where(
                    np.isclose(grid_panel, background_value),
                    grid_panel,
                )
                ax.imshow(
                    masked_grid,
                    cmap=grid_cmap or "gray",
                    origin="lower",
                    aspect=float(info["aspect"]),
                    vmin=background_value,
                    vmax=line_value,
                    alpha=grid_alpha,
                    interpolation="nearest",
                )
            else:
                if background_image is None:
                    ax.set_facecolor("black")
                rgba_overlay = _make_continuous_color_overlay(
                    grid_panel,
                    color=grid_color,
                    alpha=grid_alpha,
                    background_value=background_value,
                    peak_value=line_value,
                )
                ax.imshow(
                    rgba_overlay,
                    origin="lower",
                    aspect=float(info["aspect"]),
                    interpolation="nearest",
                )
        else:
            masked_grid = np.ma.masked_where(grid_panel <= grid_threshold, grid_panel)
            if background_image is None:
                ax.set_facecolor("black")
            ax.imshow(
                masked_grid,
                cmap=_make_single_color_cmap(grid_color or "white")
                if grid_color is not None or grid_cmap is None
                else grid_cmap,
                origin="lower",
                aspect=float(info["aspect"]),
                alpha=grid_alpha,
                interpolation="nearest",
            )

    ax.set_aspect(float(info["aspect"]))
    return info


def show_warp_grid_slice(
    warp: MedicalWarp,
    *,
    mode: GridMode,
    view: str,
    slice_index: int | None = None,
    grid_step: int = 8,
    line_thickness: int = 1,
    line_value: float = 1.0,
    background_value: float = 0.0,
    background_image: MedicalImage | None = None,
    background_cmap: str = "gray",
    background_alpha: float = 1.0,
    background_vmin: float | None = None,
    background_vmax: float | None = None,
    background_vmin_percentile: float | None = 1.0,
    background_vmax_percentile: float | None = 99.0,
    grid_cmap: str | None = None,
    grid_color: str | None = "white",
    grid_alpha: float = 1.0,
    grid_threshold: float | None = None,
    line_width: float = 1.0,
    device: str | int | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    title: str | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
    axis_on: bool = True,
    show_slice_number: bool = True,
    show_orientation_labels: bool = True,
    orientation_label_color: str = "orange",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display one backward or forward grid slice for a canonical 3D warp.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 3D displacement field to visualize.
    mode : Literal["backward", "forward"]
        Grid visualization mode.
    view : str
        Requested view name or alias.
    slice_index : int | None, optional
        One-based slice index. If omitted, the central slice is used.
    grid_step : int, optional
        Grid spacing in voxels or displayed pixels.
    line_thickness : int, optional
        Grid line thickness for backward-grid rendering.
    line_value : float, optional
        Grid line intensity used by backward-grid rendering.
    background_value : float, optional
        Background intensity used by backward-grid rendering.
    background_image : MedicalImage | None, optional
        Optional canonical image used as a background underlay.
    background_cmap : str, optional
        Colormap used for the background image.
    background_alpha : float, optional
        Alpha used for the background image.
    background_vmin, background_vmax : float | None, optional
        Explicit display bounds for the background image.
    background_vmin_percentile, background_vmax_percentile : float | None, optional
        Percentile-based display bounds for the background image.
    grid_cmap : str | None, optional
        Optional colormap used for the backward grid image when no single grid
        color is requested.
    grid_color : str | None, optional
        Single grid color used for forward-grid lines and optional backward-grid
        rendering. In continuous backward mode (`grid_threshold=None`), the
        warped-grid intensity modulates the overlay opacity of this color.
    grid_alpha : float, optional
        Grid alpha.
    grid_threshold : float | None, optional
        Backward-grid overlay threshold. Use `None` to render the continuous
        warped-grid intensities without thresholding, which is the default and
        best matches the legacy notebook behavior. Provide a float to mask
        values at or below the threshold and obtain a cleaner binary-style
        overlay.
    line_width : float, optional
        Forward-grid line width.
    device : str | int | None, optional
        Device selector used by the backward-grid deformation path.
    figsize : tuple[float, float], optional
        Figure size used when a new figure is created.
    title : str | None, optional
        Optional axes title.
    title_fontsize : float, optional
        Font size used for the title.
    label_fontsize : float, optional
        Font size used for ticks and orientation labels.
    axis_on : bool, optional
        Whether to keep axes visible.
    show_slice_number : bool, optional
        Whether to append `i/N` slice numbering to the default title. Custom
        titles are left unchanged.
    show_orientation_labels : bool, optional
        Whether to draw anatomical orientation labels.
    orientation_label_color : str, optional
        Matplotlib-compatible color for orientation labels.
    ax : Axes | None, optional
        Optional target axes.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the grid visualization.
    """

    if warp.spatial_ndim != 3:
        raise ValueError("show_warp_grid_slice currently supports only 3D warps.")

    view_name = _normalize_view_name(view)

    if ax is None:
        fig, ax_out = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
        ax_out = ax

    if mode == "backward":
        info = _render_backward_grid_slice(
            warp=warp,
            view=view_name,
            slice_index=slice_index,
            grid_step=grid_step,
            line_thickness=line_thickness,
            line_value=line_value,
            background_value=background_value,
            device=device,
            background_image=background_image,
            background_cmap=background_cmap,
            background_alpha=background_alpha,
            background_vmin=background_vmin,
            background_vmax=background_vmax,
            background_vmin_percentile=background_vmin_percentile,
            background_vmax_percentile=background_vmax_percentile,
            grid_cmap=grid_cmap,
            grid_color=grid_color,
            grid_alpha=grid_alpha,
            grid_threshold=grid_threshold,
            ax=ax_out,
        )
    elif mode == "forward":
        info = _render_background_slice(
            ax=ax_out,
            background_image=background_image,
            view=view_name,
            slice_index=slice_index,
            cmap=background_cmap,
            alpha=background_alpha,
            vmin=background_vmin,
            vmax=background_vmax,
            vmin_percentile=background_vmin_percentile,
            vmax_percentile=background_vmax_percentile,
        )
        u_display, v_display, vector_info = _extract_display_ready_vector_slice(
            warp=warp,
            view=view_name,
            slice_index=slice_index,
        )
        info = vector_info if info is None else info
        _draw_forward_grid(
            ax=ax_out,
            u_display=u_display,
            v_display=v_display,
            aspect=float(vector_info["aspect"]),
            grid_step=grid_step,
            line_width=line_width,
            line_color=grid_color or "white",
            line_alpha=grid_alpha,
        )
    else:
        raise ValueError(f"mode must be 'backward' or 'forward', got {mode!r}.")

    default_title = _format_slice_title(
        f"{mode.capitalize()} Grid | {view_name.capitalize()}",
        info,
        show_slice_number=show_slice_number,
    )
    ax_out.set_title(title or default_title, fontsize=title_fontsize)
    _apply_axis_style(
        ax_out,
        axis_on=axis_on,
        show_orientation_labels=show_orientation_labels,
        orientation_labels=info["orientation_labels"],
        label_fontsize=label_fontsize,
        orientation_label_color=orientation_label_color,
    )

    return fig, ax_out


def show_warp_grid_three_views(
    warp: MedicalWarp,
    *,
    mode: GridMode,
    sagittal_index: int | None = None,
    coronal_index: int | None = None,
    axial_index: int | None = None,
    grid_step: int = 8,
    line_thickness: int = 1,
    line_value: float = 1.0,
    background_value: float = 0.0,
    background_image: MedicalImage | None = None,
    background_cmap: str = "gray",
    background_alpha: float = 1.0,
    background_vmin: float | None = None,
    background_vmax: float | None = None,
    background_vmin_percentile: float | None = 1.0,
    background_vmax_percentile: float | None = 99.0,
    grid_cmap: str | None = None,
    grid_color: str | None = "white",
    grid_alpha: float = 1.0,
    grid_threshold: float | None = None,
    line_width: float = 1.0,
    device: str | int | None = None,
    figsize: tuple[float, float] = (15.0, 5.0),
    title: str | None = None,
    title_fontsize: float = 14.0,
    label_fontsize: float = 12.0,
    axis_on: bool = True,
    show_slice_number: bool = True,
    show_orientation_labels: bool = True,
    orientation_label_color: str = "orange",
) -> tuple[Figure, np.ndarray]:
    """Display backward or forward grid visualizations in all three orthogonal views.

    Parameters
    ----------
    warp : MedicalWarp
        Canonical 3D displacement field to visualize.
    mode : Literal["backward", "forward"]
        Grid visualization mode.
    sagittal_index, coronal_index, axial_index : int | None, optional
        One-based slice indices for the three views.
    grid_step : int, optional
        Grid spacing in voxels or displayed pixels.
    line_thickness : int, optional
        Grid line thickness for backward-grid rendering.
    line_value : float, optional
        Grid line intensity used by backward-grid rendering.
    background_value : float, optional
        Background intensity used by backward-grid rendering.
    background_image : MedicalImage | None, optional
        Optional canonical image used as a background underlay.
    background_cmap : str, optional
        Colormap used for the background image.
    background_alpha : float, optional
        Alpha used for the background image.
    background_vmin, background_vmax : float | None, optional
        Explicit display bounds for the background image.
    background_vmin_percentile, background_vmax_percentile : float | None, optional
        Percentile-based display bounds for the background image.
    grid_cmap : str | None, optional
        Optional colormap used for backward-grid rendering when no single grid
        color is requested.
    grid_color : str | None, optional
        Single grid color used for forward-grid lines and optional backward-grid
        rendering. In continuous backward mode (`grid_threshold=None`), the
        warped-grid intensity modulates the overlay opacity of this color.
    grid_alpha : float, optional
        Grid alpha.
    grid_threshold : float | None, optional
        Backward-grid overlay threshold. Use `None` to render the continuous
        warped-grid intensities without thresholding, which is the default and
        best matches the legacy notebook behavior. Provide a float to mask
        values at or below the threshold and obtain a cleaner binary-style
        overlay.
    line_width : float, optional
        Forward-grid line width.
    device : str | int | None, optional
        Device selector used by the backward-grid deformation path.
    figsize : tuple[float, float], optional
        Figure size in inches.
    title : str | None, optional
        Optional figure-level title.
    title_fontsize : float, optional
        Font size used for axes titles and the optional figure title.
    label_fontsize : float, optional
        Font size used for ticks and orientation labels.
    axis_on : bool, optional
        Whether to keep axes visible.
    show_slice_number : bool, optional
        Whether to append `i/N` slice numbering to the default per-view titles.
    show_orientation_labels : bool, optional
        Whether to draw anatomical orientation labels.
    orientation_label_color : str, optional
        Matplotlib-compatible text color for orientation labels.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and a one-dimensional array of three axes.
    """

    if warp.spatial_ndim != 3:
        raise ValueError("show_warp_grid_three_views only supports 3D warps.")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (view_name, view_index) in zip(
        axes,
        [("sagittal", sagittal_index), ("coronal", coronal_index), ("axial", axial_index)],
    ):
        show_warp_grid_slice(
            warp=warp,
            mode=mode,
            view=view_name,
            slice_index=view_index,
            grid_step=grid_step,
            line_thickness=line_thickness,
            line_value=line_value,
            background_value=background_value,
            background_image=background_image,
            background_cmap=background_cmap,
            background_alpha=background_alpha,
            background_vmin=background_vmin,
            background_vmax=background_vmax,
            background_vmin_percentile=background_vmin_percentile,
            background_vmax_percentile=background_vmax_percentile,
            grid_cmap=grid_cmap,
            grid_color=grid_color,
            grid_alpha=grid_alpha,
            grid_threshold=grid_threshold,
            line_width=line_width,
            device=device,
            title=None,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            axis_on=axis_on,
            show_slice_number=show_slice_number,
            show_orientation_labels=show_orientation_labels,
            orientation_label_color=orientation_label_color,
            ax=ax,
        )

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    if title is not None and show_orientation_labels:
        fig.subplots_adjust(top=0.82, wspace=0.5)
    elif title is not None:
        fig.subplots_adjust(top=0.88, wspace=0.35)
    elif show_orientation_labels:
        fig.subplots_adjust(top=0.92, wspace=0.5)
    else:
        fig.subplots_adjust(top=0.96, wspace=0.35)

    return fig, np.asarray(axes)
