"""Core warp container for canonical displacement-field data.

Purpose:
This module defines the shared in-memory warp object used by the warp IO and
orientation utilities. The object stores displacement fields in a canonical
channel-last layout and keeps a 4x4 affine matrix in canonical LPS+
coordinates.

Variables:
- SUPPORTED_SPATIAL_NDIMS: Supported numbers of spatial dimensions.

Functions:
- infer_warp_axis_labels: Build default axis labels for a canonical warp array.

Classes:
- MedicalWarp: Canonical displacement-field container with validation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

SUPPORTED_SPATIAL_NDIMS: tuple[int, ...] = (2, 3)


def infer_warp_axis_labels(spatial_ndim: int) -> tuple[str, ...]:
    """Infer default axis labels for a canonical channel-last warp array.

    Parameters
    ----------
    spatial_ndim : int
        Number of spatial axes in the displacement field. Supported values are
        `2` and `3`.

    Returns
    -------
    tuple[str, ...]
        Axis labels for the canonical warp array. For 2D this returns
        `("i", "j", "vector")`. For 3D this returns
        `("i", "j", "k", "vector")`.
    """
    if spatial_ndim not in SUPPORTED_SPATIAL_NDIMS:
        raise ValueError(
            f"Unsupported spatial_ndim={spatial_ndim}. "
            f"Expected one of {SUPPORTED_SPATIAL_NDIMS}."
        )

    labels: list[str] = ["i", "j"]
    if spatial_ndim == 3:
        labels.append("k")
    labels.append("vector")
    return tuple(labels)


@dataclass(slots=True)
class MedicalWarp:
    """Store canonical displacement-field data and its LPS+ voxel-to-world transform.

    Parameters
    ----------
    array : np.ndarray
        Canonical channel-last displacement field. Supported layouts are
        `(X, Y, 2)` for 2D and `(X, Y, Z, 3)` for 3D.
    affine_lps : np.ndarray
        Homogeneous 4x4 voxel-to-world affine matrix in canonical LPS+
        coordinates for the displacement lattice.
    spatial_ndim : int
        Number of spatial axes in `array`. Supported values are `2` and `3`.
    axis_labels : tuple[str, ...] | None, optional
        Optional axis labels aligned with `array`. If omitted, default labels
        are inferred from `spatial_ndim`.
    units : Literal["voxel", "world_mm"], optional
        Units used by the displacement vector components. The canonical warp IO
        currently supports reorientation only for `units="voxel"`.
    source_type : str | None, optional
        Optional source identifier such as `"nifti"`.
    metadata : dict[str, Any], optional
        Additional source metadata used for debugging or round-trip export.
    """

    array: np.ndarray
    affine_lps: np.ndarray
    spatial_ndim: int
    axis_labels: tuple[str, ...] | None = None
    units: Literal["voxel", "world_mm"] = "voxel"
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the warp object and normalize its core fields."""
        self.array = np.asarray(self.array)
        self.affine_lps = np.asarray(self.affine_lps, dtype=np.float64)

        if self.spatial_ndim not in SUPPORTED_SPATIAL_NDIMS:
            raise ValueError(
                f"Unsupported spatial_ndim={self.spatial_ndim}. "
                f"Expected one of {SUPPORTED_SPATIAL_NDIMS}."
            )
        if self.array.ndim != self.spatial_ndim + 1:
            raise ValueError(
                f"Canonical warp array must have ndim={self.spatial_ndim + 1}, "
                f"got {self.array.ndim}."
            )
        if self.array.shape[-1] != self.spatial_ndim:
            raise ValueError(
                f"Canonical warp vector axis must have length {self.spatial_ndim}, "
                f"got {self.array.shape[-1]}."
            )
        if self.affine_lps.shape != (4, 4):
            raise ValueError(
                f"affine_lps must have shape (4, 4), got {self.affine_lps.shape}."
            )
        if self.axis_labels is None:
            self.axis_labels = infer_warp_axis_labels(self.spatial_ndim)
        if len(self.axis_labels) != self.array.ndim:
            raise ValueError(
                f"axis_labels length={len(self.axis_labels)} must match "
                f"array.ndim={self.array.ndim}."
            )
        if not np.isfinite(self.affine_lps).all():
            raise ValueError("affine_lps must contain only finite values.")
        if self.units not in ("voxel", "world_mm"):
            raise ValueError(
                f"units must be 'voxel' or 'world_mm', got {self.units!r}."
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full array shape."""
        return tuple(int(v) for v in self.array.shape)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the shape of the leading spatial axes."""
        return tuple(int(v) for v in self.array.shape[: self.spatial_ndim])

    @property
    def spacing(self) -> np.ndarray:
        """Return spatial voxel sizes derived from the affine basis vectors."""
        basis = self.affine_lps[:3, : self.spatial_ndim]
        return np.linalg.norm(basis, axis=0)

    def copy(
        self,
        *,
        array: np.ndarray | None = None,
        affine_lps: np.ndarray | None = None,
    ) -> "MedicalWarp":
        """Create a shallow metadata-preserving copy of the warp object."""
        return MedicalWarp(
            array=self.array if array is None else array,
            affine_lps=self.affine_lps if affine_lps is None else affine_lps,
            spatial_ndim=self.spatial_ndim,
            axis_labels=self.axis_labels,
            units=self.units,
            source_type=self.source_type,
            metadata=dict(self.metadata),
        )
