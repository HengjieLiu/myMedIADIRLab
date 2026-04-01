"""Core image container for canonical medical image data.

Purpose:
This module defines the shared in-memory image object used by the image IO,
orientation, and display modules. The object stores voxel data in a
spatial-first layout and keeps a 4x4 affine matrix in canonical LPS+
coordinates.

Variables:
- SUPPORTED_SPATIAL_NDIMS: Supported numbers of spatial dimensions.

Functions:
- infer_axis_labels: Build default axis labels for a spatial-first array.

Classes:
- MedicalImage: Canonical image container with validation and geometry helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SUPPORTED_SPATIAL_NDIMS: tuple[int, ...] = (2, 3)


def infer_axis_labels(spatial_ndim: int, array_ndim: int) -> tuple[str, ...]:
    """Infer default axis labels for a spatial-first image array.

    Parameters
    ----------
    spatial_ndim : int
        Number of spatial axes in the image. Supported values are 2 and 3.
    array_ndim : int
        Total number of array axes, including spatial axes and trailing
        non-spatial axes such as time.

    Returns
    -------
    tuple[str, ...]
        Axis labels whose length matches ``array_ndim``. Spatial axes are
        labeled as ``i``, ``j``, ``k`` and the first trailing extra axis is
        labeled as ``t``. Additional trailing axes are labeled as ``extra_N``.
    """

    if spatial_ndim not in SUPPORTED_SPATIAL_NDIMS:
        raise ValueError(
            f"Unsupported spatial_ndim={spatial_ndim}. "
            f"Expected one of {SUPPORTED_SPATIAL_NDIMS}."
        )
    if array_ndim < spatial_ndim:
        raise ValueError(
            f"array_ndim={array_ndim} must be at least spatial_ndim={spatial_ndim}."
        )

    labels: list[str] = ["i", "j"]
    if spatial_ndim == 3:
        labels.append("k")

    extra_count = array_ndim - spatial_ndim
    if extra_count >= 1:
        labels.append("t")
    for extra_idx in range(1, extra_count):
        labels.append(f"extra_{extra_idx}")

    return tuple(labels)


@dataclass(slots=True)
class MedicalImage:
    """Store canonical image data and its LPS+ voxel-to-world transform.

    Parameters
    ----------
    array : np.ndarray
        Spatial-first image data. Supported layouts are ``(i, j)``,
        ``(i, j, k)``, and spatial-first arrays with trailing non-spatial axes
        such as ``(i, j, k, t)``.
    affine_lps : np.ndarray
        Homogeneous 4x4 voxel-to-world affine matrix in canonical LPS+
        coordinates. The first spatial columns define the image basis vectors
        in millimeters.
    spatial_ndim : int
        Number of spatial axes in ``array``. Supported values are 2 and 3.
    axis_labels : tuple[str, ...] | None, optional
        Optional axis labels aligned with ``array``. If omitted, default labels
        are inferred from ``spatial_ndim`` and ``array.ndim``.
    source_type : str | None, optional
        Optional source identifier such as ``"nifti"`` or ``"sitk"``.
    metadata : dict[str, Any], optional
        Additional source metadata used for debugging or round-trip export.

    Returns
    -------
    None
        The class constructor initializes a validated in-memory image object.
    """

    array: np.ndarray
    affine_lps: np.ndarray
    spatial_ndim: int
    axis_labels: tuple[str, ...] | None = None
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the image object and normalize its core fields.

        Parameters
        ----------
        None
            This method uses the dataclass fields already assigned.

        Returns
        -------
        None
            The image object is validated in place.
        """

        self.array = np.asarray(self.array)
        self.affine_lps = np.asarray(self.affine_lps, dtype=np.float64)

        if self.spatial_ndim not in SUPPORTED_SPATIAL_NDIMS:
            raise ValueError(
                f"Unsupported spatial_ndim={self.spatial_ndim}. "
                f"Expected one of {SUPPORTED_SPATIAL_NDIMS}."
            )
        if self.array.ndim < self.spatial_ndim:
            raise ValueError(
                f"array.ndim={self.array.ndim} must be at least "
                f"spatial_ndim={self.spatial_ndim}."
            )
        if self.affine_lps.shape != (4, 4):
            raise ValueError(
                f"affine_lps must have shape (4, 4), got {self.affine_lps.shape}."
            )
        if self.axis_labels is None:
            self.axis_labels = infer_axis_labels(
                spatial_ndim=self.spatial_ndim,
                array_ndim=self.array.ndim,
            )
        if len(self.axis_labels) != self.array.ndim:
            raise ValueError(
                f"axis_labels length={len(self.axis_labels)} must match "
                f"array.ndim={self.array.ndim}."
            )
        if not np.isfinite(self.affine_lps).all():
            raise ValueError("affine_lps must contain only finite values.")

    @property
    def ndim(self) -> int:
        """Return the total number of array dimensions.

        Parameters
        ----------
        None
            The value is derived from ``self.array``.

        Returns
        -------
        int
            Total number of array axes, including spatial and trailing
            non-spatial axes.
        """

        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full array shape.

        Parameters
        ----------
        None
            The value is derived from ``self.array``.

        Returns
        -------
        tuple[int, ...]
            Complete array shape.
        """

        return tuple(int(v) for v in self.array.shape)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the shape of the spatial axes.

        Parameters
        ----------
        None
            The value is derived from the leading axes of ``self.array``.

        Returns
        -------
        tuple[int, ...]
            Shape of the leading spatial axes only.
        """

        return tuple(int(v) for v in self.array.shape[: self.spatial_ndim])

    @property
    def extra_shape(self) -> tuple[int, ...]:
        """Return the shape of trailing non-spatial axes.

        Parameters
        ----------
        None
            The value is derived from the trailing axes of ``self.array``.

        Returns
        -------
        tuple[int, ...]
            Shape of trailing non-spatial axes such as time.
        """

        return tuple(int(v) for v in self.array.shape[self.spatial_ndim :])

    @property
    def spacing(self) -> np.ndarray:
        """Return spatial voxel sizes derived from the affine basis vectors.

        Parameters
        ----------
        None
            The value is derived from the first spatial columns of
            ``self.affine_lps``.

        Returns
        -------
        np.ndarray
            One-dimensional array of length ``spatial_ndim`` containing the
            physical size of each spatial voxel axis in millimeters.
        """

        basis = self.affine_lps[:3, : self.spatial_ndim]
        return np.linalg.norm(basis, axis=0)

    def copy(self, *, array: np.ndarray | None = None, affine_lps: np.ndarray | None = None) -> "MedicalImage":
        """Create a shallow metadata-preserving copy of the image object.

        Parameters
        ----------
        array : np.ndarray | None, optional
            Optional replacement voxel data. If ``None``, the existing array is
            reused.
        affine_lps : np.ndarray | None, optional
            Optional replacement 4x4 LPS affine. If ``None``, the existing
            affine is reused.

        Returns
        -------
        MedicalImage
            A new image object with copied metadata and either the original or
            provided array and affine.
        """

        return MedicalImage(
            array=self.array if array is None else array,
            affine_lps=self.affine_lps if affine_lps is None else affine_lps,
            spatial_ndim=self.spatial_ndim,
            axis_labels=tuple(self.axis_labels),
            source_type=self.source_type,
            metadata=dict(self.metadata),
        )
