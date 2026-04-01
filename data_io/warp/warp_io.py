"""
General description:
    Lightweight file I/O helpers for displacement-field data used by the warp
    utilities. The current implementation focuses on reading NIfTI displacement
    files into numpy arrays without reorientation or convention conversion.

    This module is intentionally minimal for now and can be extended later with
    richer metadata handling, canonical orientation conversion, and writing
    utilities.

Variable / function / class list:
    Variables:
        None

    Functions:
        read_warp_nifti
        read_warp_nifti_with_affine

    Classes:
        None
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def read_warp_nifti(path: str | Path) -> np.ndarray:
    """
    Load a displacement field from a NIfTI file into a numpy array.

    Parameters
    ----------
    path : str | Path
        Filesystem path to a displacement-field NIfTI file such as `.nii` or
        `.nii.gz`.

    Returns
    -------
    np.ndarray
        Displacement array loaded from disk in the exact axis order stored in
        the file. No orientation reformatting, channel reordering, or dtype
        conversion is performed beyond conversion to a numpy array view.
    """
    return np.asarray(nib.load(str(Path(path))).dataobj)


def read_warp_nifti_with_affine(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a displacement field and its affine matrix from a NIfTI file.

    Parameters
    ----------
    path : str | Path
        Filesystem path to a displacement-field NIfTI file such as `.nii` or
        `.nii.gz`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - the displacement array in the exact on-disk axis order, and
        - the corresponding NIfTI affine as a `(4, 4)` numpy array.
    """
    nii = nib.load(str(Path(path)))
    return np.asarray(nii.dataobj), np.asarray(nii.affine)
