from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
import SimpleITK as sitk


@dataclass
class SitkCompareResult:
    ok: bool
    summary: str
    details: Dict[str, Any]


def compare_sitk_images(
    img_a: sitk.Image,
    img_b: sitk.Image,
    *,
    name_a: str = "img_a",
    name_b: str = "img_b",
    check_pixel_type: bool = True,
    compare_array: bool = True,
    array_rtol: float = 0.0,
    array_atol: float = 0.0,
    geom_atol: float = 1e-6,
    early_exit: bool = False,
) -> SitkCompareResult:
    """
    Compare two SimpleITK images for geometry + (optionally) voxel array equality.

    Parameters
    ----------
    check_pixel_type : bool
        If True, also require same SimpleITK pixel ID.
    compare_array : bool
        If True, compare voxel values via numpy arrays.
    array_rtol, array_atol : float
        Tolerances for numpy allclose when comparing arrays.
        Use nonzero tolerances for float images.
    geom_atol : float
        Absolute tolerance for spacing/origin/direction comparisons.
    early_exit : bool
        If True, stop at first mismatch (faster for big images).

    Returns
    -------
    SitkCompareResult
        .ok indicates whether everything matched.
        .summary is a human-readable report.
        .details contains per-field comparisons.
    """
    def _tuple_close(t1: Tuple[float, ...], t2: Tuple[float, ...], atol: float) -> bool:
        a = np.asarray(t1, dtype=np.float64)
        b = np.asarray(t2, dtype=np.float64)
        return np.allclose(a, b, rtol=0.0, atol=atol)

    details: Dict[str, Any] = {}
    mismatches = []

    # ---- Size ----
    size_a = img_a.GetSize()
    size_b = img_b.GetSize()
    size_ok = (size_a == size_b)
    details["size"] = {"match": size_ok, name_a: size_a, name_b: size_b}
    if not size_ok:
        mismatches.append("size")
        if early_exit:
            return SitkCompareResult(False, _format_summary(name_a, name_b, details, mismatches), details)

    # ---- Spacing ----
    spacing_a = img_a.GetSpacing()
    spacing_b = img_b.GetSpacing()
    spacing_ok = _tuple_close(spacing_a, spacing_b, geom_atol)
    details["spacing"] = {"match": spacing_ok, name_a: spacing_a, name_b: spacing_b, "atol": geom_atol}
    if not spacing_ok:
        mismatches.append("spacing")
        if early_exit:
            return SitkCompareResult(False, _format_summary(name_a, name_b, details, mismatches), details)

    # ---- Origin ----
    origin_a = img_a.GetOrigin()
    origin_b = img_b.GetOrigin()
    origin_ok = _tuple_close(origin_a, origin_b, geom_atol)
    details["origin"] = {"match": origin_ok, name_a: origin_a, name_b: origin_b, "atol": geom_atol}
    if not origin_ok:
        mismatches.append("origin")
        if early_exit:
            return SitkCompareResult(False, _format_summary(name_a, name_b, details, mismatches), details)

    # ---- Direction ----
    direction_a = img_a.GetDirection()
    direction_b = img_b.GetDirection()
    direction_ok = _tuple_close(direction_a, direction_b, geom_atol)
    details["direction"] = {"match": direction_ok, name_a: direction_a, name_b: direction_b, "atol": geom_atol}
    if not direction_ok:
        mismatches.append("direction")
        if early_exit:
            return SitkCompareResult(False, _format_summary(name_a, name_b, details, mismatches), details)

    # ---- Pixel type ----
    if check_pixel_type:
        pida = img_a.GetPixelIDValue()
        pidb = img_b.GetPixelIDValue()
        pid_ok = (pida == pidb)
        details["pixel_id"] = {"match": pid_ok, name_a: pida, name_b: pidb}
        if not pid_ok:
            mismatches.append("pixel_id")
            if early_exit:
                return SitkCompareResult(False, _format_summary(name_a, name_b, details, mismatches), details)

    # ---- Array compare ----
    if compare_array:
        if not size_ok:
            # cannot compare voxel arrays reliably if shapes differ
            details["array"] = {"match": False, "reason": "size mismatch; array compare skipped"}
            mismatches.append("array")
        else:
            arr_a = sitk.GetArrayViewFromImage(img_a)  # (z,y,x) view
            arr_b = sitk.GetArrayViewFromImage(img_b)
            # exact match (fast-ish) or tolerance-based
            if array_rtol == 0.0 and array_atol == 0.0:
                array_ok = np.array_equal(arr_a, arr_b)
            else:
                array_ok = np.allclose(arr_a, arr_b, rtol=array_rtol, atol=array_atol)

            # If mismatch, compute a small diagnostic (min/max diff, first mismatch index)
            diag: Dict[str, Any] = {"match": bool(array_ok), "rtol": array_rtol, "atol": array_atol}
            if not array_ok:
                diff = arr_a.astype(np.float64) - arr_b.astype(np.float64)
                diag["diff_min"] = float(np.min(diff))
                diag["diff_max"] = float(np.max(diff))
                diag["diff_mean"] = float(np.mean(diff))
                # find one index to help debug
                if array_rtol == 0.0 and array_atol == 0.0:
                    bad = np.argwhere(arr_a != arr_b)
                else:
                    bad = np.argwhere(~np.isclose(arr_a, arr_b, rtol=array_rtol, atol=array_atol))
                if bad.size > 0:
                    z, y, x = map(int, bad[0])
                    diag["first_mismatch_zyx"] = (z, y, x)
                    diag[f"{name_a}_value"] = float(arr_a[z, y, x])
                    diag[f"{name_b}_value"] = float(arr_b[z, y, x])

                mismatches.append("array")

            details["array"] = diag

    ok = (len(mismatches) == 0)
    summary = _format_summary(name_a, name_b, details, mismatches)
    return SitkCompareResult(ok, summary, details)


def _format_summary(name_a: str, name_b: str, details: Dict[str, Any], mismatches: list) -> str:
    lines = []
    if not mismatches:
        lines.append(f"✅ {name_a} and {name_b} match (size/spacing/origin/direction"
                     f"{', pixel type' if 'pixel_id' in details else ''}"
                     f"{', array' if 'array' in details else ''}).")
        return "\n".join(lines)

    lines.append(f"❌ {name_a} and {name_b} do NOT match.")
    lines.append("Mismatched fields: " + ", ".join(mismatches))
    for k in mismatches:
        info = details.get(k, {})
        # keep it readable
        if k in ("size", "spacing", "origin", "direction", "pixel_id"):
            lines.append(f"- {k}: {info.get('match')} | {name_a}={info.get(name_a)} | {name_b}={info.get(name_b)}")
        elif k == "array":
            if "reason" in info:
                lines.append(f"- array: {info['reason']}")
            else:
                extra = []
                if "diff_min" in info:
                    extra.append(f"diff[min,max,mean]=({info['diff_min']:.3g}, {info['diff_max']:.3g}, {info['diff_mean']:.3g})")
                if "first_mismatch_zyx" in info:
                    extra.append(f"first_mismatch_zyx={info['first_mismatch_zyx']}")
                lines.append(f"- array: False | " + " | ".join(extra))
    return "\n".join(lines)
