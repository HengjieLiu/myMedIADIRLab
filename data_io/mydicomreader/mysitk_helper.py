import json
from pathlib import Path
from typing import Dict, Any
import SimpleITK as sitk
import numpy as np


def write_json(p: Path, data: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def format_geometry(img: sitk.Image, prefix: str = "") -> str:
    """
    Format image geometry information as a string.
    
    Args:
        img: SimpleITK image
        prefix: Optional prefix string to prepend to the output
        
    Returns:
        str: Formatted geometry string
    """
    sz = "x".join(map(str, img.GetSize()))
    sp = "x".join(f"{v:.4g}" for v in img.GetSpacing())
    ori = tuple(round(v, 6) for v in img.GetOrigin())
    direction = ", ".join(f"{v:.3f}" for v in img.GetDirection())
    
    if prefix:
        return f"{prefix}: size={sz}, spacing={sp}, origin={ori}, dir=[{direction}]"
    else:
        return f"size={sz}, spacing={sp}, origin={ori}, dir=[{direction}]"


def image_geometry_dict(img: sitk.Image, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    d = {
        "size": list(img.GetSize()),
        "spacing": [float(v) for v in img.GetSpacing()],
        "origin": [float(v) for v in img.GetOrigin()],
        "direction": [float(v) for v in img.GetDirection()],
    }
    if extra:
        d.update(extra)
    return d


def save_image_with_geometry(img: sitk.Image, out_nifti: Path, meta_path: Path, meta_extra: Dict[str, Any] = None):
    out_nifti.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_nifti))
    write_json(meta_path, image_geometry_dict(img, meta_extra or {}))


def save_image_with_meta(img: sitk.Image, out_nifti: Path, meta_path: Path, meta_extra: Dict[str, Any] = None):
    """
    Save SimpleITK image to NIfTI file and write metadata JSON.
    
    This is an alias for save_image_with_geometry with a more descriptive name.
    
    Args:
        img: SimpleITK image to save
        out_nifti: Output path for NIfTI file
        meta_path: Output path for metadata JSON file
        meta_extra: Additional metadata to include in JSON
    """
    save_image_with_geometry(img, out_nifti, meta_path, meta_extra)


def calculate_volume(img: sitk.Image, threshold: float = 0.0) -> float:
    """
    Calculate the volume of non-zero voxels in a SimpleITK image in mm^3.
    
    This function is typically used for binary masks or segmented regions.
    The volume is calculated by counting voxels above the threshold and 
    multiplying by the voxel volume (spacing[0] * spacing[1] * spacing[2]).
    
    Args:
        img: SimpleITK image (typically a binary mask)
        threshold: Threshold value for determining which voxels to count.
                   Voxels with value > threshold are counted. Default: 0.0
                   For binary masks (0/1), threshold=0.0 counts all non-zero voxels.
                   For continuous masks, specify an appropriate threshold.
    
    Returns:
        float: Volume in mm^3
        
    Example:
        >>> mask = sitk.ReadImage("mask.nii.gz")
        >>> volume_mm3 = calculate_volume(mask)
        >>> print(f"Volume: {volume_mm3:.2f} mm^3")
        
        >>> # For continuous masks, use a specific threshold
        >>> volume = calculate_volume(probability_map, threshold=0.5)
    """
    # Get image array
    arr = sitk.GetArrayFromImage(img)
    
    # Count voxels above threshold
    voxel_count = np.sum(arr > threshold)
    
    # Get voxel spacing (in mm)
    spacing = img.GetSpacing()
    
    # Calculate voxel volume in mm^3
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    # Calculate total volume
    volume_mm3 = voxel_count * voxel_volume_mm3
    
    return float(volume_mm3)


def calculate_dose_statistics(img: sitk.Image) -> Dict[str, float]:
    """
    Calculate dose statistics (min, mean, max) from a SimpleITK dose image.
    
    This function extracts the dose array from a SimpleITK image and calculates
    basic statistics. Typically used for RTDOSE images where values are in Gy.
    
    Args:
        img: SimpleITK image containing dose values (typically in Gy)
    
    Returns:
        dict: Dictionary with keys 'min', 'mean', 'max' containing dose statistics
        
    Example:
        >>> dose_img = sitk.ReadImage("dose.nii.gz")
        >>> stats = calculate_dose_statistics(dose_img)
        >>> print(f"Min: {stats['min']:.3f} Gy, Mean: {stats['mean']:.3f} Gy, Max: {stats['max']:.3f} Gy")
    """
    # Get dose array from image
    dose_array = sitk.GetArrayFromImage(img)
    
    # Calculate statistics (handling NaN values)
    dmin = float(np.nanmin(dose_array))
    dmean = float(np.nanmean(dose_array))
    dmax = float(np.nanmax(dose_array))
    
    return {
        'min': dmin,
        'mean': dmean,
        'max': dmax
    }


def format_dose_statistics(stats: Dict[str, float], unit: str = "Gy") -> str:
    """
    Format dose statistics as a string for logging.
    
    Args:
        stats: Dictionary with 'min', 'mean', 'max' keys from calculate_dose_statistics()
        unit: Unit string to append (default: "Gy")
        
    Returns:
        str: Formatted string like "min=0.000 Gy, mean=12.345 Gy, max=45.678 Gy"
    """
    return f"min={stats['min']:.3f} {unit}, mean={stats['mean']:.3f} {unit}, max={stats['max']:.3f} {unit}"


