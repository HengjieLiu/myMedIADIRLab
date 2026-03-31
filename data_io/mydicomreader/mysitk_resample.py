"""
SITK resampling functions.

List of functions:
- resample_to_reference
- resample_to_isotropic

"""


import SimpleITK as sitk


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    *,
    transform: sitk.Transform | None = None,
    interpolator: int | None = None,
    default_value: float = 0.0,
    output_pixel_type: int | None = None,
) -> sitk.Image:
    """
    Resample `moving` onto the grid of `reference`.

    Parameters
    ----------
    moving : sitk.Image
        Image to be resampled (e.g., RTDOSE).
    reference : sitk.Image
        Target grid (e.g., CT/MR DICOM image) providing spacing/origin/direction/size.
    transform : sitk.Transform | None
        Physical transform mapping reference -> moving. For same patient space, use identity.
        If you already registered images, pass that transform here.
    interpolator : int | None
        sitk interpolator. If None, auto-choose:
          - sitkLinear for scalar images (dose, CT, MR)
          - sitkNearestNeighbor for labels (mask)
    default_value : float
        Value for samples outside moving image support (dose outside grid -> 0).
    output_pixel_type : int | None
        Output pixel type. If None, keep moving's pixel type.

    Returns
    -------
    sitk.Image
        moving resampled onto reference grid.
    """
    if transform is None:
        transform = sitk.Transform(3, sitk.sitkIdentity)

    if interpolator is None:
        # Heuristic: use NN for integer-ish label images, linear otherwise.
        # RTDOSE is scalar -> linear.
        is_label_like = moving.GetPixelID() in (
            sitk.sitkUInt8, sitk.sitkInt8,
            sitk.sitkUInt16, sitk.sitkInt16,
            sitk.sitkUInt32, sitk.sitkInt32,
            sitk.sitkUInt64, sitk.sitkInt64,
        )
        interpolator = sitk.sitkNearestNeighbor if is_label_like else sitk.sitkLinear

    if output_pixel_type is None:
        output_pixel_type = moving.GetPixelID()

    return sitk.Resample(
        moving,
        reference,              # defines output grid
        transform,              # maps output (reference) physical points into moving
        interpolator,
        default_value,
        output_pixel_type,
    )



def resample_to_isotropic(
    image: sitk.Image,
    *,
    new_spacing: float = 1.0,
    interpolator: int | None = None,
    default_value: float = 0.0,
    output_pixel_type: int | None = None,
) -> sitk.Image:
    """
    Resample `image` to a 1mm isotropic resolution (or other specified spacing)
    while maintaining the same physical Field of View (FOV), origin, and direction.

    Parameters
    ----------
    image : sitk.Image
        The input image to be resampled.
    new_spacing : float
        The desired isotropic spacing in mm. Default is 1.0.
    interpolator : int | None
        sitk interpolator. If None, auto-choose:
          - sitkLinear for scalar images
          - sitkNearestNeighbor for label/integer images
    default_value : float
        Value for samples outside the original image support.
    output_pixel_type : int | None
        Output pixel type. If None, keep the input image's pixel type.

    Returns
    -------
    sitk.Image
        The resampled image with isotropic spacing.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size to maintain the same physical FOV
    # New Size = (Old Size * Old Spacing) / New Spacing
    new_size = [
        int(round(old_sz * old_spc / new_spacing))
        for old_sz, old_spc in zip(original_size, original_spacing)
    ]

    if interpolator is None:
        # Heuristic: use NN for integer-ish label images, linear otherwise.
        is_label_like = image.GetPixelID() in (
            sitk.sitkUInt8, sitk.sitkInt8,
            sitk.sitkUInt16, sitk.sitkInt16,
            sitk.sitkUInt32, sitk.sitkInt32,
            sitk.sitkUInt64, sitk.sitkInt64,
        )
        interpolator = sitk.sitkNearestNeighbor if is_label_like else sitk.sitkLinear

    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()

    # Create the resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([new_spacing] * image.GetDimension())
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform(image.GetDimension(), sitk.sitkIdentity))
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(default_value)
    resample.SetOutputPixelType(output_pixel_type)

    return resample.Execute(image)
