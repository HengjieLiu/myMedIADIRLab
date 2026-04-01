# Image and Displacement-Field Orientation Convention, and Equivalent Warping with SciPy and PyTorch

## Resources
- Learn2Reg submission guide:  
        The convention used for displacement fields depends on scipy's map_coordinates() function, expecting displacement fields in the format [X, Y, Z,[x, y, z]] or [[x, y, z], X, Y, Z], where X, Y, Z and x, y, z represent voxel displacements and image dimensions, respectively.  The evaluation script expects .nii.gz files using full-precision format  and having shapes 160x224x196x3. Further information can be found here:  
  - https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/main/L2R_LUMIR_Eval/evaluation.py
- VoxelMorph issue #213: `new_locs = new_locs[..., [2,1,0]]`
  - https://github.com/voxelmorph/voxelmorph/issues/213
  - Reference: "Surprising convention for grid sample coordinates"
    - https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/5
- PyTorch `grid_sample` docs:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
- SciPy `map_coordinates` docs:
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html

## Purpose

This document defines a consistent convention for:

1. Image orientation
2. Displacement field orientation and storage
3. The warping formula
4. Equivalent warping with:
   - `scipy.ndimage.map_coordinates`
   - `torch.nn.functional.grid_sample`

It first describes the **3D** case, then gives the corresponding **2D** version.

The main goal is to make the conventions explicit and ensure that:

- image axes are interpreted consistently,
- the displacement field has a clear physical meaning, and
- SciPy and PyTorch produce the **same warp behavior** when configured properly.

---

## Part I. 3D Convention

### 1. Image Convention in 3D

We use a 3D image array:

```python
image.shape == (X, Y, Z)
```

The three array axes are interpreted in **LPS+ physical orientation** as follows:

| Axis | Array axis | Positive direction |
| --- | --- | --- |
| X | axis 0 | Right -> Left (R -> L) |
| Y | axis 1 | Anterior -> Posterior (A -> P) |
| Z | axis 2 | Inferior -> Superior (I -> S) |

So, if `image[x, y, z]` is a voxel value, then:

- increasing `x` means moving **Left**,
- increasing `y` means moving **Posterior**,
- increasing `z` means moving **Superior**.

This is an array-axis convention together with a physical interpretation of the array axes.

### 2. Displacement Field Convention in 3D

We use a displacement field stored as:

```python
disp.shape == (X, Y, Z, 3)
```

with channel-last storage:

```python
disp[x, y, z, :] = (dX, dY, dZ)
```

The three displacement components are defined along the same three image axes:

- `disp[..., 0]` is displacement along axis `X`, i.e. `R -> L`
- `disp[..., 1]` is displacement along axis `Y`, i.e. `A -> P`
- `disp[..., 2]` is displacement along axis `Z`, i.e. `I -> S`

The displacement values are in **voxel units**, not physical units such as mm.

Examples:

- `disp[x, y, z, 0] = +1` means `+1` voxel along axis 0, i.e. sample from one voxel more **Left**
- `disp[x, y, z, 1] = +1` means sample from one voxel more **Posterior**
- `disp[x, y, z, 2] = +1` means sample from one voxel more **Superior**

### 3. Mapping Meaning: Backward / Pull Warping

This convention uses **backward warping**.

That means the displacement field answers:

> For each voxel location on the output lattice, where should we sample in the input image?

Assume:

- `moving` is the image to be sampled from,
- `warped` is the resampled output image,
- the displacement field is defined on the output lattice.

Then the warped image is:

```math
\mathrm{warped}[x, y, z]
=
\mathrm{moving}\bigl(
x + dX(x, y, z),
y + dY(x, y, z),
z + dZ(x, y, z)
\bigr)
```

where:

```math
(dX, dY, dZ) = \mathrm{disp}[x, y, z, :]
```

This means:

- the displacement does **not** say where the source voxel moves to,
- it says where the output voxel samples from in the moving image.

This is exactly the convention used by:

- `scipy.ndimage.map_coordinates`
- `torch.nn.functional.grid_sample`

when used for image warping.

### 4. Identity Grid in 3D

To implement the warp, define the identity coordinate grid:

```python
grid_x, grid_y, grid_z = np.meshgrid(
    np.arange(X),
    np.arange(Y),
    np.arange(Z),
    indexing="ij",
)
```

Then:

```python
grid_x[x, y, z] = x
grid_y[x, y, z] = y
grid_z[x, y, z] = z
```

The sampling coordinates are:

```python
sample_x = grid_x + disp[..., 0]
sample_y = grid_y + disp[..., 1]
sample_z = grid_z + disp[..., 2]
```

### 5. Challenge-Style Storage Convention

The challenge convention described as:

```text
[X, Y, Z, [x, y, z]]
```

matches the above exactly if interpreted as:

- image dimensions are `(X, Y, Z)`,
- vector components are `(dX, dY, dZ)` along those same array axes.

So the chosen convention is:

```python
disp.shape == (X, Y, Z, 3)
disp[..., 0] = displacement along axis X (R -> L)
disp[..., 1] = displacement along axis Y (A -> P)
disp[..., 2] = displacement along axis Z (I -> S)
```

This is fully consistent.

---

## Part II. Warping with SciPy in 3D

### 6. `scipy.ndimage.map_coordinates` Convention

`map_coordinates` expects coordinates in the form:

```python
coords.shape == (ndim, ...)
```

So for a 3D image with shape `(X, Y, Z)`, it expects:

```python
coords.shape == (3, X, Y, Z)
```

with:

- `coords[0, ...]` = coordinates along axis 0
- `coords[1, ...]` = coordinates along axis 1
- `coords[2, ...]` = coordinates along axis 2

Therefore, starting from:

```python
disp.shape == (X, Y, Z, 3)
```

you can convert it to SciPy coordinate format by transposing:

```python
disp_cf = disp.transpose(3, 0, 1, 2)  # (3, X, Y, Z)
```

Then define identity and sample coordinates:

```python
identity = np.stack(
    np.meshgrid(
        np.arange(X),
        np.arange(Y),
        np.arange(Z),
        indexing="ij",
    ),
    axis=0,
)  # (3, X, Y, Z)

coords = identity + disp_cf
```

Then warp:

```python
from scipy.ndimage import map_coordinates

warped = map_coordinates(
    moving,
    coords,
    order=1,
    mode="nearest",
)
```

### 7. SciPy Interpolation and Padding Choices

To match PyTorch behavior later, use:

#### Interpolation

- `order=0` <-> nearest-neighbor interpolation
- `order=1` <-> linear interpolation

#### Padding / out-of-bound behavior

- `mode="nearest"` <-> replicate border value
- `mode="constant", cval=0.0` <-> sample zero outside image

Recommended pairings for matching PyTorch:

- `order=0` <-> PyTorch `mode="nearest"`
- `order=1` <-> PyTorch `mode="bilinear"`

In 3D, PyTorch still uses the string `"bilinear"`, but the actual interpolation over a 5D tensor is **trilinear**.

And:

- SciPy `mode="nearest"` <-> PyTorch `padding_mode="border"`
- SciPy `mode="constant", cval=0.0` <-> PyTorch `padding_mode="zeros"`

### 8. Full SciPy 3D Example

```python
import numpy as np
from scipy.ndimage import map_coordinates


def warp_scipy_3d(moving, disp, order=1, mode="nearest", cval=0.0):
    """
    moving: (X, Y, Z)
    disp:   (X, Y, Z, 3), voxel units, channel-last
    """
    X, Y, Z = moving.shape

    identity = np.stack(
        np.meshgrid(
            np.arange(X),
            np.arange(Y),
            np.arange(Z),
            indexing="ij",
        ),
        axis=0,
    )  # (3, X, Y, Z)

    coords = identity + disp.transpose(3, 0, 1, 2)  # (3, X, Y, Z)

    warped = map_coordinates(
        moving,
        coords,
        order=order,
        mode=mode,
        cval=cval,
    )
    return warped
```

---

## Part III. Warping with PyTorch in 3D

### 9. PyTorch `grid_sample` Convention

For a 3D image, PyTorch expects input tensor shape:

```python
src.shape == (N, C, D, H, W)
```

So PyTorch spatial axes are ordered as:

- axis `D`
- axis `H`
- axis `W`

However, the sampling grid must be stored as:

```python
grid.shape == (N, D, H, W, 3)
```

and the last dimension must be ordered as:

```python
(x, y, z)
```

where:

- `x` indexes `W`
- `y` indexes `H`
- `z` indexes `D`

This is the key source of confusion.

PyTorch does **not** want coordinates in array-axis order `(D, H, W)`.
It wants them in Cartesian grid order `(x, y, z) = (W, H, D)`.

### 10. Relationship Between the Array Convention and the PyTorch Convention

The natural internal convention is:

```python
flow.shape == (N, 3, X, Y, Z)
flow[:, 0, ...] = displacement along X
flow[:, 1, ...] = displacement along Y
flow[:, 2, ...] = displacement along Z
```

with:

- `X = axis 0 = R -> L`
- `Y = axis 1 = A -> P`
- `Z = axis 2 = I -> S`

If a tensor is fed to PyTorch as:

```python
src.shape == (N, C, X, Y, Z)
```

then PyTorch interprets:

- `D = X`
- `H = Y`
- `W = Z`

Therefore, when constructing the final PyTorch grid:

- PyTorch `x` corresponds to `W`, which is the internal `Z`
- PyTorch `y` corresponds to `H`, which is the internal `Y`
- PyTorch `z` corresponds to `D`, which is the internal `X`

So the coordinate order must change from:

```python
(X, Y, Z)
```

to:

```python
(Z, Y, X)
```

This is why the channel reversal is needed.

It is **not** a flip of physical direction.
It is only a reordering required by the API.

### 11. Why Normalization Is Needed in PyTorch

`grid_sample` does not use voxel coordinates directly.
It expects sampling coordinates normalized to the range:

```math
[-1, 1]
```

For `align_corners=True`, the normalization is:

```math
u = 2 \cdot \frac{x}{X - 1} - 1
```

for axis `X`, and similarly for the other axes.

So if:

- voxel coordinate `0` is the first voxel center,
- voxel coordinate `X - 1` is the last voxel center,

then:

- `0` maps to `-1`
- `X - 1` maps to `+1`

This is the best match to SciPy integer-index coordinate interpretation.

That is why using `align_corners=True` is important when trying to match `map_coordinates`.

### 12. Step-by-Step Logic for PyTorch Warping in 3D

Starting from:

```python
flow.shape == (N, 3, X, Y, Z)
```

where `flow` is in voxel units and in natural axis order `(X, Y, Z)`:

#### Step 1. Build identity grid in voxel coordinates

```python
grid_x, grid_y, grid_z = torch.meshgrid(
    torch.arange(X),
    torch.arange(Y),
    torch.arange(Z),
    indexing="ij",
)
grid = torch.stack((grid_x, grid_y, grid_z), dim=0).float()  # (3, X, Y, Z)
grid = grid.unsqueeze(0)  # (1, 3, X, Y, Z)
```

#### Step 2. Add displacement in voxel units

```python
new_locs = grid + flow
```

This still has channel order:

```python
(X, Y, Z)
```

#### Step 3. Normalize each axis to `[-1, 1]`

```python
new_locs[:, 0, ...] = 2 * (new_locs[:, 0, ...] / (X - 1) - 0.5)
new_locs[:, 1, ...] = 2 * (new_locs[:, 1, ...] / (Y - 1) - 0.5)
new_locs[:, 2, ...] = 2 * (new_locs[:, 2, ...] / (Z - 1) - 0.5)
```

#### Step 4. Move channels to the last dimension

```python
new_locs = new_locs.permute(0, 2, 3, 4, 1)  # (N, X, Y, Z, 3)
```

#### Step 5. Reorder from `(X, Y, Z)` to PyTorch `(x, y, z) = (Z, Y, X)`

```python
new_locs = new_locs[..., [2, 1, 0]]
```

Now `new_locs` is in the correct format for `grid_sample`.

#### Step 6. Warp

```python
warped = torch.nn.functional.grid_sample(
    src,
    new_locs,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
)
```

### 13. Full PyTorch 3D Example

```python
import torch
import torch.nn.functional as F


def warp_torch_3d(src, flow, mode="bilinear", padding_mode="border"):
    """
    src:  (N, C, X, Y, Z)
    flow: (N, 3, X, Y, Z), voxel units in axis order (X, Y, Z)
    """
    N, C, X, Y, Z = src.shape
    device = src.device
    dtype = src.dtype

    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(X, device=device, dtype=dtype),
        torch.arange(Y, device=device, dtype=dtype),
        torch.arange(Z, device=device, dtype=dtype),
        indexing="ij",
    )

    grid = torch.stack((grid_x, grid_y, grid_z), dim=0).unsqueeze(0)  # (1, 3, X, Y, Z)

    new_locs = grid + flow

    new_locs[:, 0, ...] = 2 * (new_locs[:, 0, ...] / (X - 1) - 0.5)
    new_locs[:, 1, ...] = 2 * (new_locs[:, 1, ...] / (Y - 1) - 0.5)
    new_locs[:, 2, ...] = 2 * (new_locs[:, 2, ...] / (Z - 1) - 0.5)

    new_locs = new_locs.permute(0, 2, 3, 4, 1)  # (N, X, Y, Z, 3)
    new_locs = new_locs[..., [2, 1, 0]]  # now in PyTorch order (x, y, z) = (Z, Y, X)

    warped = F.grid_sample(
        src,
        new_locs,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped
```

---

## Part IV. Why SciPy and PyTorch Can Represent the Same Warp

### 14. Same Mathematical Warp, Different API Conventions

SciPy and PyTorch both perform **backward sampling**.

They both implement the same mathematical idea:

```math
\mathrm{warped}(\text{output voxel})
=
\mathrm{moving}(\text{sampling coordinate in moving image})
```

The difference is only in how the coordinates are encoded.

#### SciPy

SciPy uses:

```python
coords.shape == (ndim, ...)
```

with coordinates listed in array-axis order:

```python
(axis0, axis1, axis2)
```

So for an image `(X, Y, Z)`, SciPy wants coordinates ordered as:

```python
(X, Y, Z)
```

#### PyTorch

PyTorch uses:

```python
grid.shape == (N, D, H, W, 3)
```

with coordinates ordered as:

```python
(x, y, z)
```

which correspond to:

```python
(W, H, D)
```

If the tensor is shaped `(N, C, X, Y, Z)`, then:

- `D = X`
- `H = Y`
- `W = Z`

so PyTorch wants:

```python
(Z, Y, X)
```

That is why the final channel order must be reversed.

### 15. Important Clarification: Reversal Is Not Orientation Inversion

The line:

```python
new_locs = new_locs[..., [2, 1, 0]]
```

is often confusing.

It does **not** mean:

- Left becomes Right
- Posterior becomes Anterior
- Superior becomes Inferior

It only means:

- PyTorch expects coordinates in the order `(x, y, z)`, and
- in this tensor layout those correspond to `(Z, Y, X)`.

So this is only:

- an axis-order conversion,
- not a sign flip,
- not a world-orientation change.

The positive directions remain:

- `+X`: `R -> L`
- `+Y`: `A -> P`
- `+Z`: `I -> S`

before and after the reorder.

### 16. Conditions Required to Make SciPy and PyTorch Match

To make the two behave equivalently, use the following pairings.

#### Interpolation

- SciPy `order=0` <-> PyTorch `mode="nearest"`
- SciPy `order=1` <-> PyTorch `mode="bilinear"`

In 3D, PyTorch `"bilinear"` effectively means **trilinear interpolation**.

#### Boundary behavior

- SciPy `mode="nearest"` <-> PyTorch `padding_mode="border"`
- SciPy `mode="constant", cval=0.0` <-> PyTorch `padding_mode="zeros"`

#### Coordinate interpretation

- PyTorch must use `align_corners=True`

#### Displacement convention

Both must use the same displacement definition:

```math
\mathrm{warped}[x, y, z]
=
\mathrm{moving}[x + dX, y + dY, z + dZ]
```

with displacements in voxel units.

### 17. Why `align_corners=True` Is Important

If `align_corners=False`, the normalized coordinate mapping in PyTorch changes.
Then the meaning of normalized coordinates no longer matches the simple voxel-center interpretation used by SciPy.

Since SciPy uses coordinates directly in voxel index space, the cleanest matching setup is:

```python
align_corners=True
```

This makes:

- normalized `-1` correspond to the first voxel center,
- normalized `+1` correspond to the last voxel center,

which aligns naturally with SciPy indexing.

### 18. Practical Interpretation of a Positive Displacement

Suppose:

```python
disp[x, y, z, 0] = +1
disp[x, y, z, 1] = 0
disp[x, y, z, 2] = 0
```

Then:

```math
\mathrm{warped}[x, y, z] = \mathrm{moving}[x + 1, y, z]
```

Since axis `X` increases from `Right -> Left`, the output voxel at `(x, y, z)` samples from a location one voxel more **Left** in the moving image.

So the displacement means:

- sample from **Left**,
- not "push this voxel to the Right".

This is the essence of backward warping.

---

## Part V. 2D Version

### 19. Image Convention in 2D

Now consider a 2D image:

```python
image.shape == (X, Y)
```

The same style of convention applies:

| Axis | Array axis | Positive direction |
| --- | --- | --- |
| X | axis 0 | Right -> Left (R -> L) |
| Y | axis 1 | Anterior -> Posterior (A -> P) |

So:

- increasing `x` means moving **Left**,
- increasing `y` means moving **Posterior**.

### 20. Displacement Field Convention in 2D

Use:

```python
disp.shape == (X, Y, 2)
```

with:

```python
disp[x, y, :] = (dX, dY)
```

where:

- `disp[..., 0]` = displacement along axis `X` = `R -> L`
- `disp[..., 1]` = displacement along axis `Y` = `A -> P`

The warp formula becomes:

```math
\mathrm{warped}[x, y]
=
\mathrm{moving}\bigl(
x + dX(x, y),
y + dY(x, y)
\bigr)
```

### 21. SciPy in 2D

SciPy expects coordinates:

```python
coords.shape == (2, X, Y)
```

So:

```python
identity = np.stack(
    np.meshgrid(
        np.arange(X),
        np.arange(Y),
        indexing="ij",
    ),
    axis=0,
)  # (2, X, Y)

coords = identity + disp.transpose(2, 0, 1)
```

Then warp:

```python
warped = map_coordinates(
    moving,
    coords,
    order=1,
    mode="nearest",
)
```

#### Full 2D SciPy Example

```python
import numpy as np
from scipy.ndimage import map_coordinates


def warp_scipy_2d(moving, disp, order=1, mode="nearest", cval=0.0):
    """
    moving: (X, Y)
    disp:   (X, Y, 2), voxel units, channel-last
    """
    X, Y = moving.shape

    identity = np.stack(
        np.meshgrid(
            np.arange(X),
            np.arange(Y),
            indexing="ij",
        ),
        axis=0,
    )  # (2, X, Y)

    coords = identity + disp.transpose(2, 0, 1)  # (2, X, Y)

    warped = map_coordinates(
        moving,
        coords,
        order=order,
        mode=mode,
        cval=cval,
    )
    return warped
```

### 22. PyTorch in 2D

For 2D input, PyTorch expects:

```python
src.shape == (N, C, H, W)
grid.shape == (N, H, W, 2)
```

and the final grid last dimension must be:

```python
(x, y)
```

where:

- `x` indexes `W`
- `y` indexes `H`

If the tensor is stored as:

```python
src.shape == (N, C, X, Y)
```

then:

- `H = X`
- `W = Y`

so PyTorch wants coordinates in the order:

```python
(Y, X)
```

The natural internal flow order is `(X, Y)`, so it must be reordered to `(Y, X)` before calling `grid_sample`.

#### Full 2D PyTorch Example

```python
import torch
import torch.nn.functional as F


def warp_torch_2d(src, flow, mode="bilinear", padding_mode="border"):
    """
    src:  (N, C, X, Y)
    flow: (N, 2, X, Y), voxel units in axis order (X, Y)
    """
    N, C, X, Y = src.shape
    device = src.device
    dtype = src.dtype

    grid_x, grid_y = torch.meshgrid(
        torch.arange(X, device=device, dtype=dtype),
        torch.arange(Y, device=device, dtype=dtype),
        indexing="ij",
    )

    grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)  # (1, 2, X, Y)

    new_locs = grid + flow

    new_locs[:, 0, ...] = 2 * (new_locs[:, 0, ...] / (X - 1) - 0.5)
    new_locs[:, 1, ...] = 2 * (new_locs[:, 1, ...] / (Y - 1) - 0.5)

    new_locs = new_locs.permute(0, 2, 3, 1)  # (N, X, Y, 2)
    new_locs = new_locs[..., [1, 0]]  # now in PyTorch order (x, y) = (Y, X)

    warped = F.grid_sample(
        src,
        new_locs,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped
```

---

## Part VI. Summary Tables

### 23. 3D Summary

#### Image Axes

| Axis | Array index | Positive direction |
| --- | --- | --- |
| X | axis 0 | R -> L |
| Y | axis 1 | A -> P |
| Z | axis 2 | I -> S |

#### Displacement Field

| Component | Meaning |
| --- | --- |
| `disp[..., 0]` | displacement along X (`R -> L`) |
| `disp[..., 1]` | displacement along Y (`A -> P`) |
| `disp[..., 2]` | displacement along Z (`I -> S`) |

#### Warp Formula

```math
\mathrm{warped}[x, y, z]
=
\mathrm{moving}[x + dX, y + dY, z + dZ]
```

#### API Coordinate Order

| Library | Coordinate order expected |
| --- | --- |
| SciPy `map_coordinates` | `(X, Y, Z)` |
| PyTorch `grid_sample` | `(x, y, z) = (Z, Y, X)` if tensor is `(N, C, X, Y, Z)` |

### 24. 2D Summary

#### Image Axes

| Axis | Array index | Positive direction |
| --- | --- | --- |
| X | axis 0 | R -> L |
| Y | axis 1 | A -> P |

#### Displacement Field

| Component | Meaning |
| --- | --- |
| `disp[..., 0]` | displacement along X (`R -> L`) |
| `disp[..., 1]` | displacement along Y (`A -> P`) |

#### Warp Formula

```math
\mathrm{warped}[x, y]
=
\mathrm{moving}[x + dX, y + dY]
```

#### API Coordinate Order

| Library | Coordinate order expected |
| --- | --- |
| SciPy `map_coordinates` | `(X, Y)` |
| PyTorch `grid_sample` | `(x, y) = (Y, X)` if tensor is `(N, C, X, Y)` |

---

## Part VII. Recommended Matching Settings

### 25. Matching Configuration

Use the following settings to make SciPy and PyTorch behave as similarly as possible.

#### Interpolation

- SciPy `order=0` <-> PyTorch `mode="nearest"`
- SciPy `order=1` <-> PyTorch `mode="bilinear"`

#### Boundary handling

- SciPy `mode="nearest"` <-> PyTorch `padding_mode="border"`
- SciPy `mode="constant", cval=0.0` <-> PyTorch `padding_mode="zeros"`

#### Coordinate normalization

- PyTorch `align_corners=True`

#### Displacement units

- always use voxel units

#### Warp type

- always use backward / pull warping

---

## Part VIII. Final Remarks

### 26. Core Idea to Remember

This convention is internally consistent if these three facts are kept separate:

#### A. Physical orientation of array axes

The axes are defined as:

- axis 0 = `R -> L`
- axis 1 = `A -> P`
- axis 2 = `I -> S`

#### B. Displacement meaning

The displacement field is a **backward sampling field**:

- it tells each output voxel where to sample in the moving image.

#### C. Library-specific coordinate encoding

- SciPy expects coordinates in array-axis order.
- PyTorch expects coordinates in Cartesian grid order `(x, y, z)` or `(x, y)`.

The PyTorch channel reversal is only due to item **C**, not due to any change in **A** or **B**.

### 27. Minimal Practical Rule

In 3D, if the displacement is stored as:

```python
disp.shape == (X, Y, Z, 3)
disp[..., 0] = X-direction
disp[..., 1] = Y-direction
disp[..., 2] = Z-direction
```

then:

#### For SciPy

```python
coords = identity + disp.transpose(3, 0, 1, 2)
```

#### For PyTorch

```python
flow = disp.permute(...)  # to (N, 3, X, Y, Z)
new_locs = identity + flow
# normalize
# permute to channel-last
# reverse channels [2, 1, 0]
```

In 2D, the same idea becomes:

- SciPy wants `(X, Y)`
- PyTorch wants `(Y, X)`

### 28. Sanity-Check Interpretation

If in 3D:

```python
disp[..., 0] = +1
disp[..., 1] = 0
disp[..., 2] = 0
```

then both SciPy and PyTorch should implement:

```math
\mathrm{warped}[x, y, z] = \mathrm{moving}[x + 1, y, z]
```

which means sampling from one voxel further along the positive `X` axis, i.e. one voxel more **Left**.

That is the simplest way to verify the convention is being used correctly.
