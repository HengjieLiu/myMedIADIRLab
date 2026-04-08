

### Backward grid can still faint.

need code checking, currently codex say there is not an issue ...


check homebase/code_sync/myMedIADIRLab/data_io/warp/warp_display.py

the backward grid, should be a stack of 2d grid lines, it has 3 views, for each view it is the same stack of 2d grid in the remaining direction, and there should be no gap in the remaining direction, is that what is implemented




Worked for 31s
Yes. For the backward grid, the code does implement a gapless stack of identical 2D grids along the remaining axis.

The key part is warp_display.py:672 to warp_display.py:694:

Sagittal uses in-plane axes (1, 2), so the grid is drawn in YZ and repeated across all X.
Coronal uses (0, 2), so the grid is drawn in XZ and repeated across all Y.
Axial uses (0, 1), so the grid is drawn in XY and repeated across all Z.
The important detail is warp_display.py:690 to warp_display.py:692: it sets slicer = [slice(None), slice(None), slice(None)] and only restricts one of the two in-plane axes. That means the third axis stays slice(None), so each grid line extends through the full remaining direction with no gaps.

Then the backward renderer actually uses that 3D stacked volume, warps it, and extracts the requested view slice in warp_display.py:1549 to warp_display.py:1574. The view slicing itself matches the expected axes in image_display.py:222 to image_display.py:253.

So the answer is: yes, the source backward grid is a view-specific stack of identical 2D grids with no gap along the remaining direction. The only caveat is that after warping it is resampled with linear interpolation, so it may look softened, but the underlying construction is the one you described.