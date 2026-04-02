
Strictly follow the coding instructions in /homebase/code_sync/myMedIADIRLab/data_io/image/prompt_coding_rule_01basic.md

Carefully review everything in /homebase/code_sync/myMedIADIRLab/data_io/image and /homebase/code_sync/myMedIADIRLab/data_io/warp

I will follow the canonical LPS+ orientation for both image and warp!

I already have the /homebase/code_sync/myMedIADIRLab/data_io/image/image_display.py

Now I would like to work on warp_display.py

You could probably divide into 2 files display_2d and display_3d if that's more managable, you can decide to reuse 2d code for 3d or not

For 3D, we will focus on the following types of display. You can infer the 2D
A. Vector field display, (although for 3D everthing is projected onto a 2D plane)
    1. Grid (backward)
    2. Grid (forward)
    3. Quiver plot
B. Scalar field display
    1. Jacobian determinant
    2. Curl
    3. Magnitude
    4. Disp_x/y/z (single component


For the backward grid and Jacobian determinant, you can refer to the demo in /homebase/code_sync/Unsupervised-DL-DIR-Revisited/demo/example1_visualize_registration_results to get the basic ideas. and you should try to mimic it as it is good example, you can make improvement if needed. Jacobian map should be able to show regions that there is folding. also for jabobian map, add option to take log and adjust the colormap accordingly

The backward grid is essentially create a grid image and deform it with the disp field (you should use gpu verion to deform it for speed, use /homebase/code_sync/myMedIADIRLab/data_io/warp/warp_v1_unified.py)

The forward grid is different, it define grids/line segments and directly add the disp fiedl to the line segment so that it is going from fixed image to moving image domain (which is the definition of the disp). You can see the difference between forward and backward and an example of forward in:
    /homebase/code_sync/myMedIADIRLab/data_io/warp/resources/visualize_dvf_3_bailiangJ_test1.ipynb
    I would like you to improve the forward code and make sure the orienation is correct

For quiver plot, you can refer to:
    /homebase/code_sync/myMedIADIRLab/data_io/warp/resources/color_code_dvf_v3.ipynb
    /homebase/code_sync/myMedIADIRLab/data_io/warp/resources/utils_new_quiver_wip.py
You should make sure the orientation is correct!!! And make improvements

Implement Jacobian and curl in a seperate py file, for the Jacobian and curl I would like to 0 pad them back to the original size if there size are smaller


Compatible with image_display.py I would like to have 3 views in canonical format and it should be flexible in terms of colormap, colorbar, title etc ...


And in the end I would like you to test the display in /homebase/code_sync/myMedIADIRLab/data_io/unittest_warp/unittest_02_disp_visualization_musa_test1.ipynb
You should be able to reuse the data in /homebase/code_sync/myMedIADIRLab/data_io/unittest_warp/unittest_01_unified_warp_2musa_passed.ipynb

Test one plot in one cell, and I should be able to adjust for plotting settings specific to that visualization at the begininig of that cell.

Don't code yet, tell me if my plan is good, and any improvement is needed. Also plan well before coding.