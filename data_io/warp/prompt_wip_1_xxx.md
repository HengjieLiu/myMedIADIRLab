
Strictly follow the coding instructions in /homebase/code_sync/myMedIADIRLab/data_io/image/prompt_coding_rule_01basic.md

I could like you to code in /homebase/code_sync/myMedIADIRLab/data_io/warp

You should follow the convention in /homebase/code_sync/myMedIADIRLab/data_io/image

So this is the class that handle warp/deformation/registration

I will follow the previous image class that all the warp are in LPS+ for 3D
So if it is 3D we will have a warp in the format of (3, H, W, D)

### NOTES:
GPT's answer is confusing, why this has to do with simpleITK?
    https://chatgpt.com/share/69cce1a4-dcc0-83e8-aa27-a1b1bcc5a23a
    I do need to consider visualization in ITK-SNAP and 3d SLICER for cross validation




Currently I will mainly be dealing with 2 types nifti files loaded from nii.gz and SimpleITK images
The first most important thing is orientation, so for both nifti files and sitk image, we should keep track of a 4x4 affine matrix (Homogeneous transformation matrix) which defines its orientation and resolution. nifti file and sitk image already have that, so we will keep the 4x4 matrix with the image object.

For standardization, we will by default convert any 3D image into LPS+ orientation:
    meaning: 
        i+ is R->L
        j+ is A->P
        k+ is I->S
    And all the display code should adhere to this convention.

1. Note that the nifti files use RAS+ system so if the affine matrix's 3x3 part is identity it means RAS+ not LPS+
2. Note that in simpleITK, when convert to numpy array, the image goes from (x,y,z) to (z,y,x) this also needs to be adjusted,
   define a new convertion function to make sure the image is in LPS+ when going from simpleITK image to numpy array (also handle the inverse transform as well)


So I am thinking of the following coding plan:
    image_io.py: handle read/write for nifti and simpleITK images
    image_orientLPS.py: than handle anything related to orientation, and automatically convert the image array and affine matrix to LPS+ (should handle reverse transform back to simpleITK and nifti as well)
    image_display.py
        you can refer to the code in /homebase/DL_projects/fireants_wip/HN_reg/02jess/step1_preprocess/wip0319_step2_2_dirfireants_test4_CT_iso_crop_B_cropinput.ipynb and files in the same folder
        But note they don't have the same LPS standardization.
        I would like to be able to flexibly display any specified single view and 3 views:
            sagittal: slice along axis i/0
            coronal: slice along axis j/1
            axial: slice along axis k/2
        
        Note we will follow radiological convention:
            sagittal: A is on the left, S is on the top
            coronal:  R is on the left, S is on the top
            axial:    R is on the left, A is on the top

        and when displaying I should be able to adjust the display flexibly including:
            cmap, size of figure, vmin/vmax (can specify specific value or use percentile of intensity), colorbar, fontsize for title, labels, legend
            cropping, where I would specify crop_L/R/A/P/S/I respectively
            by default calculate the aspect using the affine matrix to make it phycically plausible


Don't code yet, tell me if my plan is good, and any improvement is needed.