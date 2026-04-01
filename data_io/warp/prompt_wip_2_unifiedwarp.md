
Strictly follow the coding instructions in /homebase/code_sync/myMedIADIRLab/data_io/image/prompt_coding_rule_01basic.md

I could like you to code a new py file /homebase/code_sync/myMedIADIRLab/data_io/warp/warp_v1_unified.py

You should first read these files thoroughly:
    /homebase/code_sync/myMedIADIRLab/data_io/warp/WARP_CONVENTION.md
    /homebase/code_sync/myMedIADIRLab/data_io/warp/warp_v0_spatialtrans.py

I would like a unified function to handle warping:
    1. It should handle both 2D and 3D, 
        the input and output are both numpy arrays
        For 3D images (X, Y, Z) and disp array [X, Y, Z,[x, y, z]], [[x, y, z], X, Y, Z],
        if disp array is channel first, convert to channel last first and then check if the spatial size matches and has 3 channels
        similar for 2D
    2. It should handle both:
       - `scipy.ndimage.map_coordinates`: cpu only
       - `torch.nn.functional.grid_sample`: cpu or gpu, and if gpu I can pass in device number like 0,1,2,3
       the torch version should convert numpy to torch tensor and put on device, and convert back to numpy array in the end
       I should be able to specify which method to use and whether to use gpu or not for torch
       For torch, align corners should be set to True
       and to match scipy and torch behavior I will define arguments:
           order: 0 or 1
           padding_mode: border or constant (and with value specified but default to 0)
       If debug flag is set, it should print the these things out        


Don't code yet, tell me if my plan is good, and any improvement is needed.