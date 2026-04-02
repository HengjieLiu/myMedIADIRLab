

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def quiver_plot_2d_new_v0(
    img, dvf, ax,
    orient, idx_slc,
    crop_window=None, ds_step=1, 
    scale=1., 
    plot_option=None,
    color_code=False,
    vmin=0., vmax=1.,
    vmin_q=None, vmax_q=None,
    colorbar=True,
):
    """
    Update 05/14/2024:
        newly added the color_code version:
            quiver_plot_2d_new_v0: no specific normalization, just provide vmin_q/vmax_q
            quiver_plot_2d_new_v1: specific normalization (not really convenient to work with)
    
    Previous notes:
    img:
        (160, 160, 192)
        L?R, A->P, I->S
        LPS+ or RPS+???
            In ITK-SNAP it says RAI-/LPS+, so I would assume LPS+
            R->L, A->P, I->S
    dvf:
        (3, 160, 160, 192)
    img and dvf are matched in a sense that the warping function dirhj.warp_modules.SpatialTransformer()
        can directly be used to warp the img with dvf.
    
    dvf channel orientation:
        channel 0: LR  
        channel 1: AP (+ is A->P)  
        channel 2: SI (+ is I->S)
        
    matplotlib display settings:
        In plt.imshow(), the origin (0, 0) is at the top-left corner of the image, while in plt.quiver(), the origin is at the bottom-left corner of the plot.
        To fix this orientation mismatch, you can flip the y-coordinates of the quiver plot or change the origin of the image display using the origin parameter.
        e.g.. ax.imshow(img_2d_disp, origin='lower', cmap='gray') # by default: origin='upper'
    """
    
    assert orient in ['sag', 'axi', 'cor'], 'ERROR: unrecognized orient: '+ orient
    ##########                      # orientation     # orientation after transpose
    if orient == 'sag':
        img_2d = img[idx_slc, :, :] # (A->P, I->S)    # (I->S, A->P) ("flip" needed: origin='lower')
        u   = dvf[1, idx_slc, :, :] # A->P
        v   = dvf[2, idx_slc, :, :] # I->S
        origin_imshow = 'lower'
    elif orient == 'axi':
        img_2d = img[:, :, idx_slc] # (R->L, A->P)    # (A->P, R->L)
        u   = dvf[0, :, :, idx_slc] # R->L
        v   = dvf[1, :, :, idx_slc] # A->P
        origin_imshow = 'upper'
    elif orient =='cor':
        img_2d = img[:, idx_slc, :] # (R->L, I->S)    # (I->S, R->L) ("flip" needed: origin='lower')
        u   = dvf[0, :, idx_slc, :] # R->L
        v   = dvf[2, :, idx_slc, :] # I->S
        origin_imshow = 'lower'
        
    ### meshgrid
    shape_i, shape_j = img_2d.shape
    i, j = np.meshgrid(np.arange(0,shape_i,1), np.arange(0,shape_j,1), indexing='ij') # not 'xy'
    
    ### transpose 
    # transpose is necessary
    #   1. This is coincidence that I need to transpose so that the image orientation is what I want ..
    #   2. need to verify: transpose is also necessary to match between the ij indexing meshgrid and xy indexing plot
    img_2d_disp = np.transpose(img_2d, (1,0))
    
    ### DEBUG
    # print(img_2d.shape)
    # print(img_2d_disp.shape)
    # print(i.shape, j.shape, u.shape, v.shape)
    
    ### crop (if necessary)
    if crop_window is not None:
        i1, i2, j1, j2 = crop_window
        
        img_2d_disp = img_2d_disp[i1:i2,j1:j2] # transposed here
        
        i = i[j1:j2,i1:i2]-j1 # minus the offset as i/j are coordinates
        j = j[j1:j2,i1:i2]-i1 # minus the offset as i/j are coordinates
        u = u[j1:j2,i1:i2]
        v = v[j1:j2,i1:i2]
    
    ### downsample the quiver plot (if necessary)
    if ds_step != 1:
        i = i[0::ds_step,0::ds_step]
        j = j[0::ds_step,0::ds_step]
        u = u[0::ds_step,0::ds_step]
        v = v[0::ds_step,0::ds_step]
    
    # plot with options
    if plot_option != 'quiver_only':
        ax.imshow(img_2d_disp, origin=origin_imshow, vmin=vmin, vmax=vmax, cmap='gray')
    if plot_option != 'imshow_only':
        if not color_code:
            ### plot the vector field in fixed color, magnitude coded by lengths
            ax.quiver(i, j, u, v, color='r', angles='xy', scale_units='xy', scale=scale)
        else:
            ### plot the vector field, magnitude coded by both length and colors
            
            # calculate the magnitude of each vector
            magnitude = np.sqrt(u**2 + v**2)
            
            # create a colormap (reverse Red/Yelleo/Green: red for large, yellow for medium, green for small)
            cmap = plt.cm.get_cmap('RdYlGn_r')
            
            # set the vmin and vmax for quiver magnitude
            if vmin_q is None:
                vmin_q = magnitude.min()
            if vmax_q is None:
                vmax_q = magnitude.max()
            norm = mcolors.Normalize(vmin=vmin_q, vmax=vmax_q)
            quiver = ax.quiver(i, j, u, v, magnitude, cmap=cmap, norm=norm, angles='xy', scale_units='xy', scale=scale)
                        
            # add a colorbar to the plot with original magnitude values
            if colorbar:
                cbar = plt.colorbar(quiver, ax=ax, label='Vector Magnitude')


def quiver_plot_2d_new_v1(
    img, dvf, ax,
    orient, idx_slc,
    crop_window=None, ds_step=1, 
    scale=1., 
    plot_option=None,
    color_code=False,
    vmin=0., vmax=1.,
    norm_minmax=None, norm_percentile=None,
):
    """
    Update 05/14/2024:
        newly added the color_code version:
            quiver_plot_2d_new_v0: no specific normalization, just provide vmin_q/vmax_q
            quiver_plot_2d_new_v1: specific normalization (not really convenient to work with)
    
    Previous notes:
    img:
        (160, 160, 192)
        L?R, A->P, I->S
        LPS+ or RPS+???
            In ITK-SNAP it says RAI-/LPS+, so I would assume LPS+
            R->L, A->P, I->S
    dvf:
        (3, 160, 160, 192)
    img and dvf are matched in a sense that the warping function dirhj.warp_modules.SpatialTransformer()
        can directly be used to warp the img with dvf.
    
    dvf channel orientation:
        channel 0: LR  
        channel 1: AP (+ is A->P)  
        channel 2: SI (+ is I->S)
        
    matplotlib display settings:
        In plt.imshow(), the origin (0, 0) is at the top-left corner of the image, while in plt.quiver(), the origin is at the bottom-left corner of the plot.
        To fix this orientation mismatch, you can flip the y-coordinates of the quiver plot or change the origin of the image display using the origin parameter.
        e.g.. ax.imshow(img_2d_disp, origin='lower', cmap='gray') # by default: origin='upper'
    """
    
    assert orient in ['sag', 'axi', 'cor'], 'ERROR: unrecognized orient: '+ orient
    ##########                      # orientation     # orientation after transpose
    if orient == 'sag':
        img_2d = img[idx_slc, :, :] # (A->P, I->S)    # (I->S, A->P) ("flip" needed: origin='lower')
        u   = dvf[1, idx_slc, :, :] # A->P
        v   = dvf[2, idx_slc, :, :] # I->S
        origin_imshow = 'lower'
    elif orient == 'axi':
        img_2d = img[:, :, idx_slc] # (R->L, A->P)    # (A->P, R->L)
        u   = dvf[0, :, :, idx_slc] # R->L
        v   = dvf[1, :, :, idx_slc] # A->P
        origin_imshow = 'upper'
    elif orient =='cor':
        img_2d = img[:, idx_slc, :] # (R->L, I->S)    # (I->S, R->L) ("flip" needed: origin='lower')
        u   = dvf[0, :, idx_slc, :] # R->L
        v   = dvf[2, :, idx_slc, :] # I->S
        origin_imshow = 'lower'
        
    ### meshgrid
    shape_i, shape_j = img_2d.shape
    i, j = np.meshgrid(np.arange(0,shape_i,1), np.arange(0,shape_j,1), indexing='ij') # not 'xy'
    
    ### transpose 
    # transpose is necessary
    #   1. This is coincidence that I need to transpose so that the image orientation is what I want ..
    #   2. need to verify: transpose is also necessary to match between the ij indexing meshgrid and xy indexing plot
    img_2d_disp = np.transpose(img_2d, (1,0))
    
    ### DEBUG
    # print(img_2d.shape)
    # print(img_2d_disp.shape)
    # print(i.shape, j.shape, u.shape, v.shape)
    
    ### crop (if necessary)
    if crop_window is not None:
        i1, i2, j1, j2 = crop_window
        
        img_2d_disp = img_2d_disp[i1:i2,j1:j2] # transposed here
        
        i = i[j1:j2,i1:i2]-j1 # minus the offset as i/j are coordinates
        j = j[j1:j2,i1:i2]-i1 # minus the offset as i/j are coordinates
        u = u[j1:j2,i1:i2]
        v = v[j1:j2,i1:i2]
    
    ### downsample the quiver plot (if necessary)
    if ds_step != 1:
        i = i[0::ds_step,0::ds_step]
        j = j[0::ds_step,0::ds_step]
        u = u[0::ds_step,0::ds_step]
        v = v[0::ds_step,0::ds_step]
    
    # plot with options
    if plot_option != 'quiver_only':
        ax.imshow(img_2d_disp, origin=origin_imshow, vmin=vmin, vmax=vmax, cmap='gray')
    if plot_option != 'imshow_only':
        if not color_code:
            ### plot the vector field in fixed color, magnitude coded by lengths
            ax.quiver(i, j, u, v, color='r', angles='xy', scale_units='xy', scale=scale)
        else:
            ### plot the vector field, magnitude coded by both length and colors
            
            # calculate the magnitude of each vector
            magnitude = np.sqrt(u**2 + v**2)
            
            ### manually fix colorbar (unwanted)
            # print(magnitude.shape)
            # print(magnitude[0][0])
            # magnitude[0][0] = 0
            
            # normalize the magnitude based on the specified method
            if norm_minmax is not None and norm_percentile is not None:
                raise ValueError("Only one normalization method should be specified.")
            
            if norm_minmax is not None:
                if norm_minmax == 'auto':
                    # normalize using the min and max values of the magnitude
                    vmin_norm, vmax_norm = magnitude.min(), magnitude.max()
                else:
                    # normalize using the provided min and max values
                    vmin_norm, vmax_norm = norm_minmax
                magnitude_norm = np.clip((magnitude - vmin_norm) / (vmax_norm - vmin_norm), 0, 1)
            elif norm_percentile is not None:
                # normalize using the provided percentile values
                low_percentile, high_percentile = np.percentile(magnitude, norm_percentile) # norm_percentile is a list/array
                magnitude_norm = np.clip((magnitude - low_percentile) / (high_percentile - low_percentile), 0, 1)
            else:
                # no normalization specified, use the original magnitude values
                magnitude_norm = magnitude
            
            # create a colormap
            cmap = plt.cm.get_cmap('RdYlGn_r')  # red for large, yellow for medium, green for small
            
            # plot the quiver with color-coding based on magnitude
            quiver = ax.quiver(i, j, u, v, magnitude_norm, cmap=cmap, angles='xy', scale_units='xy', scale=scale)
            
            
            # add a colorbar to the plot with original magnitude values
            cbar = plt.colorbar(quiver, ax=ax, label='Vector Magnitude')
            
            # update the colorbar ticks and labels to show original magnitude values
            cbar_ticks = np.linspace(0, 1, 5)
            if norm_minmax is not None:
                if norm_minmax == 'auto':
                    cbar_tick_labels = np.round(vmin_norm + cbar_ticks * (vmax_norm - vmin_norm), 2)
                else:
                    cbar_tick_labels = np.round(vmin_norm + cbar_ticks * (vmax_norm - vmin_norm), 2)
            elif norm_percentile is not None:
                cbar_tick_labels = np.round(low_percentile + cbar_ticks * (high_percentile - low_percentile), 2)
            else:
                cbar_tick_labels = cbar_ticks
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_tick_labels)

            
