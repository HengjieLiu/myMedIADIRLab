import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import scipy.ndimage

# --- HELPER FUNCTIONS (NUMPY/SCIPY) ---

def warp_volume_scipy(image, displacement):
    """
    Warp an image using a displacement field (Backward Warp).
    
    Args:
        image: (H, W, D) numpy array
        displacement: (3, H, W, D) numpy array of voxel offsets
    
    Returns:
        warped_image: (H, W, D) numpy array
    """
    shape = image.shape
    # Create meshgrid of indices (x, y, z)
    # indexing='ij' gives (H, W, D) grids
    ranges = [np.arange(s) for s in shape]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=0) # Shape: (3, H, W, D)
    
    # Add displacement to grid to get sample coordinates
    # sample_coords[0] = x_indices + disp_x
    # sample_coords[1] = y_indices + disp_y
    # ...
    sample_coords = grid + displacement
    
    # Map coordinates (Linear interpolation)
    # mode='nearest' avoids interpolation artifacts at boundaries for grids
    warped = scipy.ndimage.map_coordinates(image, sample_coords, order=1, mode='nearest')
    return warped

def mk_grid_img_3d_numpy(grid_step, line_thickness=1, grid_size=(160, 224, 192), axis_to_slice=0):
    """
    Generates a 3D grid image optimized for a specific viewing axis.
    """
    grid_img = np.zeros(grid_size, dtype=np.float32)
    H, W, D = grid_size
    
    if axis_to_slice == 0: # Optimized for Sagittal (Looking at Y-Z plane)
        # Lines in Y direction (varying Z) and Z direction (varying Y)
        for y in range(0, W, grid_step):
            grid_img[:, y:y+line_thickness, :] = 1
        for z in range(0, D, grid_step):
            grid_img[:, :, z:z+line_thickness] = 1
            
    elif axis_to_slice == 1: # Optimized for Coronal (Looking at X-Z plane)
        # Lines in X and Z
        for x in range(0, H, grid_step):
            grid_img[x:x+line_thickness, :, :] = 1
        for z in range(0, D, grid_step):
            grid_img[:, :, z:z+line_thickness] = 1
            
    elif axis_to_slice == 2: # Optimized for Axial (Looking at X-Y plane)
        # Lines in X and Y
        for x in range(0, H, grid_step):
            grid_img[x:x+line_thickness, :, :] = 1
        for y in range(0, W, grid_step):
            grid_img[:, y:y+line_thickness, :] = 1
            
    return grid_img

def calculate_jdet_numpy(disp):
    """
    Calculate Jacobian Determinant of a displacement field using NumPy/SciPy.
    disp: (3, H, W, D)
    """
    # Central difference kernel
    # [-0.5, 0, 0.5] applied to [x-1, x, x+1] gives 0.5*(x+1 - x-1)
    kernel = np.array([-0.5, 0, 0.5])
    
    # Initialize 3x3 Jacobian tensor for every voxel
    # du[i, j] corresponds to d(u_i) / d(x_j)
    # i=0(x), 1=y, 2=z
    du = np.zeros((3, 3) + disp.shape[1:], dtype=disp.dtype)
    
    for i in range(3): # For each displacement component
        du[i, 0] = scipy.ndimage.correlate1d(disp[i], kernel, axis=0, mode='constant', cval=0.0) # dx
        du[i, 1] = scipy.ndimage.correlate1d(disp[i], kernel, axis=1, mode='constant', cval=0.0) # dy
        du[i, 2] = scipy.ndimage.correlate1d(disp[i], kernel, axis=2, mode='constant', cval=0.0) # dz

    # Add Identity matrix (since J = I + Gradient(u))
    du[0, 0] += 1
    du[1, 1] += 1
    du[2, 2] += 1
    
    # Determinant of 3x3
    det = du[0, 0] * (du[1, 1] * du[2, 2] - du[1, 2] * du[2, 1]) - \
          du[0, 1] * (du[1, 0] * du[2, 2] - du[1, 2] * du[2, 0]) + \
          du[0, 2] * (du[1, 0] * du[2, 1] - du[1, 1] * du[2, 0])
          
    return det

def get_spacing_ijk_mm(affine):
    if affine is None: return (1.0, 1.0, 1.0)
    return tuple(float(x) for x in np.linalg.norm(affine[:3, :3], axis=0))

def get_aspect_ratio_for_view(spacing_ijk, view_type, transpose=True):
    s_i, s_j, s_k = spacing_ijk
    if view_type == 'axial': return s_j / s_i if transpose else s_i / s_j
    elif view_type == 'sagittal': return s_k / s_j if transpose else s_j / s_k
    elif view_type == 'coronal': return s_k / s_i if transpose else s_i / s_k
    return 1.0

# --- MAIN APP ---
class MedicalViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Viewer v1.6 (NumPy/SciPy Only)")
        self.root.geometry("1400x950")
        self.root.configure(bg="#2b2b2b")

        # Data State
        self.img_data = None
        self.dose_data = None
        self.mask_data = None
        
        # DVF State
        self.dvf_data = None # Numpy (3, H, W, D)
        self.jdet_data = None # Numpy (H, W, D)
        self.grid_warped = [None, None, None] # List of 3 numpy arrays
        
        self.affine = None
        self.spacing = (1.0, 1.0, 1.0)
        self.dims = [0, 0, 0]
        self.current_slice = [0, 0, 0]

        # Interaction State
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.active_zoom_ax = None
        self.active_pan_ax = None
        self.zoom_scales = [1.0, 1.0, 1.0] 
        self.pan_offsets = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]] 

        # UI Variables
        self.var_slice_x = tk.StringVar(value="0")
        self.var_slice_y = tk.StringVar(value="0")
        self.var_slice_z = tk.StringVar(value="0")
        
        # Vis settings
        self.wl_window = 0.8; self.wl_level = 0.4
        self.dose_alpha = 0.4; self.dose_max = 70.0; self.dose_min = 1.0
        self.show_dose = tk.BooleanVar(value=True)
        self.show_isolines = tk.BooleanVar(value=True)
        self.sync_zoom = tk.BooleanVar(value=True)
        self.iso_levels_str = tk.StringVar(value="5, 12, 16") 
        self.iso_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']
        
        # DVF UI Vars
        self.show_jdet = tk.BooleanVar(value=False)
        self.log_jdet = tk.BooleanVar(value=False)
        self.show_grid = tk.BooleanVar(value=False)
        self.hide_img = tk.BooleanVar(value=False)
        self.jdet_min_var = tk.StringVar(value="-1.0")
        self.jdet_max_var = tk.StringVar(value="3.0")
        self.jdet_min = -1.0
        self.jdet_max = 3.0

        self.artists = {
            'img': [None]*3, 'dose': [None]*3, 'jdet': [None]*3, 'grid': [None]*3,
            'cross_h': [None]*3, 'cross_v': [None]*3, 'contours': [None]*3, 'jdet_contours': [None]*3
        }

        self._setup_ui()
        self._init_plots()

    def _setup_ui(self):
        control_frame = tk.Frame(self.root, width=320, bg="#333333")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(control_frame, text="Controls v1.6", font=("Arial", 14, "bold"), bg="#333", fg="white").pack(pady=10)

        # Load Group
        load_grp = tk.LabelFrame(control_frame, text="Load Data", bg="#333", fg="white")
        load_grp.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(load_grp, text="Load Image (.nii)", command=lambda: self.load_file('img'), bg="#444", fg="white").pack(fill=tk.X, pady=1)
        tk.Button(load_grp, text="Load Dose (.nii)", command=lambda: self.load_file('dose'), bg="#444", fg="white").pack(fill=tk.X, pady=1)
        tk.Button(load_grp, text="Load DVF (.nii)", command=lambda: self.load_file('dvf'), bg="#444", fg="white").pack(fill=tk.X, pady=1)

        # DVF Panel
        self.dvf_grp = tk.LabelFrame(control_frame, text="DVF / Grid / JDet", bg="#333", fg="white")
        self.dvf_grp.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Checkbutton(self.dvf_grp, text="Hide Main Image", variable=self.hide_img, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Checkbutton(self.dvf_grp, text="Show Grid (Backward)", variable=self.show_grid, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        
        f_jdet = tk.Frame(self.dvf_grp, bg="#333")
        f_jdet.pack(fill=tk.X)
        tk.Checkbutton(f_jdet, text="Show JDet", variable=self.show_jdet, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(side=tk.LEFT)
        tk.Checkbutton(f_jdet, text="Log Scale", variable=self.log_jdet, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(side=tk.LEFT, padx=10)

        self.slider_jdet_min = self._create_val_control(self.dvf_grp, "JDet Min", self.jdet_min_var, -5.0, 5.0, self._on_jdet_range, -1.0)
        self.slider_jdet_max = self._create_val_control(self.dvf_grp, "JDet Max", self.jdet_max_var, -5.0, 10.0, self._on_jdet_range, 3.0)

        # View Settings
        view_grp = tk.LabelFrame(control_frame, text="View Settings", bg="#333", fg="white")
        view_grp.pack(fill=tk.X, padx=5, pady=5)
        tk.Checkbutton(view_grp, text="Sync Zoom", variable=self.sync_zoom, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Button(view_grp, text="Reset View", command=self.reset_view, bg="#555", fg="white").pack(fill=tk.X, pady=2)

        # Nav
        nav_grp = tk.LabelFrame(control_frame, text="Navigation", bg="#333", fg="white")
        nav_grp.pack(fill=tk.X, padx=5, pady=5)
        self.slider_x = self._create_nav_control(nav_grp, "Sagittal (X)", self.var_slice_x, 0, 100, 0)
        self.slider_y = self._create_nav_control(nav_grp, "Coronal (Y)", self.var_slice_y, 0, 100, 1)
        self.slider_z = self._create_nav_control(nav_grp, "Axial (Z)", self.var_slice_z, 0, 100, 2)

        # WL
        wl_grp = tk.LabelFrame(control_frame, text="Window / Level", bg="#333", fg="white")
        wl_grp.pack(fill=tk.X, padx=5, pady=5)
        self.slider_win = self._create_val_control(wl_grp, "Win", tk.StringVar(value="0.8"), 0.01, 2.0, self._on_wl, 0.8)
        self.slider_lev = self._create_val_control(wl_grp, "Lev", tk.StringVar(value="0.4"), 0.0, 1.0, self._on_wl, 0.4)

        # Info
        self.info_label = tk.Label(control_frame, text="No Data", bg="#333", fg="#888", justify=tk.LEFT, font=("Consolas", 9))
        self.info_label.pack(side=tk.BOTTOM, pady=10)

        self.plot_frame = tk.Frame(self.root, bg="black")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _create_nav_control(self, parent, label, text_var, min_val, max_val, axis_idx):
        frame = tk.Frame(parent, bg="#333")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, bg="#333", fg="#ccc", width=10, anchor="w").pack(side=tk.LEFT)
        entry = tk.Entry(frame, textvariable=text_var, width=5, bg="#555", fg="white", justify="center")
        entry.pack(side=tk.LEFT, padx=2)
        entry.bind("<Return>", lambda e: self._on_entry_slice(axis_idx))
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                          command=lambda v: self._on_slider_slice(axis_idx, v), 
                          bg="#333", fg="white", showvalue=0, highlightthickness=0, troughcolor="#555")
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return slider

    def _create_val_control(self, parent, label, text_var, min_val, max_val, command, init_val):
        frame = tk.Frame(parent, bg="#333")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, bg="#333", fg="#ccc", width=10, anchor="w").pack(side=tk.LEFT)
        entry = tk.Entry(frame, textvariable=text_var, width=5, bg="#555", fg="white", justify="center")
        entry.pack(side=tk.LEFT, padx=2)
        entry.bind("<Return>", lambda e: command(None))
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                          command=lambda v: [text_var.set(f"{float(v):.2f}"), command(v)], 
                          bg="#333", fg="white", showvalue=0, highlightthickness=0, troughcolor="#555", resolution=0.01)
        slider.set(init_val)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return slider

    def _init_plots(self):
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="black")
        self.ax_ax = self.fig.add_subplot(221); self.ax_sag = self.fig.add_subplot(222); self.ax_cor = self.fig.add_subplot(224)
        self.axes = [self.ax_ax, self.ax_sag, self.ax_cor]
        
        dummy = np.zeros((10, 10))
        for i, ax in enumerate(self.axes):
            ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
            self.artists['img'][i] = ax.imshow(dummy, cmap='gray', vmin=0, vmax=1, origin='lower', animated=True)
            # Layers
            self.artists['jdet'][i] = ax.imshow(dummy, cmap='bwr', vmin=-1, vmax=3, origin='lower', alpha=0, animated=True)
            self.artists['dose'][i] = ax.imshow(dummy, cmap='jet', vmin=0, vmax=1, origin='lower', alpha=0, animated=True)
            self.artists['grid'][i] = ax.imshow(dummy, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=0, animated=True)
            
            self.artists['cross_v'][i] = ax.axvline(5, color='#00FF00', lw=1, alpha=0.6)
            self.artists['cross_h'][i] = ax.axhline(5, color='#00FF00', lw=1, alpha=0.6)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def load_file(self, ftype):
        path = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii;*.nii.gz")])
        if not path: return
        try:
            nii = nib.load(path); nii = nib.as_closest_canonical(nii)
            data = nii.get_fdata()
            
            if ftype == 'img':
                d_min, d_max = np.min(data), np.max(data)
                self.img_data = (data - d_min) / (d_max - d_min) if d_max > d_min else data
                self.dims = data.shape; self.affine = nii.affine
                self.spacing = get_spacing_ijk_mm(self.affine)
                
                self.current_slice = [d//2 for d in self.dims]
                self.slider_x.config(to=self.dims[0]-1); self.slider_y.config(to=self.dims[1]-1); self.slider_z.config(to=self.dims[2]-1)
                
                # Aspect ratios
                self.ax_ax.set_aspect(get_aspect_ratio_for_view(self.spacing, 'axial'))
                self.ax_sag.set_aspect(get_aspect_ratio_for_view(self.spacing, 'sagittal'))
                self.ax_cor.set_aspect(get_aspect_ratio_for_view(self.spacing, 'coronal'))
                self._update_nav_controls()

            elif ftype == 'dose':
                self.dose_data = data
                
            elif ftype == 'dvf':
                # Handle shapes: Expect (3, H, W, D) internally
                if data.ndim == 4:
                    if data.shape[-1] == 3: # (H, W, D, 3) -> (3, H, W, D)
                        data = data.transpose(3, 0, 1, 2)
                
                self.dvf_data = data
                print(f"DVF Loaded. Shape: {self.dvf_data.shape}. Computing JDet & Grids...")
                
                # 1. Compute JDet (Numpy)
                self.jdet_data = calculate_jdet_numpy(self.dvf_data)
                print(f"JDet computed. Range: {self.jdet_data.min():.2f} to {self.jdet_data.max():.2f}")
                
                # 2. Compute Warped Grids (3 Volumes)
                for axis in range(3):
                    # Create clean grid for specific axis
                    grid_img = mk_grid_img_3d_numpy(grid_step=10, grid_size=self.dims, axis_to_slice=axis)
                    # Warp it
                    self.grid_warped[axis] = warp_volume_scipy(grid_img, self.dvf_data)
                
                print("Grids computed.")
                self.show_jdet.set(True) # Auto show
                
            self.update_plots()
            self._update_info()
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")
            print(e)

    def _on_wl(self, _):
        self.wl_window = float(self.slider_win.get())
        self.wl_level = float(self.slider_lev.get())
        self.update_plots()

    def _on_jdet_range(self, _):
        self.jdet_min = float(self.jdet_min_var.get())
        self.jdet_max = float(self.jdet_max_var.get())
        self.update_plots()

    def update_plots(self):
        if self.img_data is None: return
        cx, cy, cz = self.current_slice
        
        vmin = self.wl_level - self.wl_window/2
        vmax = self.wl_level + self.wl_window/2

        def update_ax(idx, slice_img, slice_dose, slice_jdet, slice_grid, cx, cy, max_x, max_y):
            # 1. Main Image
            im = self.artists['img'][idx]
            if not self.hide_img.get():
                im.set_data(slice_img); im.set_clim(vmin, vmax); im.set_extent((0, max_x, 0, max_y)); im.set_alpha(1.0)
            else:
                im.set_data(np.zeros_like(slice_img)); im.set_alpha(0.0)

            # 2. JDet
            jd = self.artists['jdet'][idx]
            if self.artists['jdet_contours'][idx]:
                for c in self.artists['jdet_contours'][idx].collections: c.remove()
                self.artists['jdet_contours'][idx] = None
                
            if self.jdet_data is not None and self.show_jdet.get():
                jd.set_data(slice_jdet); jd.set_extent((0, max_x, 0, max_y)); jd.set_alpha(1.0)
                
                if self.log_jdet.get():
                    jd_disp = np.log(np.maximum(slice_jdet, 1e-6))
                    jd.set_data(jd_disp)
                    jd.set_clim(-1, 1) # Log range
                else:
                    jd.set_clim(self.jdet_min, self.jdet_max)
                
                ax = self.axes[idx]
                self.artists['jdet_contours'][idx] = ax.contour(
                    slice_jdet, levels=[0], colors=['black'], linewidths=1.5, origin='lower', extent=(0, max_x, 0, max_y)
                )
            else:
                jd.set_alpha(0)

            # 3. Grid
            gr = self.artists['grid'][idx]
            if self.grid_warped[0] is not None and self.show_grid.get():
                masked_grid = np.ma.masked_less(slice_grid, 0.1)
                gr.set_data(masked_grid); gr.set_extent((0, max_x, 0, max_y))
                gr.set_alpha(0.8) 
            else:
                gr.set_alpha(0)

            # 4. Crosshairs / Zoom
            self.artists['cross_v'][idx].set_xdata([max_x - cx]*2)
            self.artists['cross_h'][idx].set_ydata([cy]*2)
            
            sc = self.zoom_scales[idx]; px, py = self.pan_offsets[idx]
            vcx, vcy = (max_x+1)/2 - px, (max_y+1)/2 - py
            hw, hh = ((max_x+1)/2)/sc, ((max_y+1)/2)/sc
            self.axes[idx].set_xlim(vcx - hw, vcx + hw)
            self.axes[idx].set_ylim(vcy - hh, vcy + hh)

        # Extract Slices (Transpose + Flip)
        # Axial (Z)
        s_img = np.flip(self.img_data[:, :, cz].T, 1)
        s_jdet = np.flip(self.jdet_data[:, :, cz].T, 1) if self.jdet_data is not None else None
        s_grid = np.flip(self.grid_warped[2][:, :, cz].T, 1) if self.grid_warped[2] is not None else None
        update_ax(0, s_img, None, s_jdet, s_grid, cx, cy, self.dims[0]-1, self.dims[1]-1)
        self.ax_ax.set_title(f"Axial Z={cz}")

        # Sagittal (X)
        s_img = np.flip(self.img_data[cx, :, :].T, 1)
        s_jdet = np.flip(self.jdet_data[cx, :, :].T, 1) if self.jdet_data is not None else None
        s_grid = np.flip(self.grid_warped[0][cx, :, :].T, 1) if self.grid_warped[0] is not None else None
        update_ax(1, s_img, None, s_jdet, s_grid, cy, cz, self.dims[1]-1, self.dims[2]-1)
        self.ax_sag.set_title(f"Sagittal X={cx}")

        # Coronal (Y)
        s_img = np.flip(self.img_data[:, cy, :].T, 1)
        s_jdet = np.flip(self.jdet_data[:, cy, :].T, 1) if self.jdet_data is not None else None
        s_grid = np.flip(self.grid_warped[1][:, cy, :].T, 1) if self.grid_warped[1] is not None else None
        update_ax(2, s_img, None, s_jdet, s_grid, cx, cz, self.dims[0]-1, self.dims[2]-1)
        self.ax_cor.set_title(f"Coronal Y={cy}")

        self.canvas.draw_idle()

    # --- INPUT HANDLERS (Same as before) ---
    def reset_view(self):
        self.zoom_scales = [1.0]*3; self.pan_offsets = [[0,0]]*3
        self.current_slice = [d//2 for d in self.dims]
        self._update_nav_controls(); self.update_plots()

    def _update_nav_controls(self):
        self.var_slice_x.set(str(self.current_slice[0])); self.slider_x.set(self.current_slice[0])
        self.var_slice_y.set(str(self.current_slice[1])); self.slider_y.set(self.current_slice[1])
        self.var_slice_z.set(str(self.current_slice[2])); self.slider_z.set(self.current_slice[2])

    def _on_slider_slice(self, axis, val):
        self.current_slice[axis] = int(float(val)); self._update_nav_controls(); self.update_plots(); self._update_info()

    def _on_entry_slice(self, axis):
        try:
            val = int([self.var_slice_x, self.var_slice_y, self.var_slice_z][axis].get())
            self.current_slice[axis] = max(0, min(val, self.dims[axis]-1))
            self._update_nav_controls(); self.update_plots(); self._update_info()
        except: pass

    def _on_scroll(self, event):
        if self.img_data is None or event.inaxes is None: return
        ax_map = {self.ax_sag: 0, self.ax_cor: 1, self.ax_ax: 2}
        if event.inaxes in ax_map:
            axis = ax_map[event.inaxes]
            self.current_slice[axis] = max(0, min(self.current_slice[axis] + (1 if event.step > 0 else -1), self.dims[axis]-1))
            self._update_nav_controls(); self.update_plots(); self._update_info()

    def _on_mouse_press(self, event):
        if event.inaxes is None: return
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        if event.button == 1:
            ix, iy = int(event.xdata), int(event.ydata)
            if event.inaxes == self.ax_ax: self.current_slice[0] = self.dims[0]-1 - ix; self.current_slice[1] = iy
            elif event.inaxes == self.ax_sag: self.current_slice[1] = self.dims[1]-1 - ix; self.current_slice[2] = iy
            elif event.inaxes == self.ax_cor: self.current_slice[0] = self.dims[0]-1 - ix; self.current_slice[2] = iy
            self._update_nav_controls(); self.update_plots(); self._update_info()
        elif event.button == 3: self.active_zoom_ax = event.inaxes
        elif event.button == 2: self.active_pan_ax = event.inaxes

    def _on_mouse_release(self, event): self.active_zoom_ax = None; self.active_pan_ax = None

    def _on_mouse_move(self, event):
        if event.inaxes is None: return
        dx, dy = event.x - self.last_mouse_x, event.y - self.last_mouse_y
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        if self.active_zoom_ax:
            zf = 1 + dy * 0.01
            for i, ax in enumerate(self.axes):
                if self.sync_zoom.get() or ax == self.active_zoom_ax:
                    self.zoom_scales[i] = np.clip(self.zoom_scales[i] * zf, 0.1, 20)
            self.update_plots()
        elif self.active_pan_ax == event.inaxes: pass 

    def _update_info(self):
        cx, cy, cz = self.current_slice
        txt = f"Pos: ({cx}, {cy}, {cz})\n"
        if self.img_data is not None: txt += f"Img: {self.img_data[cx,cy,cz]:.2f}\n"
        if self.jdet_data is not None: txt += f"JDet: {self.jdet_data[cx,cy,cz]:.2f}"
        self.info_label.config(text=txt)

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalViewerApp(root)
    root.mainloop()