import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import scipy.ndimage

# --- HELPER FUNCTIONS (MATH & GEOMETRY) ---

def warp_volume_scipy(image, displacement):
    """Backward Warp using Scipy."""
    shape = image.shape
    ranges = [np.arange(s) for s in shape]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=0)
    sample_coords = grid + displacement
    warped = scipy.ndimage.map_coordinates(image, sample_coords, order=1, mode='nearest')
    return warped

def mk_grid_img_3d_numpy(grid_step, line_thickness=1, grid_size=(160, 224, 192), axis_to_slice=0):
    """Generate 3D grid volume optimized for specific axis slicing."""
    grid_img = np.zeros(grid_size, dtype=np.float32)
    H, W, D = grid_size
    if axis_to_slice == 0:
        for y in range(0, W, grid_step): grid_img[:, y:y+line_thickness, :] = 1
        for z in range(0, D, grid_step): grid_img[:, :, z:z+line_thickness] = 1
    elif axis_to_slice == 1:
        for x in range(0, H, grid_step): grid_img[x:x+line_thickness, :, :] = 1
        for z in range(0, D, grid_step): grid_img[:, :, z:z+line_thickness] = 1
    elif axis_to_slice == 2:
        for x in range(0, H, grid_step): grid_img[x:x+line_thickness, :, :] = 1
        for y in range(0, W, grid_step): grid_img[:, y:y+line_thickness, :] = 1
    return grid_img

def calculate_jdet_numpy(disp):
    """Calculate Jacobian Determinant (NumPy)."""
    kernel = np.array([-0.5, 0, 0.5])
    du = np.zeros((3, 3) + disp.shape[1:], dtype=disp.dtype)
    for i in range(3):
        du[i, 0] = scipy.ndimage.correlate1d(disp[i], kernel, axis=0, mode='constant')
        du[i, 1] = scipy.ndimage.correlate1d(disp[i], kernel, axis=1, mode='constant')
        du[i, 2] = scipy.ndimage.correlate1d(disp[i], kernel, axis=2, mode='constant')
    
    du[0, 0] += 1; du[1, 1] += 1; du[2, 2] += 1
    det = du[0, 0]*(du[1, 1]*du[2, 2] - du[1, 2]*du[2, 1]) - \
          du[0, 1]*(du[1, 0]*du[2, 2] - du[1, 2]*du[2, 0]) + \
          du[0, 2]*(du[1, 0]*du[2, 1] - du[1, 1]*du[2, 0])
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

# --- DATA CONTAINER CLASS ---
class Dataset:
    def __init__(self, name="Data"):
        self.name = name
        self.img = None
        self.dose = None
        self.mask = None
        self.affine = None
        self.dims = [0, 0, 0]
        self.spacing = (1.0, 1.0, 1.0)
        self.aspects = [1.0, 1.0, 1.0]

# --- MAIN APPLICATION ---
class MedicalViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Viewer Pro v1.7 (Multi-Mode)")
        self.root.geometry("1600x950")
        self.root.configure(bg="#2b2b2b")

        # --- Data State ---
        # Slot 0 is Main/Left, Slot 1 is Secondary/Right
        self.datasets = [Dataset("Set 1"), Dataset("Set 2")]
        
        # DVF State (Specific to Mode 3)
        self.dvf_data = None
        self.jdet_data = None
        self.grid_warped = [None, None, None]

        # Global Navigation State
        self.current_slice = [0, 0, 0]
        self.global_dims = [0, 0, 0] # Master dimensions (usually Set 1)

        # UI State
        self.mode = 1 # 1=Single, 2=Compare, 3=DVF
        self.sync_zoom = tk.BooleanVar(value=True)
        self.show_dose = tk.BooleanVar(value=True)
        self.show_isolines = tk.BooleanVar(value=True)
        self.show_jdet = tk.BooleanVar(value=True) # Mode 3
        self.show_grid = tk.BooleanVar(value=False) # Mode 3
        self.log_jdet = tk.BooleanVar(value=False)
        
        # Vis Params
        self.wl_window = 0.8; self.wl_level = 0.4
        self.dose_alpha = 0.4
        self.jdet_range = [-1.0, 3.0]
        self.iso_levels_str = tk.StringVar(value="5, 12, 16")
        self.iso_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0']

        # Zoom/Pan
        # We need a list of axes to manage interactions. Rebuilt on mode switch.
        self.active_axes = [] 
        self.zoom_scales = {} # keyed by axis object
        self.pan_offsets = {} # keyed by axis object
        
        # Components
        self._setup_menu()
        self.main_container = tk.Frame(self.root, bg="#2b2b2b")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        self.control_frame = None
        self.plot_frame = None
        
        # Init
        self.switch_mode(1)

    def _setup_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        mode_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View Mode", menu=mode_menu)
        mode_menu.add_command(label="Mode 1: Single Analysis", command=lambda: self.switch_mode(1))
        mode_menu.add_command(label="Mode 2: Comparison (Side-by-Side)", command=lambda: self.switch_mode(2))
        mode_menu.add_command(label="Mode 3: Deformation Analysis", command=lambda: self.switch_mode(3))

    def switch_mode(self, mode_idx):
        self.mode = mode_idx
        # Clear UI
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Build Frames
        self.control_frame = tk.Frame(self.main_container, width=320, bg="#333")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.plot_frame = tk.Frame(self.main_container, bg="black")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls()
        self._init_plots()
        self.update_plots()

    # --- UI BUILDER ---
    def _build_controls(self):
        # Header
        modes = {1: "Single Analysis", 2: "Comparison", 3: "Deformation"}
        tk.Label(self.control_frame, text=modes[self.mode], font=("Arial", 14, "bold"), bg="#333", fg="white").pack(pady=10)

        # 1. Loaders
        load_grp = tk.LabelFrame(self.control_frame, text="Data Loading", bg="#333", fg="white")
        load_grp.pack(fill=tk.X, padx=5, pady=5)
        
        # Helper to make load buttons
        def mk_btn(parent, txt, slot, type_):
            cmd = lambda: self.load_data(slot, type_)
            tk.Button(parent, text=txt, command=cmd, bg="#444", fg="white", font=("Arial", 8)).pack(fill=tk.X, pady=1)

        if self.mode == 1:
            mk_btn(load_grp, "Load Image (.nii)", 0, 'img')
            mk_btn(load_grp, "Load Dose (.nii)", 0, 'dose')
        elif self.mode == 2:
            f1 = tk.Frame(load_grp, bg="#333"); f1.pack(fill=tk.X)
            tk.Label(f1, text="Left Panel (Set 1)", bg="#333", fg="#ccc").pack(anchor="w")
            mk_btn(f1, "Load Image 1", 0, 'img')
            mk_btn(f1, "Load Dose 1", 0, 'dose')
            
            f2 = tk.Frame(load_grp, bg="#333"); f2.pack(fill=tk.X, pady=(10,0))
            tk.Label(f2, text="Right Panel (Set 2)", bg="#333", fg="#ccc").pack(anchor="w")
            mk_btn(f2, "Load Image 2", 1, 'img')
            mk_btn(f2, "Load Dose 2", 1, 'dose')
        elif self.mode == 3:
            mk_btn(load_grp, "Load Image (.nii)", 0, 'img')
            mk_btn(load_grp, "Load DVF (.nii)", 0, 'dvf')

        # 2. View Settings
        view_grp = tk.LabelFrame(self.control_frame, text="View Control", bg="#333", fg="white")
        view_grp.pack(fill=tk.X, padx=5, pady=5)
        tk.Checkbutton(view_grp, text="Sync Zoom/Pan", variable=self.sync_zoom, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Button(view_grp, text="Reset View", command=self.reset_view, bg="#555", fg="white").pack(fill=tk.X, pady=2)
        
        # 3. Mode Specific Settings
        if self.mode == 3:
            dvf_grp = tk.LabelFrame(self.control_frame, text="DVF Display", bg="#333", fg="white")
            dvf_grp.pack(fill=tk.X, padx=5, pady=5)
            tk.Checkbutton(dvf_grp, text="Show JDet", variable=self.show_jdet, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
            tk.Checkbutton(dvf_grp, text="Log Scale", variable=self.log_jdet, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
            tk.Checkbutton(dvf_grp, text="Show Grid", variable=self.show_grid, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")

        # 4. Navigation
        nav_grp = tk.LabelFrame(self.control_frame, text="Navigation", bg="#333", fg="white")
        nav_grp.pack(fill=tk.X, padx=5, pady=5)
        
        self.var_slice = [tk.StringVar(value="0") for _ in range(3)]
        labels = ["Sag (X)", "Cor (Y)", "Ax (Z)"]
        self.sliders = []
        for i in range(3):
            f = tk.Frame(nav_grp, bg="#333")
            f.pack(fill=tk.X)
            tk.Label(f, text=labels[i], bg="#333", fg="#ccc", width=6).pack(side=tk.LEFT)
            tk.Entry(f, textvariable=self.var_slice[i], width=5, bg="#555", fg="white").pack(side=tk.LEFT)
            self.var_slice[i].trace("w", lambda *args, ax=i: self._on_entry(ax))
            s = tk.Scale(f, from_=0, to=100, orient=tk.HORIZONTAL, bg="#333", fg="white", showvalue=0, command=lambda v, ax=i: self._on_slider(ax, v))
            s.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.sliders.append(s)

        # 5. Vis Params (WL / Dose)
        vis_grp = tk.LabelFrame(self.control_frame, text="Visualization", bg="#333", fg="white")
        vis_grp.pack(fill=tk.X, padx=5, pady=5)
        
        # WL
        f_wl = tk.Frame(vis_grp, bg="#333"); f_wl.pack(fill=tk.X)
        tk.Label(f_wl, text="Win/Lev", bg="#333", fg="#ccc").pack(side=tk.LEFT)
        self.sl_win = tk.Scale(f_wl, from_=0.01, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, bg="#333", fg="white", showvalue=0, command=self.update_plots); self.sl_win.set(0.8); self.sl_win.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.sl_lev = tk.Scale(f_wl, from_=0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, bg="#333", fg="white", showvalue=0, command=self.update_plots); self.sl_lev.set(0.4); self.sl_lev.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Iso
        tk.Label(vis_grp, text="Iso Levels:", bg="#333", fg="#ccc").pack(anchor="w")
        self.ent_iso = tk.Entry(vis_grp, textvariable=self.iso_levels_str, bg="#555", fg="white")
        self.ent_iso.pack(fill=tk.X)
        self.ent_iso.bind("<Return>", lambda e: self.update_plots())
        tk.Button(vis_grp, text="Update Iso", command=self.update_plots, bg="#555", fg="white", height=1).pack(fill=tk.X, pady=2)

    def _init_plots(self):
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="black")
        self.active_axes = []
        self.zoom_scales = {}
        self.pan_offsets = {}
        self.cbar_refs = {} # To store colorbar objects to update them

        # --- LAYOUT LOGIC ---
        if self.mode == 1:
            # Mode 1: 2x2 grid (Ax, Sag, Cor, Hist)
            # Layout: 221(Ax), 222(Sag), 223(Hist), 224(Cor) ?? Or classic
            # Let's do: Top-Left: Ax, Top-Right: Sag, Bot-Left: Hist, Bot-Right: Cor
            gs = self.fig.add_gridspec(2, 2)
            self.ax_map = {
                'view': [self.fig.add_subplot(gs[0,0]), self.fig.add_subplot(gs[0,1]), self.fig.add_subplot(gs[1,1])], # Ax, Sag, Cor
                'hist': self.fig.add_subplot(gs[1,0])
            }
            # Ax mapping: 0=Ax(Z), 1=Sag(X), 2=Cor(Y)
            # Titles
            self.ax_map['view'][0].set_title("Axial", color='white')
            self.ax_map['view'][1].set_title("Sagittal", color='white')
            self.ax_map['view'][2].set_title("Coronal", color='white')
            self.active_axes = self.ax_map['view']

        elif self.mode == 2:
            # Mode 2: Left Panel (3 views), Right Panel (3 views)
            # 2 Columns, 3 Rows? Or 2 big columns each with 3 subplots.
            # Let's do 2 Columns x 3 Rows grid.
            gs = self.fig.add_gridspec(3, 2, width_ratios=[1, 1])
            self.ax_map = {'left': [], 'right': []}
            titles = ["Sagittal", "Coronal", "Axial"] # Order X, Y, Z for display rows
            # Display order: Row 0: Sag, Row 1: Cor, Row 2: Ax (Just a choice)
            # Or standard: Ax, Sag, Cor. Let's do Standard vertical stack
            titles = ["Axial", "Sagittal", "Coronal"]
            
            for i in range(3):
                ax_l = self.fig.add_subplot(gs[i, 0])
                ax_r = self.fig.add_subplot(gs[i, 1])
                ax_l.set_title(f"{titles[i]} (Set 1)", color='white', fontsize=8)
                ax_r.set_title(f"{titles[i]} (Set 2)", color='white', fontsize=8)
                self.ax_map['left'].append(ax_l)
                self.ax_map['right'].append(ax_r)
            
            # Map logical views (Ax, Sag, Cor) to the list. 
            # If I stacked them 0,1,2 corresponding to titles:
            self.active_axes = self.ax_map['left'] + self.ax_map['right']

        elif self.mode == 3:
            # Mode 3: Left (Image), Right (DVF)
            gs = self.fig.add_gridspec(3, 2)
            self.ax_map = {'img': [], 'dvf': []}
            titles = ["Axial", "Sagittal", "Coronal"]
            for i in range(3):
                ax_l = self.fig.add_subplot(gs[i, 0])
                ax_r = self.fig.add_subplot(gs[i, 1])
                ax_l.set_title(f"{titles[i]} (Img)", color='white', fontsize=8)
                ax_r.set_title(f"{titles[i]} (DVF)", color='white', fontsize=8)
                self.ax_map['img'].append(ax_l)
                self.ax_map['dvf'].append(ax_r)
            self.active_axes = self.ax_map['img'] + self.ax_map['dvf']

        # Init Axes Appearance
        for ax in self.active_axes:
            ax.set_facecolor("black")
            ax.tick_params(colors='white', labelbottom=False, labelleft=False)
            self.zoom_scales[ax] = 1.0
            self.pan_offsets[ax] = [0.0, 0.0]

        if self.mode == 1:
            self.ax_map['hist'].set_facecolor("#111")
            self.ax_map['hist'].tick_params(colors='white')
            for spine in self.ax_map['hist'].spines.values(): spine.set_edgecolor('#555')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Events
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_move)

        # Fix scrolling focus issue
        self.canvas.get_tk_widget().bind('<Enter>', lambda e: self.canvas.get_tk_widget().focus_set())

    # --- LOADING ---
    def load_data(self, slot_idx, ftype):
        path = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii;*.nii.gz")])
        if not path: return
        try:
            nii = nib.load(path); nii = nib.as_closest_canonical(nii)
            data = nii.get_fdata()
            
            if ftype == 'img':
                d_min, d_max = np.min(data), np.max(data)
                if d_max > d_min: data = (data - d_min) / (d_max - d_min)
                self.datasets[slot_idx].img = data
                self.datasets[slot_idx].affine = nii.affine
                self.datasets[slot_idx].dims = data.shape
                # Calc aspect
                sp = get_spacing_ijk_mm(nii.affine)
                self.datasets[slot_idx].spacing = sp
                self.datasets[slot_idx].aspects = [
                    get_aspect_ratio_for_view(sp, 'axial'),
                    get_aspect_ratio_for_view(sp, 'sagittal'),
                    get_aspect_ratio_for_view(sp, 'coronal')
                ]
                
                # If loading to slot 0, update globals
                if slot_idx == 0:
                    self.global_dims = data.shape
                    self.current_slice = [d//2 for d in data.shape]
                    for i in range(3): 
                        self.sliders[i].config(to=self.global_dims[i]-1)
                    self._update_nav_ui()
            
            elif ftype == 'dose':
                self.datasets[slot_idx].dose = data
            
            elif ftype == 'dvf':
                # Mode 3 only
                if data.ndim == 4 and data.shape[-1] == 3:
                    data = data.transpose(3, 0, 1, 2)
                self.dvf_data = data
                print("Computing DVF data...")
                # JDet
                self.jdet_data = calculate_jdet_numpy(data)
                # Grid
                st_grid = []
                for axis in range(3):
                    g = mk_grid_img_3d_numpy(10, 1, self.datasets[0].dims, axis)
                    st_grid.append(warp_volume_scipy(g, data))
                self.grid_warped = st_grid
                print("DVF Computed.")

            self.update_plots()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(e)

    # --- PLOTTING ---
    def update_plots(self):
        if self.datasets[0].img is None: return
        
        # Helper to draw colorbar
        def update_colorbar(ax, mappable, label):
            # We use a unique key for cbar_refs based on ax
            # If exists, update? simpler to remove and re-add or clear
            if ax not in self.cbar_refs:
                div = make_axes_locatable(ax)
                cax = div.append_axes("right", size="5%", pad=0.05)
                self.cbar_refs[ax] = cax
            else:
                cax = self.cbar_refs[ax]
                cax.clear()
            
            cbar = self.fig.colorbar(mappable, cax=cax)
            cbar.ax.tick_params(labelsize=6, colors='white')
            # cbar.set_label(label, color='white', fontsize=6) # Optional

        # Parse Levels
        try:
            levels = sorted([float(x) for x in self.iso_levels_str.get().split(',') if x.strip()])
        except: levels = []

        # Prepare Shared Data
        cx, cy, cz = self.current_slice
        wl_v = self.sl_lev.get(); wl_w = self.sl_win.get()
        vmin, vmax = wl_v - wl_w/2, wl_v + wl_w/2

        # --- DRAW FUNCTION ---
        def draw_panel(ax_list, dataset, is_dvf_panel=False):
            # ax_list order: [Ax, Sag, Cor]
            # Slicing logic:
            # 0: Axial (Z) -> Show (X, Y) -> Data is (X, Y, Z). Slice Z. Transpose -> (Y, X). Flip 1 -> (Y, -X) Correct?
            # Standard logic from v1.5:
            # Ax: flip(img[:,:,z].T, 1)
            # Sag: flip(img[x,:,:].T, 1)
            # Cor: flip(img[:,y,:].T, 1)
            
            # Slices
            try:
                if not is_dvf_panel:
                    slices = [
                        np.flip(dataset.img[:, :, cz].T, 1), # Ax
                        np.flip(dataset.img[cx, :, :].T, 1), # Sag
                        np.flip(dataset.img[:, cy, :].T, 1)  # Cor
                    ]
                    
                    dose_slices = [None]*3
                    if dataset.dose is not None and self.show_dose.get():
                        dose_slices = [
                            np.flip(dataset.dose[:, :, cz].T, 1),
                            np.flip(dataset.dose[cx, :, :].T, 1),
                            np.flip(dataset.dose[:, cy, :].T, 1)
                        ]
                else:
                    # DVF Logic (JDet or Grid)
                    # We map JDet to image slices logic
                    if self.jdet_data is not None and self.show_jdet.get():
                         slices = [
                            np.flip(self.jdet_data[:, :, cz].T, 1),
                            np.flip(self.jdet_data[cx, :, :].T, 1),
                            np.flip(self.jdet_data[:, cy, :].T, 1)
                        ]
                    elif self.grid_warped[0] is not None and self.show_grid.get():
                         # Grid slices. Note: Grid warped is list [SagGrid, CorGrid, AxGrid]
                         # But here we need specific cuts.
                         # Simplified: Use the AxGrid for Ax view, etc.
                         slices = [
                            np.flip(self.grid_warped[2][:, :, cz].T, 1),
                            np.flip(self.grid_warped[0][cx, :, :].T, 1),
                            np.flip(self.grid_warped[1][:, cy, :].T, 1)
                         ]
                    else:
                        slices = [np.zeros((10,10))]*3

            except IndexError: return # Handle out of bounds if set 2 is smaller

            titles = ["Axial", "Sagittal", "Coronal"] # Matches indices 0,1,2

            for i, ax in enumerate(ax_list):
                ax.clear()
                ax.set_title(titles[i], color='white', fontsize=8)
                
                # Base Image / Heatmap
                if not is_dvf_panel:
                    im = ax.imshow(slices[i], cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect=dataset.aspects[i])
                    # Colorbar for Image? Usually not needed for Grayscale anatomy, but requested.
                    # Let's add it only for the first view or if space permits.
                    # update_colorbar(ax, im, "Int") 

                    # Dose Overlay
                    if dose_slices[i] is not None:
                        # Mask low dose
                        d_show = np.ma.masked_less(dose_slices[i], 1.0)
                        d_im = ax.imshow(d_show, cmap='jet', alpha=self.dose_alpha, origin='lower', aspect=dataset.aspects[i])
                        # Add Dose Colorbar
                        update_colorbar(ax, d_im, "Dose")
                        
                        # Isolines
                        if self.show_isolines.get() and levels:
                            ax.contour(dose_slices[i], levels=levels, colors=self.iso_colors[:len(levels)], linewidths=1, origin='lower')
                
                else:
                    # DVF Panel
                    if self.show_jdet.get() and self.jdet_data is not None:
                        # Log Scale
                        data_plot = slices[i]
                        if self.log_jdet.get():
                            data_plot = np.log(np.maximum(data_plot, 1e-6))
                            c_min, c_max = -1, 1
                        else:
                            c_min, c_max = -1, 3
                        
                        im = ax.imshow(data_plot, cmap='bwr', vmin=c_min, vmax=c_max, origin='lower', aspect=self.datasets[0].aspects[i])
                        update_colorbar(ax, im, "JDet")
                        # Folding contour
                        ax.contour(slices[i], levels=[0], colors=['black'], linewidths=1.5, origin='lower')
                    
                    elif self.show_grid.get():
                        ax.imshow(slices[i], cmap='gray', origin='lower', aspect=self.datasets[0].aspects[i])
                
                # Crosshairs
                h, w = slices[i].shape
                # Coordinates mapping needs care.
                # Ax(0): Z fixed. Show Y(rows), X(cols). Crosshair: X=cx, Y=cy.
                # Sag(1): X fixed. Show Z(rows), Y(cols). Crosshair: Y=cy, Z=cz.
                # Cor(2): Y fixed. Show Z(rows), X(cols). Crosshair: X=cx, Z=cz.
                # NOTE: Because of Transpose + Flip, coordinates swap.
                # Flip(T, 1) -> Rows=Original Cols (flipped?), Cols=Original Rows.
                
                # Simplified Center Logic based on Extent
                # Let's trust the v1.5 math:
                # Ax: lines at x=max_x-cx, y=cy
                # Sag: lines at x=cy, y=cz
                # Cor: lines at x=max_x-cx, y=cz
                
                if i == 0: lx, ly = w - 1 - cx, cy
                elif i == 1: lx, ly = cy, cz
                elif i == 2: lx, ly = w - 1 - cx, cz
                
                ax.axvline(lx, color='#00FF00', lw=0.8, alpha=0.5)
                ax.axhline(ly, color='#00FF00', lw=0.8, alpha=0.5)

                # Zoom/Pan Application
                scale = self.zoom_scales.get(ax, 1.0)
                px, py = self.pan_offsets.get(ax, [0,0])
                
                # Center of view
                vcx, vcy = w/2 - px, h/2 - py
                hw, hh = (w/2)/scale, (h/2)/scale
                ax.set_xlim(vcx - hw, vcx + hw)
                ax.set_ylim(vcy - hh, vcy + hh)

        # --- EXECUTE DRAW BASED ON MODE ---
        if self.mode == 1:
            draw_panel(self.ax_map['view'], self.datasets[0])
            # Update Histogram
            ax_h = self.ax_map['hist']
            ax_h.clear()
            ax_h.set_facecolor("#111")
            ax_h.tick_params(colors='white', labelsize=8)
            if self.datasets[0].dose is not None:
                d = self.datasets[0].dose[self.datasets[0].dose > 0.5]
                ax_h.hist(d, bins=50, color='blue', alpha=0.5, log=True)
                # Fix Legends
                patches = []
                for idx, lvl in enumerate(levels):
                    col = self.iso_colors[idx % len(self.iso_colors)]
                    ax_h.axvline(lvl, color=col, lw=2)
                    patches.append(mpatches.Patch(color=col, label=f'{lvl} Gy'))
                if patches:
                    ax_h.legend(handles=patches, facecolor='#333', labelcolor='white', fontsize=8, loc='upper right')
        
        elif self.mode == 2:
            # Re-order axes to match Ax, Sag, Cor logic passed to draw_panel
            # ax_map['left'] contains [Ax, Sag, Cor] in that order? 
            # In init: added Ax, Sag, Cor.
            draw_panel(self.ax_map['left'], self.datasets[0])
            draw_panel(self.ax_map['right'], self.datasets[1]) # Might be None, handled in try/except
            
        elif self.mode == 3:
            draw_panel(self.ax_map['img'], self.datasets[0])
            draw_panel(self.ax_map['dvf'], self.datasets[0], is_dvf_panel=True)

        self.canvas.draw_idle()

    # --- INTERACTION ---
    def reset_view(self):
        for ax in self.active_axes:
            self.zoom_scales[ax] = 1.0
            self.pan_offsets[ax] = [0.0, 0.0]
        self.current_slice = [d//2 for d in self.global_dims]
        self._update_nav_ui()
        self.update_plots()

    def _update_nav_ui(self):
        for i in range(3):
            self.var_slice[i].set(str(self.current_slice[i]))
            self.sliders[i].set(self.current_slice[i])

    def _on_entry(self, ax_idx):
        try:
            val = int(self.var_slice[ax_idx].get())
            val = max(0, min(val, self.global_dims[ax_idx]-1))
            self.current_slice[ax_idx] = val
            self.sliders[ax_idx].set(val) # Avoid loop? set sends command? No
            self.update_plots()
        except: pass

    def _on_slider(self, ax_idx, val):
        self.current_slice[ax_idx] = int(val)
        self.var_slice[ax_idx].set(str(int(val)))
        self.update_plots()

    def _on_scroll(self, event):
        if event.inaxes is None: return
        # Find which dimension to scroll based on view type
        # Simplification: Standard mapping based on ax index in list
        # We need to reverse engineer which axis index (0=Ax, 1=Sag, 2=Cor) belongs to event.inaxes
        view_idx = -1
        for lst in [self.ax_map.get('view',[]), self.ax_map.get('left',[]), self.ax_map.get('right',[]), self.ax_map.get('img',[]), self.ax_map.get('dvf',[])]:
            if event.inaxes in lst:
                view_idx = lst.index(event.inaxes)
                break
        
        if view_idx != -1:
            # Map view_idx to slice index. 
            # 0(Ax) -> Scroll Z(2). 1(Sag) -> Scroll X(0). 2(Cor) -> Scroll Y(1).
            target_dim = {0: 2, 1: 0, 2: 1}[view_idx]
            
            step = 1 if event.button == 'up' else -1
            self.current_slice[target_dim] = max(0, min(self.current_slice[target_dim] + step, self.global_dims[target_dim]-1))
            self._update_nav_ui()
            self.update_plots()

    # Mouse Drag Logic (Zoom/Pan/Click)
    def _on_press(self, event):
        if event.inaxes in self.active_axes:
            self.last_x, self.last_y = event.x, event.y
            
            # Button 1: Left Click (Navigation / Jump)
            if event.button == 1:
                # Need to determine which axes we clicked (Axial/Sag/Cor) to know how to map x/y to slices
                view_idx = -1
                for lst in [self.ax_map.get('view',[]), self.ax_map.get('left',[]), self.ax_map.get('right',[]), self.ax_map.get('img',[]), self.ax_map.get('dvf',[])]:
                    if event.inaxes in lst:
                        view_idx = lst.index(event.inaxes)
                        break
                
                if view_idx != -1:
                    # Logic derived from draw_panel construction
                    # Slices are transposed and flipped.
                    # Ax(0): x_plot = Dim0_len - 1 - slice_x. y_plot = slice_y.
                    # Sag(1): x_plot = slice_y. y_plot = slice_z.
                    # Cor(2): x_plot = Dim0_len - 1 - slice_x. y_plot = slice_z.
                    
                    ix = int(event.xdata)
                    iy = int(event.ydata)
                    
                    if view_idx == 0: # Axial View (updates X and Y)
                        # Inverse of: lx = w - 1 - cx -> cx = w - 1 - lx
                        self.current_slice[0] = max(0, min(self.global_dims[0] - 1 - ix, self.global_dims[0]-1))
                        self.current_slice[1] = max(0, min(iy, self.global_dims[1]-1))
                    
                    elif view_idx == 1: # Sagittal View (updates Y and Z) -- Actually X is fixed slice
                        # Inverse of: lx = cy, ly = cz
                        # cx (Slice X) is the slice we are ON, so we don't change it by clicking HERE
                        self.current_slice[1] = max(0, min(ix, self.global_dims[1]-1))
                        self.current_slice[2] = max(0, min(iy, self.global_dims[2]-1))
                        
                    elif view_idx == 2: # Coronal View (updates X and Z)
                         # Inverse of: lx = w - 1 - cx, ly = cz
                        self.current_slice[0] = max(0, min(self.global_dims[0] - 1 - ix, self.global_dims[0]-1))
                        self.current_slice[2] = max(0, min(iy, self.global_dims[2]-1))

                    self._update_nav_ui()
                    self.update_plots()

            elif event.button == 3: self.drag_mode = 'zoom'; self.active_ax = event.inaxes
            elif event.button == 2: self.drag_mode = 'pan'; self.active_ax = event.inaxes

    def _on_release(self, event): self.drag_mode = None; self.active_ax = None

    def _on_move(self, event):
        if not hasattr(self, 'drag_mode') or self.drag_mode is None or event.inaxes is None: return
        dx, dy = event.x - self.last_x, event.y - self.last_y
        self.last_x, self.last_y = event.x, event.y

        axes_to_update = self.active_axes if self.sync_zoom.get() else [self.active_ax]

        if self.drag_mode == 'zoom':
            factor = 1 + dy * 0.01
            for ax in axes_to_update:
                self.zoom_scales[ax] = max(0.1, min(20.0, self.zoom_scales[ax] * factor))
            self.update_plots()
        
        elif self.drag_mode == 'pan':
            # Need to scale pixel delta to data delta
            # Approximate for performance
            for ax in axes_to_update:
                self.pan_offsets[ax][0] += dx * 0.5 
                self.pan_offsets[ax][1] += dy * 0.5
            self.update_plots()

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalViewerApp(root)
    root.mainloop()