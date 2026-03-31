import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

# --- Helper Functions for Geometry ---
def get_spacing_ijk_mm(affine):
    """
    Calculate voxel spacing (resolution) in mm along i, j, k axes (0, 1, 2)
    from the 4x4 affine matrix.
    """
    if affine is None:
        return (1.0, 1.0, 1.0)
    
    # Extract the 3x3 linear part (rotation + scaling)
    M = affine[:3, :3]
    
    # Calculate the norm of each column vector
    spacings = np.linalg.norm(M, axis=0)
    return tuple(float(x) for x in spacings)

def get_aspect_ratio_for_view(spacing_ijk, view_type, transpose=True):
    """
    Calculate aspect ratio for matplotlib imshow based on voxel spacing
    and whether the image slice is transposed before display.
    """
    s_i, s_j, s_k = spacing_ijk
    
    if view_type == 'axial':
        return s_j / s_i if transpose else s_i / s_j
    elif view_type == 'sagittal':
        return s_k / s_j if transpose else s_j / s_k
    elif view_type == 'coronal':
        return s_k / s_i if transpose else s_i / s_k
    
    return 1.0

# --- Main Application ---
class MedicalViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Medical Dose Viewer (High Performance + True Aspect)")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2b2b2b")

        # --- Data State ---
        self.img_data = None
        self.dose_data = None
        self.mask_data = None
        self.affine = None
        self.spacing = (1.0, 1.0, 1.0) # (sx, sy, sz)
        
        # Coordinate state (x, y, z)
        self.current_slice = [0, 0, 0]
        self.dims = [0, 0, 0]

        # Interaction State
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.active_zoom_ax = None
        self.active_pan_ax = None
        
        # View State [Axial, Sagittal, Coronal]
        self.zoom_scales = [1.0, 1.0, 1.0] 
        self.pan_offsets = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]] 
        self.aspect_ratios = [1.0, 1.0, 1.0] # Store calculated aspects

        # Visualization settings
        self.wl_window = 0.8
        self.wl_level = 0.4
        self.dose_alpha = 0.4
        self.dose_max = 70.0
        self.dose_min = 1.0
        self.show_dose = tk.BooleanVar(value=True)
        self.show_isolines = tk.BooleanVar(value=True)
        
        # New Sync Setting
        self.sync_zoom = tk.BooleanVar(value=True)
        
        self.iso_levels_str = tk.StringVar(value="5, 12, 16") 
        self.iso_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

        # UI Variables
        self.var_slice_x = tk.StringVar(value="0")
        self.var_slice_y = tk.StringVar(value="0")
        self.var_slice_z = tk.StringVar(value="0")
        self.var_win = tk.StringVar(value="0.8")
        self.var_lev = tk.StringVar(value="0.4")
        self.var_dose_max = tk.StringVar(value="70.0")
        self.var_dose_min = tk.StringVar(value="1.0")

        # --- Graphics Storage ---
        self.artists = {
            'img': [None, None, None],     # [Axial, Sagittal, Coronal]
            'dose': [None, None, None],    # Dose overlays
            'cross_h': [None, None, None], # Horizontal Crosshair lines
            'cross_v': [None, None, None], # Vertical Crosshair lines
            'contours': [None, None, None] # Contour sets
        }

        self._setup_ui()
        self._init_plots()

    def _setup_ui(self):
        # Left Control Panel
        control_frame = tk.Frame(self.root, width=320, bg="#333333")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(control_frame, text="Controls", font=("Arial", 14, "bold"), bg="#333333", fg="white").pack(pady=10)

        # File Loading
        load_grp = tk.LabelFrame(control_frame, text="Load Data", bg="#333333", fg="white", padx=5, pady=5)
        load_grp.pack(fill=tk.X, padx=5, pady=5)

        self.btn_img = tk.Button(load_grp, text="Load Main Image (.nii)", command=lambda: self.load_file('img'), bg="#444", fg="white")
        self.btn_img.pack(fill=tk.X, pady=2)
        
        self.btn_dose = tk.Button(load_grp, text="Load Dose (.nii)", command=lambda: self.load_file('dose'), bg="#444", fg="white")
        self.btn_dose.pack(fill=tk.X, pady=2)

        self.btn_mask = tk.Button(load_grp, text="Load Mask (.nii)", command=lambda: self.load_file('mask'), bg="#444", fg="white")
        self.btn_mask.pack(fill=tk.X, pady=2)

        # View Settings (New)
        view_grp = tk.LabelFrame(control_frame, text="View Settings", bg="#333333", fg="white", padx=5, pady=5)
        view_grp.pack(fill=tk.X, padx=5, pady=5)
        
        # Checkbox for Sync Zoom
        tk.Checkbutton(view_grp, text="Sync Zoom", variable=self.sync_zoom, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        
        # Reset Button
        tk.Button(view_grp, text="Reset View (Zoom/Pan/Slice)", command=self.reset_view, bg="#555", fg="white").pack(fill=tk.X, pady=2)

        # Navigation
        nav_grp = tk.LabelFrame(control_frame, text="Navigation", bg="#333333", fg="white", padx=5, pady=5)
        nav_grp.pack(fill=tk.X, padx=5, pady=5)

        self.slider_x = self._create_nav_control(nav_grp, "Sagittal (X)", self.var_slice_x, 0, 100, 0)
        self.slider_y = self._create_nav_control(nav_grp, "Coronal (Y)", self.var_slice_y, 0, 100, 1)
        self.slider_z = self._create_nav_control(nav_grp, "Axial (Z)", self.var_slice_z, 0, 100, 2)

        # Window/Level
        wl_grp = tk.LabelFrame(control_frame, text="Window / Level", bg="#333333", fg="white", padx=5, pady=5)
        wl_grp.pack(fill=tk.X, padx=5, pady=5)
        
        self.slider_win = self._create_val_control(wl_grp, "Window", self.var_win, 0.01, 2.0, self._on_wl_change, 0.8)
        self.slider_lev = self._create_val_control(wl_grp, "Level", self.var_lev, 0.0, 1.0, self._on_wl_change, 0.4)

        # Dose Settings
        dose_grp = tk.LabelFrame(control_frame, text="Dose & Isolines", bg="#333333", fg="white", padx=5, pady=5)
        dose_grp.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(dose_grp, text="Show Dose Overlay", variable=self.show_dose, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        
        self.slider_dose_max = self._create_val_control(dose_grp, "Dose Max", self.var_dose_max, 1.0, 100.0, self._on_dose_range_change, 70.0)
        self.slider_dose_min = self._create_val_control(dose_grp, "Dose Min", self.var_dose_min, 0.0, 50.0, self._on_dose_range_change, 1.0)

        frame_alpha = tk.Frame(dose_grp, bg="#333")
        frame_alpha.pack(fill=tk.X, pady=2)
        tk.Label(frame_alpha, text="Opacity", bg="#333", fg="#ccc", width=10, anchor="w").pack(side=tk.LEFT)
        self.slider_alpha = tk.Scale(frame_alpha, from_=0, to=100, orient=tk.HORIZONTAL, command=self._on_dose_change, bg="#333", fg="white", highlightthickness=0, troughcolor="#555")
        self.slider_alpha.set(40)
        self.slider_alpha.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        tk.Label(dose_grp, text="Iso Levels (Gy):", bg="#333", fg="#ccc").pack(anchor="w", pady=(5, 0))
        self.entry_iso = tk.Entry(dose_grp, textvariable=self.iso_levels_str, bg="#555", fg="white")
        self.entry_iso.pack(fill=tk.X, pady=2)
        self.entry_iso.bind("<Return>", lambda e: self.update_plots()) 
        tk.Button(dose_grp, text="Update Iso", command=self.update_plots, bg="#555", fg="white", height=1).pack(fill=tk.X)

        self.info_label = tk.Label(control_frame, text="No Data Loaded", bg="#333", fg="#888", justify=tk.LEFT, font=("Consolas", 9))
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
        
        self.ax_axial = self.fig.add_subplot(221)
        self.ax_sag = self.fig.add_subplot(222)
        self.ax_cor = self.fig.add_subplot(224)
        
        self.ax_hist_img = self.fig.add_subplot(425) 
        self.ax_hist_dose = self.fig.add_subplot(427)

        self.image_axes = [self.ax_axial, self.ax_sag, self.ax_cor]

        # Initialize Image Artists with Dummy Data
        dummy_data = np.zeros((10, 10))
        for i, ax in enumerate(self.image_axes):
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 1. Main Image Artist
            self.artists['img'][i] = ax.imshow(dummy_data, cmap='gray', vmin=0, vmax=1, origin='lower', aspect='auto', animated=True)
            
            # 2. Dose Image Artist
            self.artists['dose'][i] = ax.imshow(dummy_data, cmap='jet', vmin=0, vmax=1, origin='lower', aspect='auto', alpha=0, animated=True)
            
            # 3. Crosshairs
            self.artists['cross_v'][i] = ax.axvline(5, color='#00FF00', lw=1, alpha=0.6)
            self.artists['cross_h'][i] = ax.axhline(5, color='#00FF00', lw=1, alpha=0.6)

        for ax in [self.ax_hist_img, self.ax_hist_dose]:
            ax.set_facecolor("#111")
            ax.tick_params(axis='x', colors='white', labelsize=8)
            ax.tick_params(axis='y', colors='white', labelsize=8)
            for spine in ax.spines.values(): spine.set_edgecolor('#555')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def load_file(self, file_type):
        path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii;*.nii.gz")])
        if not path: return

        try:
            nii = nib.load(path)
            # Use canonical orientation to ensure consistency
            nii = nib.as_closest_canonical(nii) 
            data = nii.get_fdata()

            if file_type == 'img':
                d_min, d_max = np.min(data), np.max(data)
                if d_max > d_min: data = (data - d_min) / (d_max - d_min)
                else: data = np.zeros_like(data)
                
                self.img_data = data
                self.dims = data.shape
                self.affine = nii.affine
                
                # --- CALCULATE ASPECT RATIOS HERE ---
                self.spacing = get_spacing_ijk_mm(self.affine)
                
                # Note: We use transpose=True because our slicing logic below uses .T
                ar_ax = get_aspect_ratio_for_view(self.spacing, 'axial', transpose=True)
                ar_sag = get_aspect_ratio_for_view(self.spacing, 'sagittal', transpose=True)
                ar_cor = get_aspect_ratio_for_view(self.spacing, 'coronal', transpose=True)
                
                self.aspect_ratios = [ar_ax, ar_sag, ar_cor]
                
                # Apply aspect ratios to the axes
                self.ax_axial.set_aspect(ar_ax)
                self.ax_sag.set_aspect(ar_sag)
                self.ax_cor.set_aspect(ar_cor)
                # ------------------------------------
                
                self.zoom_scales = [1.0, 1.0, 1.0]
                self.pan_offsets = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

                self.slider_x.config(to=self.dims[0]-1)
                self.slider_y.config(to=self.dims[1]-1)
                self.slider_z.config(to=self.dims[2]-1)
                
                self.current_slice = [d // 2 for d in self.dims]
                self._update_nav_controls()
                self.btn_img.config(bg="green")
                self._update_histograms()

            elif file_type == 'dose':
                self.dose_data = data
                d_max = np.max(data)
                self.dose_max = d_max
                self.var_dose_max.set(f"{d_max:.2f}")
                
                self.slider_dose_max.config(to=d_max * 1.1) 
                self.slider_dose_max.set(d_max)
                self.slider_dose_min.config(to=d_max)
                self.btn_dose.config(bg="green")
                self._update_histograms()

            elif file_type == 'mask':
                self.mask_data = data
                self.btn_mask.config(bg="green")

            self.update_plots()
            self._update_info()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    # --- Interaction Logic ---
    def reset_view(self):
        """Resets zoom, pan, and slice scroll to defaults."""
        if self.img_data is None: return
        
        # Reset Zoom and Pan
        self.zoom_scales = [1.0, 1.0, 1.0]
        self.pan_offsets = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        
        # Reset Slices to Center
        self.current_slice = [d // 2 for d in self.dims]
        
        self._update_nav_controls()
        self.update_plots()
        self._update_info()

    def _update_nav_controls(self):
        self.slider_x.set(self.current_slice[0])
        self.var_slice_x.set(str(self.current_slice[0]))
        self.slider_y.set(self.current_slice[1])
        self.var_slice_y.set(str(self.current_slice[1]))
        self.slider_z.set(self.current_slice[2])
        self.var_slice_z.set(str(self.current_slice[2]))

    def _on_slider_slice(self, axis, val):
        self.current_slice[axis] = int(float(val))
        if axis == 0: self.var_slice_x.set(str(self.current_slice[axis]))
        elif axis == 1: self.var_slice_y.set(str(self.current_slice[axis]))
        elif axis == 2: self.var_slice_z.set(str(self.current_slice[axis]))
        self.update_plots()
        self._update_info()

    def _on_entry_slice(self, axis):
        try:
            if axis == 0: val = int(self.var_slice_x.get())
            elif axis == 1: val = int(self.var_slice_y.get())
            elif axis == 2: val = int(self.var_slice_z.get())
            
            if self.dims[axis] > 0:
                val = max(0, min(val, self.dims[axis] - 1))
            
            self.current_slice[axis] = val
            self._update_nav_controls()
            self.update_plots()
            self._update_info()
        except ValueError: pass

    def _on_wl_change(self, _):
        try:
            self.wl_window = float(self.var_win.get())
            self.wl_level = float(self.var_lev.get())
            self.slider_win.set(self.wl_window)
            self.slider_lev.set(self.wl_level)
            self.update_plots()
            self.ax_hist_img.lines[0].set_xdata([self.wl_level - self.wl_window/2]*2)
            self.ax_hist_img.lines[1].set_xdata([self.wl_level + self.wl_window/2]*2)
        except ValueError: pass

    def _on_dose_range_change(self, _):
        try:
            self.dose_max = float(self.var_dose_max.get())
            self.dose_min = float(self.var_dose_min.get())
            self.slider_dose_max.set(self.dose_max)
            self.slider_dose_min.set(self.dose_min)
            self.update_plots()
        except ValueError: pass

    def _on_dose_change(self, _):
        self.dose_alpha = float(self.slider_alpha.get()) / 100.0
        self.update_plots()

    def _on_scroll(self, event):
        if self.img_data is None or event.inaxes is None: return
        
        axis_idx = -1
        if event.inaxes == self.ax_axial: axis_idx = 2
        elif event.inaxes == self.ax_sag: axis_idx = 0
        elif event.inaxes == self.ax_cor: axis_idx = 1
            
        if axis_idx != -1:
            new_val = self.current_slice[axis_idx] + (1 if event.step > 0 else -1)
            new_val = max(0, min(new_val, self.dims[axis_idx] - 1))
            self.current_slice[axis_idx] = new_val
            self._update_nav_controls()
            self.update_plots()
            self._update_info()

    def _on_mouse_press(self, event):
        if self.img_data is None or event.inaxes is None: return
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        if event.button == 1: 
            ix, iy = event.xdata, event.ydata
            if ix is None or iy is None: return
            ix, iy = int(ix), int(iy)
            
            # Note: Mapping depends on transpose!
            if event.inaxes == self.ax_axial:
                # Axial Display: Rows=Y (max_y), Cols=X (max_x)
                w, h = self.dims[0], self.dims[1]
                d_x = max(0, min((w-1) - ix, w-1)) # Flip X
                d_y = max(0, min(iy, h-1))
                self.current_slice[0] = d_x
                self.current_slice[1] = d_y

            elif event.inaxes == self.ax_sag:
                # Sagittal Display: Rows=Z, Cols=Y
                w, h = self.dims[1], self.dims[2]
                d_y = max(0, min((w-1) - ix, w-1))
                d_z = max(0, min(iy, h-1))
                self.current_slice[1] = d_y
                self.current_slice[2] = d_z

            elif event.inaxes == self.ax_cor:
                # Coronal Display: Rows=Z, Cols=X
                w, h = self.dims[0], self.dims[2]
                d_x = max(0, min((w-1) - ix, w-1))
                d_z = max(0, min(iy, h-1))
                self.current_slice[0] = d_x
                self.current_slice[2] = d_z

            self._update_nav_controls()
            self.update_plots()
            self._update_info()
        elif event.button == 3: self.active_zoom_ax = event.inaxes
        elif event.button == 2: self.active_pan_ax = event.inaxes

    def _on_mouse_release(self, event):
        if event.button == 3: self.active_zoom_ax = None
        if event.button == 2: self.active_pan_ax = None

    def _on_mouse_move(self, event):
        if event.inaxes is None: return
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        if self.active_zoom_ax:
            # Check if we should sync zoom
            # We zoom if the mouse is currently dragging (active_zoom_ax is set)
            # We don't strictly require event.inaxes == active_zoom_ax if we are syncing, 
            # but usually interaction feels best if we are dragging on the canvas.
            
            # Identify which axis triggered the zoom (to apply if not syncing)
            idx = -1
            if self.active_zoom_ax == self.ax_axial: idx = 0
            elif self.active_zoom_ax == self.ax_sag: idx = 1
            elif self.active_zoom_ax == self.ax_cor: idx = 2
            
            if idx != -1:
                zoom_factor = 1 + dy * 0.01
                
                if self.sync_zoom.get():
                    # Apply to all axes
                    for i in range(3):
                        self.zoom_scales[i] = np.clip(self.zoom_scales[i] * zoom_factor, 0.1, 20.0)
                else:
                    # Apply only to active axis
                    self.zoom_scales[idx] = np.clip(self.zoom_scales[idx] * zoom_factor, 0.1, 20.0)
                
                self.update_plots()

        if self.active_pan_ax and self.active_pan_ax == event.inaxes:
            idx = -1
            if self.active_pan_ax == self.ax_axial: idx = 0
            elif self.active_pan_ax == self.ax_sag: idx = 1
            elif self.active_pan_ax == self.ax_cor: idx = 2
            
            if idx != -1:
                ax = self.active_pan_ax
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                bbox = ax.get_window_extent()
                if bbox.width > 0:
                    scale_x = (xlim[1] - xlim[0]) / bbox.width
                    scale_y = (ylim[1] - ylim[0]) / bbox.height
                    self.pan_offsets[idx][0] += dx * abs(scale_x) 
                    self.pan_offsets[idx][1] += dy * abs(scale_y)
                    self.update_plots()

    def _update_info(self):
        cx, cy, cz = self.current_slice
        txt = f"Pos: ({cx}, {cy}, {cz})\n"
        if self.img_data is not None: txt += f"Img: {self.img_data[cx,cy,cz]:.4f}\n"
        if self.dose_data is not None: txt += f"Dose: {self.dose_data[cx,cy,cz]:.2f} Gy"
        self.info_label.config(text=txt)

    def _parse_iso_levels(self):
        try:
            txt = self.iso_levels_str.get()
            levels = sorted([float(x.strip()) for x in txt.split(',') if x.strip()])
            return levels
        except: return []

    def _update_histograms(self):
        self.ax_hist_img.clear()
        if self.img_data is not None:
            flat_data = self.img_data[::4, ::4, ::4].flatten()
            self.ax_hist_img.hist(flat_data, bins=50, color='gray', log=True)
            self.ax_hist_img.set_title("Image Intensity", color='white', fontsize=9)
            self.ax_hist_img.axvline(self.wl_level - self.wl_window/2, color='red', ls='--')
            self.ax_hist_img.axvline(self.wl_level + self.wl_window/2, color='red', ls='--')
            self.ax_hist_img.set_xlim(0, 1)

        self.ax_hist_dose.clear()
        if self.dose_data is not None:
            valid_dose = self.dose_data[self.dose_data > 0.5]
            if len(valid_dose) > 0:
                valid_dose = valid_dose[::4]
                self.ax_hist_dose.hist(valid_dose, bins=50, color='blue', alpha=0.6, log=True)
                self.ax_hist_dose.set_title("DVH (Approx)", color='white', fontsize=9)
                iso_levels = self._parse_iso_levels()
                patches = []
                if iso_levels:
                    for idx, level in enumerate(iso_levels):
                        col = self.iso_colors[idx % len(self.iso_colors)]
                        patches.append(mpatches.Patch(color=col, label=f'{level} Gy'))
                        self.ax_hist_dose.axvline(level, color=col, linewidth=2)
                    self.ax_hist_dose.legend(handles=patches, loc='upper right', fontsize='small', facecolor='#333', labelcolor='white')

    def update_plots(self):
        if self.img_data is None: return
        
        cx, cy, cz = self.current_slice
        vmin = self.wl_level - self.wl_window/2
        vmax = self.wl_level + self.wl_window/2
        iso_levels = self._parse_iso_levels()

        # Helper to update a single view
        def update_view_idx(idx, slice_data, dose_slice, cur_cross_x, cur_cross_y, max_x, max_y):
            # 1. Update Main Image
            im_artist = self.artists['img'][idx]
            im_artist.set_data(slice_data)
            im_artist.set_clim(vmin, vmax)
            
            # --- IMPORTANT FIX: UPDATE EXTENT ---
            # This ensures the image stretches to the new slice dimensions, 
            # effectively fixing the "tiny icon" bug.
            im_artist.set_extent((0, max_x, 0, max_y))
            # ------------------------------------
            
            # 2. Update Dose
            dose_artist = self.artists['dose'][idx]
            if self.dose_data is not None and self.show_dose.get():
                masked_dose = np.ma.masked_less(dose_slice, self.dose_min)
                dose_artist.set_data(masked_dose)
                dose_artist.set_extent((0, max_x, 0, max_y)) # Update extent for dose too
                dose_artist.set_alpha(self.dose_alpha)
                dose_artist.set_clim(0, self.dose_max)
            else:
                dose_artist.set_data(np.zeros_like(slice_data))
                dose_artist.set_alpha(0)

            # 3. Update Contours (Remove old, add new)
            if self.artists['contours'][idx]:
                for c in self.artists['contours'][idx].collections:
                    c.remove()
                self.artists['contours'][idx] = None
            
            if self.dose_data is not None and self.show_isolines.get() and iso_levels:
                ax = self.image_axes[idx]
                # Contours automatically use the extent of the axes/data, but sometimes 
                # need explicit extent if the underlying image changed size
                self.artists['contours'][idx] = ax.contour(
                    dose_slice, levels=iso_levels, 
                    colors=self.iso_colors[:len(iso_levels)], 
                    linewidths=1, origin='lower',
                    extent=(0, max_x, 0, max_y)
                )

            # 4. Update Crosshairs
            line_v = self.artists['cross_v'][idx]
            line_h = self.artists['cross_h'][idx]
            line_v.set_xdata([max_x - cur_cross_x]*2)
            line_h.set_ydata([cur_cross_y]*2)

            # 5. Apply Zoom/Pan
            scale = self.zoom_scales[idx]
            pan_x, pan_y = self.pan_offsets[idx]
            
            # Viewport center
            view_cx = (max_x + 1) / 2 - pan_x
            view_cy = (max_y + 1) / 2 - pan_y
            hw = ((max_x + 1) / 2) / scale
            hh = ((max_y + 1) / 2) / scale
            
            ax = self.image_axes[idx]
            ax.set_xlim(view_cx - hw, view_cx + hw)
            ax.set_ylim(view_cy - hh, view_cy + hh)

        # --- Update 1: Axial (Y, -X) ---
        slice_ax = np.flip(self.img_data[:, :, cz].T, 1)
        d_ax = np.flip(self.dose_data[:, :, cz].T, 1) if self.dose_data is not None else None
        update_view_idx(0, slice_ax, d_ax, cx, cy, self.dims[0]-1, self.dims[1]-1)
        self.ax_axial.set_title(f"Axial (Z={cz})", color='white', fontsize=10)

        # --- Update 2: Sagittal (Z, Y) ---
        slice_sag = np.flip(self.img_data[cx, :, :].T, 1)
        d_sag = np.flip(self.dose_data[cx, :, :].T, 1) if self.dose_data is not None else None
        update_view_idx(1, slice_sag, d_sag, cy, cz, self.dims[1]-1, self.dims[2]-1)
        self.ax_sag.set_title(f"Sagittal (X={cx})", color='white', fontsize=10)

        # --- Update 3: Coronal (Z, X) ---
        slice_cor = np.flip(self.img_data[:, cy, :].T, 1)
        d_cor = np.flip(self.dose_data[:, cy, :].T, 1) if self.dose_data is not None else None
        update_view_idx(2, slice_cor, d_cor, cx, cz, self.dims[0]-1, self.dims[2]-1)
        self.ax_cor.set_title(f"Coronal (Y={cy})", color='white', fontsize=10)

        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalViewerApp(root)
    root.mainloop()