import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import os

class MedicalViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Medical Dose Viewer (ITK-SNAP Style)")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2b2b2b")

        # --- Data State ---
        self.img_data = None
        self.dose_data = None
        self.mask_data = None
        self.affine = None
        
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
        self.pan_offsets = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]] # [dx, dy] in data units

        # Visualization settings
        self.wl_window = 0.8
        self.wl_level = 0.4
        self.dose_alpha = 0.4
        self.dose_max = 70.0
        self.dose_min = 1.0
        self.show_dose = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.show_isolines = tk.BooleanVar(value=True)
        
        # Default Iso levels
        self.iso_levels_str = tk.StringVar(value="5, 12, 16") 
        self.iso_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

        # UI Variables for Entries
        self.var_slice_x = tk.StringVar(value="0")
        self.var_slice_y = tk.StringVar(value="0")
        self.var_slice_z = tk.StringVar(value="0")
        self.var_win = tk.StringVar(value="0.8")
        self.var_lev = tk.StringVar(value="0.4")
        self.var_dose_max = tk.StringVar(value="70.0")
        self.var_dose_min = tk.StringVar(value="1.0")

        # --- Layout ---
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

        # Navigation
        nav_grp = tk.LabelFrame(control_frame, text="Navigation", bg="#333333", fg="white", padx=5, pady=5)
        nav_grp.pack(fill=tk.X, padx=5, pady=5)

        # Sliders with Entries
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
        
        # Dose Range Sliders
        self.slider_dose_max = self._create_val_control(dose_grp, "Dose Max", self.var_dose_max, 1.0, 100.0, self._on_dose_range_change, 70.0)
        self.slider_dose_min = self._create_val_control(dose_grp, "Dose Min", self.var_dose_min, 0.0, 50.0, self._on_dose_range_change, 1.0)

        # Opacity Slider
        frame_alpha = tk.Frame(dose_grp, bg="#333")
        frame_alpha.pack(fill=tk.X, pady=2)
        tk.Label(frame_alpha, text="Opacity", bg="#333", fg="#ccc", width=10, anchor="w").pack(side=tk.LEFT)
        self.slider_alpha = tk.Scale(frame_alpha, from_=0, to=100, orient=tk.HORIZONTAL, command=self._on_dose_change, bg="#333", fg="white", highlightthickness=0, troughcolor="#555")
        self.slider_alpha.set(40)
        self.slider_alpha.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Iso Lines Input
        tk.Label(dose_grp, text="Iso Levels (Gy):", bg="#333", fg="#ccc").pack(anchor="w", pady=(5, 0))
        self.entry_iso = tk.Entry(dose_grp, textvariable=self.iso_levels_str, bg="#555", fg="white")
        self.entry_iso.pack(fill=tk.X, pady=2)
        self.entry_iso.bind("<Return>", lambda e: self.update_plots()) 
        tk.Button(dose_grp, text="Update Iso", command=self.update_plots, bg="#555", fg="white", height=1).pack(fill=tk.X)

        # Info Box
        self.info_label = tk.Label(control_frame, text="No Data Loaded", bg="#333", fg="#888", justify=tk.LEFT, font=("Consolas", 9))
        self.info_label.pack(side=tk.BOTTOM, pady=10)

        # Right: Plot Area
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
        entry.bind("<Return>", lambda e: command(None)) # Trigger update on enter
        
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                          command=lambda v: [text_var.set(f"{float(v):.2f}"), command(v)], 
                          bg="#333", fg="white", showvalue=0, highlightthickness=0, troughcolor="#555", resolution=0.01)
        slider.set(init_val)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return slider

    def _init_plots(self):
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="black")
        
        # 2x2 Grid
        self.ax_axial = self.fig.add_subplot(221)
        self.ax_sag = self.fig.add_subplot(222)
        self.ax_hist_img = self.fig.add_subplot(425) 
        self.ax_hist_dose = self.fig.add_subplot(427)
        self.ax_cor = self.fig.add_subplot(224)

        self.image_axes = [self.ax_axial, self.ax_sag, self.ax_cor]

        for ax in self.image_axes:
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])

        # Histograms styling
        for ax in [self.ax_hist_img, self.ax_hist_dose]:
            ax.set_facecolor("#111")
            ax.tick_params(axis='x', colors='white', labelsize=8)
            ax.tick_params(axis='y', colors='white', labelsize=8)
            for spine in ax.spines.values(): spine.set_edgecolor('#555')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Events
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def load_file(self, file_type):
        path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii;*.nii.gz")])
        if not path: return

        try:
            nii = nib.load(path)
            nii = nib.as_closest_canonical(nii) # Force RAS+
            data = nii.get_fdata()

            if file_type == 'img':
                # Normalize 0-1
                d_min, d_max = np.min(data), np.max(data)
                if d_max > d_min: data = (data - d_min) / (d_max - d_min)
                else: data = np.zeros_like(data)
                
                self.img_data = data
                self.dims = data.shape
                self.affine = nii.affine
                
                # Reset View State
                self.zoom_scales = [1.0, 1.0, 1.0]
                self.pan_offsets = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

                # Configure Controls
                self.slider_x.config(to=self.dims[0]-1)
                self.slider_y.config(to=self.dims[1]-1)
                self.slider_z.config(to=self.dims[2]-1)
                
                self.current_slice = [d // 2 for d in self.dims]
                self._update_nav_controls()
                
                self.btn_img.config(bg="green")

            elif file_type == 'dose':
                self.dose_data = data
                d_max = np.max(data)
                self.dose_max = d_max
                self.var_dose_max.set(f"{d_max:.2f}")
                
                # Update slider limits dynamically
                self.slider_dose_max.config(to=d_max * 1.1) # Add little headroom
                self.slider_dose_max.set(d_max)
                self.slider_dose_min.config(to=d_max)
                
                self.btn_dose.config(bg="green")

            elif file_type == 'mask':
                self.mask_data = data
                self.btn_mask.config(bg="green")

            self.update_plots()
            self._update_info()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    # --- Interaction Logic ---
    def _update_nav_controls(self):
        # Syncs all sliders and entries to self.current_slice
        self.slider_x.set(self.current_slice[0])
        self.var_slice_x.set(str(self.current_slice[0]))
        
        self.slider_y.set(self.current_slice[1])
        self.var_slice_y.set(str(self.current_slice[1]))
        
        self.slider_z.set(self.current_slice[2])
        self.var_slice_z.set(str(self.current_slice[2]))

    def _on_slider_slice(self, axis, val):
        self.current_slice[axis] = int(float(val))
        # Update entry text
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
            
            # Clamp
            if self.dims[axis] > 0:
                val = max(0, min(val, self.dims[axis] - 1))
            
            self.current_slice[axis] = val
            self._update_nav_controls() # Sync slider back
            self.update_plots()
            self._update_info()
        except ValueError:
            pass # Ignore invalid input

    def _on_wl_change(self, _):
        try:
            self.wl_window = float(self.var_win.get())
            self.wl_level = float(self.var_lev.get())
            # Sync sliders if input came from Entry
            self.slider_win.set(self.wl_window)
            self.slider_lev.set(self.wl_level)
            self.update_plots()
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

        if event.button == 1: # Left Click Move Crosshair
            ix, iy = event.xdata, event.ydata
            if ix is None or iy is None: return
            ix, iy = int(ix), int(iy)
            
            # IMPORTANT: Because we flipped the image horizontally (np.flip(..., 1)),
            # The visual x-coordinate 'ix' corresponds to width-1-x in the original transposed data.
            # We must invert 'ix' to get back to the data index.
            
            if event.inaxes == self.ax_axial:
                # Axial View Size: (X, Y). Displayed: (Y, X)_flipped.
                # Original Slice: (Y, X). Width is X-dim. Height is Y-dim.
                w, h = self.dims[0], self.dims[1]
                
                # Map Click (ix, iy) -> Data Indices
                # We flipped horizontal (X-axis). So data_x = (w-1) - ix
                # We did NOT flip vertical. So data_y = iy
                d_x = max(0, min((w-1) - ix, w-1))
                d_y = max(0, min(iy, h-1))
                
                self.current_slice[0] = d_x
                self.current_slice[1] = d_y

            elif event.inaxes == self.ax_sag:
                # Sagittal View Size: (Z, Y). Displayed: (Z, Y)_flipped.
                # Original Slice: (Z, Y). Width is Y-dim. Height is Z-dim.
                w, h = self.dims[1], self.dims[2]
                
                d_y = max(0, min((w-1) - ix, w-1))
                d_z = max(0, min(iy, h-1))
                
                self.current_slice[1] = d_y
                self.current_slice[2] = d_z

            elif event.inaxes == self.ax_cor:
                # Coronal View Size: (Z, X). Displayed: (Z, X)_flipped.
                # Original Slice: (Z, X). Width is X-dim. Height is Z-dim.
                w, h = self.dims[0], self.dims[2]
                
                d_x = max(0, min((w-1) - ix, w-1))
                d_z = max(0, min(iy, h-1))
                
                self.current_slice[0] = d_x
                self.current_slice[2] = d_z

            self._update_nav_controls()
            self.update_plots()
            self._update_info()

        elif event.button == 3: # Right Click Zoom
            self.active_zoom_ax = event.inaxes
            
        elif event.button == 2: # Middle Click Pan
            self.active_pan_ax = event.inaxes

    def _on_mouse_release(self, event):
        if event.button == 3: self.active_zoom_ax = None
        if event.button == 2: self.active_pan_ax = None

    def _on_mouse_move(self, event):
        if event.inaxes is None: return
        
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        # Zooming (Right Click)
        if self.active_zoom_ax and self.active_zoom_ax == event.inaxes:
            idx = -1
            if self.active_zoom_ax == self.ax_axial: idx = 0
            elif self.active_zoom_ax == self.ax_sag: idx = 1
            elif self.active_zoom_ax == self.ax_cor: idx = 2
            
            if idx != -1:
                self.zoom_scales[idx] = np.clip(self.zoom_scales[idx] * (1 + dy*0.01), 0.1, 20.0)
                self.update_plots()

        # Panning (Middle Click)
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
                
                if bbox.width > 0 and bbox.height > 0:
                    scale_x = (xlim[1] - xlim[0]) / bbox.width
                    scale_y = (ylim[1] - ylim[0]) / bbox.height
                    
                    # Important: Panning logic usually needs to respect the flip.
                    # But since we pan in data-coordinates, simple +/- works.
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
        except:
            return []

    def update_plots(self):
        if self.img_data is None: return
        
        cx, cy, cz = self.current_slice
        vmin = self.wl_level - self.wl_window/2
        vmax = self.wl_level + self.wl_window/2
        iso_levels = self._parse_iso_levels()

        # --- 1. AXIAL (Top-Left) ---
        # Data: Z-slice. X axis=LR, Y axis=PA.
        # Transpose -> (Y, X).
        # Flip(1) -> (Y, -X). Horizontal Flip.
        # Result: Screen Left is Patient Right.
        self.ax_axial.clear()
        slice_ax = np.flip(self.img_data[:, :, cz].T, 1)
        self.ax_axial.imshow(slice_ax, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        
        if self.dose_data is not None and self.show_dose.get():
            d = np.flip(self.dose_data[:, :, cz].T, 1)
            md = np.ma.masked_less(d, self.dose_min)
            self.ax_axial.imshow(md, cmap='jet', alpha=self.dose_alpha, origin='lower', aspect='auto', vmin=0, vmax=self.dose_max)
            if self.show_isolines.get() and iso_levels:
                self.ax_axial.contour(d, levels=iso_levels, colors=self.iso_colors[:len(iso_levels)], linewidths=1, origin='lower')

        # Crosshairs: Remember we flipped X.
        # Original coords: (cx, cy). Transposed: (cy, cx). Flipped: (cy, width-1-cx)
        w_ax = self.dims[0]
        self.ax_axial.axvline((w_ax - 1) - cx, color='#00FF00', lw=1, alpha=0.6)
        self.ax_axial.axhline(cy, color='#00FF00', lw=1, alpha=0.6)
        
        self._add_labels(self.ax_axial, 'A', 'P', 'R', 'L')
        self.ax_axial.set_title(f"Axial (Z={cz})", color='white', fontsize=10)

        # --- 2. SAGITTAL (Top-Right) ---
        # Data: X-slice. (Y, Z). Transpose -> (Z, Y). Flip -> (Z, -Y).
        # Screen Left is Y-Max (Anterior).
        self.ax_sag.clear()
        slice_sag = np.flip(self.img_data[cx, :, :].T, 1)
        self.ax_sag.imshow(slice_sag, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        
        if self.dose_data is not None and self.show_dose.get():
            d = np.flip(self.dose_data[cx, :, :].T, 1)
            md = np.ma.masked_less(d, self.dose_min)
            self.ax_sag.imshow(md, cmap='jet', alpha=self.dose_alpha, origin='lower', aspect='auto', vmin=0, vmax=self.dose_max)
            if self.show_isolines.get() and iso_levels:
                self.ax_sag.contour(d, levels=iso_levels, colors=self.iso_colors[:len(iso_levels)], linewidths=1, origin='lower')

        w_sag = self.dims[1] # Y-dim
        self.ax_sag.axvline((w_sag - 1) - cy, color='#00FF00', lw=1, alpha=0.6)
        self.ax_sag.axhline(cz, color='#00FF00', lw=1, alpha=0.6)
        
        self._add_labels(self.ax_sag, 'S', 'I', 'A', 'P')
        self.ax_sag.set_title(f"Sagittal (X={cx})", color='white', fontsize=10)

        # --- 3. CORONAL (Bottom-Right) ---
        # Data: Y-slice. (X, Z). Transpose -> (Z, X). Flip -> (Z, -X).
        # Screen Left is X-Max (Right).
        self.ax_cor.clear()
        slice_cor = np.flip(self.img_data[:, cy, :].T, 1)
        self.ax_cor.imshow(slice_cor, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        
        if self.dose_data is not None and self.show_dose.get():
            d = np.flip(self.dose_data[:, cy, :].T, 1)
            md = np.ma.masked_less(d, self.dose_min)
            self.ax_cor.imshow(md, cmap='jet', alpha=self.dose_alpha, origin='lower', aspect='auto', vmin=0, vmax=self.dose_max)
            if self.show_isolines.get() and iso_levels:
                self.ax_cor.contour(d, levels=iso_levels, colors=self.iso_colors[:len(iso_levels)], linewidths=1, origin='lower')

        w_cor = self.dims[0] # X-dim
        self.ax_cor.axvline((w_cor - 1) - cx, color='#00FF00', lw=1, alpha=0.6)
        self.ax_cor.axhline(cz, color='#00FF00', lw=1, alpha=0.6)
        
        self._add_labels(self.ax_cor, 'S', 'I', 'R', 'L')
        self.ax_cor.set_title(f"Coronal (Y={cy})", color='white', fontsize=10)

        # --- ZOOM & PAN APPLICATION ---
        for i, ax in enumerate([self.ax_axial, self.ax_sag, self.ax_cor]):
            if i == 0: h, w = self.dims[1], self.dims[0]
            elif i == 1: h, w = self.dims[2], self.dims[1]
            else: h, w = self.dims[2], self.dims[0]
            
            scale = self.zoom_scales[i]
            pan_x, pan_y = self.pan_offsets[i]
            
            cx_view = (w / 2) - pan_x
            cy_view = (h / 2) - pan_y
            hw = (w / 2) / scale
            hh = (h / 2) / scale
            
            ax.set_xlim(cx_view - hw, cx_view + hw)
            ax.set_ylim(cy_view - hh, cy_view + hh)

        # --- 4. HISTOGRAMS (Bottom-Left) ---
        self.ax_hist_img.clear()
        self.ax_hist_img.hist(self.img_data.flatten(), bins=100, color='gray', log=True)
        self.ax_hist_img.set_title("Image Intensity", color='white', fontsize=9)
        self.ax_hist_img.axvline(self.wl_level - self.wl_window/2, color='red', ls='--')
        self.ax_hist_img.axvline(self.wl_level + self.wl_window/2, color='red', ls='--')
        self.ax_hist_img.set_xlim(0, 1)

        self.ax_hist_dose.clear()
        if self.dose_data is not None:
            valid_dose = self.dose_data[self.dose_data > 0.5]
            if len(valid_dose) > 0:
                self.ax_hist_dose.hist(valid_dose, bins=50, color='blue', alpha=0.6, log=True)
                self.ax_hist_dose.set_title("Dose Volume Histogram", color='white', fontsize=9)
                
                patches = []
                if iso_levels:
                    for idx, level in enumerate(iso_levels):
                        col = self.iso_colors[idx % len(self.iso_colors)]
                        patches.append(mpatches.Patch(color=col, label=f'{level} Gy'))
                        self.ax_hist_dose.axvline(level, color=col, linewidth=2)
                    
                    self.ax_hist_dose.legend(handles=patches, loc='upper right', fontsize='small', facecolor='#333', labelcolor='white')

        self.canvas.draw()

    def _add_labels(self, ax, top, bot, left, right):
        props = dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='none')
        ax.text(0.5, 0.98, top, transform=ax.transAxes, color='yellow', ha='center', va='top', weight='bold', bbox=props)
        ax.text(0.5, 0.02, bot, transform=ax.transAxes, color='yellow', ha='center', va='bottom', weight='bold', bbox=props)
        ax.text(0.02, 0.5, left, transform=ax.transAxes, color='yellow', ha='left', va='center', weight='bold', bbox=props)
        ax.text(0.98, 0.5, right, transform=ax.transAxes, color='yellow', ha='right', va='center', weight='bold', bbox=props)

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalViewerApp(root)
    root.mainloop()