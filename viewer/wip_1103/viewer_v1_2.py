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
        self.dose_min_threshold = 1.0 
        self.show_dose = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.show_isolines = tk.BooleanVar(value=True)
        
        # Default Iso levels
        self.iso_levels_str = tk.StringVar(value="5, 12, 16") 
        self.iso_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

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

        self.slider_x = self._create_slider(nav_grp, "Sagittal (X)", 0, 100, self._on_slice_change)
        self.slider_y = self._create_slider(nav_grp, "Coronal (Y)", 0, 100, self._on_slice_change)
        self.slider_z = self._create_slider(nav_grp, "Axial (Z)", 0, 100, self._on_slice_change)

        # Window/Level
        wl_grp = tk.LabelFrame(control_frame, text="Window / Level", bg="#333333", fg="white", padx=5, pady=5)
        wl_grp.pack(fill=tk.X, padx=5, pady=5)
        self.slider_win = self._create_slider(wl_grp, "Window", 0.01, 2.0, self._on_wl_change, init=0.8, resolution=0.01)
        self.slider_lev = self._create_slider(wl_grp, "Level", 0.0, 1.0, self._on_wl_change, init=0.4, resolution=0.01)

        # Dose Settings
        dose_grp = tk.LabelFrame(control_frame, text="Dose & Isolines", bg="#333333", fg="white", padx=5, pady=5)
        dose_grp.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(dose_grp, text="Show Dose Overlay", variable=self.show_dose, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        self.slider_alpha = self._create_slider(dose_grp, "Opacity", 0, 100, self._on_dose_change, init=40)
        
        # Iso Lines Input
        tk.Label(dose_grp, text="Iso Levels (Gy):", bg="#333", fg="#ccc").pack(anchor="w", pady=(5, 0))
        self.entry_iso = tk.Entry(dose_grp, textvariable=self.iso_levels_str, bg="#555", fg="white")
        self.entry_iso.pack(fill=tk.X, pady=2)
        self.entry_iso.bind("<Return>", lambda e: self.update_plots()) # Update on Enter key
        tk.Button(dose_grp, text="Update Iso", command=self.update_plots, bg="#555", fg="white", height=1).pack(fill=tk.X)

        # Info Box
        self.info_label = tk.Label(control_frame, text="No Data Loaded", bg="#333", fg="#888", justify=tk.LEFT, font=("Consolas", 9))
        self.info_label.pack(side=tk.BOTTOM, pady=10)

        # Right: Plot Area
        self.plot_frame = tk.Frame(self.root, bg="black")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _create_slider(self, parent, label, min_val, max_val, command, init=0, resolution=1):
        frame = tk.Frame(parent, bg="#333")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, bg="#333", fg="#ccc", width=12, anchor="w").pack(side=tk.LEFT)
        scale = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, command=command, bg="#333", fg="white", highlightthickness=0, troughcolor="#555", resolution=resolution)
        scale.set(init)
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return scale

    def _init_plots(self):
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="black")
        
        # 2x2 Grid
        # 0,0 Axial | 0,1 Sagittal
        # 1,0 Hist  | 1,1 Coronal
        
        self.ax_axial = self.fig.add_subplot(221)
        self.ax_sag = self.fig.add_subplot(222)
        self.ax_hist_img = self.fig.add_subplot(425) # Top half of bottom-left
        self.ax_hist_dose = self.fig.add_subplot(427) # Bottom half of bottom-left
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

                # Sliders
                self.slider_x.config(to=self.dims[0]-1)
                self.slider_y.config(to=self.dims[1]-1)
                self.slider_z.config(to=self.dims[2]-1)
                self.current_slice = [d // 2 for d in self.dims]
                self.slider_x.set(self.current_slice[0])
                self.slider_y.set(self.current_slice[1])
                self.slider_z.set(self.current_slice[2])
                self.btn_img.config(bg="green")

            elif file_type == 'dose':
                self.dose_data = data
                self.btn_dose.config(bg="green")

            elif file_type == 'mask':
                self.mask_data = data
                self.btn_mask.config(bg="green")

            self.update_plots()
            self._update_info()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    # --- Handlers ---
    def _on_slice_change(self, _):
        self.current_slice = [int(self.slider_x.get()), int(self.slider_y.get()), int(self.slider_z.get())]
        self.update_plots()

    def _on_wl_change(self, _):
        self.wl_window = float(self.slider_win.get())
        self.wl_level = float(self.slider_lev.get())
        self.update_plots()

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
            
            if axis_idx == 0: self.slider_x.set(new_val)
            elif axis_idx == 1: self.slider_y.set(new_val)
            elif axis_idx == 2: self.slider_z.set(new_val)
            self.update_plots()
            self._update_info()

    def _on_mouse_press(self, event):
        if self.img_data is None or event.inaxes is None: return
        
        # Store click position for Dragging
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        if event.button == 1: # Left Click Move Crosshair
            ix, iy = event.xdata, event.ydata
            if ix is None or iy is None: return
            ix, iy = int(ix), int(iy)
            
            if event.inaxes == self.ax_axial:
                # Axial: X=DataX, Y=DataY
                self.current_slice[0] = max(0, min(ix, self.dims[0]-1))
                self.current_slice[1] = max(0, min(iy, self.dims[1]-1))
                self.slider_x.set(self.current_slice[0])
                self.slider_y.set(self.current_slice[1])

            elif event.inaxes == self.ax_sag:
                # Sagittal: X=DataY, Y=DataZ
                self.current_slice[1] = max(0, min(ix, self.dims[1]-1))
                self.current_slice[2] = max(0, min(iy, self.dims[2]-1))
                self.slider_y.set(self.current_slice[1])
                self.slider_z.set(self.current_slice[2])

            elif event.inaxes == self.ax_cor:
                # Coronal: X=DataX, Y=DataZ
                self.current_slice[0] = max(0, min(ix, self.dims[0]-1))
                self.current_slice[2] = max(0, min(iy, self.dims[2]-1))
                self.slider_x.set(self.current_slice[0])
                self.slider_z.set(self.current_slice[2])

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
                # Scale factor based on vertical drag
                self.zoom_scales[idx] = np.clip(self.zoom_scales[idx] * (1 + dy*0.01), 0.1, 20.0)
                self.update_plots()

        # Panning (Middle Click)
        if self.active_pan_ax and self.active_pan_ax == event.inaxes:
            idx = -1
            if self.active_pan_ax == self.ax_axial: idx = 0
            elif self.active_pan_ax == self.ax_sag: idx = 1
            elif self.active_pan_ax == self.ax_cor: idx = 2
            
            if idx != -1:
                # Convert pixel delta to data delta
                # Get current axis limits to determine scale
                ax = self.active_pan_ax
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                bbox = ax.get_window_extent()
                
                if bbox.width > 0 and bbox.height > 0:
                    scale_x = (xlim[1] - xlim[0]) / bbox.width
                    scale_y = (ylim[1] - ylim[0]) / bbox.height
                    
                    # Pan logic: Dragging right (positive dx) moves the viewport left (negative limits)
                    # so the image appears to move right.
                    # We accumulate offsets. positive offset = shift view center left.
                    self.pan_offsets[idx][0] += dx * abs(scale_x) # Use abs to avoid flipping issues
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
        self.ax_axial.clear()
        slice_ax = self.img_data[:, :, cz].T 
        self.ax_axial.imshow(slice_ax, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        
        if self.dose_data is not None and self.show_dose.get():
            d = self.dose_data[:, :, cz].T
            md = np.ma.masked_less(d, self.dose_min_threshold)
            self.ax_axial.imshow(md, cmap='jet', alpha=self.dose_alpha, origin='lower', aspect='auto', vmin=0, vmax=np.max(self.dose_data))
            if self.show_isolines.get() and iso_levels:
                self.ax_axial.contour(d, levels=iso_levels, colors=self.iso_colors[:len(iso_levels)], linewidths=1, origin='lower')

        self.ax_axial.axvline(cx, color='#00FF00', lw=1, alpha=0.6)
        self.ax_axial.axhline(cy, color='#00FF00', lw=1, alpha=0.6)
        
        self.ax_axial.invert_xaxis() 
        self._add_labels(self.ax_axial, 'A', 'P', 'R', 'L')
        self.ax_axial.set_title(f"Axial (Z={cz})", color='white', fontsize=10)

        # --- 2. SAGITTAL (Top-Right) ---
        slice_sag = self.img_data[cx, :, :].T
        self.ax_sag.clear()
        self.ax_sag.imshow(slice_sag, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        
        if self.dose_data is not None and self.show_dose.get():
            d = self.dose_data[cx, :, :].T
            md = np.ma.masked_less(d, self.dose_min_threshold)
            self.ax_sag.imshow(md, cmap='jet', alpha=self.dose_alpha, origin='lower', aspect='auto', vmin=0, vmax=np.max(self.dose_data))
            if self.show_isolines.get() and iso_levels:
                self.ax_sag.contour(d, levels=iso_levels, colors=self.iso_colors[:len(iso_levels)], linewidths=1, origin='lower')

        self.ax_sag.axvline(cy, color='#00FF00', lw=1, alpha=0.6)
        self.ax_sag.axhline(cz, color='#00FF00', lw=1, alpha=0.6)
        
        self.ax_sag.invert_xaxis()
        self._add_labels(self.ax_sag, 'S', 'I', 'A', 'P')
        self.ax_sag.set_title(f"Sagittal (X={cx})", color='white', fontsize=10)

        # --- 3. CORONAL (Bottom-Right) ---
        slice_cor = self.img_data[:, cy, :].T
        self.ax_cor.clear()
        self.ax_cor.imshow(slice_cor, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        
        if self.dose_data is not None and self.show_dose.get():
            d = self.dose_data[:, cy, :].T
            md = np.ma.masked_less(d, self.dose_min_threshold)
            self.ax_cor.imshow(md, cmap='jet', alpha=self.dose_alpha, origin='lower', aspect='auto', vmin=0, vmax=np.max(self.dose_data))
            if self.show_isolines.get() and iso_levels:
                self.ax_cor.contour(d, levels=iso_levels, colors=self.iso_colors[:len(iso_levels)], linewidths=1, origin='lower')

        self.ax_cor.axvline(cx, color='#00FF00', lw=1, alpha=0.6)
        self.ax_cor.axhline(cz, color='#00FF00', lw=1, alpha=0.6)
        
        self.ax_cor.invert_xaxis()
        self._add_labels(self.ax_cor, 'S', 'I', 'R', 'L')
        self.ax_cor.set_title(f"Coronal (Y={cy})", color='white', fontsize=10)

        # --- ZOOM & PAN APPLICATION ---
        # We calculate new limits based on Center, Zoom Scale, and Pan Offset
        for i, ax in enumerate([self.ax_axial, self.ax_sag, self.ax_cor]):
            if i == 0: h, w = self.dims[1], self.dims[0]
            elif i == 1: h, w = self.dims[2], self.dims[1]
            else: h, w = self.dims[2], self.dims[0]
            
            scale = self.zoom_scales[i]
            pan_x, pan_y = self.pan_offsets[i]
            
            # Current view center (pan moves the "camera" opposite to drag, so we subtract offset)
            cx_view = (w / 2) - pan_x
            cy_view = (h / 2) - pan_y
            
            # Half-width/height at current scale
            hw = (w / 2) / scale
            hh = (h / 2) / scale
            
            # Apply limits
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