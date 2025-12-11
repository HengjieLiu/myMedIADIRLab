import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

class MedicalViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Medical Dose Viewer")
        self.root.geometry("1200x800")
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
        self.last_mouse_y = 0
        self.active_zoom_ax = None
        # Zoom scales for [Axial, Sagittal, Coronal]
        self.zoom_scales = [1.0, 1.0, 1.0] 

        # Visualization settings
        # Updated defaults for 0-1 range
        self.wl_window = 0.8
        self.wl_level = 0.4
        self.dose_alpha = 0.4
        self.dose_min_threshold = 10.0 # Dose usually remains in Gy (absolute), so keep absolute
        self.show_dose = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.show_isolines = tk.BooleanVar(value=True)

        # --- Layout ---
        self._setup_ui()
        
        # Initialize empty plots
        self._init_plots()

    def _setup_ui(self):
        # Left Control Panel
        control_frame = tk.Frame(self.root, width=300, bg="#333333")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Title
        tk.Label(control_frame, text="Controls", font=("Arial", 14, "bold"), bg="#333333", fg="white").pack(pady=10)

        # File Loading Section
        load_grp = tk.LabelFrame(control_frame, text="Load Data", bg="#333333", fg="white", padx=5, pady=5)
        load_grp.pack(fill=tk.X, padx=5, pady=5)

        self.btn_img = tk.Button(load_grp, text="Load Main Image (.nii)", command=lambda: self.load_file('img'), bg="#444", fg="white")
        self.btn_img.pack(fill=tk.X, pady=2)
        
        self.btn_dose = tk.Button(load_grp, text="Load Dose (.nii)", command=lambda: self.load_file('dose'), bg="#444", fg="white")
        self.btn_dose.pack(fill=tk.X, pady=2)

        self.btn_mask = tk.Button(load_grp, text="Load Mask (.nii)", command=lambda: self.load_file('mask'), bg="#444", fg="white")
        self.btn_mask.pack(fill=tk.X, pady=2)

        # Navigation Sliders
        nav_grp = tk.LabelFrame(control_frame, text="Navigation", bg="#333333", fg="white", padx=5, pady=5)
        nav_grp.pack(fill=tk.X, padx=5, pady=5)

        self.slider_x = self._create_slider(nav_grp, "Sagittal (X)", 0, 100, self._on_slice_change)
        self.slider_y = self._create_slider(nav_grp, "Coronal (Y)", 0, 100, self._on_slice_change)
        self.slider_z = self._create_slider(nav_grp, "Axial (Z)", 0, 100, self._on_slice_change)

        # Window/Level
        wl_grp = tk.LabelFrame(control_frame, text="Window / Level (0-1)", bg="#333333", fg="white", padx=5, pady=5)
        wl_grp.pack(fill=tk.X, padx=5, pady=5)

        # Updated sliders for Normalized range 0.0 - 1.0
        self.slider_win = self._create_slider(wl_grp, "Window", 0.01, 2.0, self._on_wl_change, init=0.8, resolution=0.01)
        self.slider_lev = self._create_slider(wl_grp, "Level", 0.0, 1.0, self._on_wl_change, init=0.4, resolution=0.01)

        # Dose Settings
        dose_grp = tk.LabelFrame(control_frame, text="Dose Settings", bg="#333333", fg="white", padx=5, pady=5)
        dose_grp.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(dose_grp, text="Show Dose", variable=self.show_dose, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Checkbutton(dose_grp, text="Show Isolines", variable=self.show_isolines, command=self.update_plots, bg="#333", fg="white", selectcolor="#444").pack(anchor="w")
        
        self.slider_alpha = self._create_slider(dose_grp, "Opacity", 0, 100, self._on_dose_change, init=40)
        self.slider_thresh = self._create_slider(dose_grp, "Min Threshold", 0, 100, self._on_dose_change, init=10)

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
        # Create a Matplotlib figure with 3 subplots
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="black")
        self.ax_axial = self.fig.add_subplot(221)
        self.ax_sag = self.fig.add_subplot(222)
        self.ax_cor = self.fig.add_subplot(223)
        self.ax_hist = self.fig.add_subplot(224) # Histogram

        for ax in [self.ax_axial, self.ax_sag, self.ax_cor, self.ax_hist]:
            ax.set_facecolor("black")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

        self.ax_axial.set_title("Axial (Z)", color="white")
        self.ax_sag.set_title("Sagittal (X)", color="white")
        self.ax_cor.set_title("Coronal (Y)", color="white")
        self.ax_hist.set_title("Dose/Image Histogram", color="white")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # --- Event Connections ---
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def load_file(self, file_type):
        path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii;*.nii.gz")])
        if not path:
            return

        try:
            nii = nib.load(path)
            # Reorient to canonical (RAS+) 
            nii = nib.as_closest_canonical(nii)
            data = nii.get_fdata()

            if file_type == 'img':
                # Normalization to 0-1
                d_min = np.min(data)
                d_max = np.max(data)
                if d_max > d_min:
                    data = (data - d_min) / (d_max - d_min)
                else:
                    data = np.zeros_like(data)
                
                self.img_data = data
                self.dims = data.shape
                self.affine = nii.affine
                
                # Reset Zoom
                self.zoom_scales = [1.0, 1.0, 1.0]

                # Set sliders
                self.slider_x.config(to=self.dims[0]-1)
                self.slider_y.config(to=self.dims[1]-1)
                self.slider_z.config(to=self.dims[2]-1)
                
                # Set current slice to center
                self.current_slice = [d // 2 for d in self.dims]
                self.slider_x.set(self.current_slice[0])
                self.slider_y.set(self.current_slice[1])
                self.slider_z.set(self.current_slice[2])
                self.btn_img.config(bg="green")
                
                # Default Normalized W/L
                self.wl_window = 0.8
                self.wl_level = 0.4
                self.slider_win.set(0.8)
                self.slider_lev.set(0.4)

            elif file_type == 'dose':
                self.dose_data = data
                self.btn_dose.config(bg="green")
                self.slider_thresh.config(to=np.max(data))

            elif file_type == 'mask':
                self.mask_data = data
                self.btn_mask.config(bg="green")

            self.update_plots()
            self._update_info()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def _on_slice_change(self, _):
        self.current_slice = [
            int(self.slider_x.get()),
            int(self.slider_y.get()),
            int(self.slider_z.get())
        ]
        self.update_plots()

    def _on_wl_change(self, _):
        self.wl_window = float(self.slider_win.get())
        self.wl_level = float(self.slider_lev.get())
        self.update_plots()

    def _on_dose_change(self, _):
        self.dose_alpha = float(self.slider_alpha.get()) / 100.0
        self.dose_min_threshold = float(self.slider_thresh.get())
        self.update_plots()
        
    def _on_scroll(self, event):
        if self.img_data is None or event.inaxes is None: return

        axis_idx = -1
        if event.inaxes == self.ax_sag: axis_idx = 0   # X
        elif event.inaxes == self.ax_cor: axis_idx = 1 # Y
        elif event.inaxes == self.ax_axial: axis_idx = 2 # Z
            
        if axis_idx != -1:
            direction = 1 if event.step > 0 else -1
            new_val = self.current_slice[axis_idx] + direction
            new_val = max(0, min(new_val, self.dims[axis_idx] - 1))
            self.current_slice[axis_idx] = new_val
            
            if axis_idx == 0: self.slider_x.set(new_val)
            elif axis_idx == 1: self.slider_y.set(new_val)
            elif axis_idx == 2: self.slider_z.set(new_val)
            
            self.update_plots()
            self._update_info()

    def _on_mouse_press(self, event):
        if self.img_data is None or event.inaxes is None: return

        # Left Click: Move Crosshair
        if event.button == 1:
            ix, iy = int(event.xdata), int(event.ydata)
            
            if event.inaxes == self.ax_axial:
                # Axial (XY Plane). X=0..N, Y=0..N. Origin Lower.
                # Transposed: X is axis 0, Y is axis 1.
                new_x = max(0, min(ix, self.dims[0] - 1))
                new_y = max(0, min(iy, self.dims[1] - 1))
                self.current_slice[0] = new_x
                self.current_slice[1] = new_y
                self.slider_x.set(new_x)
                self.slider_y.set(new_y)

            elif event.inaxes == self.ax_sag:
                # Sagittal (YZ Plane). Y is axis 1, Z is axis 2.
                # Transposed: Y horizontal, Z vertical.
                new_y = max(0, min(ix, self.dims[1] - 1))
                new_z = max(0, min(iy, self.dims[2] - 1))
                self.current_slice[1] = new_y
                self.current_slice[2] = new_z
                self.slider_y.set(new_y)
                self.slider_z.set(new_z)

            elif event.inaxes == self.ax_cor:
                # Coronal (XZ Plane). X is axis 0, Z is axis 2.
                # Transposed: X horizontal, Z vertical.
                new_x = max(0, min(ix, self.dims[0] - 1))
                new_z = max(0, min(iy, self.dims[2] - 1))
                self.current_slice[0] = new_x
                self.current_slice[2] = new_z
                self.slider_x.set(new_x)
                self.slider_z.set(new_z)

            self.update_plots()
            self._update_info()
        
        # Right Click: Start Zoom
        elif event.button == 3:
            self.last_mouse_y = event.y
            self.active_zoom_ax = event.inaxes

    def _on_mouse_release(self, event):
        if event.button == 3:
            self.active_zoom_ax = None

    def _on_mouse_move(self, event):
        # Right Click Drag: Zoom
        if self.active_zoom_ax is not None and event.y is not None:
            dy = event.y - self.last_mouse_y
            self.last_mouse_y = event.y
            
            # Determine which view index (0=Axial, 1=Sag, 2=Cor)
            view_idx = -1
            if self.active_zoom_ax == self.ax_axial: view_idx = 0
            elif self.active_zoom_ax == self.ax_sag: view_idx = 1
            elif self.active_zoom_ax == self.ax_cor: view_idx = 2
            
            if view_idx != -1:
                zoom_speed = 0.01
                factor = 1.0 + (dy * zoom_speed)
                # Limit zoom range
                new_scale = np.clip(self.zoom_scales[view_idx] * factor, 0.1, 20.0)
                self.zoom_scales[view_idx] = new_scale
                self.update_plots()

    def _update_info(self):
        cx, cy, cz = self.current_slice
        txt = f"Dim: {self.dims}\nCur: ({cx}, {cy}, {cz})\n"
        if self.img_data is not None:
            val = self.img_data[cx, cy, cz]
            txt += f"Img: {val:.4f} (Norm)\n"
        if self.dose_data is not None:
            val = self.dose_data[cx, cy, cz]
            txt += f"Dose: {val:.2f} Gy\n"
        self.info_label.config(text=txt)

    def update_plots(self):
        if self.img_data is None:
            return

        cx, cy, cz = self.current_slice

        # ITK-SNAP Orientation:
        # Use Transpose (.T) and origin='lower'
        # Axial (Z):   data[:,:,z].T  ->  X horizontal, Y vertical (Ant up)
        # Sagittal (X): data[x,:,:].T ->  Y horizontal (Ant right), Z vertical (Sup up)
        # Coronal (Y):  data[:,y,:].T ->  X horizontal, Z vertical (Sup up)

        slice_ax = self.img_data[:, :, cz].T
        slice_sag = self.img_data[cx, :, :].T
        slice_cor = self.img_data[:, cy, :].T

        slices = [slice_ax, slice_sag, slice_cor]
        axes = [self.ax_axial, self.ax_sag, self.ax_cor]

        # Prepare Overlays
        dose_slices = [None]*3
        if self.dose_data is not None and self.show_dose.get():
            dose_slices[0] = self.dose_data[:, :, cz].T
            dose_slices[1] = self.dose_data[cx, :, :].T
            dose_slices[2] = self.dose_data[:, cy, :].T

        mask_slices = [None]*3
        if self.mask_data is not None and self.show_mask.get():
            mask_slices[0] = self.mask_data[:, :, cz].T
            mask_slices[1] = self.mask_data[cx, :, :].T
            mask_slices[2] = self.mask_data[:, cy, :].T

        vmin = self.wl_level - self.wl_window / 2
        vmax = self.wl_level + self.wl_window / 2

        for i, ax in enumerate(axes):
            ax.clear()
            
            # --- Base Image ---
            ax.imshow(slices[i], cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
            
            # --- Dose ---
            if dose_slices[i] is not None:
                d = dose_slices[i]
                masked_dose = np.ma.masked_less(d, self.dose_min_threshold)
                ax.imshow(masked_dose, cmap='jet', alpha=self.dose_alpha, origin='lower', aspect='auto',
                          vmin=0, vmax=np.max(self.dose_data))
                if self.show_isolines.get():
                    ax.contour(d, levels=[20, 30, 40, 50, 60], colors='white', linewidths=0.5, alpha=0.7, origin='lower')

            # --- Mask ---
            if mask_slices[i] is not None:
                m = mask_slices[i]
                masked_mask = np.ma.masked_equal(m, 0)
                ax.imshow(masked_mask, cmap='cool', alpha=0.3, origin='lower', aspect='auto')

            # --- Crosshairs (Mapped for origin='lower' + Transpose) ---
            # Axial (0): X, Y.  v=cx, h=cy
            # Sag (1):   Y, Z.  v=cy, h=cz
            # Cor (2):   X, Z.  v=cx, h=cz
            
            vline, hline = 0, 0
            if i == 0: vline, hline = cx, cy
            elif i == 1: vline, hline = cy, cz
            elif i == 2: vline, hline = cx, cz
            
            ax.axhline(hline, color='lime', linewidth=0.8, alpha=0.6)
            ax.axvline(vline, color='lime', linewidth=0.8, alpha=0.6)
            
            # --- Apply Zoom ---
            # Calculate center of view
            h, w = slices[i].shape
            center_x, center_y = w / 2, h / 2
            scale = self.zoom_scales[i]
            
            # New half-width/height
            hw = (w / 2) / scale
            hh = (h / 2) / scale
            
            ax.set_xlim(center_x - hw, center_x + hw)
            ax.set_ylim(center_y - hh, center_y + hh)

        # Histogram
        self.ax_hist.clear()
        if self.dose_data is not None:
             flat_dose = self.dose_data[self.dose_data > 1].flatten()
             if len(flat_dose) > 0:
                self.ax_hist.hist(flat_dose, bins=50, color='blue', alpha=0.7)
                self.ax_hist.set_title("Dose Histogram (> 1Gy)", color="white", fontsize=8)
        
        self.ax_axial.set_title(f"Axial (Z={cz})", color="white")
        self.ax_sag.set_title(f"Sagittal (X={cx})", color="white")
        self.ax_cor.set_title(f"Coronal (Y={cy})", color="white")
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalViewerApp(root)
    root.mainloop()