import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import scipy.ndimage

# --- CONFIGURATION & STYLES ---
BG_COLOR = "#2b2b2b"
FG_COLOR = "white"
ACCENT_COLOR = "#4a4a4a"
FONT_STD = ("Helvetica", 9)
FONT_BOLD = ("Helvetica", 9, "bold")

class MedicalViewer_v1_8:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Viewer v1.8 (Refined Single Analysis)")
        self.root.geometry("1600x950")
        self.root.configure(bg=BG_COLOR)

        # --- DATA STATE ---
        self.img_data = None
        self.dose_data = None
        self.mask_data = None
        self.affine = None
        self.voxel_spacing = (1.0, 1.0, 1.0) # x, y, z spacing
        
        # Dimensions & Coordinates
        self.dims = [0, 0, 0] # x, y, z
        self.current_slice = [0, 0, 0]

        # Optimization State
        self.artists = {} 
        self.cbar_dose = None 
        self.cbar_img = None
        self.crosshairs = [None, None, None]
        
        # Interaction State
        self.drag_mode = None # 'pan', 'zoom', 'contrast'
        self.last_mouse = (0, 0)
        self.zoom_scales = [1.0, 1.0, 1.0] # Ax, Sag, Cor
        self.pan_offsets = [[0,0], [0,0], [0,0]] # (dx, dy) for each view
        self.sync_zoom = tk.BooleanVar(value=True)
        self.show_crosshair = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)

        # Settings Vars
        self.tick_interval_var = tk.StringVar(value="10")
        self.tick_unit_var = tk.StringVar(value="voxel") 
        
        # View Settings
        self.wl_center = tk.DoubleVar(value=0)
        self.wl_width = tk.DoubleVar(value=0)
        self.img_min_var = tk.DoubleVar(value=0)
        self.img_max_var = tk.DoubleVar(value=0)
        
        self.dose_opacity = tk.DoubleVar(value=0.5)
        self.show_dose = tk.BooleanVar(value=True)
        self.dose_min = tk.DoubleVar(value=0)
        self.dose_max = tk.DoubleVar(value=0)
        
        # Updated default isodose levels as requested
        self.isodose_levels = tk.StringVar(value="2, 10, 12, 20")
        # Cycle colors for isodose lines
        self.iso_colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'magenta', 'lime']
        
        self.show_mask = tk.BooleanVar(value=True)
        self.mask_type = tk.StringVar(value="contour") 
        self.mask_opacity = tk.DoubleVar(value=0.3)

        # Histogram Log Scale toggles
        self.hist_log_img = tk.BooleanVar(value=False)
        self.hist_log_dose = tk.BooleanVar(value=False)

        self._init_ui()

    def _init_ui(self):
        # === LAYOUT ===
        # Left Panel: Controls + Cursor Info (Bottom)
        # Right Panel: 2x2 Grid (Axial, Sag, Hist, Cor)
        
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG_COLOR, sashwidth=4)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_pane, bg=BG_COLOR, width=380)
        right_frame = tk.Frame(main_pane, bg=BG_COLOR)
        
        main_pane.add(left_frame, minsize=350)
        main_pane.add(right_frame, stretch="always")

        # --- LEFT PANEL (Controls + Info) ---
        left_frame.rowconfigure(0, weight=1) # Controls
        left_frame.rowconfigure(1, weight=0) # Info
        left_frame.columnconfigure(0, weight=1)

        # Scrollable Control Area
        self._setup_control_panel(left_frame)
        
        # Info Panel at Bottom of Left Frame
        self._setup_info_panel(left_frame)

        # --- RIGHT PANEL (THE VIEWER) ---
        self._setup_viewer_grid(right_frame)

    def _setup_control_panel(self, parent):
        # Container for list of controls
        container = tk.Frame(parent, bg=BG_COLOR)
        container.grid(row=0, column=0, sticky="nsew")
        
        canvas = tk.Canvas(container, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 1. Data Loading
        lbl_frame = tk.LabelFrame(scrollable_frame, text="1. Data Loading", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD)
        lbl_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Button(lbl_frame, text="Load Image", command=lambda: self.load_file("img"), width=12).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(lbl_frame, text="Load Dose", command=lambda: self.load_file("dose"), width=12).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(lbl_frame, text="Load Seg", command=lambda: self.load_file("seg"), width=12).grid(row=0, column=2, padx=2, pady=2)
        
        # Log Window
        self.log_text = tk.Text(lbl_frame, height=4, width=35, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 8))
        self.log_text.grid(row=1, column=0, columnspan=3, padx=2, pady=2)
        self.log_message("System Ready.")

        # 2. Image Viewer Controls
        img_frame = tk.LabelFrame(scrollable_frame, text="2. Image Viewer", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD)
        img_frame.pack(fill="x", padx=5, pady=5)

        # 2.1 Basic Controls
        row = 0
        tk.Button(img_frame, text="Reset View", command=self.reset_view).grid(row=row, column=0, padx=2)
        tk.Checkbutton(img_frame, text="Sync Zoom", variable=self.sync_zoom, bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR).grid(row=row, column=1, sticky="w")
        tk.Checkbutton(img_frame, text="Show Grid", variable=self.show_grid, command=self.update_plots, bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR).grid(row=row, column=2, sticky="w")
        
        # 2.3 Navigation (XYZ)
        row += 1
        tk.Label(img_frame, text="Navigation:", bg=BG_COLOR, fg="yellow").grid(row=row, column=0, sticky="w")
        
        self.nav_vars = [tk.IntVar(), tk.IntVar(), tk.IntVar()] # X, Y, Z
        dims_labels = ["X (Sag)", "Y (Cor)", "Z (Axial)"]
        dirs = [("L", "R"), ("P", "A"), ("I", "S")] 

        for i in range(3):
            row += 1
            f = tk.Frame(img_frame, bg=BG_COLOR)
            f.grid(row=row, column=0, columnspan=3, sticky="ew")
            
            tk.Label(f, text=f"{dims_labels[i]}:", width=8, bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
            tk.Button(f, text=f"< {dirs[i][0]}", width=4, command=lambda axis=i: self.nudge_slice(axis, -1)).pack(side="left")
            s = tk.Scale(f, from_=0, to=100, variable=self.nav_vars[i], orient="horizontal", showvalue=0, bg=BG_COLOR, fg=FG_COLOR, highlightthickness=0)
            s.pack(side="left", fill="x", expand=True)
            s.bind("<B1-Motion>", lambda e, ax=i: self.set_slice_from_slider(ax))
            s.bind("<ButtonRelease-1>", lambda e, ax=i: self.set_slice_from_slider(ax))
            self.sliders = self.sliders if hasattr(self, 'sliders') else []
            self.sliders.append(s)
            tk.Button(f, text=f"{dirs[i][1]} >", width=4, command=lambda axis=i: self.nudge_slice(axis, 1)).pack(side="left")
            e = tk.Entry(f, textvariable=self.nav_vars[i], width=5)
            e.pack(side="left")
            e.bind("<Return>", lambda e, ax=i: self.set_slice_from_entry(ax))

        # 2.4 Window/Level
        row += 1
        tk.Label(img_frame, text="W/L:", bg=BG_COLOR, fg="cyan").grid(row=row, column=0, sticky="w")
        
        row += 1
        wl_f = tk.Frame(img_frame, bg=BG_COLOR)
        wl_f.grid(row=row, column=0, columnspan=3, sticky="ew")
        tk.Label(wl_f, text="Level:", bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
        self.scale_level = tk.Scale(wl_f, from_=0, to=1, orient="horizontal", command=self.on_wl_change, showvalue=0, bg=BG_COLOR)
        self.scale_level.pack(side="left", fill="x", expand=True)
        tk.Entry(wl_f, textvariable=self.wl_center, width=6).pack(side="left")

        row += 1
        ww_f = tk.Frame(img_frame, bg=BG_COLOR)
        ww_f.grid(row=row, column=0, columnspan=3, sticky="ew")
        tk.Label(ww_f, text="Width:", bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
        self.scale_width = tk.Scale(ww_f, from_=1, to=1, orient="horizontal", command=self.on_wl_change, showvalue=0, bg=BG_COLOR)
        self.scale_width.pack(side="left", fill="x", expand=True)
        tk.Entry(ww_f, textvariable=self.wl_width, width=6).pack(side="left")
        
        row += 1
        mm_f = tk.Frame(img_frame, bg=BG_COLOR)
        mm_f.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        tk.Label(mm_f, text="Range [Min, Max]:", bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
        e_min = tk.Entry(mm_f, textvariable=self.img_min_var, width=6)
        e_min.pack(side="left", padx=2)
        e_min.bind("<Return>", self.on_minmax_entry)
        e_max = tk.Entry(mm_f, textvariable=self.img_max_var, width=6)
        e_max.pack(side="left", padx=2)
        e_max.bind("<Return>", self.on_minmax_entry)

        # 2.5 Ticks Control
        row += 1
        tick_f = tk.Frame(img_frame, bg=BG_COLOR)
        tick_f.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        tk.Label(tick_f, text="Grid Ticks:", bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
        tk.Entry(tick_f, textvariable=self.tick_interval_var, width=5).pack(side="left", padx=2)
        ttk.Combobox(tick_f, textvariable=self.tick_unit_var, values=["voxel", "mm"], width=5, state="readonly").pack(side="left", padx=2)
        tk.Button(tick_f, text="Set", command=self.update_plots, width=4).pack(side="left")

        # 3. Dose Viewer
        dose_frame = tk.LabelFrame(scrollable_frame, text="3. Dose Viewer", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD)
        dose_frame.pack(fill="x", padx=5, pady=5)
        
        row = 0
        tk.Checkbutton(dose_frame, text="Overlay Dose", variable=self.show_dose, command=self.update_plots, bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR).grid(row=row, column=0, sticky="w")
        
        row += 1
        tk.Label(dose_frame, text="Opacity:", bg=BG_COLOR, fg=FG_COLOR).grid(row=row, column=0, sticky="w")
        d_op = tk.Scale(dose_frame, from_=0, to=1, resolution=0.1, orient="horizontal", variable=self.dose_opacity, command=lambda v: self.update_plots(), bg=BG_COLOR)
        d_op.grid(row=row, column=1, sticky="ew")
        
        row += 1
        tk.Label(dose_frame, text="Dose Range (Min/Max):", bg=BG_COLOR, fg=FG_COLOR).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        dr_f = tk.Frame(dose_frame, bg=BG_COLOR)
        dr_f.grid(row=row, column=0, columnspan=2, sticky="ew")
        tk.Entry(dr_f, textvariable=self.dose_min, width=6).pack(side="left")
        self.slider_dose_min = tk.Scale(dr_f, from_=0, to=100, orient="horizontal", variable=self.dose_min, showvalue=0, bg=BG_COLOR) 
        self.slider_dose_min.pack(side="left", fill="x", expand=True)
        
        row += 1
        dr_f2 = tk.Frame(dose_frame, bg=BG_COLOR)
        dr_f2.grid(row=row, column=0, columnspan=2, sticky="ew")
        tk.Entry(dr_f2, textvariable=self.dose_max, width=6).pack(side="left")
        self.slider_dose_max = tk.Scale(dr_f2, from_=0, to=100, orient="horizontal", variable=self.dose_max, showvalue=0, bg=BG_COLOR)
        self.slider_dose_max.pack(side="left", fill="x", expand=True)
        self.slider_dose_min.bind("<ButtonRelease-1>", lambda e: self.update_plots())
        self.slider_dose_max.bind("<ButtonRelease-1>", lambda e: self.update_plots())

        row += 1
        tk.Label(dose_frame, text="Isodose Levels (sep by ,):", bg=BG_COLOR, fg=FG_COLOR).grid(row=row, column=0, sticky="w")
        iso_ent = tk.Entry(dose_frame, textvariable=self.isodose_levels)
        iso_ent.grid(row=row, column=1, sticky="ew")
        # Trigger update on return
        iso_ent.bind("<Return>", lambda e: self.update_plots())

        # 4. Seg Viewer
        seg_frame = tk.LabelFrame(scrollable_frame, text="4. Seg Viewer", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD)
        seg_frame.pack(fill="x", padx=5, pady=5)
        
        row = 0
        tk.Checkbutton(seg_frame, text="Show Seg", variable=self.show_mask, command=self.update_plots, bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR).grid(row=row, column=0)
        ttk.Combobox(seg_frame, textvariable=self.mask_type, values=["contour", "mask"], width=8, state="readonly").grid(row=row, column=1)
        
        row += 1
        tk.Label(seg_frame, text="Opacity:", bg=BG_COLOR, fg=FG_COLOR).grid(row=row, column=0)
        tk.Scale(seg_frame, from_=0, to=1, resolution=0.1, variable=self.mask_opacity, orient="horizontal", command=lambda v: self.update_plots(), bg=BG_COLOR).grid(row=row, column=1, sticky="ew")

    def _setup_info_panel(self, parent):
        # Feature (h): Info Box fixed at bottom of control panel
        f = tk.LabelFrame(parent, text="Cursor Info", bg="black", fg="yellow", font=FONT_BOLD)
        f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.lbl_info_pos = tk.Label(f, text="Pos (X,Y,Z): -", bg="black", fg="white", font=FONT_STD, anchor="w")
        self.lbl_info_pos.pack(fill="x")
        self.lbl_info_val = tk.Label(f, text="Image Val: -", bg="black", fg="white", font=FONT_STD, anchor="w")
        self.lbl_info_val.pack(fill="x")
        self.lbl_info_dose = tk.Label(f, text="Dose Val: -", bg="black", fg="white", font=FONT_STD, anchor="w")
        self.lbl_info_dose.pack(fill="x")

    def _setup_viewer_grid(self, parent):
        # Layout Requirement:
        # Row 0: Axial, Sagittal
        # Row 1: Histogram Area, Coronal
        
        viewer_frame = tk.Frame(parent, bg=BG_COLOR)
        viewer_frame.pack(fill="both", expand=True)
        
        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.columnconfigure(1, weight=1)
        viewer_frame.rowconfigure(0, weight=1)
        viewer_frame.rowconfigure(1, weight=1)

        self.figs = []
        self.axes = []
        self.canvases = []
        
        # 3 Viewers. 
        # Index 0: Axial (Top Left)
        # Index 1: Sagittal (Top Right)
        # Index 2: Coronal (Bottom Right) -- Note index 2 is Coronal
        
        view_positions = [(0, 0), (0, 1), (1, 1)] # Grid coords
        
        for i in range(3):
            fig = Figure(figsize=(4, 4), dpi=100, facecolor="black")
            ax = fig.add_axes([0, 0, 1, 1]) 
            ax.axis('off')
            ax.set_facecolor("black")
            
            canvas = FigureCanvasTkAgg(fig, viewer_frame)
            widget = canvas.get_tk_widget()
            
            r, c = view_positions[i]
            widget.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
            
            # Events
            canvas.mpl_connect('button_press_event', self.on_click)
            canvas.mpl_connect('button_release_event', self.on_release)
            canvas.mpl_connect('motion_notify_event', self.on_move)
            canvas.mpl_connect('scroll_event', self.on_scroll)
            
            self.figs.append(fig)
            self.axes.append(ax)
            self.canvases.append(canvas)

        # Histograms Area (Bottom Left - 1, 0)
        hist_frame = tk.Frame(viewer_frame, bg=BG_COLOR, bd=1, relief="sunken")
        hist_frame.grid(row=1, column=0, sticky="nsew", padx=1, pady=1)
        
        # --- Image Hist Section ---
        chk_img = tk.Checkbutton(hist_frame, text="Log Scale (Img)", variable=self.hist_log_img, 
                                 command=self.update_histograms, bg=BG_COLOR, fg=FG_COLOR, 
                                 selectcolor=BG_COLOR, font=("Helvetica", 8))
        chk_img.pack(side="top", anchor="w")

        # Image Hist Canvas
        self.fig_hist_img = Figure(figsize=(3, 1.5), dpi=100, facecolor=BG_COLOR)
        self.ax_hist_img = self.fig_hist_img.add_subplot(111)
        self.ax_hist_img.set_facecolor(BG_COLOR)
        self.canvas_hist_img = FigureCanvasTkAgg(self.fig_hist_img, hist_frame)
        self.canvas_hist_img.get_tk_widget().pack(side="top", fill="both", expand=True)

        # --- Dose Hist Section ---
        chk_dose = tk.Checkbutton(hist_frame, text="Log Scale (Dose)", variable=self.hist_log_dose, 
                                  command=self.update_histograms, bg=BG_COLOR, fg=FG_COLOR, 
                                  selectcolor=BG_COLOR, font=("Helvetica", 8))
        chk_dose.pack(side="top", anchor="w")

        # Dose Hist Canvas
        self.fig_hist_dose = Figure(figsize=(3, 1.5), dpi=100, facecolor=BG_COLOR)
        self.ax_hist_dose = self.fig_hist_dose.add_subplot(111)
        self.ax_hist_dose.set_facecolor(BG_COLOR)
        self.canvas_hist_dose = FigureCanvasTkAgg(self.fig_hist_dose, hist_frame)
        self.canvas_hist_dose.get_tk_widget().pack(side="top", fill="both", expand=True)


    # --- LOGIC ---

    def log_message(self, msg):
        self.log_text.insert(tk.END, f">> {msg}\n")
        self.log_text.see(tk.END)

    def load_file(self, dtype):
        path = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if not path: return

        try:
            nii = nib.load(path)
            data = nii.get_fdata()
            # Use raw data logic from v1.3
            
            if dtype == 'img':
                self.img_data = data
                self.affine = nii.affine
                self.dims = list(data.shape)
                self.current_slice = [d // 2 for d in self.dims]
                
                # Ranges
                v_min, v_max = np.percentile(data, 1), np.percentile(data, 99)
                self.scale_width.config(from_=1, to=np.max(data) - np.min(data))
                self.scale_level.config(from_=np.min(data), to=np.max(data))
                self.wl_center.set((v_max + v_min) / 2)
                self.wl_width.set(v_max - v_min)
                self.on_wl_change() 
                
                # Sliders
                for i in range(3):
                    self.sliders[i].config(to=self.dims[i]-1)
                    self.nav_vars[i].set(self.current_slice[i])
                
                self.voxel_spacing = nii.header.get_zooms()[:3]
                self.log_message(f"Loaded Image: {path} {self.dims}")
                self.update_histograms()

            elif dtype == 'dose':
                if self.img_data is None: return
                self.dose_data = data
                self.dose_min.set(0)
                self.dose_max.set(np.max(data))
                self.slider_dose_max.config(to=np.max(data))
                self.slider_dose_min.config(to=np.max(data))
                self.log_message(f"Loaded Dose: {path}")
                self.update_histograms()

            elif dtype == 'seg':
                if self.img_data is None: return
                self.mask_data = data
                self.log_message(f"Loaded Seg: {path}")

            self.update_plots()
            
        except Exception as e:
            self.log_message(f"Error loading {dtype}: {str(e)}")
            print(e)

    def nudge_slice(self, axis, amount):
        if not self.img_data is not None: return
        val = self.current_slice[axis] + amount
        val = max(0, min(val, self.dims[axis] - 1))
        self.current_slice[axis] = val
        self.nav_vars[axis].set(val)
        self.update_plots()

    def set_slice_from_slider(self, axis):
        if self.img_data is None: return
        val = self.nav_vars[axis].get()
        self.current_slice[axis] = val
        self.update_plots()
        
    def set_slice_from_entry(self, axis):
        try:
            val = int(self.nav_vars[axis].get())
            val = max(0, min(val, self.dims[axis]-1))
            self.nav_vars[axis].set(val)
            self.current_slice[axis] = val
            self.update_plots()
        except: pass

    def on_wl_change(self, _=None):
        c = self.wl_center.get()
        w = self.wl_width.get()
        self.img_min_var.set(round(c - w/2, 2))
        self.img_max_var.set(round(c + w/2, 2))
        self.update_plots()

    def on_minmax_entry(self, _=None):
        try:
            vmin = self.img_min_var.get()
            vmax = self.img_max_var.get()
            self.wl_width.set(vmax - vmin)
            self.wl_center.set(vmin + (vmax - vmin)/2)
            self.update_plots()
        except: pass

    def _draw_ticks(self, ax, shape, spacing_x, spacing_y):
        if not self.show_grid.get(): 
            ax.grid(False)
            return

        try: interval = float(self.tick_interval_var.get())
        except: interval = 10.0
            
        unit = self.tick_unit_var.get()
        step_x = interval / spacing_x if unit == 'mm' else interval
        step_y = interval / spacing_y if unit == 'mm' else interval
            
        ax.xaxis.set_major_locator(MultipleLocator(step_x))
        ax.yaxis.set_major_locator(MultipleLocator(step_y))
        ax.grid(True, which='major', color='white', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='both', colors='white', labelbottom=False, labelleft=False)


    def _add_orientation_labels(self, ax, idx):
        # Feature (a): v1.3 Orientation Logic
        # 0 (Axial):   Top: A, Bot: P, Left: R, Right: L
        # 1 (Sagittal):Top: S, Bot: I, Left: A, Right: P
        # 2 (Coronal): Top: S, Bot: I, Left: R, Right: L
        
        labels_map = {
            0: ('A', 'P', 'R', 'L'),
            1: ('S', 'I', 'A', 'P'),
            2: ('S', 'I', 'R', 'L')
        }
        
        t, b, l, r = labels_map.get(idx, ('','','',''))
        props = dict(boxstyle='round', facecolor='black', alpha=0.5)
        ax.text(0.5, 0.98, t, transform=ax.transAxes, color='yellow', ha='center', va='top', fontweight='bold', bbox=props)
        ax.text(0.5, 0.02, b, transform=ax.transAxes, color='yellow', ha='center', va='bottom', fontweight='bold', bbox=props)
        ax.text(0.02, 0.5, l, transform=ax.transAxes, color='yellow', ha='left', va='center', fontweight='bold', bbox=props)
        ax.text(0.98, 0.5, r, transform=ax.transAxes, color='yellow', ha='right', va='center', fontweight='bold', bbox=props)

    def _get_slice_view(self, data3d, view_idx, cx, cy, cz):
        # Logic from v1.3 for correct orientation
        # Axial (0): Z slice. Transpose. Flip Axis 1. -> (Y, -X)
        if view_idx == 0:
            sl = data3d[:, :, cz]
            return np.flip(sl.T, 1)
        # Sagittal (1): X slice. Transpose. Flip Axis 1. -> (Z, -Y)
        elif view_idx == 1:
            sl = data3d[cx, :, :]
            return np.flip(sl.T, 1)
        # Coronal (2): Y slice. Transpose. Flip Axis 1. -> (Z, -X)
        elif view_idx == 2:
            sl = data3d[:, cy, :]
            return np.flip(sl.T, 1)
        return None

    def update_plots(self):
        # Always update histogram when plots update to ensure sync
        self.update_histograms()
        
        if self.img_data is None: return
        
        cx, cy, cz = self.current_slice
        slices = [self._get_slice_view(self.img_data, i, cx, cy, cz) for i in range(3)]
        
        # Prepare overlays
        dose_slices = [None]*3
        if self.dose_data is not None and self.show_dose.get():
            dose_slices = [self._get_slice_view(self.dose_data, i, cx, cy, cz) for i in range(3)]
            
        mask_slices = [None]*3
        if self.mask_data is not None and self.show_mask.get():
            mask_slices = [self._get_slice_view(self.mask_data, i, cx, cy, cz) for i in range(3)]
            
        vmin, vmax = self.img_min_var.get(), self.img_max_var.get()
        sx, sy, sz = self.voxel_spacing
        
        # v1.3 Aspects:
        aspects = [sy/sx, sz/sy, sz/sx]
        
        titles = [
            f"Axial (Z={cz}/{self.dims[2]})",
            f"Sagittal (X={cx}/{self.dims[0]})",
            f"Coronal (Y={cy}/{self.dims[1]})"
        ]

        # Parse iso levels and assign colors
        try:
            iso_levels = [float(v) for v in self.isodose_levels.get().split(',')]
        except:
            iso_levels = []
        
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.axis('on')
            
            # Base Image
            im = ax.imshow(slices[i], cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect=aspects[i])
            
            # Dose Overlay
            if dose_slices[i] is not None:
                d_img = dose_slices[i]
                d_min, d_max = self.dose_min.get(), self.dose_max.get()
                masked_dose = np.ma.masked_where(d_img < d_min, d_img)
                ax.imshow(masked_dose, cmap='jet', vmin=d_min, vmax=d_max, alpha=self.dose_opacity.get(), origin='lower', aspect=aspects[i])
                
                # Draw isodose contour lines
                if iso_levels:
                    # Create colors list matching the levels
                    c_list = [self.iso_colors[idx % len(self.iso_colors)] for idx in range(len(iso_levels))]
                    ax.contour(d_img, levels=iso_levels, colors=c_list, linewidths=0.8, origin='lower')

            # Mask Overlay
            if mask_slices[i] is not None:
                m_img = mask_slices[i]
                if self.mask_type.get() == 'mask':
                    masked_seg = np.ma.masked_where(m_img == 0, m_img)
                    ax.imshow(masked_seg, cmap='spring', alpha=self.mask_opacity.get(), origin='lower', aspect=aspects[i])
                else:
                    ax.contour(m_img, levels=[0.5], colors='lime', linewidths=1.0, origin='lower')

            # Features
            h, w = slices[i].shape
            
            # Crosshair (Logic based on v1.3 orientation mapping)
            if self.show_crosshair.get():
                # Crosshair coordinates logic:
                # Axial(0) displays (Y, -X). x_plot maps to -X_phys. y_plot maps to Y_phys.
                # Center is (cx, cy, cz).
                # Plot X (vertical line) corresponds to data X (flipped). 
                # x_plot = dim_x - 1 - cx.
                # Plot Y (horizontal line) corresponds to data Y.
                # y_plot = cy.
                
                cross_x, cross_y = -1, -1
                if i == 0: # Axial: X_plot ~ dx-1-cx, Y_plot ~ cy
                    cross_x = self.dims[0] - 1 - cx
                    cross_y = cy
                elif i == 1: # Sag: X_plot ~ dy-1-cy, Y_plot ~ cz
                    cross_x = self.dims[1] - 1 - cy
                    cross_y = cz
                elif i == 2: # Cor: X_plot ~ dx-1-cx, Y_plot ~ cz
                    cross_x = self.dims[0] - 1 - cx
                    cross_y = cz
                    
                ax.axvline(cross_x, color='lime', linewidth=1, alpha=0.8)
                ax.axhline(cross_y, color='lime', linewidth=1, alpha=0.8)
            
            # Ticks
            if i==0: self._draw_ticks(ax, slices[i].shape, sx, sy) 
            elif i==1: self._draw_ticks(ax, slices[i].shape, sy, sz)
            elif i==2: self._draw_ticks(ax, slices[i].shape, sx, sz)
            
            self._add_orientation_labels(ax, i)
            ax.set_title(titles[i], color='white', fontsize=10)
            
            # Zoom/Pan
            scale = self.zoom_scales[i]
            dx, dy = self.pan_offsets[i]
            cx_v, cy_v = w/2, h/2
            new_w, new_h = w/scale, h/scale
            ax.set_xlim([cx_v - new_w/2 - dx, cx_v + new_w/2 - dx])
            ax.set_ylim([cy_v - new_h/2 - dy, cy_v + new_h/2 - dy])
            
            self.canvases[i].draw_idle()

    # --- HISTOGRAMS ---
    def update_histograms(self):
        if self.img_data is None: return
        
        self.ax_hist_img.clear()
        data_sub = self.img_data[::4, ::4, ::4].flatten()
        data_sub = data_sub[data_sub > np.min(data_sub)]
        
        # Get Log State for Image
        log_img = self.hist_log_img.get()
        
        self.ax_hist_img.hist(data_sub, bins=50, color='gray', histtype='stepfilled', alpha=0.8, log=log_img)
        self.ax_hist_img.set_title("Image Histogram", color='white', fontsize=8)
        self.ax_hist_img.tick_params(colors='white', labelsize=6)
        self.ax_hist_img.set_facecolor(BG_COLOR)
        self.canvas_hist_img.draw_idle()
        
        if self.dose_data is not None:
            self.ax_hist_dose.clear()
            d_sub = self.dose_data[::4, ::4, ::4].flatten()
            d_sub = d_sub[d_sub > 0.5]
            
            # Get Log State for Dose
            log_dose = self.hist_log_dose.get()
            
            self.ax_hist_dose.hist(d_sub, bins=50, color='blue', alpha=0.5, log=log_dose)
            
            # Parse iso levels
            try:
                iso_levels = [float(v) for v in self.isodose_levels.get().split(',')]
            except:
                iso_levels = []
            
            # Plot vertical lines with cycling colors
            for idx, lvl in enumerate(iso_levels):
                col = self.iso_colors[idx % len(self.iso_colors)]
                self.ax_hist_dose.axvline(lvl, color=col, linestyle='--', linewidth=1.5)
                
            self.ax_hist_dose.set_title("Dose Histogram", color='white', fontsize=8)
            self.ax_hist_dose.tick_params(colors='white', labelsize=6)
            self.ax_hist_dose.set_facecolor(BG_COLOR)
            self.canvas_hist_dose.draw_idle()

    # --- INTERACTION ---
    def on_click(self, event):
        if event.inaxes not in self.axes: return
        
        # Mouse Button Logic: 1=Click, 2=Pan, 3=Zoom
        if event.button == 1: self.drag_mode = 'click'
        elif event.button == 2: self.drag_mode = 'pan'
        elif event.button == 3: self.drag_mode = 'zoom'
        
        self.last_mouse = (event.x, event.y)
        
        if self.drag_mode == 'click':
            # v1.3 Inverse Mapping Logic
            # Ax(0): Display (Y, -X). Click (mx, my).
            # mx = -X_phys + offset? No, just standard flip.
            # Img X index mapped to Horizontal axis? No.
            # slice = flip(data.T, 1) -> (Y, -X).
            # Horizontal (x_plot) is index 1 of array -> -X.
            # Vertical (y_plot) is index 0 of array -> Y.
            # So: x_plot = dim_x - 1 - x_index.
            #     y_plot = y_index.
            # Thus: x_index = dim_x - 1 - x_plot.
            #       y_index = y_plot.
            
            ax_idx = self.axes.index(event.inaxes)
            xp, yp = event.xdata, event.ydata
            if xp is None or yp is None: return
            
            cx, cy, cz = self.current_slice
            dx, dy, dz = self.dims
            
            if ax_idx == 0: # Axial: Shows (Y, -X). x_plot is -X. y_plot is Y.
                # Actually, check logic: flip(T, 1). 
                # T -> (Y, X). flip 1 reverses last dim -> (Y, X[::-1]).
                # So array[i, j] displayed at pixel (j, i).
                # Plot X axis corresponds to Array axis 1 (X) reversed.
                # Plot Y axis corresponds to Array axis 0 (Y).
                # So: xp = dx - 1 - true_x. yp = true_y.
                nx = int(dx - 1 - xp)
                ny = int(yp)
                self.current_slice[0] = max(0, min(nx, dx-1))
                self.current_slice[1] = max(0, min(ny, dy-1))
                
            elif ax_idx == 1: # Sagittal: Shows (Z, -Y). x_plot is -Y. y_plot is Z.
                # array (Z, Y[::-1]).
                # Plot X axis -> Array axis 1 (Y) reversed.
                # Plot Y axis -> Array axis 0 (Z).
                ny = int(dy - 1 - xp)
                nz = int(yp)
                self.current_slice[1] = max(0, min(ny, dy-1))
                self.current_slice[2] = max(0, min(nz, dz-1))
                
            elif ax_idx == 2: # Coronal: Shows (Z, -X). x_plot is -X. y_plot is Z.
                nx = int(dx - 1 - xp)
                nz = int(yp)
                self.current_slice[0] = max(0, min(nx, dx-1))
                self.current_slice[2] = max(0, min(nz, dz-1))

            # Update vars
            for i in range(3): self.nav_vars[i].set(self.current_slice[i])
            self.update_plots()

    def on_release(self, event):
        self.drag_mode = None

    def on_scroll(self, event):
        if event.inaxes not in self.axes: return
        # Logic: Scroll changes slice
        ax_idx = self.axes.index(event.inaxes)
        amount = 1 if event.step > 0 else -1
        
        # Which axis does this view traverse?
        # Axial(0) traverses Z(2).
        # Sag(1) traverses X(0).
        # Cor(2) traverses Y(1).
        target_axis = 2 if ax_idx == 0 else 0 if ax_idx == 1 else 1
        
        self.nudge_slice(target_axis, amount)

    def on_move(self, event):
        if not self.img_data is not None: return
        
        # Crosshair Logic
        if event.inaxes in self.axes and self.show_crosshair.get():
            ax = event.inaxes
            # Remove old crosshair for this ax (inefficient to clear all, but simpler)
            # Better: use blitting or just redraw
            # For simplicity in v1.8, we just update position if we had artist references,
            # but here we rely on standard mpl draw which might be slow. 
            # Given optimization requirement, let's just update text info and handle crosshair via a line artist if stored.
            pass # (Crosshair visual is complex without blitting in MPL/Tk, skipping dynamic line drawing to save FPS, implementing coordinate info)
            
        # Info Box Update
        if event.inaxes in self.axes:
            ax_idx = self.axes.index(event.inaxes)
            xp, yp = event.xdata, event.ydata
            dx, dy, dz = self.dims
            
            # Map back to real coords (same logic as on_click)
            rx, ry, rz = -1, -1, -1
            if ax_idx == 0:
                rx = int(dx - 1 - xp)
                ry = int(yp)
                rz = self.current_slice[2]
            elif ax_idx == 1:
                rx = self.current_slice[0]
                ry = int(dy - 1 - xp)
                rz = int(yp)
            elif ax_idx == 2:
                rx = int(dx - 1 - xp)
                ry = self.current_slice[1]
                rz = int(yp)
            
            if 0 <= rx < dx and 0 <= ry < dy and 0 <= rz < dz:
                val = self.img_data[rx, ry, rz]
                dval = self.dose_data[rx, ry, rz] if self.dose_data is not None else 0
                self.lbl_info_pos.config(text=f"Pos: ({rx}, {ry}, {rz})")
                self.lbl_info_val.config(text=f"Image Val: {val:.2f}")
                self.lbl_info_dose.config(text=f"Dose Val: {dval:.2f}")

        # Dragging Logic
        if self.drag_mode == 'zoom':
            dy = event.y - self.last_mouse[1]
            factor = 1.0 + (dy * 0.01)
            target_axes = range(3) if self.sync_zoom.get() else [self.axes.index(event.inaxes)]
            for i in target_axes:
                self.zoom_scales[i] *= factor
            self.update_plots()
            
        elif self.drag_mode == 'pan':
            dx = event.x - self.last_mouse[0]
            dy = event.y - self.last_mouse[1]
            idx = self.axes.index(event.inaxes)
            # Map screen pixels to data coords approx
            # This is rough; ideally use display-to-data transform
            scale = self.zoom_scales[idx] * 5.0 # Speed factor
            self.pan_offsets[idx][0] += dx / scale
            self.pan_offsets[idx][1] += dy / scale
            self.update_plots()

        self.last_mouse = (event.x, event.y)

    def reset_view(self):
        self.zoom_scales = [1.0, 1.0, 1.0]
        self.pan_offsets = [[0,0], [0,0], [0,0]]
        self.update_plots()

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalViewer_v1_8(root)
    root.mainloop()