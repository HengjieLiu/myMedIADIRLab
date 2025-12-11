import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

# --- CONFIGURATION & STYLES ---
BG_COLOR = "#2b2b2b"
FG_COLOR = "white"
ACCENT_COLOR = "#4a4a4a"
FONT_STD = ("Helvetica", 9)
FONT_BOLD = ("Helvetica", 9, "bold")

class MedicalViewer_Dual:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Viewer v2.0 (Dual Comparison)")
        self.root.geometry("1800x950")
        self.root.configure(bg=BG_COLOR)

        # --- DATA STATE (Lists of length 2 for [Group1, Group2]) ---
        self.img_data = [None, None]
        self.dose_data = [None, None]
        self.mask_data = [None, None]
        self.affine = [None, None]
        self.voxel_spacing = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)] 
        
        # Dimensions & Coordinates
        self.dims = [[0, 0, 0], [0, 0, 0]] 
        self.current_slice = [[0, 0, 0], [0, 0, 0]]

        # Interaction State
        # 6 plots total: [Ax1, Sag1, Cor1, Ax2, Sag2, Cor2]
        self.figs = []
        self.axes = []
        self.canvases = []
        
        # Histograms: [HistImg1, HistDose1, HistImg2, HistDose2]
        self.hist_axes = []
        self.hist_canvases = []

        self.drag_mode = None # 'pan', 'zoom', 'contrast'
        self.last_mouse = (0, 0)
        
        # Per-group view state
        self.zoom_scales = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] 
        self.pan_offsets = [[[0,0], [0,0], [0,0]], [[0,0], [0,0], [0,0]]]

        # Global Settings (Shared)
        self.sync_views = tk.BooleanVar(value=True)
        self.show_crosshair = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)

        self.tick_interval_var = tk.StringVar(value="10")
        self.tick_unit_var = tk.StringVar(value="voxel") 
        
        # View Settings (Shared)
        self.wl_center = tk.DoubleVar(value=0)
        self.wl_width = tk.DoubleVar(value=0)
        self.img_min_var = tk.DoubleVar(value=0)
        self.img_max_var = tk.DoubleVar(value=0)
        
        self.dose_opacity = tk.DoubleVar(value=0.5)
        self.show_dose = tk.BooleanVar(value=True)
        self.dose_min = tk.DoubleVar(value=0)
        self.dose_max = tk.DoubleVar(value=0)
        self.isodose_levels = tk.StringVar(value="2, 10, 12, 20")
        self.iso_colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'magenta', 'lime']
        
        self.show_mask = tk.BooleanVar(value=True)
        self.mask_type = tk.StringVar(value="contour") 
        self.mask_opacity = tk.DoubleVar(value=0.3)

        self.hist_log_img = tk.BooleanVar(value=False)
        self.hist_log_dose = tk.BooleanVar(value=False)

        self._init_ui()

    def _init_ui(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG_COLOR, sashwidth=4)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_pane, bg=BG_COLOR, width=380)
        right_frame = tk.Frame(main_pane, bg=BG_COLOR)
        
        main_pane.add(left_frame, minsize=350)
        main_pane.add(right_frame, stretch="always")

        # --- LEFT PANEL ---
        left_frame.rowconfigure(0, weight=1) 
        left_frame.rowconfigure(1, weight=0) 
        left_frame.columnconfigure(0, weight=1)

        self._setup_control_panel(left_frame)
        self._setup_info_panel(left_frame)

        # --- RIGHT PANEL (THE VIEWER) ---
        self._setup_viewer_grid(right_frame)

    def _setup_control_panel(self, parent):
        container = tk.Frame(parent, bg=BG_COLOR)
        container.grid(row=0, column=0, sticky="nsew")
        
        canvas = tk.Canvas(container, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 1. Data Loading (Split into two groups)
        lbl_frame = tk.LabelFrame(scrollable_frame, text="1. Data Loading", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD)
        lbl_frame.pack(fill="x", padx=5, pady=5)
        
        # Group 1 Loaders
        tk.Label(lbl_frame, text="Group 1 (Left)", bg=BG_COLOR, fg="cyan").grid(row=0, column=0, columnspan=3, sticky="w", padx=5)
        tk.Button(lbl_frame, text="Img 1", command=lambda: self.load_file("img", 0), width=8).grid(row=1, column=0, padx=2)
        tk.Button(lbl_frame, text="Dose 1", command=lambda: self.load_file("dose", 0), width=8).grid(row=1, column=1, padx=2)
        tk.Button(lbl_frame, text="Seg 1", command=lambda: self.load_file("seg", 0), width=8).grid(row=1, column=2, padx=2)

        # Group 2 Loaders
        tk.Label(lbl_frame, text="Group 2 (Right)", bg=BG_COLOR, fg="orange").grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(5,0))
        tk.Button(lbl_frame, text="Img 2", command=lambda: self.load_file("img", 1), width=8).grid(row=3, column=0, padx=2)
        tk.Button(lbl_frame, text="Dose 2", command=lambda: self.load_file("dose", 1), width=8).grid(row=3, column=1, padx=2)
        tk.Button(lbl_frame, text="Seg 2", command=lambda: self.load_file("seg", 1), width=8).grid(row=3, column=2, padx=2)
        
        # Log
        self.log_text = tk.Text(lbl_frame, height=4, width=35, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 8))
        self.log_text.grid(row=4, column=0, columnspan=3, padx=2, pady=5)
        self.log_message("System Ready.")

        # 2. Controls
        ctrl_frame = tk.LabelFrame(scrollable_frame, text="2. Shared Controls", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD)
        ctrl_frame.pack(fill="x", padx=5, pady=5)

        # Sync Checkbox
        row = 0
        tk.Checkbutton(ctrl_frame, text="Sync Views (Nav/Zoom)", variable=self.sync_views, command=self.on_sync_toggle, 
                       bg=BG_COLOR, fg="yellow", selectcolor=BG_COLOR, font=FONT_BOLD).grid(row=row, column=0, columnspan=2, sticky="w")
        
        tk.Checkbutton(ctrl_frame, text="Show Grid", variable=self.show_grid, command=self.update_plots, bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR).grid(row=row, column=2, sticky="w")
        
        # Navigation (Primary/Shared)
        row += 1
        tk.Label(ctrl_frame, text="Navigation (Group 1 / Master):", bg=BG_COLOR, fg=FG_COLOR).grid(row=row, column=0, columnspan=3, sticky="w", pady=(5,0))
        
        self.nav_vars = [tk.IntVar(), tk.IntVar(), tk.IntVar()]
        dims_labels = ["X (Sag)", "Y (Cor)", "Z (Axial)"]
        dirs = [("L", "R"), ("P", "A"), ("I", "S")] 

        self.sliders = []
        for i in range(3):
            row += 1
            f = tk.Frame(ctrl_frame, bg=BG_COLOR)
            f.grid(row=row, column=0, columnspan=3, sticky="ew")
            tk.Label(f, text=f"{dims_labels[i]}:", width=8, bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
            tk.Button(f, text="<", width=3, command=lambda axis=i: self.nudge_slice_ui(axis, -1)).pack(side="left")
            s = tk.Scale(f, from_=0, to=100, variable=self.nav_vars[i], orient="horizontal", showvalue=0, bg=BG_COLOR, fg=FG_COLOR, highlightthickness=0)
            s.pack(side="left", fill="x", expand=True)
            s.bind("<B1-Motion>", lambda e, ax=i: self.set_slice_from_slider(ax))
            s.bind("<ButtonRelease-1>", lambda e, ax=i: self.set_slice_from_slider(ax))
            self.sliders.append(s)
            tk.Button(f, text=">", width=3, command=lambda axis=i: self.nudge_slice_ui(axis, 1)).pack(side="left")
            e = tk.Entry(f, textvariable=self.nav_vars[i], width=4)
            e.pack(side="left")
            e.bind("<Return>", lambda e, ax=i: self.set_slice_from_entry(ax))

        # W/L Settings
        row += 1
        tk.Label(ctrl_frame, text="Window / Level (Shared):", bg=BG_COLOR, fg="cyan").grid(row=row, column=0, sticky="w", pady=(5,0))
        
        row += 1
        wl_f = tk.Frame(ctrl_frame, bg=BG_COLOR)
        wl_f.grid(row=row, column=0, columnspan=3, sticky="ew")
        tk.Label(wl_f, text="L:", bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
        self.scale_level = tk.Scale(wl_f, from_=0, to=1, orient="horizontal", command=self.on_wl_change, showvalue=0, bg=BG_COLOR)
        self.scale_level.pack(side="left", fill="x", expand=True)
        tk.Entry(wl_f, textvariable=self.wl_center, width=6).pack(side="left")

        row += 1
        ww_f = tk.Frame(ctrl_frame, bg=BG_COLOR)
        ww_f.grid(row=row, column=0, columnspan=3, sticky="ew")
        tk.Label(ww_f, text="W:", bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
        self.scale_width = tk.Scale(ww_f, from_=1, to=1, orient="horizontal", command=self.on_wl_change, showvalue=0, bg=BG_COLOR)
        self.scale_width.pack(side="left", fill="x", expand=True)
        tk.Entry(ww_f, textvariable=self.wl_width, width=6).pack(side="left")

        # 3. Dose & Seg
        ds_frame = tk.LabelFrame(scrollable_frame, text="3. Overlays (Shared)", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD)
        ds_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Checkbutton(ds_frame, text="Dose", variable=self.show_dose, command=self.update_plots, bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR).grid(row=0, column=0, sticky="w")
        tk.Scale(ds_frame, from_=0, to=1, resolution=0.1, variable=self.dose_opacity, orient="horizontal", command=lambda v: self.update_plots(), bg=BG_COLOR, showvalue=0).grid(row=0, column=1, sticky="ew")
        
        tk.Checkbutton(ds_frame, text="Seg", variable=self.show_mask, command=self.update_plots, bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR).grid(row=1, column=0, sticky="w")
        tk.Scale(ds_frame, from_=0, to=1, resolution=0.1, variable=self.mask_opacity, orient="horizontal", command=lambda v: self.update_plots(), bg=BG_COLOR, showvalue=0).grid(row=1, column=1, sticky="ew")

        tk.Label(ds_frame, text="Isodoses:", bg=BG_COLOR, fg=FG_COLOR).grid(row=2, column=0, sticky="w")
        e_iso = tk.Entry(ds_frame, textvariable=self.isodose_levels)
        e_iso.grid(row=2, column=1, sticky="ew")
        e_iso.bind("<Return>", lambda e: self.update_plots())

    def _setup_info_panel(self, parent):
        f = tk.LabelFrame(parent, text="Cursor Info", bg="black", fg="yellow", font=FONT_BOLD)
        f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.lbl_info_pos = tk.Label(f, text="Pos: -", bg="black", fg="white", font=FONT_STD, anchor="w")
        self.lbl_info_pos.pack(fill="x")
        self.lbl_info_val1 = tk.Label(f, text="Grp1: -", bg="black", fg="cyan", font=FONT_STD, anchor="w")
        self.lbl_info_val1.pack(fill="x")
        self.lbl_info_val2 = tk.Label(f, text="Grp2: -", bg="black", fg="orange", font=FONT_STD, anchor="w")
        self.lbl_info_val2.pack(fill="x")

    def _setup_viewer_grid(self, parent):
        # 2x4 Layout. 
        # Left 2x2: Group 1. Right 2x2: Group 2.
        viewer_frame = tk.Frame(parent, bg=BG_COLOR)
        viewer_frame.pack(fill="both", expand=True)
        
        for i in range(4): viewer_frame.columnconfigure(i, weight=1)
        for i in range(2): viewer_frame.rowconfigure(i, weight=1)

        # 6 Viewers (3 per group). 
        # Indices: 0-2 (Grp1), 3-5 (Grp2)
        # 0: Ax1, 1: Sag1, 2: Cor1
        # 3: Ax2, 4: Sag2, 5: Cor2
        
        # Grid positions (row, col)
        # Grp 1
        pos_map = {
            0: (0, 0), # Ax1
            1: (0, 1), # Sag1
            2: (1, 1), # Cor1
            # Grp 2
            3: (0, 2), # Ax2
            4: (0, 3), # Sag2
            5: (1, 3)  # Cor2
        }

        for i in range(6):
            fig = Figure(figsize=(3, 3), dpi=100, facecolor="black")
            ax = fig.add_axes([0, 0, 1, 1]) 
            ax.axis('off')
            ax.set_facecolor("black")
            
            canvas = FigureCanvasTkAgg(fig, viewer_frame)
            widget = canvas.get_tk_widget()
            
            r, c = pos_map[i]
            widget.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
            
            # Events
            canvas.mpl_connect('button_press_event', self.on_click)
            canvas.mpl_connect('button_release_event', self.on_release)
            canvas.mpl_connect('motion_notify_event', self.on_move)
            canvas.mpl_connect('scroll_event', self.on_scroll)
            
            self.figs.append(fig)
            self.axes.append(ax)
            self.canvases.append(canvas)

        # Histograms Areas
        # Hist 1 at (1, 0), Hist 2 at (1, 2)
        hist_locs = [(1, 0), (1, 2)]
        
        for idx, (r, c) in enumerate(hist_locs):
            h_frame = tk.Frame(viewer_frame, bg=BG_COLOR, bd=1, relief="sunken")
            h_frame.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
            
            # Tiny toolbar
            tb = tk.Frame(h_frame, bg=BG_COLOR)
            tb.pack(side="top", fill="x")
            lbl = tk.Label(tb, text=f"Hist {idx+1}", bg=BG_COLOR, fg=FG_COLOR, font=("Helvetica", 7)).pack(side="left")
            
            # Fig
            fig = Figure(figsize=(3, 1.5), dpi=100, facecolor=BG_COLOR)
            # Img Hist Ax
            ax_i = fig.add_subplot(211) # Top half
            ax_i.set_facecolor(BG_COLOR)
            # Dose Hist Ax
            ax_d = fig.add_subplot(212) # Bot half
            ax_d.set_facecolor(BG_COLOR)
            
            canvas = FigureCanvasTkAgg(fig, h_frame)
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            
            self.hist_axes.append((ax_i, ax_d)) # Store pair
            self.hist_canvases.append(canvas)


    # --- LOGIC ---

    def log_message(self, msg):
        self.log_text.insert(tk.END, f">> {msg}\n")
        self.log_text.see(tk.END)

    def on_sync_toggle(self):
        if self.sync_views.get():
            # Snap Group 2 to Group 1
            self.log_message("Sync ENABLED: Snap Grp2 -> Grp1")
            self.zoom_scales[1] = self.zoom_scales[0][:]
            self.pan_offsets[1] = [p[:] for p in self.pan_offsets[0]]
            
            # Sync Slice (clamping to G2 dims)
            if self.img_data[1] is not None:
                for i in range(3):
                    val = self.current_slice[0][i]
                    max_dim = self.dims[1][i] - 1
                    self.current_slice[1][i] = max(0, min(val, max_dim))
            self.update_plots()
        else:
            self.log_message("Sync DISABLED")

    def load_file(self, dtype, group_idx):
        path = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if not path: return

        try:
            nii = nib.load(path)
            data = nii.get_fdata()
            
            if dtype == 'img':
                self.img_data[group_idx] = data
                self.affine[group_idx] = nii.affine
                self.dims[group_idx] = list(data.shape)
                self.voxel_spacing[group_idx] = nii.header.get_zooms()[:3]
                
                # Reset slice for this group
                self.current_slice[group_idx] = [d // 2 for d in self.dims[group_idx]]
                
                # If this is the first image loaded (or group 0), set global ranges
                if group_idx == 0 or self.img_data[0] is None:
                    v_min, v_max = np.percentile(data, 1), np.percentile(data, 99)
                    self.scale_width.config(from_=1, to=np.max(data) - np.min(data))
                    self.scale_level.config(from_=np.min(data), to=np.max(data))
                    self.wl_center.set((v_max + v_min) / 2)
                    self.wl_width.set(v_max - v_min)
                    self.on_wl_change()
                    
                    # Set UI sliders based on Group 1
                    for i in range(3):
                        self.sliders[i].config(to=self.dims[0][i]-1)
                        self.nav_vars[i].set(self.current_slice[0][i])
                
                self.log_message(f"Loaded G{group_idx+1} Img: {self.dims[group_idx]}")

            elif dtype == 'dose':
                if self.img_data[group_idx] is None: 
                    self.log_message(f"Err: Load G{group_idx+1} Img first")
                    return
                self.dose_data[group_idx] = data
                # Set ranges if not set
                if self.dose_max.get() == 0:
                    self.dose_max.set(np.max(data))
                self.log_message(f"Loaded G{group_idx+1} Dose")

            elif dtype == 'seg':
                if self.img_data[group_idx] is None: return
                self.mask_data[group_idx] = data
                self.log_message(f"Loaded G{group_idx+1} Seg")

            self.update_plots()
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            print(e)

    # --- SHARED CONTROL HANDLERS ---
    def nudge_slice_ui(self, axis, amount):
        # Nudges Group 1 (Master), handles sync inside helper
        self.apply_nudge(0, axis, amount)

    def set_slice_from_slider(self, axis):
        val = self.nav_vars[axis].get()
        self.set_slice_absolute(0, axis, val)

    def set_slice_from_entry(self, axis):
        try:
            val = int(self.nav_vars[axis].get())
            self.set_slice_absolute(0, axis, val)
        except: pass

    # --- LOGIC HELPERS ---
    def apply_nudge(self, source_group, axis, amount):
        if self.img_data[source_group] is None: return
        
        # Helper to set value
        def update_single(g_idx, ax, delta):
            if self.img_data[g_idx] is None: return
            val = self.current_slice[g_idx][ax] + delta
            val = max(0, min(val, self.dims[g_idx][ax] - 1))
            self.current_slice[g_idx][ax] = val
            # Update UI if Group 0
            if g_idx == 0: self.nav_vars[ax].set(val)

        update_single(source_group, axis, amount)

        # Sync Logic
        if self.sync_views.get():
            target = 1 if source_group == 0 else 0
            update_single(target, axis, amount)
        
        self.update_plots()

    def set_slice_absolute(self, source_group, axis, val):
        if self.img_data[source_group] is None: return
        
        def set_single(g_idx, ax, v):
            if self.img_data[g_idx] is None: return
            v = max(0, min(v, self.dims[g_idx][ax] - 1))
            self.current_slice[g_idx][ax] = v
            if g_idx == 0: self.nav_vars[ax].set(v)
            
        set_single(source_group, axis, val)
        
        if self.sync_views.get():
            target = 1 if source_group == 0 else 0
            # If dimensions differ, index matching might map out of bounds. Clamp it.
            set_single(target, axis, val)

        self.update_plots()

    def on_wl_change(self, _=None):
        c = self.wl_center.get()
        w = self.wl_width.get()
        self.img_min_var.set(round(c - w/2, 2))
        self.img_max_var.set(round(c + w/2, 2))
        self.update_plots()

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

    def _get_slice_view(self, data3d, view_idx, cx, cy, cz):
        # 0:Axial(Z), 1:Sag(X), 2:Cor(Y)
        if view_idx == 0: return np.flip(data3d[:, :, cz].T, 1)
        elif view_idx == 1: return np.flip(data3d[cx, :, :].T, 1)
        elif view_idx == 2: return np.flip(data3d[:, cy, :].T, 1)
        return None

    def update_plots(self):
        # This handles both groups
        self.update_histograms()
        
        vmin, vmax = self.img_min_var.get(), self.img_max_var.get()
        titles_template = ["Axial", "Sagittal", "Coronal"]
        
        try:
            iso_levels = [float(v) for v in self.isodose_levels.get().split(',')]
        except: iso_levels = []

        # Iterate over all 6 axes
        # indices 0-2: Group 0. indices 3-5: Group 1.
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.axis('on')
            
            group_idx = 0 if i < 3 else 1
            view_idx = i % 3 # 0, 1, 2
            
            if self.img_data[group_idx] is None:
                ax.text(0.5, 0.5, f"No Data G{group_idx+1}", color="gray", ha="center")
                self.canvases[i].draw_idle()
                continue
                
            data = self.img_data[group_idx]
            cx, cy, cz = self.current_slice[group_idx]
            sx, sy, sz = self.voxel_spacing[group_idx]
            
            # Slice extraction
            sl = self._get_slice_view(data, view_idx, cx, cy, cz)
            
            # Aspect Ratio
            aspects = [sy/sx, sz/sy, sz/sx] # for ax, sag, cor
            aspect = aspects[view_idx]
            
            # Plot Base
            ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect=aspect)
            
            # Plot Dose
            if self.dose_data[group_idx] is not None and self.show_dose.get():
                d_sl = self._get_slice_view(self.dose_data[group_idx], view_idx, cx, cy, cz)
                d_min, d_max = self.dose_min.get(), self.dose_max.get()
                masked = np.ma.masked_where(d_sl < d_min, d_sl)
                ax.imshow(masked, cmap='jet', vmin=d_min, vmax=d_max, alpha=self.dose_opacity.get(), origin='lower', aspect=aspect)
                if iso_levels:
                    c_list = [self.iso_colors[x % len(self.iso_colors)] for x in range(len(iso_levels))]
                    ax.contour(d_sl, levels=iso_levels, colors=c_list, linewidths=0.8, origin='lower')

            # Plot Mask
            if self.mask_data[group_idx] is not None and self.show_mask.get():
                m_sl = self._get_slice_view(self.mask_data[group_idx], view_idx, cx, cy, cz)
                if self.mask_type.get() == 'mask':
                    masked_seg = np.ma.masked_where(m_sl == 0, m_sl)
                    ax.imshow(masked_seg, cmap='spring', alpha=self.mask_opacity.get(), origin='lower', aspect=aspect)
                else:
                    ax.contour(m_sl, levels=[0.5], colors='lime', linewidths=1.0, origin='lower')

            # Ticks
            h, w = sl.shape
            if view_idx==0: self._draw_ticks(ax, (h,w), sx, sy)
            elif view_idx==1: self._draw_ticks(ax, (h,w), sy, sz)
            elif view_idx==2: self._draw_ticks(ax, (h,w), sx, sz)
            
            # Crosshair
            if self.show_crosshair.get():
                # Logic: Ax(0)->(Y, -X), Sag(1)->(Z, -Y), Cor(2)->(Z, -X)
                # Correction: See v1.8 logic.
                # Ax(0): x_plot ~ dx-1-cx, y_plot ~ cy
                dims = self.dims[group_idx]
                cross_x, cross_y = -1, -1
                if view_idx == 0: 
                    cross_x, cross_y = dims[0] - 1 - cx, cy
                elif view_idx == 1:
                    cross_x, cross_y = dims[1] - 1 - cy, cz
                elif view_idx == 2:
                    cross_x, cross_y = dims[0] - 1 - cx, cz
                
                ax.axvline(cross_x, color='lime', lw=1, alpha=0.8)
                ax.axhline(cross_y, color='lime', lw=1, alpha=0.8)

            # Labels/Title
            ax.set_title(f"G{group_idx+1}: {titles_template[view_idx]}", color='white', fontsize=9)
            
            # Zoom/Pan
            scale = self.zoom_scales[group_idx][view_idx]
            dx_pan, dy_pan = self.pan_offsets[group_idx][view_idx]
            
            cx_v, cy_v = w/2, h/2
            new_w, new_h = w/scale, h/scale
            ax.set_xlim([cx_v - new_w/2 - dx_pan, cx_v + new_w/2 - dx_pan])
            ax.set_ylim([cy_v - new_h/2 - dy_pan, cy_v + new_h/2 - dy_pan])
            
            self.canvases[i].draw_idle()

    def update_histograms(self):
        # Update both histogram sets
        for g_idx in range(2):
            ax_img, ax_dose = self.hist_axes[g_idx]
            ax_img.clear()
            ax_dose.clear()
            
            if self.img_data[g_idx] is not None:
                data = self.img_data[g_idx][::4,::4,::4].flatten()
                data = data[data > np.min(data)]
                ax_img.hist(data, bins=50, color='gray' if g_idx==0 else '#555', alpha=0.8, log=self.hist_log_img.get())
            
            if self.dose_data[g_idx] is not None:
                d = self.dose_data[g_idx][::4,::4,::4].flatten()
                d = d[d > 0.5]
                ax_dose.hist(d, bins=50, color='blue', alpha=0.5, log=self.hist_log_dose.get())
            
            # Styling
            for ax in [ax_img, ax_dose]:
                ax.tick_params(colors='white', labelsize=5)
                ax.set_facecolor(BG_COLOR)
                for spine in ax.spines.values(): spine.set_edgecolor('gray')
            
            ax_img.set_title(f"G{g_idx+1} Hist", color='white', fontsize=7, pad=2)
            self.hist_canvases[g_idx].draw_idle()

    # --- INTERACTION ---
    
    def on_click(self, event):
        if event.inaxes not in self.axes: return
        
        idx = self.axes.index(event.inaxes)
        group_idx = 0 if idx < 3 else 1
        view_idx = idx % 3
        
        if event.button == 1: self.drag_mode = 'click'
        elif event.button == 2: self.drag_mode = 'pan'
        elif event.button == 3: self.drag_mode = 'zoom'
        
        self.last_mouse = (event.x, event.y)
        
        if self.drag_mode == 'click':
            # Reposition slice based on click
            if self.img_data[group_idx] is None: return
            
            xp, yp = event.xdata, event.ydata
            if xp is None or yp is None: return
            
            dx, dy, dz = self.dims[group_idx]
            
            # Inverse logic matching v1.8
            # Ax(0): x=-X, y=Y. Sag(1): x=-Y, y=Z. Cor(2): x=-X, y=Z
            if view_idx == 0:
                nx = int(dx - 1 - xp)
                ny = int(yp)
                self.set_slice_absolute(group_idx, 0, nx)
                self.set_slice_absolute(group_idx, 1, ny)
            elif view_idx == 1:
                ny = int(dy - 1 - xp)
                nz = int(yp)
                self.set_slice_absolute(group_idx, 1, ny)
                self.set_slice_absolute(group_idx, 2, nz)
            elif view_idx == 2:
                nx = int(dx - 1 - xp)
                nz = int(yp)
                self.set_slice_absolute(group_idx, 0, nx)
                self.set_slice_absolute(group_idx, 2, nz)

    def on_release(self, event):
        self.drag_mode = None

    def on_scroll(self, event):
        if event.inaxes not in self.axes: return
        idx = self.axes.index(event.inaxes)
        group_idx = 0 if idx < 3 else 1
        view_idx = idx % 3
        
        # Determine axis traversed
        target_axis = 2 if view_idx == 0 else 0 if view_idx == 1 else 1
        amount = 1 if event.step > 0 else -1
        
        self.apply_nudge(group_idx, target_axis, amount)

    def on_move(self, event):
        # 1. Update Info Box
        if event.inaxes in self.axes:
            idx = self.axes.index(event.inaxes)
            g_idx = 0 if idx < 3 else 1
            v_idx = idx % 3
            
            if self.img_data[g_idx] is not None:
                dx, dy, dz = self.dims[g_idx]
                xp, yp = event.xdata, event.ydata
                
                # Inverse map logic
                rx, ry, rz = -1, -1, -1
                cx, cy, cz = self.current_slice[g_idx]
                
                if v_idx == 0:   rx, ry, rz = int(dx-1-xp), int(yp), cz
                elif v_idx == 1: rx, ry, rz = cx, int(dy-1-xp), int(yp)
                elif v_idx == 2: rx, ry, rz = int(dx-1-xp), cy, int(yp)
                
                if 0 <= rx < dx and 0 <= ry < dy and 0 <= rz < dz:
                    val = self.img_data[g_idx][rx, ry, rz]
                    self.lbl_info_pos.config(text=f"Pos: ({rx}, {ry}, {rz})")
                    if g_idx == 0:
                        self.lbl_info_val1.config(text=f"Grp1 Img: {val:.2f}")
                        # Also attempt to show Grp2 value if synced/indices match
                        if self.img_data[1] is not None:
                            try: 
                                v2 = self.img_data[1][rx, ry, rz]
                                self.lbl_info_val2.config(text=f"Grp2 Img: {v2:.2f}")
                            except: self.lbl_info_val2.config(text="Grp2: OOB")
                    else:
                        self.lbl_info_val2.config(text=f"Grp2 Img: {val:.2f}")

        # 2. Drag Logic (Zoom/Pan)
        if self.drag_mode in ['zoom', 'pan'] and event.inaxes in self.axes:
            idx = self.axes.index(event.inaxes)
            active_g_idx = 0 if idx < 3 else 1
            
            # If synced, we affect both groups. If not, only active group.
            groups_to_update = [0, 1] if self.sync_views.get() else [active_g_idx]
            
            dx = event.x - self.last_mouse[0]
            dy = event.y - self.last_mouse[1]
            
            if self.drag_mode == 'zoom':
                factor = 1.0 + (dy * 0.01)
                for g in groups_to_update:
                    # Sync Zoom affects all views in the group
                    for v in range(3):
                        self.zoom_scales[g][v] *= factor
                        
            elif self.drag_mode == 'pan':
                for g in groups_to_update:
                    # Pan applies to the specific view index (e.g. Axial pans Axial)
                    # We need to map the event view index to the target group's view index
                    # Since view structure is identical, v_idx is the same.
                    v_idx = idx % 3
                    scale = self.zoom_scales[g][v_idx] * 5.0
                    self.pan_offsets[g][v_idx][0] += dx / scale
                    self.pan_offsets[g][v_idx][1] += dy / scale
            
            self.last_mouse = (event.x, event.y)
            self.update_plots()


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalViewer_Dual(root)
    root.mainloop()