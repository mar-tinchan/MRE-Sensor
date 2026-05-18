import serial
import time
import tkinter as tk
import numpy as np
import json
import os
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# ArduinoDataReader
# ─────────────────────────────────────────────────────────────────────────────

class ArduinoDataReader:

    SENSOR_KEYS      = ['top', 'left', 'middle', 'right', 'bottom']
    SENSOR_ROTATIONS = {'top': 270, 'left': 0, 'middle': 90, 'right': 90, 'bottom': 180}

    def __init__(self, port: str, baudrate: int = 2000000, timeout: float = 0.1):
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.ser      = None

        self.active_channels: set[int] = set()
        self.raw_values:  dict = {}
        self.baseline:    dict = {}
        self.diff_values: dict = {}
        self.raw_history: dict = {}

        self.lpf_alpha          = 0.2
        self.median_window_size = 5
        self._leftover          = ""

        # Sampling rate tracking
        self._frame_times: deque = deque(maxlen=30)
        self.sample_rate_hz: float = 0.0

        self.connect()

    def connect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.ser.reset_input_buffer()
            self._leftover = ""
            print(f"Connected to {self.port}")
        except Exception as e:
            print(f"Waiting for device on {self.port}... ({e})")

    def _init_channel(self, ch: int):
        zero = np.zeros(3)
        for key in self.SENSOR_KEYS:
            k = (ch, key)
            self.raw_values[k]  = zero.copy()
            self.baseline[k]    = zero.copy()
            self.diff_values[k] = zero.copy()
            self.raw_history[k] = deque(maxlen=self.median_window_size)
        self.active_channels.add(ch)
        print(f"  -> Channel {ch} detected and initialised.")

    @staticmethod
    def rotate_xy(x, y, angle):
        # Normal Rotation
        #if   angle ==  90: return -y,  x
        #elif angle == 180: return -x, -y
        #elif angle == 270: return  y, -x
        #return x, y
        
        #Mirrored about Y axis (to match the physical sensor orientation)
        if   angle ==  90: return y,  x
        elif angle == 180: return x, -y
        elif angle == 270: return -y, -x
        return -x, y

    def zero_sensors(self, channels=None):
        targets = channels if channels is not None else list(self.active_channels)
        for ch in targets:
            for key in self.SENSOR_KEYS:
                k = (ch, key)
                if k in self.raw_values:
                    self.baseline[k]    = self.raw_values[k].copy()
                    self.diff_values[k] = np.zeros(3)
        print(f"Zeroed channels: {targets}")
        return targets

    def _parse_line(self, line: str) -> bool:
        parts = line.split(',')
        if len(parts) != 16:
            return False
        tag = parts[0]
        if not tag.startswith('CH'):
            return False
        try:
            ch = int(tag[2:])
        except ValueError:
            return False
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            return False

        if ch not in self.active_channels:
            self._init_channel(ch)

        for i, key in enumerate(self.SENSOR_KEYS):
            k   = (ch, key)
            idx = i * 3
            rx_raw, ry_raw, rz = vals[idx], vals[idx + 1], vals[idx + 2]
            rx, ry = self.rotate_xy(rx_raw, ry_raw, self.SENSOR_ROTATIONS[key])
            rx, ry = ry, -rx
            rx, rz = -rx, -rz
            new_raw = np.array([rx, ry, rz])
            self.raw_history[k].append(new_raw)
            median_raw = np.median(self.raw_history[k], axis=0)
            self.raw_values[k] = median_raw
            target_diff = median_raw - self.baseline[k]
            self.diff_values[k] = (
                self.lpf_alpha * target_diff
                + (1 - self.lpf_alpha) * self.diff_values[k]
            )
        return True

    def read(self) -> bool:
        if self.ser is None or not self.ser.is_open:
            self.connect()
            return False
        try:
            if self.ser.in_waiting == 0:
                return False
            chunk = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
            text  = self._leftover + chunk
            lines = text.split('\n')
            self._leftover = lines[-1]
            lines = [l.strip('\r\n ') for l in lines[:-1]]

            frames, current_frame = [], None
            for line in lines:
                if line == 'ST,':
                    current_frame = []
                elif line == 'EN' and current_frame is not None:
                    frames.append(current_frame)
                    current_frame = None
                elif current_frame is not None and line:
                    current_frame.append(line)

            if not frames:
                return False

            latest: dict[int, str] = {}
            for line in frames[-1]:
                if not line.startswith('CH'):
                    continue
                parts = line.split(',')
                if len(parts) != 16:
                    continue
                try:
                    ch = int(parts[0][2:])
                    latest[ch] = line
                except ValueError:
                    continue

            if not latest:
                return False

            updated = False
            for ch in sorted(latest):
                if self._parse_line(latest[ch]):
                    updated = True

            if updated:
                now = time.monotonic()
                self._frame_times.append(now)
                if len(self._frame_times) >= 2:
                    dt = self._frame_times[-1] - self._frame_times[0]
                    if dt > 0:
                        self.sample_rate_hz = (len(self._frame_times) - 1) / dt

            return updated
        except Exception as e:
            print(f"Read error: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Constants / theme
# ─────────────────────────────────────────────────────────────────────────────

NUM_CHANNELS  = 8
SENSOR_LABELS = ['Top', 'Left', 'Middle', 'Right', 'Bottom']
SENSOR_KEYS   = ['top', 'left', 'middle', 'right', 'bottom']

BG_HEADER   = '#2b2d42'
FG_HEADER   = '#edf2f4'
BG_SENSOR   = '#d8d8d8'   
BG_LOCKED   = '#e2e2e2'   # Darker background for the uneditable Z Pos cell
BG_EDITABLE = '#ffffff'   
BG_ROOT     = '#f0f0f0'
BG_RIGHT    = '#e8e8e8'
BG_CANVAS   = '#1a1a2e'
ACCENT      = '#ef233c'
BTN_ALT     = '#457b9d'
BTN_ALT_ACT = '#1d3557'
GRID_CLR    = '#888888'
SUBCOL_W    = 6           # Slightly narrower to fit 3 subcolumns comfortably

CH_COLORS = [
    '#e63946', '#2a9d8f', '#e9c46a', '#f4a261',
    '#a8dadc', '#c77dff', '#90e0ef', '#f8ad9d',
]

SENSOR_MARKERS = {
    'top':    '▲',
    'left':   '◀',
    'middle': '●',
    'right':  '▶',
    'bottom': '▼',
}

ARROW_HALF = 22   

CONFIG_FILENAME = 'force_sensor_position.json' 

# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

class SensorTableApp:

    def __init__(self, root: tk.Tk, reader: ArduinoDataReader):
        self.root   = root
        self.reader = reader
        root.title("ReSkin Live Monitor")
        root.configure(bg=BG_ROOT)
        
        root.geometry("1300x650") # Slightly wider initial resolution for the extra Z columns
        root.resizable(True, True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.sensor_pos_vars: dict = {}
        self.disp_vars: dict = {}
        self.rot_vars: dict = {}
        self.cutoff_var = tk.StringVar(value='0.0') 
        self.is_edit_mode = False
        self.shear_mode   = False   # False = Net Pos, True = Shear vector mode
        self.raw_mode     = False   # False = differential, True = raw readings
        # Per-channel side assignment: 'left' or 'right' (default 'left')
        self.plot_side_vars: dict = {}   # keyed by ch index -> tk.StringVar('left'/'right')
        self.sample_rate_var = tk.StringVar(value='— Hz')

        # Auto-zeroing state  ('off' | 'individual' | 'average')
        self.auto_zero_mode = 'off'
        self.is_zero_settings_mode = False   # True = showing zero settings table

        # Recording state
        self.is_recording    = False
        self._record_start   = 0.0
        self._record_writers: dict = {}
        self._record_files:   dict = {}
        self._record_mode_snapshot = 'netpos'   # 'netpos' | 'shear'
        self._record_max_cops = 0
        self.snapshot_save_csv = False   # toggled by the Save CSV half-button

        # Last-computed world positions for display and recording
        self._net_pos: tuple | None = None        # (wx, wy) world coords
        self._net_pos_left: tuple | None = None   # net pos for left-side sensors
        self._net_pos_right: tuple | None = None  # net pos for right-side sensors
        # Shear displacement vectors — same value the plot draws
        self._shear_disp: tuple | None = None
        self._shear_disp_left: tuple | None = None
        self._shear_disp_right: tuple | None = None
        # Shear limit (saturation radius)
        self.shear_limit_var = tk.StringVar(value='10.0')
        # Shear origin tracking: when a reading appears after no reading,
        # that first reading becomes the displacement baseline (origin).
        self._shear_origin: tuple | None = None          # (wx, wy) world baseline
        self._shear_origin_left: tuple | None = None
        self._shear_origin_right: tuple | None = None
        self._shear_had_reading: bool = False            # was there a reading last frame?
        self._shear_had_reading_left: bool = False
        self._shear_had_reading_right: bool = False

        # ── Shear CoM smoothing (lerp, matching ReadDataV24 Mode-2) ──────────
        # Smoothed CoM positions (updated every poll via lerp)
        self._shear_com_smooth: tuple | None = None        # global
        self._shear_com_smooth_left: tuple | None = None
        self._shear_com_smooth_right: tuple | None = None
        # Frozen CoM at the moment "Zero All" is pressed — delta is the shear vec
        self._shear_com_baseline: tuple | None = None
        self._shear_com_baseline_left: tuple | None = None
        self._shear_com_baseline_right: tuple | None = None
        # Alpha for exponential lerp smoothing (0 = no update, 1 = instant)
        self.shear_com_lerp_alpha = 0.5
        # Per-sensor cutoff thresholds for auto-zeroing (keyed by (ch, key))
        self.az_cutoff_vars: dict = {}
        self.az_deriv_threshold_var = tk.StringVar(value='0.5')
        self.az_settling_time_var   = tk.StringVar(value='1.0')
        # Per-sensor derivative tracking for auto-zeroing
        self._az_prev_diff: dict = {}
        self._az_deriv:     dict = {}
        self._az_settle_start: dict = {}
        # RBF sigma as a percentage of canvas diagonal (0–100), default 15 %
        
        self._scale  = 40.0         
        self._origin = (0.0, 0.0)   
        self._tbl_frame: tk.Frame | None = None

        self._saved_config = {}
        self._load_config()

        self._build_ui()
        self._poll()

    def _load_config(self):
        if os.path.exists(CONFIG_FILENAME):
            try:
                with open(CONFIG_FILENAME, 'r') as f:
                    self._saved_config = json.load(f)
            except Exception as e:
                print(f"Failed to load config: {e}")
                self._saved_config = {}
        else:
            self._saved_config = {}

    def _save_config(self):
        config = {}
        for (ch, key, axis), var in self.sensor_pos_vars.items():
            if axis != 'z': # Don't bother saving Z since it's forced 0
                config[f"pos_{ch}_{key}_{axis}"] = var.get()
        for ch, var in self.rot_vars.items():
            config[f"rot_{ch}"] = var.get()
        
        config["global_cutoff"] = self.cutoff_var.get()
        config["shear_limit"]   = self.shear_limit_var.get()

        # Plot side assignments
        for ch, var in self.plot_side_vars.items():
            config[f"plot_side_{ch}"] = var.get()

        # Auto-zeroing settings
        config["az_deriv_threshold"] = self.az_deriv_threshold_var.get()
        config["az_settling_time"]   = self.az_settling_time_var.get()
        for (ch, key), var in self.az_cutoff_vars.items():
            config[f"az_cutoff_{ch}_{key}"] = var.get()

        try:
            with open(CONFIG_FILENAME, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def _on_closing(self):
        if self.is_recording:
            self.is_recording = False
            for fh in self._record_files.values():
                fh.close()
        self._save_config() 
        if self.reader.ser and self.reader.ser.is_open:
            self.reader.ser.close()
        self.root.destroy() 

    def _hdr(self, parent, text, row, col, colspan=1, rowspan=1):
        lbl = tk.Label(parent, text=text,
                       bg=BG_HEADER, fg=FG_HEADER,
                       font=('Helvetica', 9, 'bold'),
                       padx=4, pady=5, relief='flat')
        lbl.grid(row=row, column=col, columnspan=colspan, rowspan=rowspan,
                 sticky='nsew', padx=1, pady=1)
        return lbl

    def _build_ui(self):
        left  = tk.Frame(self.root, bg=BG_ROOT)
        right = tk.Frame(self.root, bg=BG_RIGHT, padx=18, pady=18)

        right.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=8, pady=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)

        self._build_tables(left)
        self._build_canvas(left)
        self._build_controls(right)

    # ─── Left Panel: Swappable Tables ─────────────────────────────────────────

    def _build_tables(self, parent):
        self.table_container = tk.Frame(parent, bg=BG_ROOT)
        self.table_container.pack(anchor='nw')
        self.table_container.rowconfigure(0, weight=1)
        self.table_container.columnconfigure(0, weight=1)
        
        self._tbl_frame = self.table_container 

        self.frame_data         = tk.Frame(self.table_container, bg=GRID_CLR)
        self.frame_edit         = tk.Frame(self.table_container, bg=GRID_CLR)
        self.frame_zero_settings = tk.Frame(self.table_container, bg=GRID_CLR)

        self.frame_data.grid(row=0, column=0, sticky='nsew')
        self.frame_edit.grid(row=0, column=0, sticky='nsew')
        self.frame_zero_settings.grid(row=0, column=0, sticky='nsew')

        self._populate_data_table(self.frame_data)
        self._populate_edit_table(self.frame_edit)
        self._populate_zero_settings_table(self.frame_zero_settings)

        self.frame_data.tkraise()

    def _populate_data_table(self, tbl):
        self._hdr_title_label = self._hdr(tbl, 'Live Data', row=0, col=0, rowspan=2)
        self._data_hdr_labels = []   # axis sub-headers whose text we swap

        for ch in range(NUM_CHANNELS):
            self._hdr(tbl, f'Chn {ch}', row=0, col=1 + ch * 3, colspan=3)
            for ai, (diff_txt, raw_txt) in enumerate([('Diff X', 'Raw X'),
                                                       ('Diff Y', 'Raw Y'),
                                                       ('Diff Z', 'Raw Z')]):
                lbl = self._hdr(tbl, diff_txt, row=1, col=1 + ch * 3 + ai)
                lbl._diff_text = diff_txt
                lbl._raw_text  = raw_txt
                self._data_hdr_labels.append(lbl)

        for si, (label, key) in enumerate(zip(SENSOR_LABELS, SENSOR_KEYS)):
            row = 2 + si
            self._hdr(tbl, label, row=row, col=0)
            for ch in range(NUM_CHANNELS):
                base = 1 + ch * 3
                for ai, axis in enumerate(['x', 'y', 'z']):
                    var = tk.StringVar(value='—')
                    self.disp_vars[(ch, key, axis)] = var
                    tk.Label(tbl, textvariable=var,
                             bg=BG_SENSOR, fg='#111111',
                             font=('Courier', 8),
                             width=SUBCOL_W,
                             anchor='e', relief='flat', bd=0,
                             padx=2, pady=3
                             ).grid(row=row, column=base + ai,
                                    sticky='nsew', padx=1, pady=1)

        # Row 7: show sampling rate; Row 8: blank spacer (matches edit frame rows)
        self._hdr(tbl, 'Sample Rate', row=7, col=0)
        tk.Label(tbl, textvariable=self.sample_rate_var,
                 bg=BG_SENSOR, fg='#111111',
                 font=('Courier', 8),
                 anchor='center', relief='flat', bd=0,
                 padx=2, pady=3
                 ).grid(row=7, column=1, columnspan=NUM_CHANNELS * 3,
                        sticky='nsew', padx=1, pady=1)

        self._hdr(tbl, 'Avg Diff / Deriv', row=8, col=0)
        self.avg_diff_var  = tk.StringVar(value='— / —')
        tk.Label(tbl, textvariable=self.avg_diff_var,
                 bg=BG_SENSOR, fg='#111111',
                 font=('Courier', 8),
                 anchor='center', relief='flat', bd=0,
                 padx=2, pady=3
                 ).grid(row=8, column=1, columnspan=NUM_CHANNELS * 3,
                        sticky='nsew', padx=1, pady=1)

        tbl.columnconfigure(0, weight=0)
        for col in range(1, 1 + NUM_CHANNELS * 3):
            tbl.columnconfigure(col, weight=1)

    def _populate_edit_table(self, tbl):
        self._hdr(tbl, 'Positions', row=0, col=0, rowspan=2)   
        for ch in range(NUM_CHANNELS):
            self._hdr(tbl, f'Chn {ch}', row=0, col=1 + ch * 3, colspan=3)
            self._hdr(tbl, 'X Pos', row=1, col=1 + ch * 3)
            self._hdr(tbl, 'Y Pos', row=1, col=2 + ch * 3)
            self._hdr(tbl, 'Z Pos', row=1, col=3 + ch * 3)

        for si, (label, key) in enumerate(zip(SENSOR_LABELS, SENSOR_KEYS)):
            row = 2 + si
            self._hdr(tbl, label, row=row, col=0)
            for ch in range(NUM_CHANNELS):
                base = 1 + ch * 3
                for ai, axis in enumerate(['x', 'y', 'z']):
                    
                    if axis == 'z':
                        loaded_val = '0' # Force Z to always be 0
                    else:
                        config_key = f"pos_{ch}_{key}_{axis}"
                        loaded_val = self._saved_config.get(config_key, '0')
                    
                    var = tk.StringVar(value=loaded_val)
                    self.sensor_pos_vars[(ch, key, axis)] = var
                    
                    # Lock the Z column visually and functionally
                    state_mode = 'readonly' if axis == 'z' else 'normal'
                    bg_color   = BG_LOCKED  if axis == 'z' else BG_EDITABLE

                    e = tk.Entry(tbl, textvariable=var,
                                 width=SUBCOL_W,
                                 state=state_mode,
                                 readonlybackground=bg_color,
                                 bg=bg_color, fg='#111111',
                                 insertbackground='black',
                                 relief='flat', bd=0,
                                 font=('Courier', 9),
                                 justify='right')
                    e.grid(row=row, column=base + ai, sticky='nsew', padx=1, pady=1)
                    var.trace_add('write', lambda *_, c=ch: self._on_pos_changed(c))

        self._hdr(tbl, 'Rotation CW°', row=7, col=0)
        for ch in range(NUM_CHANNELS):
            base = 1 + ch * 3
            
            config_key = f"rot_{ch}"
            loaded_val = self._saved_config.get(config_key, '0')
            
            var  = tk.StringVar(value=loaded_val)
            self.rot_vars[ch] = var
            e = tk.Entry(tbl, textvariable=var,
                         width=SUBCOL_W * 3 + 2, # Spans across 3 subcolumns now
                         bg=BG_EDITABLE, fg='#111111',
                         insertbackground='black',
                         relief='flat', bd=0,
                         font=('Courier', 9),
                         justify='center')
            e.grid(row=7, column=base, columnspan=3,
                   sticky='nsew', padx=1, pady=1)

        # Plot Side Row (Row 8) — toggle which canvas each channel feeds
        self._hdr(tbl, 'Plot Side', row=8, col=0)
        for ch in range(NUM_CHANNELS):
            base = 1 + ch * 3
            saved_side = self._saved_config.get(f"plot_side_{ch}", 'left')
            var = tk.StringVar(value=saved_side)
            self.plot_side_vars[ch] = var

            btn = tk.Button(tbl, textvariable=var,
                            bg='#3d5a80' if saved_side == 'left' else '#6b3a20',
                            fg='white',
                            font=('Courier', 8, 'bold'),
                            relief='flat', bd=0,
                            cursor='hand2')
            btn.grid(row=8, column=base, columnspan=3,
                     sticky='nsew', padx=1, pady=1)

            def _make_toggle(b=btn, v=var, c=ch):
                def _toggle():
                    new = 'right' if v.get() == 'left' else 'left'
                    v.set(new)
                    b.config(bg='#3d5a80' if new == 'left' else '#6b3a20')
                return _toggle
            btn.config(command=_make_toggle())

        # Cutoff Row (Row 9) — left half: Noise Cutoff, right half: Shear Limit
        half_cols = (NUM_CHANNELS * 3) // 2

        self._hdr(tbl, 'Noise Cutoff', row=9, col=0)
        cutoff_val = self._saved_config.get("global_cutoff", '0.0')
        self.cutoff_var = tk.StringVar(value=cutoff_val)
        e = tk.Entry(tbl, textvariable=self.cutoff_var,
                     bg=BG_EDITABLE, fg='#111111',
                     insertbackground='black',
                     relief='flat', bd=0,
                     font=('Courier', 9, 'bold'),
                     justify='center')
        e.grid(row=9, column=1, columnspan=half_cols,
               sticky='nsew', padx=1, pady=1)

        # Shear Limit — same row, right half
        self._hdr(tbl, 'Shear Limit', row=9, col=1 + half_cols)
        shear_val = self._saved_config.get("shear_limit", '10.0')
        self.shear_limit_var.set(shear_val)
        e_shear = tk.Entry(tbl, textvariable=self.shear_limit_var,
                           bg='#1e2e1e', fg='#88ff88',
                           insertbackground='#88ff88',
                           relief='flat', bd=0,
                           font=('Courier', 9, 'bold'),
                           justify='center')
        remaining_cols = NUM_CHANNELS * 3 - half_cols - 1
        e_shear.grid(row=9, column=2 + half_cols, columnspan=remaining_cols,
                     sticky='nsew', padx=1, pady=1)

        tbl.columnconfigure(0, weight=0)
        for col in range(1, 1 + NUM_CHANNELS * 3):
            tbl.columnconfigure(col, weight=1)

    def _populate_zero_settings_table(self, tbl):
        """Table for auto-zeroing thresholds — mirrors the layout of the data table."""
        self._hdr(tbl, 'Zero Settings', row=0, col=0, rowspan=2)

        for ch in range(NUM_CHANNELS):
            self._hdr(tbl, f'Chn {ch}', row=0, col=1 + ch * 3, colspan=3)
            self._hdr(tbl, 'Cutoff', row=1, col=1 + ch * 3, colspan=3)

        for si, (label, key) in enumerate(zip(SENSOR_LABELS, SENSOR_KEYS)):
            row = 2 + si
            self._hdr(tbl, label, row=row, col=0)
            for ch in range(NUM_CHANNELS):
                base = 1 + ch * 3
                config_key = f"az_cutoff_{ch}_{key}"
                loaded_val = self._saved_config.get(config_key, '5.0')
                var = tk.StringVar(value=loaded_val)
                self.az_cutoff_vars[(ch, key)] = var
                e = tk.Entry(tbl, textvariable=var,
                             width=SUBCOL_W * 3 + 2,
                             bg=BG_EDITABLE, fg='#111111',
                             insertbackground='black',
                             relief='flat', bd=0,
                             font=('Courier', 9),
                             justify='center')
                e.grid(row=row, column=base, columnspan=3,
                       sticky='nsew', padx=1, pady=1)

        # Row 7: Derivative threshold (left half) + Set-All Cutoff (right half)
        self._hdr(tbl, 'Deriv. Threshold', row=7, col=0)

        half = (NUM_CHANNELS * 3) // 2   # 12 cols split into two halves of 12 each

        loaded_deriv = self._saved_config.get("az_deriv_threshold", '0.5')
        self.az_deriv_threshold_var.set(loaded_deriv)
        e_deriv = tk.Entry(tbl, textvariable=self.az_deriv_threshold_var,
                           bg=BG_EDITABLE, fg='#111111',
                           insertbackground='black',
                           relief='flat', bd=0,
                           font=('Courier', 9, 'bold'),
                           justify='center')
        e_deriv.grid(row=7, column=1, columnspan=half,
                     sticky='nsew', padx=1, pady=1)

        # "Set All Cutoffs" entry — typing a valid number propagates to every cutoff cell
        self._hdr(tbl, 'Set All Cutoffs', row=7, col=1 + half)
        self.az_set_all_var = tk.StringVar(value='')

        def _on_set_all_write(*_):
            raw = self.az_set_all_var.get()
            try:
                float(raw)          # only propagate when it parses as a number
            except ValueError:
                return
            for var in self.az_cutoff_vars.values():
                var.set(raw)

        self.az_set_all_var.trace_add('write', _on_set_all_write)

        # Place the entry spanning the remaining columns (leave 1 col for the header label above)
        remaining = NUM_CHANNELS * 3 - half - 1
        e_set_all = tk.Entry(tbl, textvariable=self.az_set_all_var,
                             bg='#e8f4e8', fg='#111111',
                             insertbackground='black',
                             relief='flat', bd=0,
                             font=('Courier', 9, 'bold'),
                             justify='center')
        e_set_all.grid(row=7, column=2 + half, columnspan=remaining,
                       sticky='nsew', padx=1, pady=1)

        # Row 8: Settling time (in the "blank" position)
        self._hdr(tbl, 'Settling Time (s)', row=8, col=0)
        loaded_settle = self._saved_config.get("az_settling_time", '1.0')
        self.az_settling_time_var.set(loaded_settle)
        e_settle = tk.Entry(tbl, textvariable=self.az_settling_time_var,
                            bg=BG_EDITABLE, fg='#111111',
                            insertbackground='black',
                            relief='flat', bd=0,
                            font=('Courier', 9, 'bold'),
                            justify='center')
        e_settle.grid(row=8, column=1, columnspan=NUM_CHANNELS * 3,
                      sticky='nsew', padx=1, pady=1)

        tbl.columnconfigure(0, weight=0)
        for col in range(1, 1 + NUM_CHANNELS * 3):
            tbl.columnconfigure(col, weight=1)

    def _build_canvas(self, parent):
        outer = tk.Frame(parent, bg=BG_ROOT)
        outer.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        # ── "Vector Field" header ─────────────────────────────────────────────
        self.bar_vector = tk.Frame(outer, bg=BG_ROOT)
        self.bar_vector.pack(fill=tk.X, pady=(0, 4))

        tk.Label(self.bar_vector, text='Vector Field',
                 bg=BG_ROOT, fg='#2b2d42',
                 font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=4)

        leg = tk.Frame(self.bar_vector, bg=BG_ROOT)
        leg.pack(side=tk.LEFT, padx=(20, 0))
        for ch in range(NUM_CHANNELS):
            tk.Label(leg, text=f'■ Ch{ch}',
                     bg=BG_ROOT, fg=CH_COLORS[ch],
                     font=('Courier', 8, 'bold')).pack(side=tk.LEFT, padx=3)

        # ── Dual canvas frame ─────────────────────────────────────────────────
        self.canvas_frame = tk.Frame(outer, bg=BG_ROOT)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left canvas
        left_wrap = tk.Frame(self.canvas_frame, bg=BG_ROOT)
        left_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        tk.Label(left_wrap, text='Left', bg=BG_ROOT, fg='#888888',
                 font=('Helvetica', 8, 'bold')).pack(anchor='n')
        self.canvas_left_frame = tk.Frame(left_wrap, bg=BG_ROOT)
        self.canvas_left_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas_left = tk.Canvas(self.canvas_left_frame, bg=BG_CANVAS,
                                     highlightthickness=1,
                                     highlightbackground=GRID_CLR)
        self.canvas_left.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Right canvas
        right_wrap = tk.Frame(self.canvas_frame, bg=BG_ROOT)
        right_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        tk.Label(right_wrap, text='Right', bg=BG_ROOT, fg='#888888',
                 font=('Helvetica', 8, 'bold')).pack(anchor='n')
        self.canvas_right_frame = tk.Frame(right_wrap, bg=BG_ROOT)
        self.canvas_right_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas_right = tk.Canvas(self.canvas_right_frame, bg=BG_CANVAS,
                                      highlightthickness=1,
                                      highlightbackground=GRID_CLR)
        self.canvas_right.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Keep self.canvas pointing to left for backward-compat with snapshot image sizing
        self.canvas = self.canvas_left

        # Snapshot overlay — top-right of the RIGHT canvas frame
        self.snapshot_overlay_var = tk.StringVar(value='')
        self._snapshot_overlay_lbl = tk.Label(
            self.canvas_right_frame,
            textvariable=self.snapshot_overlay_var,
            bg='#0d0d1e', fg='#00E5FF',
            font=('Courier', 8),
            justify='right',
            padx=6, pady=4,
            relief='flat'
        )
        self._snapshot_overlay_lbl.place(relx=1.0, rely=0.0, anchor='ne')

        # Scale/origin are shared (sensors positions determine layout)
        self.canvas_frame.bind('<Configure>', self._on_canvas_resize)
        self.canvas_left_frame.bind('<Configure>', self._on_canvas_resize)
        self.canvas_right_frame.bind('<Configure>', self._on_canvas_resize)

    def _on_canvas_resize(self, event):
        # Resize both canvases to be square, fitting in their respective frames
        for canvas, frame in [(self.canvas_left, self.canvas_left_frame),
                              (self.canvas_right, self.canvas_right_frame)]:
            w = frame.winfo_width()
            h = frame.winfo_height()
            size = min(w, h)
            if size > 10:
                canvas.config(width=size, height=size)
                canvas.update_idletasks()
        self._autoscale()
        self._draw_grid()
        self._draw_vectors()

    # ─── autoscale ────────────────────────────────────────────────────────────

    def _on_pos_changed(self, ch: int):
        if ch not in self.reader.active_channels:
            return
        self._autoscale()

    def _autoscale(self):
        # In shear mode: center at (0,0), scale so shear_limit fills exactly to the canvas edge
        if self.shear_mode:
            cw    = self.canvas_left.winfo_width()
            ch_px = self.canvas_left.winfo_height()
            if cw < 10 or ch_px < 10:
                return
            try:
                shear_limit = float(self.shear_limit_var.get())
            except ValueError:
                shear_limit = 10.0
            if shear_limit <= 0:
                shear_limit = 10.0
            # half the shorter dimension maps exactly to shear_limit world units
            half = min(cw, ch_px) / 2
            self._scale  = half / shear_limit
            self._origin = (0.0, 0.0)
            return

        active = self.reader.active_channels
        if not active: return

        pts = []
        for ch in active:
            for key in SENSOR_KEYS:
                try:
                    px = float(self.sensor_pos_vars[(ch, key, 'x')].get())
                    py = float(self.sensor_pos_vars[(ch, key, 'y')].get())
                    pts.append((px, py))
                except (ValueError, KeyError):
                    pass

        if not pts: return

        cw = self.canvas.winfo_width()
        ch_px = self.canvas.winfo_height()
        if cw < 10 or ch_px < 10: return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w_range = (x_max - x_min) or 1.0
        h_range = (y_max - y_min) or 1.0

        margin = 0.15
        sx = cw  * (1 - 2 * margin) / w_range
        sy = ch_px * (1 - 2 * margin) / h_range
        self._scale  = min(sx, sy)
        self._origin = ((x_min + x_max) / 2, (y_min + y_max) / 2)

    def _w2c(self, wx, wy, canvas=None):
        if canvas is None:
            canvas = self.canvas_left
        cw = canvas.winfo_width()  / 2
        ch = canvas.winfo_height() / 2
        px = cw + (wx - self._origin[0]) * self._scale
        py = ch - (wy - self._origin[1]) * self._scale
        return px, py

    # ─── drawing ──────────────────────────────────────────────────────────────

    def _draw_grid(self):
        for canvas in (self.canvas_left, self.canvas_right):
            canvas.delete('grid')
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            ox, oy = self._w2c(0, 0, canvas)
            if 0 <= ox <= w:
                canvas.create_line(ox, 0, ox, h, fill='#2a2a4a', width=1, tags='grid')
            if 0 <= oy <= h:
                canvas.create_line(0, oy, w, oy, fill='#2a2a4a', width=1, tags='grid')

    def _compute_net_pos_for_sensors(self, ch_keys, cutoff):
        """Compute weighted net position for a list of (ch, key) pairs.
        Returns (wx, wy) or None."""
        total_w = 0.0
        sum_x   = 0.0
        sum_y   = 0.0
        for (ch, key) in ch_keys:
            k = (ch, key)
            src = self.reader.raw_values if self.raw_mode else self.reader.diff_values
            vec = src.get(k)
            if vec is None:
                continue
            dx, dy, dz = float(vec[0]), float(vec[1]), float(vec[2])
            if abs(dx) <= cutoff: dx = 0.0
            if abs(dy) <= cutoff: dy = 0.0
            if abs(dz) <= cutoff: dz = 0.0
            mag = float(np.linalg.norm([dx, dy, dz]))
            if mag < 1e-6:
                continue
            try:
                wx = float(self.sensor_pos_vars[(ch, key, 'x')].get())
                wy = float(self.sensor_pos_vars[(ch, key, 'y')].get())
            except (ValueError, KeyError):
                continue
            total_w += mag
            sum_x   += mag * wx
            sum_y   += mag * wy
        if total_w > 1e-6:
            return (sum_x / total_w, sum_y / total_w)
        return None

    def _sensor_centroid(self, ch_keys):
        """Unweighted geometric centroid of a set of (ch, key) sensor positions."""
        xs, ys = [], []
        for (ch, key) in ch_keys:
            try:
                xs.append(float(self.sensor_pos_vars[(ch, key, 'x')].get()))
                ys.append(float(self.sensor_pos_vars[(ch, key, 'y')].get()))
            except (ValueError, KeyError):
                pass
        if xs:
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return (0.0, 0.0)

    def _compute_raw_com_for_sensors(self, ch_keys):
        """Raw-magnitude-weighted CoM using absolute world positions.
        Identical formula to the live shear accumulation loop so that
        the saved baseline and the live CoM are in the same frame."""
        total_w = 0.0
        sum_x   = 0.0
        sum_y   = 0.0
        for (ch, key) in ch_keys:
            k = (ch, key)
            vec = self.reader.raw_values.get(k)
            if vec is None:
                continue
            w = float(np.linalg.norm(vec))
            try:
                wx = float(self.sensor_pos_vars[(ch, key, 'x')].get())
                wy = float(self.sensor_pos_vars[(ch, key, 'y')].get())
            except (ValueError, KeyError):
                continue
            total_w += w
            sum_x   += w * wx
            sum_y   += w * wy
        if total_w > 1e-5:
            return (sum_x / total_w, sum_y / total_w)
        return (0.0, 0.0)

    def _draw_shear_vector(self, canvas, net_pos, shear_limit, label_prefix=''):
        """Draw a displacement/shear vector from 0,0 to net_pos, saturated to shear_limit."""
        if net_pos is None:
            return None, None
        nx, ny = net_pos
        mag = np.hypot(nx, ny)
        angle_deg = np.degrees(np.arctan2(ny, nx))
        # Saturate
        if mag > shear_limit and shear_limit > 0:
            scale = shear_limit / mag
            sx, sy = nx * scale, ny * scale
        else:
            sx, sy = nx, ny

        # Draw from canvas center (origin) to shear point
        ox, oy = self._w2c(0, 0, canvas)
        ex, ey = self._w2c(sx, sy, canvas)

        t_color = '#FF4444'  # Red for shear vector
        saturated = (mag > shear_limit and shear_limit > 0)
        line_color = '#FF8800' if saturated else t_color

        # Draw the shear vector arrow
        canvas.create_line(ox, oy, ex, ey,
                           fill=line_color, width=3,
                           arrow=tk.LAST,
                           arrowshape=(10, 12, 5),
                           tags='vector')
        # Draw origin crosshair
        cr = 5
        canvas.create_oval(ox - cr, oy - cr, ox + cr, oy + cr,
                           outline='#888888', width=1, tags='vector')

        # Label with angle and magnitude (always of the actual unsaturated vector)
        label_txt = f'{label_prefix}∠{angle_deg:.1f}°  |{mag:.2f}|'
        if saturated:
            label_txt += ' [SAT]'
        canvas.create_text(ex + 8, ey - 8,
                           text=label_txt,
                           fill='#00E5FF',
                           font=('Helvetica', 8, 'bold'),
                           tags='vector')
        return angle_deg, mag

    def _draw_vectors_on_canvas(self, canvas, active, cutoff, side_filter=None):
        """Draw sensor vectors and net pos / shear on a single canvas.
        side_filter: None = all sensors, 'left' or 'right' = only those sides.
        Returns (raw_net_pos, angle_deg, mag_val) where raw_net_pos is the
        weighted centre in world coords (before shear-origin subtraction)."""
        shear_mode = self.shear_mode
        try:
            shear_limit = float(self.shear_limit_var.get())
        except ValueError:
            shear_limit = 10.0

        global_sum_norm   = 0.0
        global_sum_norm_x = 0.0
        global_sum_norm_y = 0.0

        for ch in range(NUM_CHANNELS):
            if ch not in active:
                continue
            # Check side assignment
            side_var = self.plot_side_vars.get(ch)
            ch_side = side_var.get() if side_var else 'left'
            if side_filter is not None and ch_side != side_filter:
                continue

            try:
                rot_deg = float(self.rot_vars[ch].get())
            except (ValueError, KeyError):
                rot_deg = 0.0
            rot_rad = -np.deg2rad(rot_deg)
            color = CH_COLORS[ch]

            for key in SENSOR_KEYS:
                try:
                    wx = float(self.sensor_pos_vars[(ch, key, 'x')].get())
                    wy = float(self.sensor_pos_vars[(ch, key, 'y')].get())
                except (ValueError, KeyError):
                    continue

                cx, cy = self._w2c(wx, wy, canvas)
                r = 3
                if not shear_mode:
                    canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                       fill=color, outline='white',
                                       width=1, tags='vector')
                if key == 'middle' and not shear_mode:
                    canvas.create_text(cx + r + 5, cy - r - 5,
                                       text=f'Ch{ch}', fill=color,
                                       font=('Helvetica', 7, 'bold'),
                                       tags='vector')

                k = (ch, key)
                if k not in self.reader.diff_values:
                    continue
                src = self.reader.raw_values if self.raw_mode else self.reader.diff_values
                vec = src.get(k)
                if vec is None:
                    continue
                dx, dy, dz = float(vec[0]), float(vec[1]), float(vec[2])
                if abs(dx) <= cutoff: dx = 0.0
                if abs(dy) <= cutoff: dy = 0.0
                if abs(dz) <= cutoff: dz = 0.0

                mag_3d = float(np.linalg.norm([dx, dy, dz]))
                # In shear mode always weight by raw magnitude so the CoM
                # reference is independent of zeroing and raw/diff toggle.
                if shear_mode:
                    raw_vec = self.reader.raw_values.get(k)
                    w = float(np.linalg.norm(raw_vec)) if raw_vec is not None else 0.0
                else:
                    w = mag_3d
                global_sum_norm   += w
                global_sum_norm_x += w * wx
                global_sum_norm_y += w * wy

                if not shear_mode:
                    mag_2d = np.hypot(dx, dy)
                    if mag_2d < 1e-6:
                        canvas.create_line(cx - 4, cy, cx + 4, cy,
                                           fill=color, width=1, dash=(2, 3), tags='vector')
                        canvas.create_line(cx, cy - 4, cx, cy + 4,
                                           fill=color, width=1, dash=(2, 3), tags='vector')
                        continue
                    ux, uy = dx / mag_2d, dy / mag_2d
                    ux2 =  ux * np.cos(rot_rad) + uy * np.sin(rot_rad)
                    uy2 = -ux * np.sin(rot_rad) + uy * np.cos(rot_rad)
                    tx =  ux2 * ARROW_HALF
                    ty = -uy2 * ARROW_HALF
                    x1, y1 = cx - tx, cy - ty
                    x2, y2 = cx + tx, cy + ty
                    canvas.create_line(x1, y1, x2, y2,
                                       fill=color, width=2,
                                       arrow=tk.LAST,
                                       arrowshape=(8, 10, 4),
                                       tags='vector')
                    marker = SENSOR_MARKERS.get(key, '·')
                    canvas.create_text(x2 + 7, y2 - 7,
                                       text=marker, fill=color,
                                       font=('Helvetica', 7),
                                       tags='vector')

        # Net pos / shear indicator
        net_pos = None
        angle_deg, mag_val = None, None
        if global_sum_norm > 1e-6:
            center_wx = global_sum_norm_x / global_sum_norm
            center_wy = global_sum_norm_y / global_sum_norm
            net_pos = (center_wx, center_wy)

            if shear_mode:
                # ── Lerp-smooth the raw CoM, then compute delta from baseline ──
                # Select the right smoothed-state and baseline slots
                if side_filter == 'left':
                    smooth = self._shear_com_smooth_left
                    baseline = self._shear_com_baseline_left
                elif side_filter == 'right':
                    smooth = self._shear_com_smooth_right
                    baseline = self._shear_com_baseline_right
                else:
                    smooth = self._shear_com_smooth
                    baseline = self._shear_com_baseline

                # Update the lerp-smoothed CoM
                alpha = self.shear_com_lerp_alpha
                if smooth is None:
                    smooth = (center_wx, center_wy)
                else:
                    smooth = (
                        alpha * center_wx + (1 - alpha) * smooth[0],
                        alpha * center_wy + (1 - alpha) * smooth[1],
                    )

                # Write back to the correct slot
                if side_filter == 'left':
                    self._shear_com_smooth_left = smooth
                elif side_filter == 'right':
                    self._shear_com_smooth_right = smooth
                else:
                    self._shear_com_smooth = smooth

                # Compute displacement vector from the frozen baseline
                if baseline is not None:
                    disp_pos = (smooth[0] - baseline[0], smooth[1] - baseline[1])
                else:
                    disp_pos = smooth   # no baseline yet — show raw CoM

                # Store displacement so status bar / snapshot / CSV use the same value
                if side_filter == 'left':
                    self._shear_disp_left = disp_pos
                elif side_filter == 'right':
                    self._shear_disp_right = disp_pos
                else:
                    self._shear_disp = disp_pos
                # Saturate to shear_limit and draw
                angle_deg, mag_val = self._draw_shear_vector(
                    canvas, disp_pos, shear_limit,
                    label_prefix=f'{side_filter.capitalize()} ' if side_filter else '')

                # Keep net_pos as the smoothed CoM (used for shear origin tracking)
                net_pos = smooth
            else:
                ccx, ccy = self._w2c(center_wx, center_wy, canvas)
                cr = 8
                t_color = '#00E5FF'
                canvas.create_oval(ccx - cr, ccy - cr, ccx + cr, ccy + cr,
                                   outline=t_color, width=2, tags='vector')
                canvas.create_line(ccx - cr - 4, ccy, ccx + cr + 4, ccy,
                                   fill=t_color, width=2, tags='vector')
                canvas.create_line(ccx, ccy - cr - 4, ccx, ccy + cr + 4,
                                   fill=t_color, width=2, tags='vector')
                canvas.create_text(ccx + cr + 6, ccy - cr - 6,
                                   text='Net Pos', fill=t_color,
                                   font=('Helvetica', 8, 'bold'),
                                   tags='vector')
        else:
            # No signal — decay smoothed CoM toward baseline (or clear it)
            if shear_mode:
                if side_filter == 'left':
                    self._shear_com_smooth_left = None
                elif side_filter == 'right':
                    self._shear_com_smooth_right = None
                else:
                    self._shear_com_smooth = None

        return net_pos, angle_deg, mag_val

    def _draw_vectors(self):
        self.canvas_left.delete('vector')
        self.canvas_right.delete('vector')
        self._net_pos       = None
        self._net_pos_left  = None
        self._net_pos_right = None
        self._shear_disp       = None
        self._shear_disp_left  = None
        self._shear_disp_right = None
        active = self.reader.active_channels

        try:
            cutoff = float(self.cutoff_var.get())
        except ValueError:
            cutoff = 0.0

        # Draw on left canvas (left-side sensors)
        left_result = self._draw_vectors_on_canvas(
            self.canvas_left, active, cutoff, side_filter='left')
        self._net_pos_left = left_result[0]

        # Draw on right canvas (right-side sensors)
        right_result = self._draw_vectors_on_canvas(
            self.canvas_right, active, cutoff, side_filter='right')
        self._net_pos_right = right_result[0]

        # Global net pos for status bar / CSV — computed directly, no drawing,
        # so it never touches or double-steps the smoothed shear CoM state.
        self._net_pos = self._compute_net_pos_for_sensors(
            [(ch, key) for ch in active for key in SENSOR_KEYS], cutoff)

        # ── Shear CoM state is maintained inside _draw_vectors_on_canvas ────────
        # Reset legacy origin fields when not in shear mode
        if not self.shear_mode:
            self._shear_origin = None
            self._shear_origin_left = None
            self._shear_origin_right = None
            self._shear_had_reading = False
            self._shear_had_reading_left = False
            self._shear_had_reading_right = False

    # ─── Right Panel: Controls ────────────────────────────────────────────────

    def _build_controls(self, parent):
        top_frame = tk.Frame(parent, bg=BG_RIGHT)
        top_frame.pack(fill=tk.X)

        tk.Label(top_frame, text='Controls',
                 bg=BG_RIGHT, fg='#2b2d42',
                 font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=(0, 14))

        tk.Button(top_frame, text='Zero All Sensors',
                  command=self._zero_all,
                  bg=ACCENT, fg='white',
                  font=('Helvetica', 10, 'bold'),
                  relief='flat', padx=10, pady=8,
                  cursor='hand2').pack(fill=tk.X, pady=4)

        # Auto-zeroing toggle
        self.btn_auto_zero = tk.Button(top_frame, text='Auto Zero: OFF',
                                       command=self._toggle_auto_zero,
                                       bg='#4a4a6a', fg='white',
                                       font=('Helvetica', 10, 'bold'),
                                       relief='flat', padx=10, pady=8,
                                       cursor='hand2')
        self.btn_auto_zero.pack(fill=tk.X, pady=4)

        # Auto-zeroing settings
        self.btn_zero_settings = tk.Button(top_frame, text='Zero Settings',
                                           command=self._toggle_zero_settings_mode,
                                           bg='#3d5a80', fg='white',
                                           font=('Helvetica', 10, 'bold'),
                                           relief='flat', padx=10, pady=8,
                                           cursor='hand2')
        self.btn_zero_settings.pack(fill=tk.X, pady=4)

        self.btn_raw_mode = tk.Button(top_frame, text='Show Raw Readings',
                                      command=self._toggle_raw_mode,
                                      bg='#7a5c1e', fg='white',
                                      font=('Helvetica', 10, 'bold'),
                                      relief='flat', padx=10, pady=8,
                                      cursor='hand2')
        self.btn_raw_mode.pack(fill=tk.X, pady=4)

        self.btn_toggle_mode = tk.Button(top_frame, text='Edit Positions',
                                         command=self._toggle_mode,
                                         bg=BTN_ALT, fg='white',
                                         font=('Helvetica', 10, 'bold'),
                                         relief='flat', padx=10, pady=8,
                                         cursor='hand2')
        self.btn_toggle_mode.pack(fill=tk.X, pady=4)

        self.btn_toggle_viz = tk.Button(top_frame, text='Shear Mode',
                                        command=self._toggle_shear_mode,
                                        bg='#4a6741', fg='white',
                                        font=('Helvetica', 10, 'bold'),
                                        relief='flat', padx=10, pady=8,
                                        cursor='hand2')
        self.btn_toggle_viz.pack(fill=tk.X, pady=4)

        self.btn_record = tk.Button(top_frame, text='⏺  Start Recording',
                                    command=self._toggle_recording,
                                    bg='#6b2020', fg='white',
                                    font=('Helvetica', 10, 'bold'),
                                    relief='flat', padx=10, pady=8,
                                    cursor='hand2')
        self.btn_record.pack(fill=tk.X, pady=4)

        # Snapshot row: left=Snapshot, right=Save CSV toggle — same total width as other buttons
        snap_row = tk.Frame(top_frame, bg=BG_RIGHT)
        snap_row.pack(fill=tk.X, pady=4)
        snap_row.columnconfigure(0, weight=3)
        snap_row.columnconfigure(1, weight=1)

        self.btn_snapshot = tk.Button(snap_row, text='📸  Snapshot',
                                      command=self._take_snapshot,
                                      bg='#4a3b6b', fg='white',
                                      font=('Helvetica', 10, 'bold'),
                                      relief='flat', padx=10, pady=8,
                                      cursor='hand2')
        self.btn_snapshot.grid(row=0, column=0, sticky='nsew', padx=(0, 2))

        self.btn_snap_csv = tk.Button(snap_row, text='CSV: OFF',
                                      command=self._toggle_snapshot_csv,
                                      bg='#3a3a3a', fg='white',
                                      font=('Helvetica', 9, 'bold'),
                                      relief='flat', padx=4, pady=8,
                                      cursor='hand2')
        self.btn_snap_csv.grid(row=0, column=1, sticky='nsew')

        self.status_var = tk.StringVar(value='Waiting for data…')
        tk.Label(top_frame, textvariable=self.status_var,
                 bg=BG_RIGHT, fg='#444444',
                 font=('Helvetica', 9),
                 wraplength=200, justify='left').pack(anchor='w', pady=(14, 0))

    def _toggle_recording(self):
        import csv, datetime
        if not self.is_recording:
            # ── Start recording ───────────────────────────────────────────────
            self.is_recording  = True
            self._record_start = time.monotonic()
            self._record_writers.clear()
            self._record_files.clear()

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            active = sorted(self.reader.active_channels)
            if not active:
                # Still open files for all channels — data will just be empty rows
                active = list(range(NUM_CHANNELS))

            header = ['time_elapsed_s']
            for key in SENSOR_KEYS:
                for axis in ['x', 'y', 'z']:
                    header.append(f'{key}_raw_{axis}')
            for key in SENSOR_KEYS:
                for axis in ['x', 'y', 'z']:
                    header.append(f'{key}_diff_{axis}')
            for key in SENSOR_KEYS:
                header.append(f'{key}_deriv_norm')
            if self.shear_mode:
                header += ['shear_angle_deg', 'shear_magnitude',
                           'shear_left_angle_deg', 'shear_left_magnitude',
                           'shear_right_angle_deg', 'shear_right_magnitude']
            else:
                header += ['net_pos_x', 'net_pos_y', 'net_pos_angle_deg', 'net_pos_magnitude',
                           'net_pos_left_x', 'net_pos_left_y', 'net_pos_left_angle_deg', 'net_pos_left_magnitude',
                           'net_pos_right_x', 'net_pos_right_y', 'net_pos_right_angle_deg', 'net_pos_right_magnitude']
            self._record_mode_snapshot = 'shear' if self.shear_mode else 'netpos'

            for ch in active:
                fname = f'recording_ch{ch}_{timestamp}.csv'
                fh = open(fname, 'w', newline='')
                writer = csv.writer(fh)
                writer.writerow(header)
                self._record_files[ch]   = fh
                self._record_writers[ch] = writer

            self.btn_record.config(text='⏹  Stop Recording', bg='#b71c1c')
            self.status_var.set(f'Recording to {len(active)} CSV file(s)…')
        else:
            # ── Stop recording ────────────────────────────────────────────────
            self.is_recording = False
            for fh in self._record_files.values():
                fh.close()
            self._record_files.clear()
            self._record_writers.clear()
            self.btn_record.config(text='⏺  Start Recording', bg='#6b2020')
            self.status_var.set('Recording stopped.')

    def _write_csv_rows(self):
        """Called every poll tick while recording is active."""
        elapsed = time.monotonic() - self._record_start

        for ch, writer in self._record_writers.items():
            row = [f'{elapsed:.4f}']

            for key in SENSOR_KEYS:
                rv = self.reader.raw_values.get((ch, key), [float('nan')] * 3)
                for v in rv:
                    row.append(f'{float(v):.4f}')

            for key in SENSOR_KEYS:
                dv = self.reader.diff_values.get((ch, key), [float('nan')] * 3)
                for v in dv:
                    row.append(f'{float(v):.4f}')

            for key in SENSOR_KEYS:
                k = (ch, key)
                dv  = self.reader.diff_values.get(k)
                prv = self._az_prev_diff.get(k)
                if dv is not None and prv is not None:
                    deriv = float(np.linalg.norm(dv - prv))
                else:
                    deriv = float('nan')
                row.append(f'{deriv:.4f}')

            if self._record_mode_snapshot == 'shear':
                try:
                    shear_limit = float(self.shear_limit_var.get())
                except ValueError:
                    shear_limit = 10.0
                def _shear_vals(disp):
                    if disp is not None:
                        mag = float(np.hypot(disp[0], disp[1]))
                        ang = float(np.degrees(np.arctan2(disp[1], disp[0])))
                        return f'{ang:.4f}', f'{mag:.4f}'
                    return 'nan', 'nan'
                ga, gm = _shear_vals(self._shear_disp)
                la, lm = _shear_vals(self._shear_disp_left)
                ra, rm = _shear_vals(self._shear_disp_right)
                row += [ga, gm, la, lm, ra, rm]
            else:
                def _netpos_vals(pos):
                    if pos is not None:
                        x, y = pos
                        mag = float(np.hypot(x, y))
                        ang = float(np.degrees(np.arctan2(y, x)))
                        return f'{x:.4f}', f'{y:.4f}', f'{ang:.4f}', f'{mag:.4f}'
                    return 'nan', 'nan', 'nan', 'nan'
                gx, gy, ga, gm = _netpos_vals(self._net_pos)
                lx, ly, la, lm = _netpos_vals(self._net_pos_left)
                rx, ry, ra, rm = _netpos_vals(self._net_pos_right)
                row += [gx, gy, ga, gm, lx, ly, la, lm, rx, ry, ra, rm]

            writer.writerow(row)

    def _toggle_snapshot_csv(self):
        self.snapshot_save_csv = not self.snapshot_save_csv
        if self.snapshot_save_csv:
            self.btn_snap_csv.config(text='CSV: ON', bg='#1b6b3a')
        else:
            self.btn_snap_csv.config(text='CSV: OFF', bg='#3a3a3a')

    def _take_snapshot(self):
        """Compute Net Pos / Shear from current filtered diff data."""
        import csv, datetime

        active = sorted(self.reader.active_channels)
        if not active:
            self.status_var.set('Snapshot: no active channels.')
            return

        try:
            cutoff = float(self.cutoff_var.get())
        except ValueError:
            cutoff = 0.0
        try:
            shear_limit = float(self.shear_limit_var.get())
        except ValueError:
            shear_limit = 10.0

        # ── 1. Collect all (ch, key) pairs split by side ───────────────────────
        all_pairs   = [(ch, key) for ch in active for key in SENSOR_KEYS]
        left_pairs  = [(ch, key) for ch in active for key in SENSOR_KEYS
                       if self.plot_side_vars.get(ch, tk.StringVar(value='left')).get() == 'left']
        right_pairs = [(ch, key) for ch in active for key in SENSOR_KEYS
                       if self.plot_side_vars.get(ch, tk.StringVar(value='left')).get() == 'right']

        net_pos       = self._compute_net_pos_for_sensors(all_pairs,   cutoff)
        net_pos_left  = self._compute_net_pos_for_sensors(left_pairs,  cutoff)
        net_pos_right = self._compute_net_pos_for_sensors(right_pairs, cutoff)

        # ── 2. Build display text ──────────────────────────────────────────────
        timestamp  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mode_label = 'Shear' if self.shear_mode else 'Net Pos'
        lines = [f'Snapshot  {timestamp}', f'Mode: {mode_label}']

        def _pos_line(label, pos, disp=None):
            if self.shear_mode:
                if disp is not None:
                    mag = float(np.hypot(disp[0], disp[1]))
                    ang = float(np.degrees(np.arctan2(disp[1], disp[0])))
                    return f'{label}  ∠{ang:+.2f}°  |{mag:.4f}|'
                return f'{label}  —'
            else:
                if pos:
                    return f'{label}  X={pos[0]:+.4f}  Y={pos[1]:+.4f}'
                return f'{label}  —'
        lines.append(_pos_line('Global', net_pos, self._shear_disp))
        lines.append(_pos_line('Left  ', net_pos_left, self._shear_disp_left))
        lines.append(_pos_line('Right ', net_pos_right, self._shear_disp_right))

        overlay_text = '\n'.join(lines)
        self.snapshot_overlay_var.set(overlay_text)
        print(overlay_text)

        # ── 3. Optionally save CSV + screenshot ───────────────────────────────
        snap_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.snapshot_save_csv:
            fname = f'snapshot_{snap_ts}.csv'
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                if self.shear_mode:
                    writer.writerow(['label', 'angle_deg', 'magnitude', 'shear_limit'])
                    for lbl, pos in [('global', self._shear_disp), ('left', self._shear_disp_left), ('right', self._shear_disp_right)]:
                        if pos is not None:
                            mag = float(np.hypot(pos[0], pos[1]))
                            ang = float(np.degrees(np.arctan2(pos[1], pos[0])))
                            writer.writerow([lbl, f'{ang:.6f}', f'{mag:.6f}', f'{shear_limit:.4f}'])
                        else:
                            writer.writerow([lbl, 'nan', 'nan', f'{shear_limit:.4f}'])
                else:
                    writer.writerow(['label', 'x', 'y'])
                    for lbl, pos in [('global', net_pos), ('left', net_pos_left), ('right', net_pos_right)]:
                        if pos:
                            writer.writerow([lbl, f'{pos[0]:.6f}', f'{pos[1]:.6f}'])
                        else:
                            writer.writerow([lbl, 'nan', 'nan'])

            # ── Render image ───────────────────────────────────────────────────
            try:
                from PIL import Image, ImageDraw
                IMG_SIZE = 800
                iw = ih = IMG_SIZE
                img  = Image.new('RGB', (iw, ih), (26, 26, 46))
                draw = ImageDraw.Draw(img)

                def w2i(wx, wy):
                    px = iw / 2 + (wx - self._origin[0]) * self._scale
                    py = ih / 2 - (wy - self._origin[1]) * self._scale
                    return px, py

                def _hex_to_rgb(h):
                    h = h.lstrip('#')
                    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

                ox, oy = w2i(0, 0)
                if 0 <= ox <= iw:
                    draw.line([(ox, 0), (ox, ih)], fill=(42, 42, 74), width=1)
                if 0 <= oy <= ih:
                    draw.line([(0, oy), (iw, oy)], fill=(42, 42, 74), width=1)

                # Draw sensor dots and vectors
                for ch_id in active:
                    color_rgb = _hex_to_rgb(CH_COLORS[ch_id])
                    try: rot_deg = float(self.rot_vars[ch_id].get())
                    except: rot_deg = 0.0
                    rot_rad = -np.deg2rad(rot_deg)
                    for key in SENSOR_KEYS:
                        try:
                            wx2 = float(self.sensor_pos_vars[(ch_id, key, 'x')].get())
                            wy2 = float(self.sensor_pos_vars[(ch_id, key, 'y')].get())
                        except (ValueError, KeyError):
                            continue
                        ipx, ipy = w2i(wx2, wy2)
                        r_dot = 4
                        draw.ellipse([ipx-r_dot, ipy-r_dot, ipx+r_dot, ipy+r_dot],
                                     fill=color_rgb, outline=(255,255,255), width=1)
                        if not self.shear_mode:
                            k = (ch_id, key)
                            vec = self.reader.diff_values.get(k)
                            if vec is not None:
                                dx2, dy2, dz2 = float(vec[0]), float(vec[1]), float(vec[2])
                                if abs(dx2) <= cutoff: dx2 = 0.0
                                if abs(dy2) <= cutoff: dy2 = 0.0
                                mag_2d = np.hypot(dx2, dy2)
                                if mag_2d > 1e-6:
                                    ux, uy = dx2 / mag_2d, dy2 / mag_2d
                                    ux2r = ux * np.cos(rot_rad) + uy * np.sin(rot_rad)
                                    uy2r = -ux * np.sin(rot_rad) + uy * np.cos(rot_rad)
                                    tx = ux2r * ARROW_HALF
                                    ty = -uy2r * ARROW_HALF
                                    draw.line([(ipx-tx, ipy-ty), (ipx+tx, ipy+ty)],
                                              fill=color_rgb, width=2)

                tc = (0, 229, 255)
                if self.shear_mode:
                    for pos_label, pos in [('Global', self._shear_disp), ('Left', self._shear_disp_left), ('Right', self._shear_disp_right)]:
                        if pos is not None:
                            nx, ny = pos
                            mag = float(np.hypot(nx, ny))
                            ang = float(np.degrees(np.arctan2(ny, nx)))
                            if mag > shear_limit > 0:
                                sc = shear_limit / mag
                                sx, sy = nx * sc, ny * sc
                            else:
                                sx, sy = nx, ny
                            ox2, oy2 = w2i(0, 0)
                            ex, ey   = w2i(sx, sy)
                            lc = (255, 136, 0) if (mag > shear_limit > 0) else (255, 68, 68)
                            draw.line([(ox2, oy2), (ex, ey)], fill=lc, width=3)
                            draw.text((ex+8, ey-8),
                                      f'{pos_label} ∠{ang:.1f}° |{mag:.2f}|',
                                      fill=tc)
                else:
                    if net_pos:
                        ipx, ipy = w2i(net_pos[0], net_pos[1])
                        cr2 = 10
                        draw.ellipse([ipx-cr2, ipy-cr2, ipx+cr2, ipy+cr2], outline=tc, width=2)
                        draw.line([(ipx-cr2-5, ipy), (ipx+cr2+5, ipy)], fill=tc, width=2)
                        draw.line([(ipx, ipy-cr2-5), (ipx, ipy+cr2+5)], fill=tc, width=2)
                        draw.text((ipx+cr2+7, ipy-cr2-7),
                                  f'NetPos ({net_pos[0]:.2f},{net_pos[1]:.2f})', fill=tc)

                _stamp = mode_label
                for _dx2, _dy2 in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
                    draw.text((8+_dx2, 8+_dy2), _stamp, fill=(0, 0, 0))
                draw.text((8, 8), _stamp, fill=(255, 255, 255))

                img_fname = f'snapshot_{snap_ts}_{mode_label.replace(" ", "_")}.png'
                img.save(img_fname)
                self.status_var.set(f'Snapshot saved → {fname} + {img_fname}')
            except Exception as e:
                import traceback; traceback.print_exc()
                self.status_var.set(f'Snapshot saved → {fname}  (image failed: {e})')
        else:
            if self.shear_mode and self._shear_disp is not None:
                mag = float(np.hypot(self._shear_disp[0], self._shear_disp[1]))
                ang = float(np.degrees(np.arctan2(self._shear_disp[1], self._shear_disp[0])))
                self.status_var.set(f'Snapshot | Shear ∠{ang:.2f}°  |{mag:.4f}|')
            else:
                self.status_var.set(
                    f'Snapshot | NetPos: {"({:.2f},{:.2f})".format(*net_pos) if net_pos else "—"}'
                )

    def _zero_all(self):
        # ── Capture shear CoM baseline BEFORE zeroing (raw values are live) ──
        if self.shear_mode:
            active = sorted(self.reader.active_channels)
            all_pairs   = [(ch, key) for ch in active for key in SENSOR_KEYS]
            left_pairs  = [(ch, key) for ch in active for key in SENSOR_KEYS
                           if self.plot_side_vars.get(ch, tk.StringVar(value='left')).get() == 'left']
            right_pairs = [(ch, key) for ch in active for key in SENSOR_KEYS
                           if self.plot_side_vars.get(ch, tk.StringVar(value='left')).get() == 'right']

            bx, by = self._compute_raw_com_for_sensors(all_pairs)
            self._shear_com_baseline       = (bx, by)
            self._shear_com_smooth         = (bx, by)   # reset smoothing to baseline

            lx, ly = self._compute_raw_com_for_sensors(left_pairs)
            self._shear_com_baseline_left  = (lx, ly)
            self._shear_com_smooth_left    = (lx, ly)

            rx, ry = self._compute_raw_com_for_sensors(right_pairs)
            self._shear_com_baseline_right = (rx, ry)
            self._shear_com_smooth_right   = (rx, ry)

        zeroed = self.reader.zero_sensors()
        active = sorted(self.reader.active_channels)
        if active:
            self.status_var.set(f'Zeroed. Active: {active}')
        else:
            self.status_var.set('Zeroed. No active channels detected yet.')

    def _toggle_raw_mode(self):
        self.raw_mode = not self.raw_mode
        if self.raw_mode:
            self.btn_raw_mode.config(text='Show Diff Readings', bg='#3b5e7a')
            # Update column headers in the data table to "Raw X/Y/Z"
            for lbl in self._data_hdr_labels:
                lbl.config(text=lbl._raw_text)
            self._hdr_title_label.config(text='Live Data (Raw)')
        else:
            self.btn_raw_mode.config(text='Show Raw Readings', bg='#7a5c1e')
            for lbl in self._data_hdr_labels:
                lbl.config(text=lbl._diff_text)
            self._hdr_title_label.config(text='Live Data')

    def _toggle_auto_zero(self):
        # Cycle: off -> individual -> average -> off
        if self.auto_zero_mode == 'off':
            self.auto_zero_mode = 'individual'
            self.btn_auto_zero.config(text='Auto Zero: Individual', bg='#1b6b3a')
            self._az_prev_diff.clear()
            self._az_deriv.clear()
            self._az_settle_start.clear()
        elif self.auto_zero_mode == 'individual':
            self.auto_zero_mode = 'average'
            self.btn_auto_zero.config(text='Auto Zero: Average', bg='#1b4d6b')
            self._az_prev_diff.clear()
            self._az_deriv.clear()
            self._az_settle_start.clear()
        else:
            self.auto_zero_mode = 'off'
            self.btn_auto_zero.config(text='Auto Zero: OFF', bg='#4a4a6a')

    def _toggle_zero_settings_mode(self):
        self.is_zero_settings_mode = not self.is_zero_settings_mode
        if self.is_zero_settings_mode:
            # Leave edit mode first if active
            self.is_edit_mode = False
            self.frame_zero_settings.tkraise()
            self.btn_zero_settings.config(text='View Live Data', bg='#1d3557')
            # btn_toggle_mode now goes back to zero settings, not live data
            self.btn_toggle_mode.config(text='Edit Positions', bg=BTN_ALT)
        else:
            self.frame_data.tkraise()
            self.btn_zero_settings.config(text='Zero Settings', bg='#3d5a80')
            self.btn_toggle_mode.config(text='Edit Positions', bg=BTN_ALT)

    def _toggle_mode(self):
        self.is_edit_mode = not self.is_edit_mode
        if self.is_edit_mode:
            self.frame_edit.tkraise()
            self.btn_toggle_mode.config(text='View Live Data', bg=BTN_ALT_ACT)
            # Track what we came from so the back-button label is correct
            if self.is_zero_settings_mode:
                self.btn_toggle_mode.config(text='Zero Settings', bg=BTN_ALT_ACT)
        else:
            if self.is_zero_settings_mode:
                self.frame_zero_settings.tkraise()
                self.btn_toggle_mode.config(text='Edit Positions', bg=BTN_ALT)
            else:
                self.frame_data.tkraise()
                self.btn_toggle_mode.config(text='Edit Positions', bg=BTN_ALT)

    def _toggle_shear_mode(self):
        # Cycle: Net Pos -> Shear -> Net Pos
        self.shear_mode = not self.shear_mode
        if self.shear_mode:
            self.btn_toggle_viz.config(text='Net Pos Mode', bg='#7a3b3b')
            # Seed CoM smoothing/baseline from current raw values
            active = sorted(self.reader.active_channels)
            all_pairs   = [(ch, key) for ch in active for key in SENSOR_KEYS]
            left_pairs  = [(ch, key) for ch in active for key in SENSOR_KEYS
                           if self.plot_side_vars.get(ch, tk.StringVar(value='left')).get() == 'left']
            right_pairs = [(ch, key) for ch in active for key in SENSOR_KEYS
                           if self.plot_side_vars.get(ch, tk.StringVar(value='left')).get() == 'right']
            bx, by = self._compute_raw_com_for_sensors(all_pairs)
            self._shear_com_baseline       = (bx, by)
            self._shear_com_smooth         = (bx, by)
            lx, ly = self._compute_raw_com_for_sensors(left_pairs)
            self._shear_com_baseline_left  = (lx, ly)
            self._shear_com_smooth_left    = (lx, ly)
            rx, ry = self._compute_raw_com_for_sensors(right_pairs)
            self._shear_com_baseline_right = (rx, ry)
            self._shear_com_smooth_right   = (rx, ry)
        else:
            self.btn_toggle_viz.config(text='Shear Mode', bg='#4a6741')
            # Reset all shear CoM state
            self._shear_com_smooth         = None
            self._shear_com_smooth_left    = None
            self._shear_com_smooth_right   = None
            self._shear_com_baseline       = None
            self._shear_com_baseline_left  = None
            self._shear_com_baseline_right = None
            self._shear_origin = None
            self._shear_origin_left = None
            self._shear_origin_right = None
            self._shear_had_reading = False
            self._shear_had_reading_left = False
            self._shear_had_reading_right = False
        self._autoscale()
        self._draw_grid()
        self._draw_vectors()

    # ─── poll ─────────────────────────────────────────────────────────────────

    def _poll(self):
        updated = self.reader.read()

        try:
            cutoff = float(self.cutoff_var.get())
        except ValueError:
            cutoff = 0.0

        if updated:
            active = self.reader.active_channels
            self.status_var.set(
                f"Active: {sorted(active)}" if active else "Connected — no channels yet."
            )

            # Update sampling rate display
            hz = self.reader.sample_rate_hz
            self.sample_rate_var.set(f'{hz:.1f} Hz' if hz > 0 else '— Hz')

            # Update avg diff / avg derivative display (row 8)
            _diff_norms  = []
            _deriv_norms = []
            for ch in active:
                for key in SENSOR_KEYS:
                    k = (ch, key)
                    dv = self.reader.diff_values.get(k)
                    if dv is None:
                        continue
                    _diff_norms.append(float(np.linalg.norm(dv)))
                    prev = self._az_prev_diff.get(k)
                    if prev is not None:
                        _deriv_norms.append(float(np.linalg.norm(dv - prev)))
            if _diff_norms:
                avg_d = sum(_diff_norms)  / len(_diff_norms)
                avg_r = sum(_deriv_norms) / len(_deriv_norms) if _deriv_norms else 0.0
                if self.shear_mode:
                    disp = self._shear_disp
                    if disp is not None:
                        mag = float(np.hypot(disp[0], disp[1]))
                        ang = float(np.degrees(np.arctan2(disp[1], disp[0])))
                        pos_str = f'  |  Shear ∠{ang:.1f}°  |{mag:.2f}|'
                    else:
                        pos_str = '  |  Shear (—)'
                    self.avg_diff_var.set(f'Diff {avg_d:.2f}  |  Deriv {avg_r:.2f}{pos_str}')
                else:
                    pos = self._net_pos
                    pos_str = (f'  |  NetPos ({pos[0]:.2f}, {pos[1]:.2f})'
                               if pos else '  |  NetPos (—, —)')
                    self.avg_diff_var.set(f'Diff {avg_d:.2f}  |  Deriv {avg_r:.2f}{pos_str}')
            else:
                self.avg_diff_var.set('— / —')
            
            for ch in range(NUM_CHANNELS):
                is_active = ch in active
                for key in SENSOR_KEYS:
                    for axis in ['x', 'y', 'z']:
                        var = self.disp_vars[(ch, key, axis)]
                        if is_active and (ch, key) in self.reader.diff_values:
                            # Choose source dict based on raw_mode flag
                            src = self.reader.raw_values if self.raw_mode else self.reader.diff_values
                            vec = src.get((ch, key), None)
                            if vec is None:
                                var.set('—')
                                continue

                            if axis == 'x':   num = vec[0]
                            elif axis == 'y': num = vec[1]
                            else:             num = vec[2]

                            # Apply cutoff only in diff mode (raw is absolute)
                            if not self.raw_mode and abs(num) <= cutoff:
                                num = 0.0

                            var.set(f'{num:+.1f}')
                        else:
                            var.set('—')

            # ── Auto-zeroing algorithm ────────────────────────────────────────
            if self.auto_zero_mode != 'off':
                try:
                    deriv_thresh = float(self.az_deriv_threshold_var.get())
                except ValueError:
                    deriv_thresh = 0.5
                try:
                    settling_time = float(self.az_settling_time_var.get())
                except ValueError:
                    settling_time = 1.0

                now = time.monotonic()

                if self.auto_zero_mode == 'individual':
                    for ch in active:
                        for key in SENSOR_KEYS:
                            k = (ch, key)
                            diff_vec = self.reader.diff_values.get(k)
                            if diff_vec is None:
                                continue

                            try:
                                az_cutoff = float(self.az_cutoff_vars.get(k, tk.StringVar(value='5.0')).get())
                            except ValueError:
                                az_cutoff = 5.0

                            prev = self._az_prev_diff.get(k)
                            if prev is not None:
                                deriv = float(np.linalg.norm(diff_vec - prev))
                            else:
                                deriv = float('inf')
                            self._az_prev_diff[k] = diff_vec.copy()
                            self._az_deriv[k] = deriv

                            reading_norm = float(np.linalg.norm(diff_vec))
                            cond_deriv  = deriv < deriv_thresh
                            cond_cutoff = reading_norm < az_cutoff

                            if cond_deriv and cond_cutoff:
                                if k not in self._az_settle_start:
                                    self._az_settle_start[k] = now
                                elif now - self._az_settle_start[k] >= settling_time:
                                    self.reader.zero_sensors(channels=[ch])
                                    self._az_settle_start.pop(k, None)
                                    self._az_prev_diff.pop(k, None)
                            else:
                                self._az_settle_start.pop(k, None)

                elif self.auto_zero_mode == 'average':
                    # Collect norms, cutoffs, and derivatives for every active sensor
                    reading_norms = []
                    cutoff_vals   = []
                    deriv_vals    = []
                    for ch in active:
                        for key in SENSOR_KEYS:
                            k = (ch, key)
                            diff_vec = self.reader.diff_values.get(k)
                            if diff_vec is None:
                                continue
                            reading_norms.append(float(np.linalg.norm(diff_vec)))
                            try:
                                az_cutoff = float(self.az_cutoff_vars.get(k, tk.StringVar(value='5.0')).get())
                            except ValueError:
                                az_cutoff = 5.0
                            cutoff_vals.append(az_cutoff)

                            # Compute per-sensor derivative magnitude
                            prev = self._az_prev_diff.get(k)
                            if prev is not None:
                                deriv = float(np.linalg.norm(diff_vec - prev))
                            else:
                                deriv = float('inf')
                            self._az_prev_diff[k] = diff_vec.copy()
                            deriv_vals.append(abs(deriv))

                    if reading_norms and cutoff_vals and deriv_vals:
                        avg_reading = sum(reading_norms) / len(reading_norms)
                        avg_cutoff  = sum(cutoff_vals)   / len(cutoff_vals)
                        avg_deriv   = sum(deriv_vals)    / len(deriv_vals)

                        if avg_reading < avg_cutoff and avg_deriv < deriv_thresh:
                            if 'avg' not in self._az_settle_start:
                                self._az_settle_start['avg'] = now
                            elif now - self._az_settle_start['avg'] >= settling_time:
                                self.reader.zero_sensors()
                                self._az_settle_start.pop('avg', None)
                                self._az_prev_diff.clear()
                        else:
                            self._az_settle_start.pop('avg', None)

        self._draw_grid()
        self._draw_vectors()

        if self.is_recording:
            self._write_csv_rows()

        self.root.after(50, self._poll)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    PORT = 'COM4'   # <- change to your port

    reader = ArduinoDataReader(PORT)

    root = tk.Tk()
    app  = SensorTableApp(root, reader)
    root.mainloop()