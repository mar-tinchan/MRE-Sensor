# Standard and third-party imports used throughout the application.
# serial: reads data from the Arduino over USB.
# tkinter: builds the GUI window, widgets, and canvas.
# numpy: numerical operations
# json/os: persist and load sensor position configuration from disk.
# deque: fixed-length queues for rolling median and sample-rate tracking.
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

# Manages the serial connection to the Arduino and parses incoming sensor frames.
# Stores raw readings, baseline values, and smoothed differential values for
# every (channel, sensor-position) pair.
class ArduinoDataReader:

    # Ordered names for the five sensor positions on each ReSkin patch.
    SENSOR_KEYS      = ['top', 'left', 'middle', 'right', 'bottom']
    # Physical rotation (in degrees) applied to each sensor's XY data so that
    # all sensors share a consistent coordinate frame after parsing.
    SENSOR_ROTATIONS = {'top': 270, 'left': 0, 'middle': 90, 'right': 90, 'bottom': 180}

    def __init__(self, port: str, baudrate: int = 2000000, timeout: float = 0.1):
        # Serial connection parameters; high baud rate is required for the sensor data rate.
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.ser      = None

        # Tracking structures keyed by (channel_index, sensor_key).
        # active_channels: set of channel numbers seen in incoming data.
        # raw_values:  latest median-filtered magnetometer reading per sensor.
        # baseline:    snapshot taken at the last zero, subtracted to get diff_values.
        # diff_values: low-pass-filtered difference from baseline.
        # raw_history: sliding window of recent raw readings used for the median filter.
        self.active_channels: set[int] = set()
        self.raw_values:  dict = {}
        self.baseline:    dict = {}
        self.diff_values: dict = {}
        self.raw_history: dict = {}

        # lpf_alpha: blending weight for the exponential (low-pass) filter on diff values.
        #            Lower values = more smoothing, higher values = faster response.
        # median_window_size: number of consecutive raw frames used for the median filter.
        # _leftover: partial line carried over between serial read() calls.
        self.lpf_alpha          = 0.2
        self.median_window_size = 5
        self._leftover          = ""

        # Sampling rate tracking
        self._frame_times: deque = deque(maxlen=30)
        self.sample_rate_hz: float = 0.0

        self.connect()

    def connect(self):
        # Open (or re-open) the serial port and flush any stale bytes in the buffer.
        # If the port is unavailable, prints a waiting message without crashing.
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
        # Called the first time data arrives for a previously unseen channel.
        # Pre-fills every data dictionary with zero arrays so later code can
        # safely read any key without a KeyError.
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
        # Rotates XY sensor data to account for how each sensor is physically
        # oriented on the patch. The transform is mirrored about the Y axis
        # to match the physical sensor orientation 
        # Normal Rotation
        #if   angle ==  90: return -y,  x
        #elif angle == 180: return -x, -y
        #elif angle == 270: return  y, -x
        #return x, y
        
        #Mirrored about Y axis (to have a traditional x-y frame facing into the MRE)
        if   angle ==  90: return y,  x
        elif angle == 180: return x, -y
        elif angle == 270: return -y, -x
        return -x, y

    def zero_sensors(self, channels=None):
        # Captures the current raw_values as the new baseline for the specified
        # channels (or all active channels if none are given). After zeroing,
        # diff_values resets to zero and subsequent readings are relative to this
        # new baseline.
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
        # Parses one CSV data line from the Arduino with the format:
        #   CH<n>,x0,y0,z0,x1,y1,z1,...  (16 comma-separated fields total).
        # For each of the 5 sensor positions the method:
        #   1. Applies the per-sensor rotation to align XY with a common frame.
        #   2. Applies additional fixed axis flips to match the physical layout.
        #   3. Pushes the corrected vector into a rolling window and takes the median.
        #   4. Runs an exponential low-pass filter on the difference from baseline.
        # Returns True if the line was valid and data was updated.
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
            idx = i * 3   # Each sensor occupies 3 consecutive values (x, y, z).
            rx_raw, ry_raw, rz = vals[idx], vals[idx + 1], vals[idx + 2]

            # Step 1: Apply the sensor-specific rotation to align XY with a shared frame.
            rx, ry = self.rotate_xy(rx_raw, ry_raw, self.SENSOR_ROTATIONS[key])
            # Step 2: Additional fixed axis remapping to match the physical board orientation.
            rx, ry = ry, -rx
            rx, rz = -rx, -rz

            # Step 3: Accumulate into the rolling history window and compute the median
            # to suppress impulse noise in the raw signal.
            new_raw = np.array([rx, ry, rz])
            self.raw_history[k].append(new_raw)
            median_raw = np.median(self.raw_history[k], axis=0)
            self.raw_values[k] = median_raw

            # Step 4: Compute the difference from the baseline, then blend it into the
            # existing diff_values using an exponential low-pass filter to reduce jitter.
            target_diff = median_raw - self.baseline[k]
            self.diff_values[k] = (
                self.lpf_alpha * target_diff
                + (1 - self.lpf_alpha) * self.diff_values[k]
            )
        return True

    def read(self) -> bool:
        # Reads all available bytes from the serial buffer, reconstructs complete
        # frames delimited by "ST," (start) and "EN" (end) markers, then processes
        # only the most recent complete frame to avoid processing stale data.
        # Also updates the estimated sample rate using a rolling timestamp window.
        # Returns True if at least one sensor was successfully updated this call.
        if self.ser is None or not self.ser.is_open:
            self.connect()
            return False
        try:
            if self.ser.in_waiting == 0:
                return False
            # Read all bytes currently in the OS buffer and prepend any partial
            # line left over from the previous call to avoid losing split packets.
            chunk = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
            text  = self._leftover + chunk
            lines = text.split('\n')
            # The last element may be an incomplete line; save it for next call.
            self._leftover = lines[-1]
            lines = [l.strip('\r\n ') for l in lines[:-1]]

            # Reconstruct complete frames bounded by 'ST,' (start) and 'EN' (end).
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

            # Use only the most recent complete frame to discard any queued-up
            # stale data and keep the display latency minimal.
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

            # Parse each channel line in ascending channel order.
            updated = False
            for ch in sorted(latest):
                if self._parse_line(latest[ch]):
                    updated = True

            # Update the rolling sample-rate estimate using the timestamps of the
            # last 30 successfully processed frames.
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

# Maximum number of multiplexed Arduino channels supported by the UI.
NUM_CHANNELS  = 8
# Human-readable labels and dictionary keys for the five sensor positions per patch.
SENSOR_LABELS = ['Top', 'Left', 'Middle', 'Right', 'Bottom']
SENSOR_KEYS   = ['top', 'left', 'middle', 'right', 'bottom']

# UI color palette used throughout the application.
# Header/background/accent colors for the main window and data table cells.
BG_HEADER   = '#2b2d42'
FG_HEADER   = '#edf2f4'
BG_SENSOR   = '#d8d8d8'   
BG_LOCKED   = '#e2e2e2'
BG_EDITABLE = '#ffffff'   
BG_ROOT     = '#f0f0f0'
BG_RIGHT    = '#e8e8e8'
BG_CANVAS   = '#1a1a2e'
ACCENT      = '#ef233c'
BTN_ALT     = '#457b9d'
BTN_ALT_ACT = '#1d3557'
GRID_CLR    = '#888888'
# Width (in characters) for each axis sub-column in the data table.
SUBCOL_W    = 6

# One distinct color per channel so vectors/dots from different channels
# can be visually distinguished on the canvas.
CH_COLORS = [
    '#e63946', '#2a9d8f', '#e9c46a', '#f4a261',
    '#a8dadc', '#c77dff', '#90e0ef', '#f8ad9d',
]

# Unicode arrow/dot characters drawn on the canvas next to each sensor vector
# so the user can identify which sensor position each arrow belongs to.
SENSOR_MARKERS = {
    'top':    '▲',
    'left':   '◀',
    'middle': '●',
    'right':  '▶',
    'bottom': '▼',
}

# Half-length in pixels of the vector arrow drawn on the canvas for each sensor.
# The arrow is drawn symmetrically around the sensor dot, so the full arrow spans
# 2 * ARROW_HALF pixels.
ARROW_HALF = 22   

# JSON file used to persist sensor XY positions, per-channel rotations, the noise
# cutoff value, and auto-zeroing thresholds between application sessions.
CONFIG_FILENAME = 'sensor_positions.json' 

# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

# Main GUI application. Owns the tkinter root window, all widgets, the sensor
# position/rotation configuration, visualization state, recording state, and
# the periodic polling loop that drives updates.
class SensorTableApp:

    def __init__(self, root: tk.Tk, reader: ArduinoDataReader):
        self.root   = root
        self.reader = reader
        root.title("ReSkin Live Monitor")
        root.configure(bg=BG_ROOT)
        
        root.geometry("1300x650")
        root.resizable(True, True)

        # Register a cleanup handler so files are flushed and the serial port
        # is closed gracefully when the user closes the window.
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Tkinter StringVar dicts for sensor XY positions (edit table) and
        # live axis readings (data table), keyed by (ch, key, axis).
        self.sensor_pos_vars: dict = {}
        self.disp_vars: dict = {}
        # Per-channel clockwise rotation (degrees) applied to displayed vectors.
        self.rot_vars: dict = {}
        # Global noise cutoff: axis values below this threshold are treated as zero.
        self.cutoff_var = tk.StringVar(value='0.0') 
        # Flags controlling which data view is shown in the table and canvas.
        self.is_edit_mode = False
        self.heatmap_mode = False   # False = Net Pos, True = Gaussian Heatmap
        self.raw_mode     = False   # False = differential, True = raw readings
        # StringVars bound to the status bar labels at the bottom of the data table.
        self.sample_rate_var = tk.StringVar(value='— Hz')
        self.sum_abs_z_var   = tk.StringVar(value='Σ|Z|: —')

        # Auto-zeroing state  ('off' | 'individual' | 'average')
        self.auto_zero_mode = 'off'
        self.is_zero_settings_mode = False   # True = showing zero settings table

        # Recording state: tracks whether CSV recording is active, when it started,
        # and holds open file handles and csv.writer objects keyed by channel.
        self.is_recording    = False
        self._record_start   = 0.0
        self._record_writers: dict = {}
        self._record_files:   dict = {}
        # Snapshot of the visualization mode when recording started, used to
        # write the correct position columns (Net Pos vs CoP) for the entire session.
        self._record_mode_snapshot = 'netpos'   # 'netpos' | 'heatmap'
        self._record_max_cops = 0
        # When True, the Snapshot button also writes a CSV and PNG to disk.
        self.snapshot_save_csv = False   # toggled by the Save CSV half-button

        # Last-computed world positions for display and recording
        self._net_pos: tuple | None = None        # (wx, wy) world coords
        self._cop_positions: list   = []          # list of (wx, wy) world coords – one per local peak
        # Per-sensor cutoff thresholds for auto-zeroing (keyed by (ch, key))
        self.az_cutoff_vars: dict = {}
        self.az_deriv_threshold_var = tk.StringVar(value='0.5')
        self.az_settling_time_var   = tk.StringVar(value='1.0')
        # Per-sensor derivative tracking for auto-zeroing
        self._az_prev_diff: dict = {}
        self._az_deriv:     dict = {}
        self._az_settle_start: dict = {}
        # RBF sigma as a percentage of canvas diagonal (0–100), default 15 %
        self.rbf_sigma_pct = tk.DoubleVar(value=15.0)

        # Canvas coordinate transform: pixels-per-world-unit and world-space origin.
        self._scale  = 40.0         
        self._origin = (0.0, 0.0)   
        self._tbl_frame: tk.Frame | None = None

        self._saved_config = {}
        self._load_config()

        self._build_ui()
        self._poll()

    def _load_config(self):
        # Reads sensor positions, rotations, cutoff, and auto-zero settings from
        # the JSON config file if it exists. Falls back to an empty dict (defaults)
        # if the file is missing or unreadable.
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
        # Serializes current sensor XY positions (Z is always 0 and not saved),
        # per-channel rotations, the global noise cutoff, and all auto-zero
        # thresholds to the JSON config file for the next session.
        config = {}
        for (ch, key, axis), var in self.sensor_pos_vars.items():
            if axis != 'z': # Don't bother saving Z since it's forced 0
                config[f"pos_{ch}_{key}_{axis}"] = var.get()
        for ch, var in self.rot_vars.items():
            config[f"rot_{ch}"] = var.get()
        
        config["global_cutoff"] = self.cutoff_var.get()

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
        # Called when the user closes the window. Stops any active recording,
        # saves the current configuration to disk, closes the serial port, and
        # destroys the tkinter root.
        if self.is_recording:
            self.is_recording = False
            for fh in self._record_files.values():
                fh.close()
        self._save_config() 
        if self.reader.ser and self.reader.ser.is_open:
            self.reader.ser.close()
        self.root.destroy() 

    def _hdr(self, parent, text, row, col, colspan=1, rowspan=1):
        # Convenience method that creates a styled header Label and places it in
        # the grid. Returns the Label so callers can store a reference if needed.
        lbl = tk.Label(parent, text=text,
                       bg=BG_HEADER, fg=FG_HEADER,
                       font=('Helvetica', 9, 'bold'),
                       padx=4, pady=5, relief='flat')
        lbl.grid(row=row, column=col, columnspan=colspan, rowspan=rowspan,
                 sticky='nsew', padx=1, pady=1)
        return lbl

    def _build_ui(self):
        # Divides the root window into a left panel (data tables + canvas) and a
        # right panel (control buttons). Delegates population to dedicated helpers.
        left  = tk.Frame(self.root, bg=BG_ROOT)
        right = tk.Frame(self.root, bg=BG_RIGHT, padx=18, pady=18)

        right.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=8, pady=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)

        self._build_tables(left)
        self._build_canvas(left)
        self._build_controls(right)

    # ─── Left Panel: Swappable Tables ─────────────────────────────────────────

    def _build_tables(self, parent):
        # Creates three overlapping frames (live data, edit positions, zero settings)
        # stacked in a container. Only one is visible at a time; the visible frame is
        # raised via tkraise() when the user switches modes.
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
        # Builds the live data grid: channel headers across the top, sensor-position
        # labels down the left, and a Label per (channel, sensor, axis) cell whose
        # textvariable is updated each poll cycle. The column headers store both their
        # "Diff X/Y/Z" and "Raw X/Y/Z" text so _toggle_raw_mode can swap them in place.
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

        # Row 7: show sampling rate on left half, Sigma|Z| on right half
        self._hdr(tbl, 'Sample Rate', row=7, col=0)
        half = (NUM_CHANNELS * 3) // 2
        tk.Label(tbl, textvariable=self.sample_rate_var,
                 bg=BG_SENSOR, fg='#111111',
                 font=('Courier', 8),
                 anchor='center', relief='flat', bd=0,
                 padx=2, pady=3
                 ).grid(row=7, column=1, columnspan=half,
                        sticky='nsew', padx=1, pady=1)
        tk.Label(tbl, textvariable=self.sum_abs_z_var,
                 bg=BG_SENSOR, fg='#111111',
                 font=('Courier', 8),
                 anchor='center', relief='flat', bd=0,
                 padx=2, pady=3
                 ).grid(row=7, column=1 + half, columnspan=NUM_CHANNELS * 3 - half,
                        sticky='nsew', padx=1, pady=1)

        self._avg_diff_hdr_label = self._hdr(tbl, 'Avg Diff / Deriv', row=8, col=0)
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
        # Builds the sensor position editor. Each cell is an Entry widget bound to a
        # StringVar in sensor_pos_vars. The Z column is locked to 0 (read-only, greyed
        # background) because the application only supports 2-D sensor layouts.
        # A rotation Entry per channel and a global noise cutoff Entry occupy the
        # last two rows.
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

        # Cutoff Row (Row 8)
        self._hdr(tbl, 'Noise Cutoff', row=8, col=0)
        cutoff_val = self._saved_config.get("global_cutoff", '0.0')
        self.cutoff_var = tk.StringVar(value=cutoff_val)
        
        e = tk.Entry(tbl, textvariable=self.cutoff_var,
                     bg=BG_EDITABLE, fg='#111111',
                     insertbackground='black',
                     relief='flat', bd=0,
                     font=('Courier', 9, 'bold'),
                     justify='center')
        # Span across all channel columns
        e.grid(row=8, column=1, columnspan=NUM_CHANNELS * 3,
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
        # Builds the lower-left visualization area. Contains two swappable header
        # bars (vector field legend vs heatmap controls with sigma slider) and a
        # square tkinter Canvas where vectors or the RBF heatmap are drawn each poll.
        # A snapshot text overlay and a sigma value overlay are placed on top of the
        # canvas frame as floating Labels.
        outer = tk.Frame(parent, bg=BG_ROOT)
        outer.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        # ── "Vector Field" header (visible in vector mode) ────────────────────
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

        # ── Heatmap controls header (visible in heatmap mode) ─────────────────
        self.bar_heatmap = tk.Frame(outer, bg=BG_ROOT)
        # (packed/unpacked by _toggle_viz_mode — not shown initially)

        tk.Label(self.bar_heatmap, text='Heatmap — σ (% diagonal):',
                 bg=BG_ROOT, fg='#2b2d42',
                 font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=4)

        self.sigma_slider = tk.Scale(
            self.bar_heatmap,
            variable=self.rbf_sigma_pct,
            from_=1.0, to=60.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            length=260,
            bg=BG_ROOT, fg='#2b2d42',
            highlightthickness=0,
            troughcolor='#cccccc',
            activebackground='#4a6741',
            font=('Helvetica', 8),
            label=None,
        )
        self.sigma_slider.pack(side=tk.LEFT, padx=(4, 0))

        self.sigma_val_lbl = tk.Label(self.bar_heatmap, text='15.0 %',
                                      bg=BG_ROOT, fg='#444444',
                                      font=('Helvetica', 9))
        self.sigma_val_lbl.pack(side=tk.LEFT, padx=(6, 0))

        # keep label in sync with slider
        def _on_sigma_move(*_):
            self.sigma_val_lbl.config(text=f'{self.rbf_sigma_pct.get():.1f} %')
        self.rbf_sigma_pct.trace_add('write', _on_sigma_move)

        # legend repeated in heatmap bar
        leg2 = tk.Frame(self.bar_heatmap, bg=BG_ROOT)
        leg2.pack(side=tk.LEFT, padx=(20, 0))
        for ch in range(NUM_CHANNELS):
            tk.Label(leg2, text=f'■ Ch{ch}',
                     bg=BG_ROOT, fg=CH_COLORS[ch],
                     font=('Courier', 8, 'bold')).pack(side=tk.LEFT, padx=3)

        # ── canvas itself ─────────────────────────────────────────────────────
        self.canvas_frame = tk.Frame(outer, bg=BG_ROOT)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = tk.Canvas(self.canvas_frame, bg=BG_CANVAS,
                                highlightthickness=1,
                                highlightbackground=GRID_CLR)
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Snapshot overlay — shown in the top-right corner of the canvas frame
        self.snapshot_overlay_var = tk.StringVar(value='')
        self._snapshot_overlay_lbl = tk.Label(
            self.canvas_frame,
            textvariable=self.snapshot_overlay_var,
            bg='#0d0d1e', fg='#00E5FF',
            font=('Courier', 8),
            justify='right',
            padx=6, pady=4,
            relief='flat'
        )
        self._snapshot_overlay_lbl.place(relx=1.0, rely=0.0, anchor='ne')

        # Gaussian sigma overlay — top-right of canvas, only visible in heatmap mode
        self._sigma_overlay_var = tk.StringVar(value='')
        self._sigma_overlay_lbl = tk.Label(
            self.canvas_frame,
            textvariable=self._sigma_overlay_var,
            bg='#1a1a2e', fg='#FFD600',
            font=('Courier', 9, 'bold'),
            justify='right',
            padx=6, pady=4,
            relief='flat'
        )
        self._sigma_overlay_lbl.place_forget()   # hidden until heatmap mode is on

        def _on_sigma_overlay_update(*_):
            if self.heatmap_mode:
                self._sigma_overlay_var.set(f'σ = {self.rbf_sigma_pct.get():.1f}% diag')
        self.rbf_sigma_pct.trace_add('write', _on_sigma_overlay_update)

        self.canvas_frame.bind('<Configure>', self._on_canvas_resize)

    def _on_canvas_resize(self, event):
        # Fires when the canvas container is resized. Keeps the canvas square
        # by using the smaller of the two dimensions, then recalculates the
        # world-to-canvas scale and redraws.
        w = event.width
        h = event.height
        size = min(w, h)
        if size > 10:
            self.canvas.config(width=size, height=size)
            self.canvas.update_idletasks()
            self._autoscale()
            self._draw_grid()
            self._draw_vectors()

    # ─── autoscale ────────────────────────────────────────────────────────────

    def _on_pos_changed(self, ch: int):
        # Triggered by a trace on any sensor position StringVar for the given channel.
        # Recalculates the canvas scale so the updated layout fits in the view.
        if ch not in self.reader.active_channels:
            return
        self._autoscale()

    def _autoscale(self):
        # Computes _scale and _origin so all configured sensor positions for active
        # channels fit inside the canvas with a 15% margin on each side. Called
        # whenever sensor positions are edited or the canvas is resized.
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

    def _w2c(self, wx, wy):
        # Converts world-space coordinates (wx, wy) to canvas pixel coordinates.
        # World Y increases upward; canvas Y increases downward, so Y is negated.
        cw = self.canvas.winfo_width()  / 2
        ch = self.canvas.winfo_height() / 2
        px = cw + (wx - self._origin[0]) * self._scale
        py = ch - (wy - self._origin[1]) * self._scale
        return px, py

    # ─── drawing ──────────────────────────────────────────────────────────────

    def _draw_grid(self):
        # Redraws the X and Y world-axis lines on the canvas (only if the origin
        # falls within the visible canvas area).
        self.canvas.delete('grid')
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        ox, oy = self._w2c(0, 0)
        if 0 <= ox <= w:
            self.canvas.create_line(ox, 0, ox, h, fill='#2a2a4a', width=1, tags='grid')
        if 0 <= oy <= h:
            self.canvas.create_line(0, oy, w, oy, fill='#2a2a4a', width=1, tags='grid')

    # ─── Gaussian RBF Heatmap ─────────────────────────────────────────────────

    def _draw_heatmap(self, active, cutoff):
        """Render a Gaussian RBF contour/heat map on the canvas."""
        self._cop_positions = []
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # ── collect sensor positions and weights ──────────────────────────────
        positions = []   # list of (canvas_x, canvas_y)
        weights   = []   # corresponding L2-norm weights

        for ch_id in range(NUM_CHANNELS):
            if ch_id not in active:
                continue
            color = CH_COLORS[ch_id]
            for key in SENSOR_KEYS:
                try:
                    wx = float(self.sensor_pos_vars[(ch_id, key, 'x')].get())
                    wy = float(self.sensor_pos_vars[(ch_id, key, 'y')].get())
                except (ValueError, KeyError):
                    continue

                k = (ch_id, key)
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

                w = np.linalg.norm([dx, dy, dz])
                if w < 1e-6:
                    continue

                cx, cy = self._w2c(wx, wy)
                positions.append((cx, cy))
                weights.append(w)

                # Draw sensor dot marker
                r = 3
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=color, outline='white',
                                        width=1, tags='vector')
                if key == 'middle':
                    self.canvas.create_text(cx + r + 5, cy - r - 5,
                                            text=f'Ch{ch_id}', fill=color,
                                            font=('Helvetica', 7, 'bold'),
                                            tags='vector')

        if not positions:
            return

        # ── build scalar field via Gaussian RBF ───────────────────────────────
        GRID = 60   # resolution: higher = smoother but slower
        xs = np.linspace(0, cw, GRID)
        ys = np.linspace(0, ch, GRID)
        gx, gy = np.meshgrid(xs, ys)   # shape (GRID, GRID)

        # RBF bandwidth: driven by the σ slider (% of canvas diagonal)
        sigma = (self.rbf_sigma_pct.get() / 100.0) * np.hypot(cw, ch)

        field = np.zeros((GRID, GRID))
        for (px, py), w in zip(positions, weights):
            dist2 = (gx - px) ** 2 + (gy - py) ** 2
            field += w * np.exp(-dist2 / (2 * sigma ** 2))

        f_max = field.max()
        if f_max < 1e-9:
            return
        field_norm = field / f_max   # [0, 1]

        # ── colourise and paint rectangles ────────────────────────────────────
        cell_w = cw / GRID
        cell_h = ch / GRID

        # Pre-build colour LUT (256 entries: blue→cyan→green→yellow→red)
        def _field_to_hex(v):
            """Map [0,1] → colour string via a 4-stop gradient."""
            v = float(np.clip(v, 0.0, 1.0))
            if v < 0.25:
                t = v / 0.25
                r, g, b = 0, int(255 * t), 255
            elif v < 0.5:
                t = (v - 0.25) / 0.25
                r, g, b = 0, 255, int(255 * (1 - t))
            elif v < 0.75:
                t = (v - 0.5) / 0.25
                r, g, b = int(255 * t), 255, 0
            else:
                t = (v - 0.75) / 0.25
                r, g, b = 255, int(255 * (1 - t)), 0
            return f'#{r:02x}{g:02x}{b:02x}'

        ALPHA_THRESHOLD = 0.04   # skip nearly-zero cells for performance
        for row in range(GRID):
            for col in range(GRID):
                v = field_norm[row, col]
                if v < ALPHA_THRESHOLD:
                    continue
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = x0 + cell_w + 1   # +1 avoids hairline gaps
                y1 = y0 + cell_h + 1
                hex_col = _field_to_hex(v)
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=hex_col, outline='',
                                             tags='vector')

        COP_COLORS = ['#00E5FF', '#FFD600', '#FF6D00', '#69F0AE', '#EA80FC']
        cr = 8       # Crosshair radius in pixels.
        cw_half = cw / 2
        ch_half = ch / 2

        # ── Discrete local maxima + sub-cell weighted centroid ───────────────────
        # Cells below this fraction of the peak value are ignored as background noise.
        PEAK_THRESH = 0.15
        # Two peak cells within this many grid cells of each other are merged into one.
        MERGE_CELLS = 3

        # Pad the field by one cell so the 3x3 neighbourhood check never goes
        # out of bounds at the edges of the grid.
        padded = np.pad(field_norm, 1, mode='constant', constant_values=0)
        peak_cells = []
        for r in range(GRID):
            for c in range(GRID):
                v = field_norm[r, c]
                if v < PEAK_THRESH:
                    continue
                rp, cp = r + 1, c + 1
                neighbourhood = padded[rp-1:rp+2, cp-1:cp+2]
                # A cell is a local maximum if it equals the highest value in its
                # 3x3 neighbourhood (ties kept; first encountered wins after sort).
                if v == neighbourhood.max():
                    peak_cells.append((r, c, v))

        # Sort strongest peaks first, then greedily suppress any peak that falls
        # within MERGE_CELLS of an already-accepted peak.
        peak_cells.sort(key=lambda p: p[2], reverse=True)
        merged_cells = []
        for r, c, v in peak_cells:
            too_close = any(
                abs(r - mr) <= MERGE_CELLS and abs(c - mc) <= MERGE_CELLS
                for mr, mc, _ in merged_cells
            )
            if not too_close:
                merged_cells.append((r, c, v))

        # Refine each merged peak to sub-cell precision using a weighted centroid
        # over the 3x3 patch surrounding the peak cell.
        refined = []
        for r, c, _ in merged_cells:
            r0, r1 = max(0, r - 1), min(GRID - 1, r + 1)
            c0, c1 = max(0, c - 1), min(GRID - 1, c + 1)
            patch = field_norm[r0:r1+1, c0:c1+1]
            total = patch.sum()
            if total < 1e-12:
                # Degenerate patch: fall back to the cell centre in canvas pixels.
                refined.append((c * cell_w + cell_w / 2, r * cell_h + cell_h / 2))
                continue
            cols_idx = np.arange(c0, c1 + 1)
            rows_idx = np.arange(r0, r1 + 1)
            col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)
            cx_idx = (col_grid * patch).sum() / total
            cy_idx = (row_grid * patch).sum() / total
            refined.append((cx_idx * cell_w + cell_w / 2, cy_idx * cell_h + cell_h / 2))

        # Convert refined canvas-pixel CoP positions to world coordinates and
        # store them for the status bar and CSV recording.
        self._cop_positions = []
        for mx, my in refined:
            if self._scale > 0:
                wx = (mx - cw_half) / self._scale + self._origin[0]
                wy = (ch_half - my) / self._scale + self._origin[1]
            else:
                wx, wy = 0.0, 0.0
            self._cop_positions.append((wx, wy))

        for i, (mx, my) in enumerate(refined):
            t_color = COP_COLORS[i % len(COP_COLORS)]
            self.canvas.create_oval(mx - cr, my - cr, mx + cr, my + cr,
                                    outline=t_color, width=2, tags='vector')
            self.canvas.create_line(mx - cr - 4, my, mx + cr + 4, my,
                                    fill=t_color, width=2, tags='vector')
            self.canvas.create_line(mx, my - cr - 4, mx, my + cr + 4,
                                    fill=t_color, width=2, tags='vector')
            label = f'CoP{i+1}' if len(refined) > 1 else 'CoP'
            wx_lbl, wy_lbl = self._cop_positions[i]
            self.canvas.create_text(mx + cr + 6, my - cr - 6,
                                    text=f'{label} ({wx_lbl:.1f},{wy_lbl:.1f})',
                                    fill=t_color,
                                    font=('Helvetica', 8, 'bold'),
                                    tags='vector')

    def _draw_vectors(self):
        # Clears and redraws all vector field elements on the canvas.
        # In heatmap mode, delegates to _draw_heatmap.
        # In vector mode, draws one arrow per sensor showing the XY component of
        # the differential (or raw) reading, rotated by the channel's configured angle.
        # Also computes and draws the global weighted centroid (Net Pos) as a crosshair.
        self.canvas.delete('vector')
        self._net_pos = None
        self._cop_positions = []
        active = self.reader.active_channels

        try:
            cutoff = float(self.cutoff_var.get())
        except ValueError:
            cutoff = 0.0

        if self.heatmap_mode:
            self._draw_heatmap(active, cutoff)
            return

        # Running weighted sums used to compute the global Net Pos centroid.
        global_sum_norm = 0.0
        global_sum_norm_x = 0.0
        global_sum_norm_y = 0.0

        for ch in range(NUM_CHANNELS):
            if ch not in active: continue

            # Per-channel clockwise rotation applied to displayed arrow directions.
            try: rot_deg = float(self.rot_vars[ch].get())
            except (ValueError, KeyError): rot_deg = 0.0
            rot_rad = -np.deg2rad(rot_deg)

            color = CH_COLORS[ch]

            for key in SENSOR_KEYS:
                try:
                    wx = float(self.sensor_pos_vars[(ch, key, 'x')].get())
                    wy = float(self.sensor_pos_vars[(ch, key, 'y')].get())
                except (ValueError, KeyError):
                    continue

                # Convert world-space sensor position to canvas pixel position and
                # draw a small filled dot to mark the sensor location.
                cx, cy = self._w2c(wx, wy)

                r = 3
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=color, outline='white',
                                        width=1, tags='vector')

                # Label only the middle sensor of each channel to avoid clutter.
                if key == 'middle':
                    self.canvas.create_text(cx + r + 5, cy - r - 5,
                                            text=f'Ch{ch}', fill=color,
                                            font=('Helvetica', 7, 'bold'),
                                            tags='vector')

                k = (ch, key)
                if k not in self.reader.diff_values:
                    continue

                # Select either raw or differential data based on the current display mode.
                src = self.reader.raw_values if self.raw_mode else self.reader.diff_values
                vec = src.get(k)
                if vec is None:
                    continue
                dx, dy, dz = float(vec[0]), float(vec[1]), float(vec[2])
                
                # Apply Cutoff Zeroing strictly on individual components
                if abs(dx) <= cutoff: dx = 0.0
                if abs(dy) <= cutoff: dy = 0.0
                if abs(dz) <= cutoff: dz = 0.0
                
                # Compute L2 Norm (3D) of the sensor's filtered differential data
                mag_3d = np.linalg.norm([dx, dy, dz]) 
                
                # Add to GLOBAL tracking sums for Net Pos
                global_sum_norm += mag_3d
                global_sum_norm_x += mag_3d * wx
                global_sum_norm_y += mag_3d * wy

                # 2D magnitude for the arrow direction; Z contributes to Net Pos weight but
                # is not drawn on the 2D canvas.
                mag_2d = np.hypot(dx, dy)

                # Draw a small crosshair instead of an arrow when the XY component is
                # negligible (pure-Z press or below cutoff).
                if mag_2d < 1e-6:
                    self.canvas.create_line(cx - 4, cy, cx + 4, cy, fill=color, width=1, dash=(2, 3), tags='vector')
                    self.canvas.create_line(cx, cy - 4, cx, cy + 4, fill=color, width=1, dash=(2, 3), tags='vector')
                    continue

                # Normalize XY to a unit vector, then rotate by the channel angle.
                ux, uy = dx / mag_2d, dy / mag_2d
                ux2 =  ux * np.cos(rot_rad) + uy * np.sin(rot_rad)
                uy2 = -ux * np.sin(rot_rad) + uy * np.cos(rot_rad)

                # Scale the unit vector to ARROW_HALF pixels and compute arrow endpoints
                # centered on the sensor dot.
                tx =  ux2 * ARROW_HALF
                ty = -uy2 * ARROW_HALF   # Negate Y because canvas Y increases downward.

                x1, y1 = cx - tx, cy - ty   # Tail
                x2, y2 = cx + tx, cy + ty   # Head (arrowhead drawn here)

                self.canvas.create_line(x1, y1, x2, y2,
                                        fill=color, width=2,
                                        arrow=tk.LAST,
                                        arrowshape=(8, 10, 4),
                                        tags='vector')

                # Draw the sensor-position marker symbol near the arrowhead.
                marker = SENSOR_MARKERS.get(key, '·')
                self.canvas.create_text(x2 + 7, y2 - 7,
                                        text=marker, fill=color,
                                        font=('Helvetica', 7),
                                        tags='vector')

        # --- Draw the single GLOBAL computed weighted center target ---
        # Net Pos is the 3D-magnitude-weighted centroid of all sensor world positions.
        # Only drawn when there is a non-trivial total weight.
        if global_sum_norm > 1e-6:
            center_wx = global_sum_norm_x / global_sum_norm
            center_wy = global_sum_norm_y / global_sum_norm
            self._net_pos = (center_wx, center_wy)
            
            ccx, ccy = self._w2c(center_wx, center_wy)
            
            cr = 8       # Crosshair radius
            t_color = '#00E5FF'   # Cyan
            
            self.canvas.create_oval(ccx - cr, ccy - cr, ccx + cr, ccy + cr,
                                    outline=t_color, width=2, tags='vector')
            self.canvas.create_line(ccx - cr - 4, ccy, ccx + cr + 4, ccy, fill=t_color, width=2, tags='vector')
            self.canvas.create_line(ccx, ccy - cr - 4, ccx, ccy + cr + 4, fill=t_color, width=2, tags='vector')
            
            self.canvas.create_text(ccx + cr + 6, ccy - cr - 6,
                                    text='Net Pos', fill=t_color,
                                    font=('Helvetica', 8, 'bold'),
                                    tags='vector')

    # ─── Right Panel: Controls ────────────────────────────────────────────────

    def _build_controls(self, parent):
        # Populates the right-hand control panel with buttons for zeroing, auto-zero
        # cycling, raw/diff toggling, edit mode, visualization mode, recording,
        # and snapshots. Also adds a status label at the bottom.
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

        self.btn_toggle_viz = tk.Button(top_frame, text='Show Heatmap',
                                        command=self._toggle_viz_mode,
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
        # Starts or stops CSV recording. On start, opens one CSV file per active
        # channel (named with a timestamp) and writes a header row. On stop, closes
        # all open file handles. While recording, _write_csv_rows is called each poll.
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

            # Build the CSV header dynamically based on the current visualization mode.
            # Each channel file records: elapsed time, raw XYZ per sensor, diff XYZ per
            # sensor, derivative norm per sensor, and either Net Pos or up to 5 CoP
            # coordinates, followed by the sum of absolute Z values.
            header = ['time_elapsed_s']
            for key in SENSOR_KEYS:
                for axis in ['x', 'y', 'z']:
                    header.append(f'{key}_raw_{axis}')
            for key in SENSOR_KEYS:
                for axis in ['x', 'y', 'z']:
                    header.append(f'{key}_diff_{axis}')
            for key in SENSOR_KEYS:
                header.append(f'{key}_deriv_norm')
            if self.heatmap_mode:
                MAX_COPS = 5
                for i in range(1, MAX_COPS + 1):
                    header += [f'cop{i}_x', f'cop{i}_y']
            else:
                header += ['net_pos_x', 'net_pos_y']
            header.append('sum_abs_z')
            # Snapshot the mode so the correct position columns are written for the
            # entire recording session even if the user switches modes mid-session.
            self._record_max_cops = 5 if self.heatmap_mode else 0
            self._record_mode_snapshot = 'heatmap' if self.heatmap_mode else 'netpos'

            # Open one CSV file per active channel and write the shared header row.
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
        # Called every poll tick while recording is active. Writes one row per
        # channel containing elapsed time, raw XYZ, diff XYZ, derivative norms,
        # position data (CoPs or Net Pos), and the sum of absolute Z values.
        elapsed = time.monotonic() - self._record_start

        for ch, writer in self._record_writers.items():
            row = [f'{elapsed:.4f}']

            # Raw magnetometer readings for each sensor position on this channel.
            for key in SENSOR_KEYS:
                rv = self.reader.raw_values.get((ch, key), [float('nan')] * 3)
                for v in rv:
                    row.append(f'{float(v):.4f}')

            # Differential (baseline-subtracted, low-pass-filtered) readings.
            for key in SENSOR_KEYS:
                dv = self.reader.diff_values.get((ch, key), [float('nan')] * 3)
                for v in dv:
                    row.append(f'{float(v):.4f}')

            # Per-sensor derivative: L2 norm of the change in diff_values since
            # the previous poll. NaN if no previous value is available.
            for key in SENSOR_KEYS:
                k = (ch, key)
                dv  = self.reader.diff_values.get(k)
                prv = self._az_prev_diff.get(k)
                if dv is not None and prv is not None:
                    deriv = float(np.linalg.norm(dv - prv))
                else:
                    deriv = float('nan')
                row.append(f'{deriv:.4f}')

            # Position columns: write up to MAX_COPS CoP pairs (padded with NaN)
            # in heatmap mode, or a single Net Pos pair in vector mode.
            if self._record_mode_snapshot == 'heatmap':
                cops = self._cop_positions
                for i in range(self._record_max_cops):
                    if i < len(cops):
                        row += [f'{cops[i][0]:.4f}', f'{cops[i][1]:.4f}']
                    else:
                        row += ['nan', 'nan']
            else:
                net_x, net_y = self._net_pos if self._net_pos else (float('nan'), float('nan'))
                row += [f'{net_x:.4f}', f'{net_y:.4f}']

            # Σ|Z|: sum of absolute Z diff values across all active sensors
            _sum_z = sum(
                abs(float(self.reader.diff_values[(ch2, key)][2]))
                for ch2 in self.reader.active_channels
                for key in SENSOR_KEYS
                if (ch2, key) in self.reader.diff_values
            )
            row.append(f'{_sum_z:.4f}')

            writer.writerow(row)

    def _toggle_snapshot_csv(self):
        # Toggles whether the Snapshot button also writes a CSV and PNG image to disk
        # (in addition to updating the on-screen overlay). Updates the button label.
        self.snapshot_save_csv = not self.snapshot_save_csv
        if self.snapshot_save_csv:
            self.btn_snap_csv.config(text='CSV: ON', bg='#1b6b3a')
        else:
            self.btn_snap_csv.config(text='CSV: OFF', bg='#3a3a3a')

    def _take_snapshot(self):
        """Compute Net Pos and all CoPs from the current filtered diff data,
        independently of whichever display mode is active."""
        import csv, datetime

        active = sorted(self.reader.active_channels)
        if not active:
            self.status_var.set('Snapshot: no active channels.')
            return

        try:
            cutoff = float(self.cutoff_var.get())
        except ValueError:
            cutoff = 0.0

        # ── 1. Net Pos ─────────────────────────────────────────────────────────
        # Compute the 3D-magnitude-weighted centroid of all active sensor positions,
        # independent of the current canvas display mode.
        net_sum   = 0.0
        net_sum_x = 0.0
        net_sum_y = 0.0

        for ch in active:
            for key in SENSOR_KEYS:
                k = (ch, key)
                vec = self.reader.diff_values.get(k)
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
                net_sum   += mag
                net_sum_x += mag * wx
                net_sum_y += mag * wy

        net_pos = (net_sum_x / net_sum, net_sum_y / net_sum) if net_sum > 1e-6 else None

        # ── 2. CoP via Local Maxima on a fresh RBF field ───────────────────────
        # Re-runs the full Gaussian RBF heatmap + peak-finding algorithm on the
        # current sensor data at the snapshot resolution (SNAP_GRID), regardless
        # of which visualization mode is currently displayed on the canvas.
        SNAP_GRID = 60
        cw    = self.canvas.winfo_width()
        ch_px = self.canvas.winfo_height()

        # Collect canvas-pixel positions and 3D-magnitude weights for each sensor.
        positions, weights = [], []
        for ch in active:
            for key in SENSOR_KEYS:
                k = (ch, key)
                vec = self.reader.diff_values.get(k)
                if vec is None:
                    continue
                dx, dy, dz = float(vec[0]), float(vec[1]), float(vec[2])
                if abs(dx) <= cutoff: dx = 0.0
                if abs(dy) <= cutoff: dy = 0.0
                if abs(dz) <= cutoff: dz = 0.0
                w = float(np.linalg.norm([dx, dy, dz]))
                if w < 1e-6:
                    continue
                try:
                    wx = float(self.sensor_pos_vars[(ch, key, 'x')].get())
                    wy = float(self.sensor_pos_vars[(ch, key, 'y')].get())
                except (ValueError, KeyError):
                    continue
                cx, cy = self._w2c(wx, wy)
                positions.append((cx, cy))
                weights.append(w)

        cop_positions = []
        if positions and cw > 10 and ch_px > 10:
            # Build the RBF field at snapshot resolution using the same sigma as the
            # live heatmap, then find and refine local maxima for CoP positions.
            sigma = (self.rbf_sigma_pct.get() / 100.0) * np.hypot(cw, ch_px)
            xs = np.linspace(0, cw, SNAP_GRID)
            ys = np.linspace(0, ch_px, SNAP_GRID)
            gx, gy = np.meshgrid(xs, ys)
            field = np.zeros((SNAP_GRID, SNAP_GRID))
            for (px, py), sw in zip(positions, weights):
                dist2 = (gx - px) ** 2 + (gy - py) ** 2
                field += sw * np.exp(-dist2 / (2 * sigma ** 2))
            f_max = field.max()
            if f_max > 1e-9:
                field_norm = field / f_max
                cell_w = cw    / SNAP_GRID
                cell_h = ch_px / SNAP_GRID
                cw_half = cw    / 2
                ch_half = ch_px / 2

                padded = np.pad(field_norm, 1, mode='constant', constant_values=0)
                peak_cells = []
                for r in range(SNAP_GRID):
                    for c in range(SNAP_GRID):
                        v = field_norm[r, c]
                        if v < 0.15:
                            continue
                        if v == padded[r:r+3, c:c+3].max():
                            peak_cells.append((r, c, v))

                peak_cells.sort(key=lambda p: p[2], reverse=True)
                merged_cells = []
                for r, c, v in peak_cells:
                    if not any(abs(r - mr) <= 3 and abs(c - mc) <= 3
                               for mr, mc, _ in merged_cells):
                        merged_cells.append((r, c, v))

                for r, c, _ in merged_cells:
                    r0, r1 = max(0, r-1), min(SNAP_GRID-1, r+1)
                    c0, c1 = max(0, c-1), min(SNAP_GRID-1, c+1)
                    patch = field_norm[r0:r1+1, c0:c1+1]
                    total = patch.sum()
                    if total < 1e-12:
                        can_x = c * cell_w + cell_w / 2
                        can_y = r * cell_h + cell_h / 2
                    else:
                        col_grid, row_grid = np.meshgrid(
                            np.arange(c0, c1+1), np.arange(r0, r1+1))
                        can_x = (col_grid * patch).sum() / total * cell_w + cell_w / 2
                        can_y = (row_grid * patch).sum() / total * cell_h + cell_h / 2
                    if self._scale > 0:
                        wx = (can_x - cw_half) / self._scale + self._origin[0]
                        wy = (ch_half - can_y) / self._scale + self._origin[1]
                    else:
                        wx, wy = 0.0, 0.0
                    cop_positions.append((wx, wy))

        # ── 3. Build display text ──────────────────────────────────────────────
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mode_label = 'Heatmap' if self.heatmap_mode else 'Net Pos'
        lines = [f'Snapshot  {timestamp}', f'Mode: {mode_label}']
        if self.heatmap_mode:
            sigma_val = self.rbf_sigma_pct.get()
            lines.append(f'σ = {sigma_val:.1f}% diag')
        if net_pos:
            lines.append(f'NetPos  X={net_pos[0]:+.4f}  Y={net_pos[1]:+.4f}')
        else:
            lines.append('NetPos  —')
        if cop_positions:
            for i, (wx, wy) in enumerate(cop_positions):
                lbl = f'CoP{i+1}' if len(cop_positions) > 1 else 'CoP '
                lines.append(f'{lbl}    X={wx:+.4f}  Y={wy:+.4f}')
        else:
            lines.append('CoP     —')

        overlay_text = '\n'.join(lines)
        self.snapshot_overlay_var.set(overlay_text)
        print(overlay_text)

        # ── 4. Optionally save CSV + screenshot ───────────────────────────────
        snap_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.snapshot_save_csv:
            fname = f'snapshot_{snap_ts}.csv'
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                header_row = ['type', 'index', 'x', 'y']
                if self.heatmap_mode:
                    header_row.append('sigma_pct')
                header_row.append('sum_abs_z')
                writer.writerow(header_row)
                # Compute Σ|Z| once for this snapshot
                _snap_sum_z = sum(
                    abs(float(self.reader.diff_values[(ch, key)][2]))
                    for ch in active
                    for key in SENSOR_KEYS
                    if (ch, key) in self.reader.diff_values
                )
                if net_pos:
                    row = ['net_pos', 1, f'{net_pos[0]:.6f}', f'{net_pos[1]:.6f}']
                    if self.heatmap_mode:
                        row.append(f'{self.rbf_sigma_pct.get():.2f}')
                    row.append(f'{_snap_sum_z:.4f}')
                    writer.writerow(row)
                for i, (wx, wy) in enumerate(cop_positions):
                    row = ['cop', i + 1, f'{wx:.6f}', f'{wy:.6f}']
                    if self.heatmap_mode:
                        row.append(f'{self.rbf_sigma_pct.get():.2f}')
                    row.append(f'{_snap_sum_z:.4f}')
                    writer.writerow(row)

            # ── Render image directly from snapshot data (no screen capture) ──
            # Uses Pillow (PIL) to draw an 800x800 pixel image reproducing the current
            # canvas contents (heatmap or vector field) at higher resolution. Falls back
            # gracefully if Pillow is not installed, leaving only the CSV output.
            try:
                from PIL import Image, ImageDraw

                IMG_SIZE = 800   # output image resolution
                iw = ih = IMG_SIZE
                bg_color = (26, 26, 46)   # matches BG_CANVAS #1a1a2e

                img = Image.new('RGB', (iw, ih), bg_color)
                draw = ImageDraw.Draw(img)

                # World-to-image coordinate transform, mirroring _w2c but for the
                # fixed-size output image rather than the live canvas.
                def w2i(wx, wy):
                    px = iw / 2 + (wx - self._origin[0]) * self._scale
                    py = ih / 2 - (wy - self._origin[1]) * self._scale
                    return px, py

                # Draw grid axes
                ox, oy = w2i(0, 0)
                if 0 <= ox <= iw:
                    draw.line([(ox, 0), (ox, ih)], fill=(42, 42, 74), width=1)
                if 0 <= oy <= ih:
                    draw.line([(0, oy), (iw, oy)], fill=(42, 42, 74), width=1)

                # Helper to convert a '#rrggbb' hex string to an (R, G, B) tuple
                # for Pillow draw calls.
                def _hex_to_rgb(h):
                    h = h.lstrip('#')
                    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

                # RGB equivalents of the CoP crosshair colors defined elsewhere.
                COP_COLORS_RGB = [
                    (0, 229, 255), (255, 214, 0), (255, 109, 0),
                    (105, 240, 174), (234, 128, 252)
                ]

                if self.heatmap_mode:
                    # ── Render Gaussian heatmap from positions/weights ──────────
                    # Recomputes the RBF field at full image resolution (100x100 grid)
                    # by converting the stored canvas-pixel sensor positions to world
                    # coordinates and then to image-pixel coordinates.
                    if positions:
                        GRID = 100
                        xs_g = np.linspace(0, iw, GRID)
                        ys_g = np.linspace(0, ih, GRID)
                        gx, gy = np.meshgrid(xs_g, ys_g)
                        sigma_i = (self.rbf_sigma_pct.get() / 100.0) * np.hypot(iw, ih)
                        # Re-map canvas positions → image positions
                        cw_c = self.canvas.winfo_width()
                        ch_c = self.canvas.winfo_height()
                        field_i = np.zeros((GRID, GRID))
                        for (cpx, cpy), sw in zip(positions, weights):
                            # canvas coords → world → image coords
                            wpx = (cpx - cw_c / 2) / self._scale + self._origin[0]
                            wpy = (ch_c / 2 - cpy) / self._scale + self._origin[1]
                            ipx, ipy = w2i(wpx, wpy)
                            dist2 = (gx - ipx) ** 2 + (gy - ipy) ** 2
                            field_i += sw * np.exp(-dist2 / (2 * sigma_i ** 2))
                        f_max = field_i.max()
                        if f_max > 1e-9:
                            field_i /= f_max
                            cell_w_i = iw / GRID
                            cell_h_i = ih / GRID
                            pix = img.load()
                            # Paint each non-zero grid cell with the same 4-stop
                            # blue → cyan → green → yellow → red gradient used on
                            # the live canvas, writing directly into the pixel buffer.
                            for row in range(GRID):
                                for col in range(GRID):
                                    v = float(field_i[row, col])
                                    if v < 0.04:
                                        continue
                                    v = max(0.0, min(1.0, v))
                                    if v < 0.25:
                                        t = v / 0.25
                                        r2, g2, b2 = 0, int(255*t), 255
                                    elif v < 0.5:
                                        t = (v - 0.25) / 0.25
                                        r2, g2, b2 = 0, 255, int(255*(1-t))
                                    elif v < 0.75:
                                        t = (v - 0.5) / 0.25
                                        r2, g2, b2 = int(255*t), 255, 0
                                    else:
                                        t = (v - 0.75) / 0.25
                                        r2, g2, b2 = 255, int(255*(1-t)), 0
                                    x0i = int(col * cell_w_i)
                                    y0i = int(row * cell_h_i)
                                    x1i = int(x0i + cell_w_i) + 1
                                    y1i = int(y0i + cell_h_i) + 1
                                    for py2 in range(y0i, min(y1i, ih)):
                                        for px2 in range(x0i, min(x1i, iw)):
                                            pix[px2, py2] = (r2, g2, b2)

                    # Draw sensor dots
                    for ch_id in active:
                        color_rgb = _hex_to_rgb(CH_COLORS[ch_id])
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

                    # Draw CoP crosshairs
                    for i, (wx2, wy2) in enumerate(cop_positions):
                        ipx, ipy = w2i(wx2, wy2)
                        tc = COP_COLORS_RGB[i % len(COP_COLORS_RGB)]
                        cr2 = 10
                        draw.ellipse([ipx-cr2, ipy-cr2, ipx+cr2, ipy+cr2],
                                     outline=tc, width=2)
                        draw.line([(ipx-cr2-5, ipy), (ipx+cr2+5, ipy)], fill=tc, width=2)
                        draw.line([(ipx, ipy-cr2-5), (ipx, ipy+cr2+5)], fill=tc, width=2)
                        lbl2 = f'CoP{i+1}' if len(cop_positions) > 1 else 'CoP'
                        draw.text((ipx+cr2+7, ipy-cr2-7),
                                  f'{lbl2} ({wx2:.1f},{wy2:.1f})',
                                  fill=tc)

                else:
                    # ── Render Net Pos vector field ─────────────────────────────
                    for ch_id in active:
                        color_rgb = _hex_to_rgb(CH_COLORS[ch_id])
                        try:
                            rot_deg = float(self.rot_vars[ch_id].get())
                        except (ValueError, KeyError):
                            rot_deg = 0.0
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

                            k = (ch_id, key)
                            vec = self.reader.diff_values.get(k)
                            if vec is None:
                                continue
                            dx2, dy2, dz2 = float(vec[0]), float(vec[1]), float(vec[2])
                            if abs(dx2) <= cutoff: dx2 = 0.0
                            if abs(dy2) <= cutoff: dy2 = 0.0
                            if abs(dz2) <= cutoff: dz2 = 0.0
                            mag_2d = np.hypot(dx2, dy2)
                            if mag_2d < 1e-6:
                                continue
                            ux, uy = dx2 / mag_2d, dy2 / mag_2d
                            ux2r = ux * np.cos(rot_rad) + uy * np.sin(rot_rad)
                            uy2r = -ux * np.sin(rot_rad) + uy * np.cos(rot_rad)
                            tx = ux2r * ARROW_HALF
                            ty = -uy2r * ARROW_HALF
                            ax1, ay1 = ipx - tx, ipy - ty
                            ax2, ay2 = ipx + tx, ipy + ty
                            draw.line([(ax1, ay1), (ax2, ay2)], fill=color_rgb, width=2)

                    # Net Pos crosshair
                    if net_pos:
                        ipx, ipy = w2i(net_pos[0], net_pos[1])
                        tc = (0, 229, 255)
                        cr2 = 10
                        draw.ellipse([ipx-cr2, ipy-cr2, ipx+cr2, ipy+cr2],
                                     outline=tc, width=2)
                        draw.line([(ipx-cr2-5, ipy), (ipx+cr2+5, ipy)], fill=tc, width=2)
                        draw.line([(ipx, ipy-cr2-5), (ipx, ipy+cr2+5)], fill=tc, width=2)
                        draw.text((ipx+cr2+7, ipy-cr2-7),
                                  f'NetPos ({net_pos[0]:.2f},{net_pos[1]:.2f})',
                                  fill=tc)

                # Timestamp + mode label in top-left — white with black outline
                _stamp = (f'{mode_label}' +
                          (f'  |  σ={self.rbf_sigma_pct.get():.1f}%' if self.heatmap_mode else ''))
                for _dx, _dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
                    draw.text((8 + _dx, 8 + _dy), _stamp, fill=(0, 0, 0))
                draw.text((8, 8), _stamp, fill=(255, 255, 255))

                img_fname = f'snapshot_{snap_ts}_{mode_label.replace(" ", "_")}.png'
                img.save(img_fname)
                self.status_var.set(f'Snapshot saved → {fname} + {img_fname}')
            except Exception as e:
                import traceback; traceback.print_exc()
                self.status_var.set(f'Snapshot saved → {fname}  (image failed: {e})')
        else:
            self.status_var.set(
                f'Snapshot | NetPos: {"({:.2f},{:.2f})".format(*net_pos) if net_pos else "—"}'
                f' | CoPs: {len(cop_positions)}'
            )

    def _zero_all(self):
        # Zeros all active sensor channels and updates the status bar to list the
        # channels that were zeroed.
        zeroed = self.reader.zero_sensors()
        active = sorted(self.reader.active_channels)
        if active:
            self.status_var.set(f'Zeroed. Active: {active}')
        else:
            self.status_var.set('Zeroed. No active channels detected yet.')

    def _toggle_raw_mode(self):
        # Switches the data table and canvas between showing differential readings
        # (relative to the last zero) and raw absolute magnetometer readings.
        # Updates the column header labels and the title row text accordingly.
        self.raw_mode = not self.raw_mode
        if self.raw_mode:
            self.btn_raw_mode.config(text='Show Diff Readings', bg='#3b5e7a')
            # Update column headers in the data table to "Raw X/Y/Z"
            for lbl in self._data_hdr_labels:
                lbl.config(text=lbl._raw_text)
            self._hdr_title_label.config(text='Live Data (Raw)')
            self._avg_diff_hdr_label.config(text='Range (Min/Max/Span)')
        else:
            self.btn_raw_mode.config(text='Show Raw Readings', bg='#7a5c1e')
            for lbl in self._data_hdr_labels:
                lbl.config(text=lbl._diff_text)
            self._hdr_title_label.config(text='Live Data')
            self._avg_diff_hdr_label.config(text='Avg Diff / Deriv')

    def _toggle_auto_zero(self):
        # Cycles through the three auto-zero modes: off -> individual -> average -> off.
        # In 'individual' mode each sensor is zeroed independently when it has settled.
        # In 'average' mode all sensors are zeroed together when the average reading
        # has settled. Clears tracking state whenever the mode changes.
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
        # Shows or hides the auto-zero settings table (per-sensor cutoffs, derivative
        # threshold, settling time). Exits edit mode first if it is currently active.
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
        # Toggles between the edit-positions table and whichever view was previously
        # shown (live data or zero settings). Updates the button label so it always
        # describes the destination, not the current state.
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

    def _toggle_viz_mode(self):
        # Switches the canvas visualization between the vector field (Net Pos) and
        # the Gaussian RBF heatmap. Swaps the header bar (legend vs sigma slider)
        # and shows or hides the sigma overlay label on the canvas.
        self.heatmap_mode = not self.heatmap_mode
        if self.heatmap_mode:
            self.btn_toggle_viz.config(text='Show Net Pos', bg='#7a3b3b')
            self.bar_vector.pack_forget()
            self.bar_heatmap.pack(fill=tk.X, pady=(0, 4), before=self.canvas_frame)
            # Show Gaussian sigma overlay in top-right of canvas
            self._sigma_overlay_var.set(f'σ = {self.rbf_sigma_pct.get():.1f}% diag')
            self._sigma_overlay_lbl.place(relx=1.0, rely=0.0, anchor='ne')
        else:
            self.btn_toggle_viz.config(text='Show Heatmap', bg='#4a6741')
            self.bar_heatmap.pack_forget()
            self.bar_vector.pack(fill=tk.X, pady=(0, 4), before=self.canvas_frame)
            # Hide sigma overlay
            self._sigma_overlay_lbl.place_forget()

    # ─── poll ─────────────────────────────────────────────────────────────────

    def _poll(self):
        # Main update loop, scheduled every 50 ms via root.after().
        # On each tick it reads new serial data, updates the status bar, refreshes
        # every cell in the data table, runs the auto-zero algorithm, redraws the
        # canvas, and writes a CSV row if recording is active.
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

            # Update sum of absolute Z values across all active sensors
            _sum_abs_z = 0.0
            for ch in active:
                for key in SENSOR_KEYS:
                    k = (ch, key)
                    src = self.reader.raw_values if self.raw_mode else self.reader.diff_values
                    vec = src.get(k)
                    if vec is not None:
                        _sum_abs_z += abs(float(vec[2]))
            self.sum_abs_z_var.set(f'\u03a3|Z|: {_sum_abs_z:.1f}')

            # Update row 8 display: range (raw mode) or avg diff/deriv (diff mode)
            if self.raw_mode:
                _raw_norms = []
                for ch in active:
                    for key in SENSOR_KEYS:
                        k = (ch, key)
                        rv = self.reader.raw_values.get(k)
                        if rv is not None:
                            _raw_norms.append(float(np.linalg.norm(rv)))
                if _raw_norms:
                    rng_min  = min(_raw_norms)
                    rng_max  = max(_raw_norms)
                    rng_span = rng_max - rng_min
                    self.avg_diff_var.set(
                        f'Min {rng_min:.2f}  |  Max {rng_max:.2f}  |  Span {rng_span:.2f}'
                    )
                else:
                    self.avg_diff_var.set('— / — / —')
            else:
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
                    if self.heatmap_mode:
                        cops = self._cop_positions
                        if cops:
                            cop_str = '  |  ' + '  '.join(
                                f'CoP{i+1}({p[0]:.2f},{p[1]:.2f})' if len(cops) > 1
                                else f'CoP({p[0]:.2f},{p[1]:.2f})'
                                for i, p in enumerate(cops)
                            )
                        else:
                            cop_str = '  |  CoP (—, —)'
                        self.avg_diff_var.set(f'Diff {avg_d:.2f}  |  Deriv {avg_r:.2f}{cop_str}')
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

                            # Compute the frame-to-frame change (derivative) for this sensor.
                            # If no previous value exists, treat the derivative as infinite
                            # so the sensor is not zeroed until at least two frames have passed.
                            prev = self._az_prev_diff.get(k)
                            if prev is not None:
                                deriv = float(np.linalg.norm(diff_vec - prev))
                            else:
                                deriv = float('inf')
                            self._az_prev_diff[k] = diff_vec.copy()
                            self._az_deriv[k] = deriv

                            reading_norm = float(np.linalg.norm(diff_vec))
                            cond_deriv  = deriv < deriv_thresh    # signal is stable
                            cond_cutoff = reading_norm < az_cutoff  # signal is close to zero

                            # Both conditions must hold continuously for settling_time seconds
                            # before the sensor is zeroed. Any instability resets the timer.
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

                        # Zero all sensors together only when the whole array has
                        # settled: average reading below the average cutoff threshold
                        # and average derivative below the global derivative threshold.
                        if avg_reading < avg_cutoff and avg_deriv < deriv_thresh:
                            if 'avg' not in self._az_settle_start:
                                self._az_settle_start['avg'] = now
                            elif now - self._az_settle_start['avg'] >= settling_time:
                                self.reader.zero_sensors()
                                self._az_settle_start.pop('avg', None)
                                self._az_prev_diff.clear()
                        else:
                            self._az_settle_start.pop('avg', None)

        # Redraw the canvas and (if recording) append a CSV row, then reschedule.
        self._draw_grid()
        self._draw_vectors()

        if self.is_recording:
            self._write_csv_rows()

        self.root.after(50, self._poll)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Entry point: configure the serial port, instantiate the reader and the GUI,
    # then start the tkinter event loop.
    PORT = 'COM4'   # <- change to your port

    reader = ArduinoDataReader(PORT)

    root = tk.Tk()
    app  = SensorTableApp(root, reader)
    root.mainloop()