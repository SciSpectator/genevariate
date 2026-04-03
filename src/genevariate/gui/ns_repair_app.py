"""
GeneVariate -- NS Repair GUI Application

Tkinter-based control panel for the NS (Not Specified) repair pipeline.
Dark-themed interface with:
  - Hardware info panel (CPU cores, RAM, GPU VRAM)
  - Separate GPU / CPU worker spinboxes
  - Per-column progress bars (Tissue, Condition, Treatment)
  - Latency, ETA, GSE progress, watchdog status
  - Memory-efficient LLM agent monitoring
"""

import os
import time
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from datetime import datetime, timedelta

import psutil

from genevariate.core.ns_repair_pipeline import pipeline, ALL_GPLS
from genevariate.core.ollama_manager import (
    DEFAULT_MODEL, DEFAULT_URL, MODEL_RAM_GB, DEFAULT_MODEL_GB,
    detect_gpus, compute_ollama_parallel,
    ollama_server_ok, model_available, ollama_binary_exists,
    kill_ollama, check_ollama_gpu,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Dark Theme Colour Palette
# ═══════════════════════════════════════════════════════════════════════════════

BG = "#1e1e2e"; BG2 = "#2a2a3e"; BG3 = "#313145"
ACCENT = "#7c5cbf"; ACCENT2 = "#5c9fd4"
SUCCESS = "#4caf76"; WARNING = "#e0a84a"; ERROR = "#e05c5c"
FG = "#e0e0f0"; FG2 = "#a0a0c0"

REPAIR_COLS = ("Tissue", "Condition", "Treatment")


# ═══════════════════════════════════════════════════════════════════════════════
#  NSRepairApp
# ═══════════════════════════════════════════════════════════════════════════════

class NSRepairApp(tk.Tk):
    """Main NS-repair controller window with hardware-aware worker management."""

    def __init__(self):
        super().__init__()
        self.title("GeneVariate -- NS Repair Pipeline")
        self.configure(bg=BG)
        self.minsize(1160, 800)
        self.geometry("1300x860")

        # shared state
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._stop_event = threading.Event()
        self._worker_thread = None

        # progress counters
        self._col_progress = {c: (0, 0) for c in REPAIR_COLS}
        self._total_done = 0
        self._total_count = 0
        self._start_time = 0.0

        # hardware info (populated by _detect_workers)
        self._hw_gpus = []
        self._hw_cpu_count = os.cpu_count() or 1
        self._hw_ram_gb = psutil.virtual_memory().total / 1e9

        # tk variables
        self._var_dir = tk.StringVar(value="")
        self._var_gpl = tk.StringVar(value="GPL570")
        self._var_limit = tk.StringVar(value="")
        self._var_mode = tk.StringVar(value="repair")
        self._var_gsm_file = tk.StringVar(value="")
        self._var_gpu_workers = tk.IntVar(value=1)
        self._var_cpu_workers = tk.IntVar(value=2)
        self._var_model = tk.StringVar(value=DEFAULT_MODEL)
        self._var_url = tk.StringVar(value=DEFAULT_URL)

        self._build_ui()
        self._apply_theme()
        self._check_env_async()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._poll_queue)
        self.after(600, lambda: self._detect_workers(silent=True))

    # ── ttk theme ─────────────────────────────────────────────────────────

    def _apply_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=FG, fieldbackground=BG2,
                         borderwidth=0, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=BG2)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Card.TLabel", background=BG2, foreground=FG)
        style.configure("Dim.TLabel", background=BG2, foreground=FG2,
                         font=("Segoe UI", 9))
        style.configure("HW.TLabel", background=BG2, foreground=ACCENT2,
                         font=("Consolas", 9))
        style.configure("Badge.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("TRadiobutton", background=BG2, foreground=FG,
                         indicatorcolor=ACCENT)
        style.map("TRadiobutton",
                  background=[("active", BG3)],
                  indicatorcolor=[("selected", ACCENT)])
        style.configure("TEntry", fieldbackground=BG3, foreground=FG,
                         insertcolor=FG)
        style.configure("TButton", background=ACCENT, foreground="#ffffff",
                         padding=(12, 6), font=("Segoe UI", 10, "bold"))
        style.map("TButton",
                  background=[("active", "#6b4dab"), ("disabled", BG3)])
        style.configure("Stop.TButton", background=ERROR, foreground="#ffffff")
        style.map("Stop.TButton",
                  background=[("active", "#c94444"), ("disabled", BG3)])
        style.configure("Green.Horizontal.TProgressbar",
                         troughcolor=BG3, background=SUCCESS)
        style.configure("Accent.Horizontal.TProgressbar",
                         troughcolor=BG3, background=ACCENT)
        style.configure("TLabelframe", background=BG2, foreground=ACCENT2)
        style.configure("TLabelframe.Label", background=BG2, foreground=ACCENT2,
                         font=("Segoe UI", 10, "bold"))

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)

        # ── LEFT PANEL ────────────────────────────────────────────────

        # 1. Data card
        card_data = ttk.LabelFrame(left, text="  Data  ", padding=10)
        card_data.pack(fill="x", pady=(0, 6))

        ttk.Label(card_data, text="Harmonized Labels Folder:",
                  style="Card.TLabel").pack(anchor="w")
        frm_dir = ttk.Frame(card_data, style="Card.TFrame")
        frm_dir.pack(fill="x", pady=(2, 6))
        ttk.Entry(frm_dir, textvariable=self._var_dir, width=36
                  ).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(frm_dir, text="Browse", command=self._browse_dir
                   ).pack(side="right")

        frm_geo = ttk.Frame(card_data, style="Card.TFrame")
        frm_geo.pack(fill="x")
        ttk.Label(frm_geo, text="GEOmetadb:", style="Dim.TLabel").pack(side="left")
        self._lbl_geodb = ttk.Label(frm_geo, text="...", style="Badge.TLabel")
        self._lbl_geodb.pack(side="left", padx=6)

        # 2. Platform card
        card_plat = ttk.LabelFrame(left, text="  Platform  ", padding=10)
        card_plat.pack(fill="x", pady=(0, 6))
        for gpl in ALL_GPLS:
            ttk.Radiobutton(card_plat, text=gpl, variable=self._var_gpl,
                            value=gpl).pack(anchor="w", pady=1)
        self._lbl_files = ttk.Label(card_plat, text="", style="Dim.TLabel")
        self._lbl_files.pack(anchor="w", pady=(4, 0))

        # 3. Hardware Info card
        card_hw = ttk.LabelFrame(left, text="  Hardware  ", padding=10)
        card_hw.pack(fill="x", pady=(0, 6))

        self._lbl_hw_cpu = ttk.Label(card_hw, text="CPU: scanning...",
                                     style="HW.TLabel")
        self._lbl_hw_cpu.pack(anchor="w")
        self._lbl_hw_ram = ttk.Label(card_hw, text="RAM: scanning...",
                                     style="HW.TLabel")
        self._lbl_hw_ram.pack(anchor="w")
        self._lbl_hw_gpu = ttk.Label(card_hw, text="GPU: scanning...",
                                     style="HW.TLabel")
        self._lbl_hw_gpu.pack(anchor="w")
        self._lbl_hw_vram = ttk.Label(card_hw, text="VRAM: --",
                                      style="HW.TLabel")
        self._lbl_hw_vram.pack(anchor="w")

        # Separator
        ttk.Separator(card_hw, orient="horizontal").pack(fill="x", pady=6)

        # GPU workers
        frm_gpu_w = ttk.Frame(card_hw, style="Card.TFrame")
        frm_gpu_w.pack(fill="x", pady=2)
        ttk.Label(frm_gpu_w, text="GPU Workers:", style="Card.TLabel"
                  ).pack(side="left")
        self._spn_gpu = tk.Spinbox(
            frm_gpu_w, from_=0, to=16, width=4,
            textvariable=self._var_gpu_workers,
            bg=BG3, fg=FG, buttonbackground=BG3,
            insertbackground=FG, highlightthickness=0)
        self._spn_gpu.pack(side="right")

        # CPU workers
        frm_cpu_w = ttk.Frame(card_hw, style="Card.TFrame")
        frm_cpu_w.pack(fill="x", pady=2)
        ttk.Label(frm_cpu_w, text="CPU Workers:", style="Card.TLabel"
                  ).pack(side="left")
        self._spn_cpu = tk.Spinbox(
            frm_cpu_w, from_=0, to=64, width=4,
            textvariable=self._var_cpu_workers,
            bg=BG3, fg=FG, buttonbackground=BG3,
            insertbackground=FG, highlightthickness=0)
        self._spn_cpu.pack(side="right")

        # Total display + auto-detect
        frm_total_w = ttk.Frame(card_hw, style="Card.TFrame")
        frm_total_w.pack(fill="x", pady=(4, 2))
        self._lbl_total_workers = ttk.Label(
            frm_total_w, text="Total: --", style="Badge.TLabel",
            foreground=ACCENT2)
        self._lbl_total_workers.pack(side="left")
        ttk.Button(frm_total_w, text="Auto-Detect", width=10,
                   command=self._detect_workers).pack(side="right")

        self._lbl_hw_note = ttk.Label(
            card_hw,
            text="GPU workers run on VRAM (fast). CPU workers\n"
                 "run on system RAM when VRAM is full.",
            style="Dim.TLabel", justify="left")
        self._lbl_hw_note.pack(anchor="w", pady=(4, 0))

        # Update total when spinboxes change
        def _update_total(*_):
            g = self._var_gpu_workers.get()
            c = self._var_cpu_workers.get()
            self._lbl_total_workers.configure(text=f"Total: {g + c}")
        self._var_gpu_workers.trace_add("write", _update_total)
        self._var_cpu_workers.trace_add("write", _update_total)

        # 4. Options card
        card_opts = ttk.LabelFrame(left, text="  Options  ", padding=10)
        card_opts.pack(fill="x", pady=(0, 6))

        frm_limit = ttk.Frame(card_opts, style="Card.TFrame")
        frm_limit.pack(fill="x", pady=2)
        ttk.Label(frm_limit, text="Test Limit (0 = all):",
                  style="Card.TLabel").pack(side="left")
        ttk.Entry(frm_limit, textvariable=self._var_limit,
                  width=8).pack(side="right")

        frm_mode = ttk.Frame(card_opts, style="Card.TFrame")
        frm_mode.pack(fill="x", pady=4)
        ttk.Label(frm_mode, text="Mode:", style="Card.TLabel").pack(side="left")
        for val, txt in (("repair", "Repair NS"), ("scratch", "From Scratch")):
            ttk.Radiobutton(frm_mode, text=txt, variable=self._var_mode,
                            value=val).pack(side="left", padx=(8, 0))

        frm_gsm = ttk.Frame(card_opts, style="Card.TFrame")
        frm_gsm.pack(fill="x", pady=2)
        ttk.Label(frm_gsm, text="GSM List (opt):", style="Card.TLabel"
                  ).pack(anchor="w")
        frm_gsm2 = ttk.Frame(card_opts, style="Card.TFrame")
        frm_gsm2.pack(fill="x", pady=(0, 2))
        ttk.Entry(frm_gsm2, textvariable=self._var_gsm_file,
                  width=28).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(frm_gsm2, text="...", width=3,
                   command=self._browse_gsm_file).pack(side="right")

        # 5. Model card
        card_model = ttk.LabelFrame(left, text="  Model  ", padding=10)
        card_model.pack(fill="x", pady=(0, 6))

        frm_mn = ttk.Frame(card_model, style="Card.TFrame")
        frm_mn.pack(fill="x", pady=2)
        ttk.Label(frm_mn, text="Name:", style="Card.TLabel").pack(side="left")
        ttk.Entry(frm_mn, textvariable=self._var_model, width=20
                  ).pack(side="right")

        frm_mu = ttk.Frame(card_model, style="Card.TFrame")
        frm_mu.pack(fill="x", pady=2)
        ttk.Label(frm_mu, text="Ollama URL:", style="Card.TLabel").pack(side="left")
        ttk.Entry(frm_mu, textvariable=self._var_url, width=20
                  ).pack(side="right")

        frm_badges = ttk.Frame(card_model, style="Card.TFrame")
        frm_badges.pack(fill="x", pady=(4, 0))
        for lbl_txt, attr in [("Ollama:", "_lbl_ollama"),
                               ("Model:", "_lbl_model"),
                               ("GPU:", "_lbl_gpu")]:
            ttk.Label(frm_badges, text=lbl_txt, style="Dim.TLabel"
                      ).pack(side="left")
            w = ttk.Label(frm_badges, text="...", style="Badge.TLabel")
            w.pack(side="left", padx=(3, 8))
            setattr(self, attr, w)

        # 6. Action buttons
        frm_btns = ttk.Frame(left)
        frm_btns.pack(fill="x", pady=(8, 0))
        self._btn_start = ttk.Button(frm_btns, text="Start Repair",
                                     command=self._start)
        self._btn_start.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._btn_stop = ttk.Button(frm_btns, text="Stop",
                                    command=self._stop, style="Stop.TButton",
                                    state="disabled")
        self._btn_stop.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # ── RIGHT PANEL ───────────────────────────────────────────────

        # Main progress bar
        frm_prog = ttk.Frame(right)
        frm_prog.pack(fill="x", pady=(0, 4))
        self._pbar = ttk.Progressbar(frm_prog, mode="determinate",
                                     style="Green.Horizontal.TProgressbar")
        self._pbar.pack(fill="x", pady=(0, 2))
        self._lbl_progress = ttk.Label(frm_prog, text="Idle",
                                       font=("Segoe UI", 10))
        self._lbl_progress.pack(anchor="w")

        # Per-column progress section
        frm_col_prog = ttk.LabelFrame(right, text="  Per-Field Progress  ",
                                       padding=8)
        frm_col_prog.pack(fill="x", pady=(0, 4))

        self._col_labels = {}
        self._col_bars = {}
        self._col_pct_labels = {}
        for i, col in enumerate(REPAIR_COLS):
            frm_row = ttk.Frame(frm_col_prog, style="Card.TFrame")
            frm_row.pack(fill="x", pady=2)

            lbl_name = ttk.Label(frm_row, text=f"{col}:", width=12,
                                 style="Card.TLabel", anchor="w")
            lbl_name.pack(side="left")

            bar = ttk.Progressbar(frm_row, mode="determinate",
                                  style="Accent.Horizontal.TProgressbar",
                                  length=200)
            bar.pack(side="left", fill="x", expand=True, padx=(4, 4))
            self._col_bars[col] = bar

            lbl_stat = ttk.Label(frm_row, text="0/0", width=18,
                                 style="Dim.TLabel", anchor="w")
            lbl_stat.pack(side="left")
            self._col_labels[col] = lbl_stat

            lbl_pct = ttk.Label(frm_row, text="--", width=6,
                                style="Badge.TLabel", foreground=FG2)
            lbl_pct.pack(side="right")
            self._col_pct_labels[col] = lbl_pct

        # Stats row: latency, ETA, GSE progress, watchdog
        frm_stats = ttk.Frame(right)
        frm_stats.pack(fill="x", pady=(0, 4))

        self._lbl_latency = ttk.Label(frm_stats, text="Latency: --",
                                      font=("Consolas", 9), foreground=ACCENT2)
        self._lbl_latency.pack(side="left", padx=(0, 16))

        self._lbl_gse_prog = ttk.Label(frm_stats, text="GSEs: --",
                                       font=("Consolas", 9), foreground=FG2)
        self._lbl_gse_prog.pack(side="left", padx=(0, 16))

        self._lbl_elapsed = ttk.Label(frm_stats, text="",
                                      font=("Consolas", 9), foreground=FG2)
        self._lbl_elapsed.pack(side="right")

        self._lbl_watchdog = ttk.Label(frm_stats, text="Watchdog: idle",
                                       font=("Consolas", 9), foreground=FG2)
        self._lbl_watchdog.pack(side="right", padx=(0, 16))

        # Log area
        self._log = scrolledtext.ScrolledText(
            right, wrap="word", state="disabled",
            bg=BG3, fg=FG, insertbackground=FG,
            font=("Consolas", 10), relief="flat",
            selectbackground=ACCENT, selectforeground="#ffffff",
            padx=8, pady=6)
        self._log.pack(fill="both", expand=True)
        for tag, color in [("INFO", FG), ("OK", SUCCESS), ("WARN", WARNING),
                           ("ERR", ERROR), ("ACCENT", ACCENT2), ("DIM", FG2),
                           ("TIMESTAMP", FG2)]:
            self._log.tag_configure(tag, foreground=color)
        self._log.tag_configure("TIMESTAMP", font=("Consolas", 9))

    # ── browse helpers ────────────────────────────────────────────────────

    def _browse_dir(self):
        d = filedialog.askdirectory(title="Select Harmonized Labels Folder")
        if d:
            self._var_dir.set(d)
            self._check_files()
            self._check_geodb(d)

    def _browse_gsm_file(self):
        f = filedialog.askopenfilename(
            title="Select GSM List File",
            filetypes=[("Text/CSV", "*.txt *.csv"), ("All", "*.*")])
        if f:
            self._var_gsm_file.set(f)

    # ── environment checks ────────────────────────────────────────────────

    def _check_env_async(self):
        def _check():
            if not ollama_binary_exists():
                self._queue.put(("badge_ollama", "NOT FOUND", ERROR))
            elif ollama_server_ok(self._var_url.get()):
                self._queue.put(("badge_ollama", "OK", SUCCESS))
                mdl = self._var_model.get()
                if model_available(mdl, self._var_url.get()):
                    self._queue.put(("badge_model", "OK", SUCCESS))
                else:
                    self._queue.put(("badge_model", "MISSING", WARNING))
                try:
                    info = check_ollama_gpu(self._var_url.get())
                    if info and info[0] == "gpu":
                        self._queue.put(("badge_gpu", f"YES ({info[1]:.0f}GB)", SUCCESS))
                    else:
                        self._queue.put(("badge_gpu", "CPU", WARNING))
                except Exception:
                    self._queue.put(("badge_gpu", "?", FG2))
            else:
                self._queue.put(("badge_ollama", "DOWN", WARNING))
                self._queue.put(("badge_model", "-", FG2))
                self._queue.put(("badge_gpu", "-", FG2))
                self._queue.put(("log", "Ollama not running. Will auto-start.", "WARN"))

            d = self._var_dir.get()
            if d and os.path.isdir(d):
                self._queue.put(("check_files", None, None))
                self._queue.put(("check_geodb", d, None))

        threading.Thread(target=_check, daemon=True).start()

    def _check_files(self):
        d = self._var_dir.get()
        gpl = self._var_gpl.get()
        if not d or not os.path.isdir(d):
            self._lbl_files.configure(text="Select a folder.")
            return
        expected = f"harmonized_labels_{gpl}.csv"
        path = os.path.join(d, expected)
        if os.path.isfile(path):
            sz = os.path.getsize(path) / (1024 * 1024)
            self._lbl_files.configure(text=f"{expected} ({sz:.1f} MB)",
                                      foreground=SUCCESS)
        else:
            self._lbl_files.configure(text=f"{expected} not found",
                                      foreground=ERROR)

    def _check_geodb(self, folder):
        for c in [os.path.join(folder, "GEOmetadb.sqlite"),
                  os.path.join(folder, "GEOmetadb.sqlite.gz"),
                  os.path.join(folder, "..", "data", "GEOmetadb.sqlite.gz")]:
            if os.path.isfile(c):
                sz = os.path.getsize(c) / (1024 * 1024)
                self._lbl_geodb.configure(text=f"OK ({sz:.0f} MB)",
                                          foreground=SUCCESS)
                return
        self._lbl_geodb.configure(text="NOT FOUND", foreground=WARNING)

    # ── hardware detection ────────────────────────────────────────────────

    def _detect_workers(self, silent=False):
        """Scan hardware and set GPU/CPU worker counts."""
        def _run():
            gpus = detect_gpus()
            mdl = self._var_model.get()
            total_w, gpu_w, cpu_w = compute_ollama_parallel(mdl)
            model_gb = MODEL_RAM_GB.get(mdl.lower(), DEFAULT_MODEL_GB)

            cpu_count = os.cpu_count() or 1
            ram = psutil.virtual_memory()
            free_ram = ram.available / 1e9
            total_ram = ram.total / 1e9

            # Build hardware info lines
            cpu_info = f"CPU: {cpu_count} cores"
            ram_info = f"RAM: {free_ram:.1f} / {total_ram:.1f} GB free"

            if gpus:
                gpu_names = []
                total_vram = 0.0
                free_vram = 0.0
                for g in gpus:
                    gpu_names.append(f"{g['name']}")
                    total_vram += g.get("vram_gb", 0)
                    free_vram += g.get("free_vram_gb", 0)
                gpu_info = f"GPU: {' + '.join(gpu_names)}"
                vram_info = f"VRAM: {free_vram:.1f} / {total_vram:.1f} GB free"
            else:
                gpu_info = "GPU: None detected (CPU-only mode)"
                vram_info = "VRAM: N/A"

            # Update GUI (thread-safe via queue)
            self._queue.put(("hw_info", {
                "cpu": cpu_info,
                "ram": ram_info,
                "gpu": gpu_info,
                "vram": vram_info,
                "gpu_workers": gpu_w,
                "cpu_workers": cpu_w,
            }, None))

            if not silent:
                # Build detailed report
                lines = [
                    f"{'=' * 50}",
                    f"  Hardware Scan -- Worker Allocation",
                    f"{'=' * 50}",
                    f"",
                    f"  {cpu_info}",
                    f"  {ram_info}",
                    f"  {gpu_info}",
                    f"  {vram_info}",
                    f"",
                    f"  Model: {mdl} (~{model_gb:.1f} GB per instance)",
                    f"",
                ]
                if gpus:
                    for g in gpus:
                        lines.append(
                            f"  GPU {g['id']}: {g['name']}  "
                            f"VRAM {g['vram_gb']:.1f} GB  "
                            f"(free {g['free_vram_gb']:.1f} GB)")
                lines += [
                    f"",
                    f"  {'=' * 44}",
                    f"  GPU workers : {gpu_w}  (run on VRAM -- fast)",
                    f"  CPU workers : {cpu_w}  (run on system RAM -- overflow)",
                    f"  TOTAL       : {gpu_w + cpu_w}",
                    f"  {'=' * 44}",
                    f"",
                    f"  When GPU VRAM is full, new requests spill",
                    f"  to CPU workers automatically.",
                ]
                self._queue.put(("popup_hw", "\n".join(lines), None))
                self._queue.put(("log", f"Hardware: {gpu_w} GPU + {cpu_w} CPU "
                                 f"workers ({mdl})", "ACCENT"))

        threading.Thread(target=_run, daemon=True).start()

    # ── start / stop ──────────────────────────────────────────────────────

    def _start(self):
        folder = self._var_dir.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Select a valid data folder.")
            return

        # reset
        self._total_done = 0
        self._total_count = 0
        self._col_progress = {c: (0, 0) for c in REPAIR_COLS}
        self._pbar["value"] = 0
        for col in REPAIR_COLS:
            self._col_bars[col]["value"] = 0
            self._col_labels[col].configure(text="0/0")
            self._col_pct_labels[col].configure(text="--", foreground=FG2)
        self._lbl_progress.configure(text="Starting...")
        self._lbl_latency.configure(text="Latency: --")
        self._lbl_gse_prog.configure(text="GSEs: --")
        self._lbl_watchdog.configure(text="Watchdog: idle")
        self._start_time = time.time()

        self._stop_event.clear()
        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._running = True

        gpu_w = self._var_gpu_workers.get()
        cpu_w = self._var_cpu_workers.get()
        max_workers = gpu_w + cpu_w

        config = {
            "mode": self._var_mode.get(),
            "base_dir": folder,
            "output_dir": folder,
            "platforms": [self._var_gpl.get()],
            "gsm_file": self._var_gsm_file.get().strip() or "",
            "platform_id": self._var_gpl.get(),
            "model": self._var_model.get(),
            "ollama_url": self._var_url.get(),
            "geo_db_path": "",
            "memory_dir": "",
            "max_workers": max_workers,
            "gpu_workers": gpu_w,
            "cpu_workers": cpu_w,
            "enable_watchdog": True,
            "label_cols": None,
        }

        limit_str = self._var_limit.get().strip()
        if limit_str:
            try:
                config["test_limit"] = int(limit_str)
            except ValueError:
                pass

        self._log_msg(
            f"Launching: {self._var_gpl.get()}  "
            f"workers={gpu_w}GPU+{cpu_w}CPU={max_workers}  "
            f"mode={self._var_mode.get()}", "ACCENT")

        def _run():
            try:
                pipeline(config, q=self._queue)
            except Exception as exc:
                import traceback
                self._queue.put(("log",
                    f"Pipeline error: {exc}\n{traceback.format_exc()}", "ERR"))
            finally:
                self._queue.put(("done", None, None))

        self._worker_thread = threading.Thread(target=_run, daemon=True)
        self._worker_thread.start()

    def _stop(self):
        if self._running:
            self._stop_event.set()
            self._log_msg("Stop signal sent...", "WARN")
            self._btn_stop.configure(state="disabled")

    # ── queue polling ─────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            for _ in range(50):  # process up to 50 msgs per tick
                msg = self._queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        if self._running and self._start_time:
            elapsed = time.time() - self._start_time
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            self._lbl_elapsed.configure(text=f"Elapsed: {h}:{m:02d}:{s:02d}")

        self.after(200, self._poll_queue)

    def _handle_msg(self, msg):
        # Dict messages from pipeline
        if isinstance(msg, dict):
            kind = msg.get("type", "")
            if kind == "log":
                self._log_msg(msg.get("msg", ""))
            elif kind == "progress":
                pct = msg.get("pct", 0)
                label = msg.get("label", "")
                self._pbar["value"] = pct
                self._lbl_progress.configure(text=label or f"{pct}%")
            elif kind == "stats_live":
                per_col = msg.get("per_col", {})
                for col, data in per_col.items():
                    fixed = data.get("fixed", 0)
                    ns = data.get("ns", 0)
                    total_c = fixed + ns
                    if col in self._col_bars and total_c > 0:
                        pct_c = 100 * fixed / total_c
                        self._col_bars[col]["value"] = pct_c
                        self._col_labels[col].configure(
                            text=f"{fixed:,}/{total_c:,} resolved")
                        self._col_pct_labels[col].configure(
                            text=f"{pct_c:.0f}%",
                            foreground=SUCCESS if pct_c > 50 else FG2)
                # Latency / ETA
                latency = msg.get("latency_ms", 0)
                speed = msg.get("speed", 0)
                eta = msg.get("eta", "")
                if latency > 0:
                    lat_str = (f"{latency:.0f}ms" if latency < 1000
                               else f"{latency / 1000:.1f}s")
                    self._lbl_latency.configure(
                        text=f"Latency: {lat_str}/sample  "
                             f"({speed:.1f}/s)  ETA: {eta}")
                # GSE progress
                gse_done = msg.get("gse_done", 0)
                gse_total = msg.get("gse_total", 0)
                if gse_total > 0:
                    self._lbl_gse_prog.configure(
                        text=f"GSEs: {gse_done}/{gse_total}")
                # Sample counts
                self._total_done = msg.get("sample_num", 0)
                self._total_count = msg.get("total", 0)
            elif kind == "watchdog":
                self._lbl_watchdog.configure(
                    text=f"Watchdog: {msg.get('msg', '')[:60]}")
            elif kind == "done":
                self._finish_run()
            return

        # Tuple messages from internal GUI
        kind = msg[0]
        if kind == "log":
            self._log_msg(msg[1], msg[2] or "INFO")
        elif kind == "progress":
            _, done, total = msg
            self._total_done = done
            self._total_count = total
            if total > 0:
                self._pbar["value"] = done / total * 100
                self._lbl_progress.configure(text=f"{done}/{total}")
        elif kind == "badge_ollama":
            self._lbl_ollama.configure(text=msg[1], foreground=msg[2])
        elif kind == "badge_model":
            self._lbl_model.configure(text=msg[1], foreground=msg[2])
        elif kind == "badge_gpu":
            self._lbl_gpu.configure(text=msg[1], foreground=msg[2])
        elif kind == "hw_info":
            info = msg[1]
            self._lbl_hw_cpu.configure(text=info["cpu"])
            self._lbl_hw_ram.configure(text=info["ram"])
            self._lbl_hw_gpu.configure(text=info["gpu"])
            self._lbl_hw_vram.configure(text=info["vram"])
            self._var_gpu_workers.set(info["gpu_workers"])
            self._var_cpu_workers.set(info["cpu_workers"])
        elif kind == "check_files":
            self._check_files()
        elif kind == "check_geodb":
            self._check_geodb(msg[1])
        elif kind == "popup_hw":
            messagebox.showinfo("Hardware Detection", msg[1])
        elif kind == "done":
            self._finish_run()

    def _finish_run(self):
        self._running = False
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")

        elapsed = time.time() - self._start_time if self._start_time else 0
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)

        if self._stop_event.is_set():
            self._log_msg(f"Stopped by user after {h}:{m:02d}:{s:02d}.", "WARN")
            self._lbl_progress.configure(text="Stopped")
        else:
            self._log_msg(
                f"Complete: {self._total_done}/{self._total_count} "
                f"in {h}:{m:02d}:{s:02d}", "OK")
            self._lbl_progress.configure(text="Complete")
            self._pbar["value"] = 100

    # ── logging ───────────────────────────────────────────────────────────

    def _log_msg(self, text, tag="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.configure(state="normal")
        self._log.insert("end", f"[{ts}] ", "TIMESTAMP")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    # ── window close ──────────────────────────────────────────────────────

    def _on_close(self):
        if self._running:
            if not messagebox.askokcancel("Pipeline Running",
                    "A job is running. Stop and quit?"):
                return
            self._stop_event.set()
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=3)
        try:
            kill_ollama()
        except Exception:
            pass
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = NSRepairApp()
    app.mainloop()


if __name__ == "__main__":
    main()
