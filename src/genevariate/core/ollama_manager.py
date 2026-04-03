"""
Ollama Manager - GPU detection, server management, and resource watchdog.

Handles:
    - GPU detection (NVIDIA + AMD)
    - Ollama server lifecycle (install, start, stop, pull models)
    - Parallel worker computation (GPU + CPU hybrid)
    - Resource watchdog (RAM/VRAM monitoring with auto-pause)
"""

import os
import re
import sys
import time
import shutil
import signal
import subprocess
import threading
import platform as _platform
from typing import List, Optional
from datetime import timedelta

import requests
import psutil

# Constants
DEFAULT_MODEL = "gemma2:9b"
DEFAULT_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CPU_OLLAMA_URL = "http://localhost:11435"

MODEL_RAM_GB = {
    "gemma2:2b": 2.0, "gemma2:2b-q4_0": 1.8,
    "gemma2:9b": 5.4, "gemma2:9b-q4_0": 5.0, "gemma2:9b-q8_0": 9.5,
    "gemma2:27b": 18.0,
    "llama3:8b": 5.5, "llama3.1:8b": 5.5, "llama3:70b": 48.0,
    "mistral:7b": 4.8, "mistral:7b-q4_0": 4.5,
    "qwen2.5:7b": 4.4,
}
DEFAULT_MODEL_GB = 5.4


# ── GPU Detection ──

def detect_gpus():
    """Detect NVIDIA and AMD GPUs with VRAM info."""
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append({
                    "id": int(parts[0]), "name": parts[1],
                    "vram_gb": round(int(parts[2]) / 1024, 1),
                    "free_vram_gb": round(int(parts[3]) / 1024, 1),
                    "type": "nvidia"
                })
    except Exception:
        pass
    if not gpus:
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                stderr=subprocess.DEVNULL, text=True, timeout=5)
            for i, line in enumerate(out.strip().splitlines()[1:]):
                parts = line.split(",")
                if len(parts) >= 2:
                    gpus.append({
                        "id": i, "name": f"AMD GPU {i}",
                        "vram_gb": round(int(parts[-1].strip()) / 1e6, 1),
                        "free_vram_gb": 0, "type": "amd"
                    })
        except Exception:
            pass
    return gpus


def get_vram_usage():
    """Return (used_mb, total_mb, percent)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=3)
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        u, t = int(parts[0]), int(parts[1])
        return u, t, 100.0 * u / t if t else 0.0
    except Exception:
        return 0, 0, 0.0


def vram_utilisation_pct() -> float:
    try:
        gpus = detect_gpus()
        if not gpus:
            return 0.0
        used = sum(g.get("used_vram_gb", 0) for g in gpus)
        total = sum(g.get("vram_gb", 1) for g in gpus)
        return 100.0 * used / total if total else 0.0
    except Exception:
        return 0.0


def compute_ollama_parallel(model: str, reserve_gb: float = 4.0,
                            extra_vram_gb: float = 0.0) -> tuple:
    """
    Compute hybrid worker count: GPU workers + CPU workers.
    Resource-aware: on low-RAM devices, reserves more for the OS and caps workers.
    Returns (total_workers, gpu_workers, cpu_workers).
    """
    try:
        gpus = detect_gpus()
        model_key = os.path.basename(model).strip().lower()
        slot_gb = MODEL_RAM_GB.get(model_key,
                  MODEL_RAM_GB.get(model.strip().lower(), DEFAULT_MODEL_GB))
        total_ram_gb = psutil.virtual_memory().total / 1e9
        free_gb = psutil.virtual_memory().available / 1e9

        # On low-RAM devices, reserve more for OS stability
        if total_ram_gb <= 6:
            reserve_gb = max(reserve_gb, 2.5)
        elif total_ram_gb <= 10:
            reserve_gb = max(reserve_gb, 3.0)

        if gpus:
            total_vram = sum(g["vram_gb"] for g in gpus)
            try:
                ps = requests.get(f"{DEFAULT_URL}/api/ps", timeout=2).json()
                loaded_vram_gb = sum(
                    m.get("size_vram", 0) / 1e9
                    for m in ps.get("models", []))
            except Exception:
                loaded_vram_gb = slot_gb

            kv_per_slot = max(0.3, slot_gb * 0.15)
            headroom = total_vram - loaded_vram_gb - 1.0 - extra_vram_gb
            gpu_workers = max(1, min(8, int(headroom / kv_per_slot)))
        else:
            gpu_workers = 0

        ram_after_gpu = free_gb - reserve_gb
        ram_slots = max(0, int(ram_after_gpu / slot_gb))
        cpu_count = os.cpu_count() or 4
        usable_cpu = max(1, cpu_count - 2)
        cpu_workers = min(ram_slots, usable_cpu)

        # Cap total workers based on device tier
        try:
            from genevariate.config import RESOURCE_TIER
            max_w = RESOURCE_TIER.get('watchdog_max_workers', 210)
        except Exception:
            max_w = 210

        total = max(1, min(max_w, gpu_workers + cpu_workers))
        return total, gpu_workers, min(cpu_workers, total - gpu_workers)
    except Exception:
        return 1, 0, 1


def check_ollama_gpu(base_url=DEFAULT_URL):
    try:
        r = requests.get(f"{base_url}/api/ps", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            if models:
                vram = models[0].get("size_vram", 0)
                total = models[0].get("size", 1)
                if vram > total * 0.5:
                    return "gpu", round(vram / 1e9, 1)
                return "cpu", 0
    except Exception:
        pass
    return "unknown", 0


# ── Ollama Server Management ──

def ollama_server_ok(base_url=DEFAULT_URL, timeout=3):
    try:
        return requests.get(f"{base_url}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def ollama_binary_exists():
    return shutil.which("ollama") is not None


def model_available(model, base_url=DEFAULT_URL):
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(model.split(":")[0] in n for n in names)
    except Exception:
        pass
    return False


def install_ollama_blocking(log_fn):
    os_name = _platform.system().lower()
    if os_name not in ("linux", "darwin"):
        log_fn("[ERROR] Auto-install supported only on Linux/macOS.")
        return False
    log_fn("Installing Ollama via official script...")
    proc = subprocess.Popen(
        "curl -fsSL https://ollama.com/install.sh | sh",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log_fn("  " + line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        log_fn("[ERROR] Install failed.")
        return False
    log_fn("Ollama installed.")
    return True


def start_ollama_server_blocking(log_fn, num_parallel: int = 1):
    gpus = detect_gpus()
    if gpus:
        gpu_ids = ",".join(str(g["id"]) for g in gpus)
        names = " + ".join(f"{g['name']} ({g['vram_gb']}GB)" for g in gpus)
        log_fn(f"  GPU(s): {names}")
    else:
        gpu_ids = "0"
        log_fn("  No GPU detected")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KEEP_ALIVE"] = "-1"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

    log_fn(f"Starting Ollama | GPU={gpu_ids} PARALLEL={num_parallel}")
    if _platform.system().lower() == "windows":
        proc = subprocess.Popen(["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env, preexec_fn=os.setsid)

    for i in range(40):
        time.sleep(1)
        if ollama_server_ok():
            log_fn(f"Ollama ready ({i+1}s) | {num_parallel} parallel slots")
            return proc
        if i % 5 == 4:
            log_fn(f"  waiting ({i+1}s)")
    log_fn("[ERROR] Server did not start in 40s.")
    proc.terminate()
    return None


def start_ollama_cpu_server(log_fn, num_parallel: int = 2):
    """Launch CPU-only Ollama on port 11435."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OLLAMA_HOST"] = "0.0.0.0:11435"
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    env["OLLAMA_KEEP_ALIVE"] = "-1"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["OLLAMA_FLASH_ATTENTION"] = "0"
    env["OLLAMA_MODELS"] = os.path.expanduser("~/.ollama/models")

    log_fn(f"  Starting CPU Ollama on port 11435 ({num_parallel} workers)")
    try:
        if _platform.system().lower() == "windows":
            proc = subprocess.Popen(["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            proc = subprocess.Popen(["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, preexec_fn=os.setsid)

        for i in range(30):
            time.sleep(1)
            if ollama_server_ok(CPU_OLLAMA_URL):
                log_fn(f"  CPU Ollama ready ({i+1}s)")
                return proc
        log_fn("  [WARN] CPU Ollama did not start")
        proc.terminate()
        return None
    except Exception as e:
        log_fn(f"  [WARN] Could not start CPU Ollama: {e}")
        return None


def kill_ollama(log_fn=None):
    """Kill any running Ollama serve process."""
    try:
        subprocess.run(["sudo", "-n", "systemctl", "stop", "ollama"],
                       capture_output=True, timeout=5)
    except Exception:
        pass
    for sig in ["TERM", "KILL"]:
        try:
            subprocess.run(["sudo", "-n", "pkill", f"-{sig}", "-f", "ollama serve"],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        try:
            subprocess.run(["pkill", f"-{sig}", "-f", "ollama serve"],
                           capture_output=True, timeout=5)
        except Exception:
            pass
    import socket
    for i in range(15):
        try:
            s = socket.create_connection(("127.0.0.1", 11434), timeout=0.5)
            s.close()
            time.sleep(1)
        except Exception:
            break
    return True


def pull_model_blocking(model, log_fn, progress_fn=None):
    log_fn(f"Pulling model '{model}'...")
    try:
        import json
        with requests.post(f"{DEFAULT_URL}/api/pull",
                           json={"name": model}, stream=True, timeout=3600) as r:
            r.raise_for_status()
            last = ""
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                status = d.get("status", "")
                total = d.get("total", 0)
                done = d.get("completed", 0)
                if status != last:
                    log_fn(f"  {status}")
                    last = status
                if progress_fn and total and done:
                    progress_fn(int(100 * done / total))
        log_fn(f"Model '{model}' ready.")
        return True
    except Exception as exc:
        log_fn(f"[ERROR] Pull failed: {exc}")
        return False


# ── Watchdog (fluid scaling) ──

class Watchdog:
    """Resource watchdog with fluid worker scaling.

    Instead of hard-pausing on CPU/RAM pressure, scales workers up/down
    dynamically. Hard pauses are reserved for extreme emergencies only
    (thermal protection, OOM risk).

    Resource-aware: all thresholds adapt to the device's available RAM so
    the pipeline works on "garbage local devices" (4 GB RAM) as well as
    high-end workstations.

    Fluid scaling approach:
        - Above HIGH threshold: scale down workers
        - Below LOW threshold: scale back up
        - Thermal limits: hard pause (hardware safety)
        - Near-OOM: hard pause
    """

    CHECK_INTERVAL = 3

    # ── Thermal protection (hard-pause — hardware safety, same everywhere) ──
    CPU_TEMP_PAUSE_C = 88.0
    CPU_TEMP_RESUME_C = 72.0
    GPU_TEMP_PAUSE_C = 85.0
    GPU_TEMP_RESUME_C = 70.0

    VRAM_PAUSE_PCT = 98.0
    VRAM_RESUME_PCT = 90.0
    CPU_HIGH_PCT = 92.0
    CPU_LOW_PCT = 75.0

    SCALE_DOWN_FACTOR = 0.6
    SCALE_COOLDOWN_S = 10

    def __init__(self, log_fn=None, stat_fn=None):
        self._log = log_fn or (lambda m: None)
        self._stat = stat_fn or (lambda m: None)
        self._gate = threading.Event()
        self._gate.set()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._calls: List[float] = []
        self._reason = None
        self._last_scale_time = 0.0

        # ── Resource-aware defaults from config (adapts to device tier) ──
        try:
            from genevariate.config import RESOURCE_TIER
            tier = RESOURCE_TIER
        except Exception:
            tier = {}

        self.RAM_HIGH_PCT = tier.get('ram_high_pct', 92.0)
        self.RAM_LOW_PCT = tier.get('ram_low_pct', 80.0)
        self.RAM_PAUSE_PCT = tier.get('ram_pause_pct', 99.0)
        self.RAM_RESUME_PCT = min(self.RAM_LOW_PCT + 5, self.RAM_PAUSE_PCT - 5)
        self.MIN_WORKERS = tier.get('watchdog_min_workers', 4)
        self.SCALE_UP_STEP = tier.get('watchdog_scale_up_step', 20)
        self._max_workers = tier.get('watchdog_max_workers', 210)

        tier_name = tier.get('tier', 'unknown')
        ram_gb = tier.get('total_ram_gb', '?')
        self._log(f"  Watchdog init: {tier_name} tier ({ram_gb} GB RAM) | "
                  f"workers {self.MIN_WORKERS}-{self._max_workers} | "
                  f"RAM thresholds {self.RAM_LOW_PCT}/{self.RAM_HIGH_PCT}/"
                  f"{self.RAM_PAUSE_PCT}%")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Watchdog")
        # Dynamic concurrency adjustment (set by pipeline)
        self._adjust_concurrency = None  # callable(new_n) or None
        self._target_parallel = 0
        self._model = "gemma2:9b"
        self._inhibit_fd = None

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._gate.set()
        self._release_sleep()

    def wait_if_paused(self, timeout: float = 120.0):
        """Block LLM threads while paused. Hard pauses are rare (thermal/OOM only).
        Auto-resumes after timeout to prevent silent hang."""
        if self._gate.is_set():
            return
        elapsed = 0.0
        interval = 15.0
        while not self._gate.is_set():
            self._log(f"  Paused ({self._reason or '?'}) -- waiting ({elapsed:.0f}s)")
            self._gate.wait(timeout=interval)
            elapsed += interval
            if elapsed >= timeout:
                self._log(f"[WATCHDOG] Pause exceeded {timeout:.0f}s -- resuming "
                          f"to avoid silent hang.")
                self._gate.set()
                self._reason = None
                return

    def record_call(self):
        with self._lock:
            now = time.time()
            self._calls.append(now)
            self._calls = [t for t in self._calls if now - t <= 60]

    def calls_per_min(self):
        with self._lock:
            now = time.time()
            return len([t for t in self._calls if now - t <= 60])

    def _pause(self, reason, detail):
        if self._gate.is_set():
            self._gate.clear()
            self._reason = reason
            self._log(f"Watchdog PAUSED: {detail} -- LLM calls suspended.")
            if self._stat:
                self._stat(f"PAUSED: {detail}")

    def _resume(self, detail):
        if not self._gate.is_set():
            self._gate.set()
            self._reason = None
            self._log(f"Watchdog RESUMED: {detail} -- LLM calls restarting.")
            if self._stat:
                self._stat(f"RESUMED: {detail}")

    # ── Thermal monitoring ──

    @staticmethod
    def _read_cpu_temp() -> float:
        """Read highest CPU core temperature in C. Returns 0.0 if unavailable."""
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return 0.0
            for name in ("coretemp", "k10temp", "cpu_thermal", "zenpower", "acpitz"):
                if name in temps:
                    return max(s.current for s in temps[name] if s.current > 0)
            all_readings = [s.current for entries in temps.values()
                            for s in entries if s.current > 0]
            return max(all_readings) if all_readings else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _read_gpu_temp() -> float:
        """Read GPU temperature in C via nvidia-smi. Returns 0.0 if unavailable."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True, timeout=3)
            vals = [int(v.strip()) for v in out.strip().splitlines() if v.strip()]
            return float(max(vals)) if vals else 0.0
        except Exception:
            return 0.0

    # ── Sleep prevention ──

    def _prevent_sleep(self):
        """Acquire OS-level sleep/idle inhibitor for long-running pipelines.
        Uses systemd D-Bus on Linux. No-op on other platforms."""
        if _platform.system() != "Linux":
            return

        # Method 1: gdbus (no root needed)
        try:
            result = subprocess.run(
                ["gdbus", "call", "--system",
                 "--dest", "org.freedesktop.login1",
                 "--object-path", "/org/freedesktop/login1",
                 "--method", "org.freedesktop.login1.Manager.Inhibit",
                 "sleep:idle:handle-lid-switch",
                 "GeneVariate",
                 "Long-running biomedical annotation pipeline",
                 "block"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self._log("Sleep inhibitor acquired (systemd D-Bus)")
                return
        except Exception:
            pass

        # Method 2: xset as fallback (X11 only)
        try:
            subprocess.Popen(["xset", "s", "off", "-dpms"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            self._log("Sleep: disabled screensaver via xset")
            return
        except Exception:
            pass

        self._log("[WARN] Could not acquire sleep inhibitor -- "
                  "run: 'systemd-inhibit genevariate' to prevent sleep manually")

    def _release_sleep(self):
        """Release the inhibitor lock when the run finishes."""
        try:
            if self._inhibit_fd is not None:
                self._inhibit_fd.close()
                self._inhibit_fd = None
        except Exception:
            pass

    # ── Main monitoring loop with fluid scaling ──

    def _loop(self):
        total_ram = psutil.virtual_memory().total / 1e6
        self._prevent_sleep()
        # Prime the CPU % sampler (first call always returns 0)
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while not self._stop.is_set():
            try:
                vm = psutil.virtual_memory()
                ram_pct = vm.percent
                ram_mb = vm.used / 1e6
                cpu_pct = psutil.cpu_percent(interval=None)
                vu, vt, vpct = get_vram_usage()
                has_gpu = vt > 0
                cpu_temp = self._read_cpu_temp()
                gpu_temp = self._read_gpu_temp() if has_gpu else 0.0
                cpm = self.calls_per_min()
                cur_w = getattr(self, "_target_parallel", 0)
                max_w = getattr(self, "_max_workers", cur_w)
                state = ("running" if self._gate.is_set()
                         else f"PAUSED ({self._reason or '?'})")
                wk_str = f"W:{cur_w}/{max_w}" if cur_w > 0 else ""

                # ── Status line with thermal info ──
                temp_str = ""
                if cpu_temp > 0:
                    temp_str += f"CPU:{cpu_temp:.0f}C "
                if gpu_temp > 0:
                    temp_str += f"GPU:{gpu_temp:.0f}C "

                if has_gpu:
                    self._stat(
                        f"RAM {ram_mb:.0f}/{total_ram:.0f} MB ({ram_pct:.0f}%) | "
                        f"CPU {cpu_pct:.0f}% | {temp_str}| "
                        f"VRAM {vu:,}/{vt:,} MB ({vpct:.0f}%) | "
                        f"LLM/min:{cpm} | {wk_str} | {state}")
                else:
                    self._stat(
                        f"RAM {ram_mb:.0f}/{total_ram:.0f} MB ({ram_pct:.0f}%) | "
                        f"CPU {cpu_pct:.0f}% | {temp_str}| "
                        f"LLM/min:{cpm} | {wk_str} | {state}")

                # --- FLUID WORKER SCALING ---
                now_t = time.time()
                adj_fn = getattr(self, "_adjust_concurrency", None)
                current_workers = getattr(self, "_target_parallel", 0)
                cooldown_ok = (now_t - self._last_scale_time) >= self.SCALE_COOLDOWN_S

                # 1) THERMAL -- hard pause (hardware safety, non-negotiable)
                if (cpu_temp >= self.CPU_TEMP_PAUSE_C and cpu_temp > 0
                        and self._gate.is_set()):
                    self._pause("THERMAL",
                                f"CPU temp {cpu_temp:.0f}C >= {self.CPU_TEMP_PAUSE_C:.0f}C")
                elif (gpu_temp >= self.GPU_TEMP_PAUSE_C and gpu_temp > 0
                      and self._gate.is_set()):
                    self._pause("THERMAL",
                                f"GPU temp {gpu_temp:.0f}C >= {self.GPU_TEMP_PAUSE_C:.0f}C")

                # 2) EXTREME RAM -- hard pause (OOM risk)
                elif ram_pct >= self.RAM_PAUSE_PCT and self._gate.is_set():
                    self._pause("RAM", f"RAM at {ram_pct:.0f}% -- OOM risk")

                # 3) Resume from hard pause if conditions are safe
                elif not self._gate.is_set():
                    cpu_cool = cpu_temp < self.CPU_TEMP_RESUME_C or cpu_temp == 0
                    gpu_cool = gpu_temp < self.GPU_TEMP_RESUME_C or gpu_temp == 0
                    rok = ram_pct < self.RAM_RESUME_PCT
                    if rok and cpu_cool and gpu_cool:
                        self._resume(f"RAM {ram_pct:.0f}% CPU {cpu_pct:.0f}%")

                # 4) FLUID SCALING -- scale down workers when resources are high
                elif (cooldown_ok and adj_fn
                      and current_workers > self.MIN_WORKERS):
                    pressure = max(ram_pct >= self.RAM_HIGH_PCT,
                                   cpu_pct >= self.CPU_HIGH_PCT)
                    if pressure:
                        new_n = max(self.MIN_WORKERS,
                                    int(current_workers * self.SCALE_DOWN_FACTOR))
                        if new_n < current_workers:
                            adj_fn(new_n)
                            self._last_scale_time = now_t
                            self._log(f"  Workers: {current_workers} -> {new_n} "
                                      f"(RAM {ram_pct:.0f}% CPU {cpu_pct:.0f}%)")

                    # 5) FLUID SCALING -- scale up workers when resources are low
                    elif ram_pct < self.RAM_LOW_PCT and cpu_pct < self.CPU_LOW_PCT:
                        if current_workers < max_w:
                            new_n = min(max_w,
                                        current_workers + self.SCALE_UP_STEP)
                            if new_n > current_workers:
                                adj_fn(new_n)
                                self._last_scale_time = now_t
                                self._log(f"  Workers: {current_workers} -> {new_n} "
                                          f"(RAM {ram_pct:.0f}% CPU {cpu_pct:.0f}%)")
            except Exception:
                pass
            self._stop.wait(self.CHECK_INTERVAL)
