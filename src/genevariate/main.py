#!/usr/bin/env python
"""
GeneVariate Main Entry Point
Launches the GeneVariate GUI application.
"""

import subprocess
import sys
from pathlib import Path

# Add the parent directory to Python path
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


# ── Auto-install missing packages ──────────────────────────────────────────
# Makes the program reproducible on any local device without manual pip setup.

def _ensure_pkg(pip_name: str, import_name: str = None):
    """Try to import a package; auto-install via pip if missing."""
    try:
        __import__(import_name or pip_name)
    except ImportError:
        print(f"[SETUP] Installing {pip_name} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name,
             "--break-system-packages", "-q"],
            check=False,
        )


# Core packages that must be present before anything else
_REQUIRED = [
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("matplotlib", None),
    ("seaborn", None),
    ("scikit-learn", "sklearn"),
    ("GEOparse", "GEOparse"),
    ("requests", None),
    ("psutil", None),
    ("Pillow", "PIL"),
]


def check_dependencies():
    """Check and auto-install all required packages."""
    print("Checking dependencies...")

    # tkinter is system-level -- can't pip install
    try:
        __import__("tkinter")
    except ImportError:
        print("\nMissing: tkinter (system package)")
        print("  Ubuntu/Debian : sudo apt install python3-tk")
        print("  Fedora        : sudo dnf install python3-tkinter")
        print("  macOS         : brew install python-tk")
        return False

    for pip_name, import_name in _REQUIRED:
        _ensure_pkg(pip_name, import_name)

    # Verify all imports work after install
    missing = []
    for pip_name, import_name in _REQUIRED:
        try:
            __import__(import_name or pip_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"\nCould not install: {', '.join(missing)}")
        print(f"  pip install {' '.join(missing)}")
        return False

    print("All required packages found")
    return True


def check_data_directory():
    """Check if data directory and GEOmetadb exist."""
    try:
        from genevariate.config import CONFIG
    except ModuleNotFoundError as e:
        print(f"\n❌ Failed to import genevariate.config: {e}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Python path: {sys.path[:3]}")
        return False
    
    print("Checking for GEOmetadb...")
    
    data_dir = CONFIG['paths']['data']
    geo_db = CONFIG['paths']['geo_db']
    
    if not data_dir.exists():
        print(f"\n⚠️  Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    if not geo_db.exists():
        # Try auto-searching the local filesystem before warning the user
        try:
            from genevariate.core.db_loader import find_geometadb
            found = find_geometadb(log_fn=lambda m: None)
        except Exception:
            found = None

        if found:
            size_gb = Path(found).stat().st_size / (1024**3)
            print(f"✓ GEOmetadb auto-discovered: {found} ({size_gb:.1f} GB)")
        else:
            print("\n" + "=" * 60)
            print("WARNING: GEOmetadb.sqlite.gz Not Found")
            print("=" * 60)
            print(f"Expected location: {geo_db}")
            print("(also auto-searched ~/Desktop, ~/Downloads, ~, "
                  "project dirs - nothing found)")
            print()
            print("Download it using one of these methods:")
            print()
            print("  Option 1 - Git LFS:")
            print("    git lfs install && git lfs pull")
            print()
            print("  Option 2 - Direct download (wget):")
            print(f"    wget -O {geo_db} \\")
            print("      https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz")
            print()
            print("  Option 3 - Direct download (curl):")
            print(f"    curl -L -o {geo_db} \\")
            print("      https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz")
            print()
            print("Note: The application will start, but Step 1 (GSE Extraction)")
            print("      will not work without this database.")
            print("=" * 60)
    else:
        size_gb = geo_db.stat().st_size / (1024**3)
        print(f"✓ GEOmetadb found ({size_gb:.1f} GB)")

    return True


def initialize_directories():
    """Create all required directories."""
    try:
        from genevariate.config import init_directories
        init_directories()
        print("✓ Required directories created")
        return True
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        import traceback
        traceback.print_exc()
        return False


def install_hang_diagnostics():
    """Make a frozen window explain itself, on any machine, without root.

    A GUI that stops responding gives the user nothing to report: the process
    is alive, it is using no CPU, and the log stops. Recovering the stack
    afterwards needs a debugger attached to a running process, which most
    Linux installs forbid by default (``kernel.yama.ptrace_scope``), so the one
    moment the information exists is the moment it is lost.

    Registering :mod:`faulthandler` here means the program can always be asked
    for its own answer::

        kill -USR1 <pid>

    Every thread's Python stack is then written to the log, no debugger, no
    elevated permissions and no change to how the program runs.
    """
    try:
        import faulthandler
        faulthandler.enable()
        if hasattr(faulthandler, "register"):
            import signal
            faulthandler.register(signal.SIGUSR1, all_threads=True,
                                  chain=True)
            print("  Hang report   : kill -USR1 %d" % __import__("os").getpid())
    except Exception:
        pass


def show_resource_tier():
    """Display the auto-detected resource tier at startup."""
    try:
        from genevariate.config import RESOURCE_TIER
        tier = RESOURCE_TIER
        tier_name = tier['tier'].upper()
        ram = tier['total_ram_gb']
        db_mode = "in-memory" if tier['db_in_memory'] else "disk (low-RAM mode)"
        max_w = tier.get('watchdog_max_workers', '?')
        print(f"  Hardware tier : {tier_name} ({ram} GB RAM)")
        print(f"  GEOmetadb     : {db_mode}")
        print(f"  Max workers   : {max_w}")
    except Exception:
        print("  Hardware tier : unknown (could not detect)")


def main():
    """Main entry point. Use --llm-extract to launch the LLM-GEO label extractor."""
    # Redirected to a file, stdout is block buffered, so the progress the
    # program prints only reaches the file in 8 kB chunks and a run that
    # stalls appears to have stopped minutes before it did.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print(" GeneVariate 0.9.0")
    print(" Gene Expression Variability Analysis Platform")
    print("=" * 60)

    # Must run before anything imports pandas/tkinter. Skipping it meant a
    # machine without python3-tk failed later, deep inside the GUI import,
    # as an unexplained "FATAL ERROR" traceback instead of the one-line
    # apt/dnf/brew instruction this prints.
    if not check_dependencies():
        sys.exit(1)

    show_resource_tier()
    install_hang_diagnostics()
    print("=" * 60)

    # Check for LLM-GEO label extractor mode (replaces the old --ns-repair).
    if "--llm-extract" in sys.argv:
        print("Launching LLM-GEO Label Extractor...")
        print("=" * 60 + "\n")
        idx = sys.argv.index("--llm-extract")
        forwarded = sys.argv[idx + 1:]
        try:
            from genevariate.core import geo_extract_driver as drv
            drv._ensure_path()
            drv.configure_backend()
            from geo_label_extractor import geo_pipeline
            sys.argv = ["geo_pipeline"] + forwarded
            geo_pipeline.main()
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error launching LLM-GEO Label Extractor: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    if not check_data_directory():
        sys.exit(1)

    if not initialize_directories():
        sys.exit(1)

    try:
        print("Initializing application...")
        print("(This may take a moment...)")
        print("=" * 60 + "\n")

        from genevariate.gui.app import GeoWorkflowGUI

        app = GeoWorkflowGUI()
        app.mainloop()

    except KeyboardInterrupt:
        print("\nApplication closed by user")
        sys.exit(0)

    except Exception as e:
        print("\n" + "=" * 60)
        print("FATAL ERROR")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
