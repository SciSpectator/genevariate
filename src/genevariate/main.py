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
    ("ollama", None),
    ("qnorm", None),
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
        print("\n" + "=" * 60)
        print("WARNING: GEOmetadb.sqlite.gz Not Found")
        print("=" * 60)
        print(f"Expected location: {geo_db}")
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
    """Main entry point. Use --ns-repair to launch the NS Repair tool."""
    print("\n" + "=" * 60)
    print(" GeneVariate 2.0")
    print(" Gene Expression Variability Analysis Platform")
    print("=" * 60)

    show_resource_tier()
    print("=" * 60)

    # Check for NS repair mode
    if "--ns-repair" in sys.argv:
        print("Launching NS Repair Tool...")
        print("=" * 60 + "\n")
        try:
            from genevariate.gui.ns_repair_app import main as ns_repair_main
            ns_repair_main()
        except Exception as e:
            print(f"Error launching NS Repair: {e}")
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
