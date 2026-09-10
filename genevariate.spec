# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for GeneVariate
Builds a single-directory distributable with the application icon.
Works on Windows, macOS, and Linux.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
ROOT = Path(SPECPATH)
ASSETS = ROOT / "src" / "genevariate" / "assets"

# Platform-specific icon. There is no icon.icns in assets/, so asking for one
# aborted the macOS build; PyInstaller accepts a .png and converts it.
if sys.platform == "win32":
    icon_file = str(ASSETS / "icon.ico")
else:
    icon_file = str(ASSETS / "icon.png")

a = Analysis(
    [str(ROOT / "src" / "genevariate" / "main.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(ASSETS / "icon.png"), "genevariate/assets"),
        (str(ASSETS / "icon.ico"), "genevariate/assets"),
        # app.py loads each source's logo from assets/icons/ at runtime.
        (str(ASSETS / "icons"), "genevariate/assets/icons"),
    ],
    # Collected wholesale so a module added to the package cannot be silently
    # left out of the build. The previous hand-written list had drifted: seven
    # of its entries named modules that no longer exist, while most of the
    # analysis, chatbot and window modules that do exist were absent.
    hiddenimports=collect_submodules("genevariate") + [
        # Imported by name only after _ensure_path() puts the vendored
        # directory on sys.path, so nothing static can discover it.
        "geo_label_extractor",
        "tkinter",
        "tkinter.ttk",
        "matplotlib",
        "matplotlib.backends.backend_tkagg",
        "numpy",
        "pandas",
        "scipy",
        "seaborn",
        "sklearn",
        "GEOparse",
        "requests",
        "psutil",
        "ollama",
        "qnorm",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GeneVariate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window - GUI app
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="GeneVariate",
)

# macOS .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="GeneVariate.app",
        icon=icon_file,
        bundle_identifier="com.scispectator.genevariate",
        info_plist={
            "CFBundleName": "GeneVariate",
            "CFBundleDisplayName": "GeneVariate",
            "CFBundleVersion": "1.0.0",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
