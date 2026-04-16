# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for GeneVariate
Builds a single-directory distributable with the application icon.
Works on Windows, macOS, and Linux.
"""

import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH)
ASSETS = ROOT / "src" / "genevariate" / "assets"

# Platform-specific icon
if sys.platform == "win32":
    icon_file = str(ASSETS / "icon.ico")
elif sys.platform == "darwin":
    icon_file = str(ASSETS / "icon.icns")
else:
    icon_file = str(ASSETS / "icon.png")

a = Analysis(
    [str(ROOT / "src" / "genevariate" / "main.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(ASSETS / "icon.png"), "genevariate/assets"),
        (str(ASSETS / "icon.ico"), "genevariate/assets"),
    ],
    hiddenimports=[
        "genevariate",
        "genevariate.config",
        "genevariate.core",
        "genevariate.core.ai_engine",
        "genevariate.core.db_loader",
        "genevariate.core.extraction",
        "genevariate.core.gpl_downloader",
        "genevariate.core.gse_context",
        "genevariate.core.gse_worker",
        "genevariate.core.memory_agent",
        "genevariate.core.nlp",
        "genevariate.core.ns_repair_pipeline",
        "genevariate.core.ollama_manager",
        "genevariate.core.statistics",
        "genevariate.gui",
        "genevariate.gui.app",
        "genevariate.gui.compare_analysis",
        "genevariate.gui.deterministic_extraction",
        "genevariate.gui.evaluation",
        "genevariate.gui.label_reextraction",
        "genevariate.gui.ns_repair_app",
        "genevariate.gui.region_analysis",
        "genevariate.gui.standalone_extraction",
        "genevariate.gui.windows",
        "genevariate.gui.windows.compare_dist",
        "genevariate.gui.windows.dialogs",
        "genevariate.gui.windows.interactive_subset",
        "genevariate.utils",
        "genevariate.utils.plotting",
        "genevariate.utils.workers",
        "genevariate.memory",
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
    console=False,  # No console window — GUI app
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
        icon=str(ASSETS / "icon.icns"),
        bundle_identifier="com.scispectator.genevariate",
        info_plist={
            "CFBundleName": "GeneVariate",
            "CFBundleDisplayName": "GeneVariate",
            "CFBundleVersion": "1.0.0",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
