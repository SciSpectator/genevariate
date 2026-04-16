#!/usr/bin/env python3
"""
GeneVariate Installer
Creates a platform-specific desktop shortcut/launcher with the application icon.
Run after: pip install -e .
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ASSETS = Path(__file__).parent / "src" / "genevariate" / "assets"
ICON_PNG = ASSETS / "icon.png"
ICON_ICO = ASSETS / "icon.ico"
APP_NAME = "GeneVariate"


def install_windows():
    """Create a Windows shortcut (.lnk) on the Desktop."""
    desktop = Path(os.environ.get("USERPROFILE", "~")) / "Desktop"
    desktop = desktop.expanduser()

    # Find the installed genevariate-gui script
    scripts_dir = Path(sys.executable).parent / "Scripts"
    gui_exe = scripts_dir / "genevariate-gui.exe"
    if not gui_exe.exists():
        gui_exe = scripts_dir / "genevariate.exe"
    if not gui_exe.exists():
        # Fallback: launch via python
        gui_exe = None

    # Use PowerShell to create .lnk shortcut
    lnk_path = desktop / f"{APP_NAME}.lnk"
    target = str(gui_exe) if gui_exe else f'{sys.executable} -m genevariate.main'
    icon = str(ICON_ICO) if ICON_ICO.exists() else str(ICON_PNG)

    ps_script = f'''
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut("{lnk_path}")
$shortcut.TargetPath = "{target}"
$shortcut.IconLocation = "{icon}"
$shortcut.Description = "GeneVariate - Gene Expression Variability Analysis"
$shortcut.WorkingDirectory = "{Path(__file__).parent}"
$shortcut.Save()
'''
    subprocess.run(["powershell", "-Command", ps_script], check=True)
    print(f"Desktop shortcut created: {lnk_path}")


def install_mac():
    """Create a macOS .command launcher on the Desktop."""
    desktop = Path.home() / "Desktop"
    launcher = desktop / f"{APP_NAME}.command"

    # Find the installed entry point
    entry = shutil.which("genevariate") or shutil.which("genevariate-gui")
    if entry:
        cmd = entry
    else:
        cmd = f"{sys.executable} -m genevariate.main"

    launcher.write_text(f"""#!/bin/bash
# GeneVariate Launcher
cd "{Path(__file__).parent}"
{cmd}
""")
    launcher.chmod(0o755)
    print(f"Desktop launcher created: {launcher}")
    print(f"Tip: Right-click → Get Info → drag icon.png onto the file icon to set the app icon.")


def install_linux():
    """Create a .desktop entry for Linux."""
    apps_dir = Path.home() / ".local" / "share" / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)

    entry = shutil.which("genevariate") or shutil.which("genevariate-gui")
    if entry:
        exec_cmd = entry
    else:
        exec_cmd = f"{sys.executable} -m genevariate.main"

    desktop_entry = f"""[Desktop Entry]
Version=1.0
Type=Application
Name={APP_NAME}
Comment=Gene Expression Variability Analysis Platform
Exec={exec_cmd}
Icon={ICON_PNG}
Terminal=false
Categories=Science;Education;
StartupWMClass=GeneVariate
"""
    desktop_file = apps_dir / f"{APP_NAME.lower()}.desktop"
    desktop_file.write_text(desktop_entry)
    desktop_file.chmod(0o755)
    print(f"Application menu entry created: {desktop_file}")

    # Also copy to Desktop if it exists
    desktop = Path.home() / "Desktop"
    if desktop.exists():
        desk_shortcut = desktop / f"{APP_NAME}.desktop"
        desk_shortcut.write_text(desktop_entry)
        desk_shortcut.chmod(0o755)
        print(f"Desktop shortcut created: {desk_shortcut}")


def main():
    print(f"\n{'='*50}")
    print(f" {APP_NAME} Installer")
    print(f"{'='*50}\n")

    # First install the package
    print("Installing GeneVariate package...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(Path(__file__).parent)],
        check=True,
    )
    print()

    # Create platform-specific launcher
    if sys.platform == "win32":
        install_windows()
    elif sys.platform == "darwin":
        install_mac()
    else:
        install_linux()

    print(f"\n{'='*50}")
    print(" Installation complete!")
    print(f" Run GeneVariate from your desktop or type: genevariate")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
