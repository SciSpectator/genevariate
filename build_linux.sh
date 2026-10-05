#!/bin/bash
# ============================================================
#  GeneVariate - Linux Build Script
#  Builds standalone binary + .desktop launcher with icon
# ============================================================

set -e
echo "============================================================"
echo " GeneVariate Linux Build"
echo "============================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Install with: sudo apt install python3 python3-pip python3-tk"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt
pip3 install pyinstaller Pillow

# Build with PyInstaller
echo "Building GeneVariate binary..."
pyinstaller genevariate.spec --noconfirm

# Create .desktop file for application menu / desktop shortcut
DIST_DIR="$(pwd)/dist/GeneVariate"
ICON_PATH="$DIST_DIR/genevariate/assets/icon.png"
DESKTOP_FILE="$DIST_DIR/GeneVariate.desktop"

if [ -d "$DIST_DIR" ]; then
    # Copy icon into dist if not already there by PyInstaller
    mkdir -p "$DIST_DIR/genevariate/assets"
    cp -f "src/genevariate/assets/icon.png" "$ICON_PATH"

    cat > "$DESKTOP_FILE" << DESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=GeneVariate
Comment=Gene Expression Variability Analysis Platform
Exec=$DIST_DIR/GeneVariate
Icon=$ICON_PATH
Terminal=false
Categories=Science;Education;
StartupWMClass=GeneVariate
DESKTOP
    chmod +x "$DESKTOP_FILE"
fi

echo ""
echo "============================================================"
if [ -f "$DIST_DIR/GeneVariate" ]; then
    echo " BUILD SUCCESSFUL"
    echo " Binary:   $DIST_DIR/GeneVariate"
    echo " Launcher: $DESKTOP_FILE"
    echo ""
    echo " To add to your application menu:"
    echo "   cp $DESKTOP_FILE ~/.local/share/applications/"
    echo ""
    echo " Or double-click GeneVariate.desktop on your Desktop."
else
    echo " BUILD FAILED - check errors above"
fi
echo "============================================================"
