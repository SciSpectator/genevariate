#!/bin/bash
# ============================================================
#  GeneVariate - macOS Build Script
#  Builds GeneVariate.app with embedded icon
# ============================================================

set -e
echo "============================================================"
echo " GeneVariate macOS Build"
echo "============================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Install from python.org or: brew install python"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt
pip3 install pyinstaller Pillow

# Generate .icns from PNG
ASSETS="src/genevariate/assets"
if [ ! -f "$ASSETS/icon.icns" ]; then
    echo "Generating icon.icns from icon.png..."
    ICONSET="$ASSETS/icon.iconset"
    mkdir -p "$ICONSET"

    # Generate all required sizes for macOS iconset
    for size in 16 32 64 128 256 512; do
        sips -z $size $size "$ASSETS/icon.png" --out "$ICONSET/icon_${size}x${size}.png" > /dev/null 2>&1
        double=$((size * 2))
        if [ $double -le 1024 ]; then
            sips -z $double $double "$ASSETS/icon.png" --out "$ICONSET/icon_${size}x${size}@2x.png" > /dev/null 2>&1
        fi
    done
    sips -z 512 512 "$ASSETS/icon.png" --out "$ICONSET/icon_512x512.png" > /dev/null 2>&1
    sips -z 1024 1024 "$ASSETS/icon.png" --out "$ICONSET/icon_512x512@2x.png" > /dev/null 2>&1

    iconutil -c icns "$ICONSET" -o "$ASSETS/icon.icns"
    rm -rf "$ICONSET"
    echo "Created icon.icns"
fi

# Build with PyInstaller
echo "Building GeneVariate.app..."
pyinstaller genevariate.spec --noconfirm

echo ""
echo "============================================================"
if [ -d "dist/GeneVariate.app" ]; then
    echo " BUILD SUCCESSFUL"
    echo " Application: dist/GeneVariate.app"
    echo ""
    echo " To install: drag GeneVariate.app to /Applications"
elif [ -d "dist/GeneVariate" ]; then
    echo " BUILD SUCCESSFUL (directory mode)"
    echo " Application: dist/GeneVariate/"
    echo ""
    echo " Run with: ./dist/GeneVariate/GeneVariate"
else
    echo " BUILD FAILED - check errors above"
fi
echo "============================================================"
