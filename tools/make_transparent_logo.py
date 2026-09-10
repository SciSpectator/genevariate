#!/usr/bin/env python3
"""
Convert a Gemini-generated logo (with checkerboard fake-transparency)
into a clean PNG with proper alpha-channel transparency.

Usage:
    python3 tools/make_transparent_logo.py SOURCE.png

The script writes the result to:
    docs/logo.png
    src/genevariate/assets/icon.png
    src/genevariate/assets/icon_1024.png
    src/genevariate/assets/icon.ico

Algorithm
---------
The source image has the colored logo painted on a 1024x1024 canvas
with a checkerboard pattern (white + 202-gray squares) overlaid as
"fake" transparency. The Gemini render also includes a soft white
glow around every element which appears as a halo on dark backgrounds.

To get a clean transparent result we use STRICT classification at
high resolution + LANCZOS downsample:

  1. Upscale source 4x with LANCZOS for sub-pixel precision.
  2. Per-pixel binary classification:
       opaque = (saturation > SAT_T) OR (luminance < LUM_T)
     This keeps fully-saturated logo colors and dark text pixels and
     rejects everything that is gray-and-bright (checker + halo glow).
  3. Morphological close to fill tiny holes inside logo shapes.
  4. Convert binary mask to alpha and downsample 4x with LANCZOS.
     This gives anti-aliased edges with no halo residue.
  5. For partial-alpha pixels at the boundary, un-mix the neutral
     checker color so the logo stays pure on dark backgrounds.

The thresholds are intentionally aggressive: any 'glow' from the
original Gemini render is sacrificed in exchange for clean edges.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

# Tunables
UPSCALE = 4
SAT_T = 28.0      # saturation threshold -- pixels with color
LUM_T = 130.0     # luminance threshold -- dark text/outlines
ERODE_RADIUS = 6  # pixels (at supersampled scale) to shrink to remove glow
NEUTRAL_BG = np.array([228.0, 228.0, 228.0], dtype=np.float32)  # avg of 255 + 202


def binary_close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Tiny morphological close (dilate then erode) using PIL."""
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.filter(ImageFilter.MaxFilter(2 * radius + 1))
    img = img.filter(ImageFilter.MinFilter(2 * radius + 1))
    return (np.array(img) > 127)


def binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological erosion -- shrinks the mask by `radius` px."""
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.filter(ImageFilter.MinFilter(2 * radius + 1))
    return (np.array(img) > 127)


def extract(source_path: str) -> Image.Image:
    img = Image.open(source_path).convert("RGBA")
    print(f"Source size: {img.size}")

    big_size = (img.width * UPSCALE, img.height * UPSCALE)
    big = img.resize(big_size, Image.LANCZOS)
    src = np.array(big).astype(np.float32)
    rgb = src[..., :3]

    sat = rgb.max(axis=-1) - rgb.min(axis=-1)
    lum = rgb.mean(axis=-1)

    # Strict binary mask: only saturated colors or dark pixels survive
    opaque = (sat > SAT_T) | (lum < LUM_T)

    # Fill tiny holes inside logo shapes
    opaque = binary_close(opaque, radius=2)

    # Erode by a few pixels (at the supersampled scale) so the soft
    # white "outer glow" of the source logo doesn't survive as a halo.
    opaque = binary_erode(opaque, radius=ERODE_RADIUS)
    # Re-close to undo any minor breakage of thin features
    opaque = binary_close(opaque, radius=1)

    # Convert to alpha (uint8) and downsample 4x for AA
    alpha_big = Image.fromarray((opaque.astype(np.uint8) * 255))
    alpha = np.array(alpha_big.resize((img.width, img.height), Image.LANCZOS)) \
        .astype(np.float32) / 255.0

    # Color: downsample the source RGB too (already at big size)
    rgb_big_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    rgb_small = np.array(
        rgb_big_img.resize((img.width, img.height), Image.LANCZOS)
    ).astype(np.float32)

    # Un-mix neutral bg from partial-alpha pixels
    a3 = alpha[..., None]
    safe_a = np.where(a3 > 0.04, a3, 1.0)
    F = (rgb_small - (1 - a3) * NEUTRAL_BG) / safe_a
    F = np.clip(F, 0, 255)
    F = np.where(a3 > 0.04, F, 0)

    out = np.concatenate([F, alpha[..., None] * 255], axis=-1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def write_outputs(img: Image.Image, repo_root: str) -> None:
    targets = [
        os.path.join(repo_root, "docs", "logo.png"),
        os.path.join(repo_root, "src", "genevariate", "assets", "icon.png"),
        os.path.join(repo_root, "src", "genevariate", "assets", "icon_1024.png"),
    ]
    for p in targets:
        img.save(p, "PNG", optimize=True)
        print("Wrote:", p)
    ico = os.path.join(repo_root, "src", "genevariate", "assets", "icon.ico")
    img.save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("Wrote:", ico)

    a8 = np.array(img)[..., 3]
    print(
        f"Alpha: {len(np.unique(a8))} unique values | "
        f"transparent {(a8 == 0).mean()*100:.1f}% | "
        f"opaque {(a8 == 255).mean()*100:.1f}% | "
        f"partial {((a8 > 0) & (a8 < 255)).mean()*100:.1f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to the source PNG with checker bg")
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print(f"Error: source not found: {args.source}", file=sys.stderr)
        return 1

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Repo root:", repo_root)

    img = extract(args.source)
    # Quick sanity preview against dark bg, written to /tmp
    preview = Image.new("RGB", img.size, (13, 17, 23))
    preview.paste(img, mask=img.split()[3])
    preview_path = "/tmp/logo_preview_dark.png"
    preview.save(preview_path)
    print("Preview (dark bg):", preview_path)

    write_outputs(img, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
