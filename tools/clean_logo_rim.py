#!/usr/bin/env python3
"""
Repaint the washed-out inner rim of an alpha-PNG logo.

After stripping the source's checkerboard fake-transparency we are left
with a thin band of nearly-opaque, low-saturation pixels just inside
each shape edge. Those are the source's outer-glow bleeding into the
logo's RGB channel; on a dark background they composite to a visible
white-ish ring even though the alpha is correct.

This script keeps the alpha channel exactly as-is and only touches RGB.
For every pixel that is partially or fully visible (alpha > 0) but whose
color cannot be trusted (too desaturated and too bright to be part of
the actual logo), we replace its RGB with the RGB of the nearest pixel
that we *do* trust -- a fully-opaque, saturated-or-dark "core" pixel.

Result: the silhouette is preserved bit-for-bit, but the rim takes on
the color of the saturated interior next to it instead of the glow's
washed-out gray-teal.

Usage:
    python3 tools/clean_logo_rim.py [INPUT_PNG]

If INPUT_PNG is omitted, defaults to docs/logo.png.

Outputs (overwrites in place):
    docs/logo.png
    src/genevariate/assets/icon.png
    src/genevariate/assets/icon_1024.png  (only if it already exists)
    src/genevariate/assets/icon.ico
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


# Trust criteria for "core" pixels (these define the color palette
# we will copy outward into the rim).
CORE_ALPHA_MIN = 250        # essentially fully opaque
CORE_SAT_MIN = 35.0         # colorful enough to be a real logo color
CORE_DARK_MAX = 70.0        # OR dark enough to be text/outline

# Pixels considered "rim" (will be repainted from the nearest core pixel).
# Anything that is visible (alpha > 0) and not in the core set.
# We deliberately do NOT touch fully-transparent pixels -- they stay (0,0,0,0).


def repaint_rim(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3]

    sat = rgb.max(axis=-1) - rgb.min(axis=-1)
    lum = rgb.mean(axis=-1)

    core = (alpha >= CORE_ALPHA_MIN) & ((sat >= CORE_SAT_MIN) | (lum <= CORE_DARK_MAX))
    visible = alpha > 0
    needs_repaint = visible & ~core

    print(
        f"Pixels: visible={visible.sum():,}  "
        f"core(trusted)={core.sum():,}  "
        f"to repaint={needs_repaint.sum():,}"
    )

    if not core.any():
        print("ERROR: no trusted core pixels found; cannot repaint.", file=sys.stderr)
        return img
    if not needs_repaint.any():
        print("Nothing to repaint -- returning input unchanged.")
        return img

    # For every non-core pixel, find the (y, x) of the nearest core pixel.
    # distance_transform_edt computes distance from non-zero pixels to the
    # nearest zero pixel, so we feed it ~core (zeros = core, ones = elsewhere).
    _, (iy, ix) = distance_transform_edt(~core, return_indices=True)

    new_rgb = rgb.copy()
    ys, xs = np.nonzero(needs_repaint)
    new_rgb[ys, xs] = rgb[iy[ys, xs], ix[ys, xs]]

    out = np.empty_like(arr)
    out[..., :3] = np.clip(new_rgb, 0, 255).astype(np.uint8)
    out[..., 3] = alpha
    # Keep the convention of zeroing RGB where alpha is zero
    out[alpha == 0, :3] = 0

    return Image.fromarray(out, mode="RGBA")


def write_outputs(img: Image.Image, repo_root: str) -> None:
    targets = [
        os.path.join(repo_root, "docs", "logo.png"),
        os.path.join(repo_root, "src", "genevariate", "assets", "icon.png"),
    ]
    high_res = os.path.join(repo_root, "src", "genevariate", "assets", "icon_1024.png")
    if os.path.exists(high_res):
        targets.append(high_res)
    for p in targets:
        img.save(p, "PNG", optimize=True)
        print("Wrote:", p)
    ico = os.path.join(repo_root, "src", "genevariate", "assets", "icon.ico")
    img.save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("Wrote:", ico)

    a8 = np.array(img)[..., 3]
    rgb = np.array(img)[..., :3].astype(np.float32)
    # Sanity report: how bright is the boundary ring now?
    band = (a8 >= 200) & (a8 < 255)
    if band.any():
        print(
            f"Boundary band (alpha 200-254): "
            f"{band.sum():,} pixels  mean lum={rgb[band].mean(axis=1).mean():.1f}"
        )


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=os.path.join(repo_root, "docs", "logo.png"),
        help="Path to the alpha-PNG to clean (default: docs/logo.png)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    print("Repo root:", repo_root)
    print("Input:    ", args.input)

    src = Image.open(args.input)
    print(f"Source size: {src.size}, mode: {src.mode}")

    cleaned = repaint_rim(src)

    # Quick preview against dark bg
    preview = Image.new("RGB", cleaned.size, (13, 17, 23))
    preview.paste(cleaned, mask=cleaned.split()[3])
    preview_path = "/tmp/logo_clean_preview_dark.png"
    preview.save(preview_path)
    print("Preview (dark bg):", preview_path)

    write_outputs(cleaned, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
