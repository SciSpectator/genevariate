#!/usr/bin/env python3
"""
Smooth the silhouette edges of the genevariate alpha-PNG logo.

The current logo (after the rim-cleanup pass) has clean colors but
stepped/jagged edges in places because the alpha mask started life
as a binary mask + morphological ops + LANCZOS downsample. This tool
softens those edges by:

  1. Reading the existing alpha-PNG.
  2. Upscaling 3x with LANCZOS to gain sub-pixel resolution.
  3. Applying a small Gaussian blur to the alpha channel only
     (RGB is left alone) -- the blur radius is in the upscaled
     coordinate system, so 1.8 px upscaled = 0.6 px at native.
  4. To stop the silhouette from shrinking, the alpha is brightened
     slightly before blur using a soft S-curve so the 50%-alpha
     contour stays in the same place.
  5. Downsampling back to native size with LANCZOS for a final
     anti-aliased result.

The colour channels are *not* blurred -- that would re-introduce the
washed-out boundary the previous tools removed. Only the alpha is
smoothed, which gives properly anti-aliased silhouettes against any
background.

Usage:
    python3 tools/smooth_logo_edges.py [INPUT_PNG]

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
from PIL import Image, ImageFilter


UPSCALE = 3
ALPHA_BLUR_RADIUS = 1.8   # in upscaled pixels (~0.6 px at native)
# Mild S-curve to keep the 50%-alpha contour stable after blur.
# alpha' = clip( (alpha - 0.5) * GAIN + 0.5 + BIAS, 0, 1 )
SCURVE_GAIN = 1.06
SCURVE_BIAS = 0.0


def smooth_alpha(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    # Upscale RGB and alpha separately
    big_size = (w * UPSCALE, h * UPSCALE)
    rgb_big = Image.fromarray(rgb).resize(big_size, Image.LANCZOS)
    alpha_big = Image.fromarray(alpha).resize(big_size, Image.LANCZOS)

    # Soft S-curve to compensate for the slight silhouette shrink
    # that Gaussian blur introduces, then blur.
    a_arr = np.array(alpha_big).astype(np.float32) / 255.0
    a_arr = np.clip((a_arr - 0.5) * SCURVE_GAIN + 0.5 + SCURVE_BIAS, 0, 1)
    a_curved = Image.fromarray((a_arr * 255).astype(np.uint8))
    a_blurred = a_curved.filter(ImageFilter.GaussianBlur(radius=ALPHA_BLUR_RADIUS))

    # Downsample both back to native size
    rgb_small = rgb_big.resize((w, h), Image.LANCZOS)
    alpha_small = a_blurred.resize((w, h), Image.LANCZOS)

    out = np.empty_like(arr)
    out[..., :3] = np.array(rgb_small)
    out[..., 3] = np.array(alpha_small)
    # Where alpha collapses to zero, also clear RGB
    out[out[..., 3] == 0, :3] = 0

    return Image.fromarray(out)


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

    a = np.array(img)[..., 3]
    n_unique = len(np.unique(a))
    n_partial = ((a > 0) & (a < 255)).sum()
    print(
        f"Alpha: {n_unique} unique values | "
        f"partial-alpha pixels (smooth band): {n_partial:,}"
    )


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=os.path.join(repo_root, "docs", "logo.png"),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    print("Repo root:", repo_root)
    print("Input:    ", args.input)

    src = Image.open(args.input)
    print(f"Source size: {src.size}, mode: {src.mode}")

    smoothed = smooth_alpha(src)

    preview = Image.new("RGB", smoothed.size, (13, 17, 23))
    preview.paste(smoothed, mask=smoothed.split()[3])
    preview_path = "/tmp/logo_smoothed_dark.png"
    preview.save(preview_path)
    print("Preview (dark bg):", preview_path)

    light_preview = Image.new("RGB", smoothed.size, (245, 245, 245))
    light_preview.paste(smoothed, mask=smoothed.split()[3])
    light_path = "/tmp/logo_smoothed_light.png"
    light_preview.save(light_path)
    print("Preview (light bg):", light_path)

    write_outputs(smoothed, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
