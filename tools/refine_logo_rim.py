#!/usr/bin/env python3
"""
Second-pass rim refiner for the genevariate logo.

Reads the existing alpha-PNG at docs/logo.png and produces a cleaner
version by re-classifying pixels with a tighter "trusted core" rule
and a separate "text" preservation rule.

Why this exists:
  After the first rim cleanup we still have two artifacts:
    * The DNA helix shows bright low-saturation rims because the prior
      core threshold (sat >= 35) admitted glow-bleached pixels whose
      RGB is e.g. (157, 198, 171) -- saturated by 41, but min channel
      157 means it's really just bright haze, not a deep logo color.
    * The "GenveVariate" wordmark is mid-gray (sat ~5, lum ~65) and so
      gets repainted from a nearby ribbon, dyeing it green/blue.

Fix:
  1. Tighter core: a pixel is trusted only if alpha == 255 AND its
     minimum channel < 120 (so it's a deep/dark color, not haze) AND
     either (sat >= 50) or (lum <= 60).
  2. Text-preserve: pixels that are mostly opaque, neutral (sat <= 20),
     and mid-luminance (50..160) are treated as text -- their RGB is
     left untouched so they stay the original dark gray.
  3. Everything else visible (alpha > 0) gets repainted from the
     nearest trusted core pixel via Euclidean distance transform.

Alpha is NEVER modified -- silhouette is preserved bit-for-bit.

Usage:
    python3 tools/refine_logo_rim.py [INPUT_PNG]

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


# Tighter core criteria:
#   - fully opaque
#   - at least one channel is dark-ish (min < 120) so it isn't bright haze
#   - AND either saturated (>=50) or genuinely dark (lum <= 60)
CORE_ALPHA = 255
CORE_MIN_CHANNEL_MAX = 120
CORE_SAT_MIN = 50.0
CORE_DARK_MAX = 60.0

# Text preservation: mostly-opaque neutral mid-gray pixels that should
# keep their existing RGB rather than being overwritten by a ribbon color.
TEXT_ALPHA_MIN = 100
TEXT_SAT_MAX = 20.0
TEXT_LUM_MIN = 50.0
TEXT_LUM_MAX = 160.0


def refine(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3]

    sat = rgb.max(axis=-1) - rgb.min(axis=-1)
    lum = rgb.mean(axis=-1)
    min_ch = rgb.min(axis=-1)

    core = (
        (alpha == CORE_ALPHA)
        & (min_ch < CORE_MIN_CHANNEL_MAX)
        & ((sat >= CORE_SAT_MIN) | (lum <= CORE_DARK_MAX))
    )

    text = (
        (alpha >= TEXT_ALPHA_MIN)
        & (sat <= TEXT_SAT_MAX)
        & (lum >= TEXT_LUM_MIN)
        & (lum <= TEXT_LUM_MAX)
    )

    visible = alpha > 0
    keep_unchanged = core | text
    needs_repaint = visible & ~keep_unchanged

    print(
        f"Pixels: visible={visible.sum():,}  "
        f"core={core.sum():,}  text-preserve={text.sum():,}  "
        f"to repaint={needs_repaint.sum():,}"
    )

    if not core.any():
        print("ERROR: no core pixels under the tighter rule.", file=sys.stderr)
        return img

    new_rgb = rgb.copy()
    if needs_repaint.any():
        _, (iy, ix) = distance_transform_edt(~core, return_indices=True)
        ys, xs = np.nonzero(needs_repaint)
        new_rgb[ys, xs] = rgb[iy[ys, xs], ix[ys, xs]]

    out = np.empty_like(arr)
    out[..., :3] = np.clip(new_rgb, 0, 255).astype(np.uint8)
    out[..., 3] = alpha
    out[alpha == 0, :3] = 0

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

    arr = np.array(img)
    a8 = arr[..., 3]
    rgb = arr[..., :3].astype(np.float32)
    band = (a8 >= 200) & (a8 < 255)
    if band.any():
        print(
            f"Boundary band (alpha 200-254): "
            f"{band.sum():,} pixels  mean lum={rgb[band].mean(axis=1).mean():.1f}"
        )
    bright_inside = (a8 == 255) & (rgb.mean(axis=-1) > 170) & (
        rgb.max(axis=-1) - rgb.min(axis=-1) < 50
    )
    print(f"Opaque bright low-sat pixels remaining: {bright_inside.sum():,}")


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
    refined = refine(src)

    preview = Image.new("RGB", refined.size, (13, 17, 23))
    preview.paste(refined, mask=refined.split()[3])
    preview_path = "/tmp/logo_refined_dark.png"
    preview.save(preview_path)
    print("Preview (dark bg):", preview_path)

    write_outputs(refined, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
