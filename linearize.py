#!/usr/bin/env python3
"""
Remove the gamma encoding from slide pixels: stored RGB -> values proportional to light.

usage:
  python linearize.py slide.svs out_dir --region X Y W H [--level 0]     # region read with OpenSlide
  python linearize.py patch1.png patch2.tif ... out_dir                    # patches that are still gamma-encoded

  --curve gamma   pure power law, exponent from the slide's metadata (aperio.Gamma) or --gamma (default 2.2)
  --curve srgb    the sRGB curve (linear toe + 2.4 power), as used elsewhere in plism-icc
  --out uint16    16-bit TIFF, 0..65535 = 0..1 linear (default)   |   --out float32   float TIFF, 0..1

Output: <name>_linear.tif. 8-bit output is not offered: in 8 bits, linear values keep only a few
levels for everything darker than about L* 20.

Only use this on pixels that are still gamma-encoded, e.g. regions read straight from an SVS with
OpenSlide. Images that were already decoded somewhere in an export pipeline must not be decoded again.
Requires numpy, tifffile, pillow; openslide-python for slides.
"""
import argparse
import os

import numpy as np
import tifffile
from PIL import Image


def decode(x, curve, gamma):
    x = np.clip(x, 0, 1)
    if curve == "srgb":
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x ** gamma


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="slide (with --region) or patch images, followed by the output directory")
    ap.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="region in level-0 coordinates (slides)")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--curve", choices=["gamma", "srgb"], default="gamma")
    ap.add_argument("--gamma", type=float, help="exponent for --curve gamma (default: aperio.Gamma, else 2.2)")
    ap.add_argument("--out", choices=["uint16", "float32"], default="uint16")
    a = ap.parse_args()
    *srcs, out_dir = a.inputs
    os.makedirs(out_dir, exist_ok=True)
    for src in srcs:
        g = a.gamma
        if a.region:
            import openslide
            s = openslide.OpenSlide(src)
            g = g or float(s.properties.get("aperio.Gamma", 2.2))
            x, y, w, h = a.region
            rgb = np.asarray(s.read_region((x, y), a.level, (w, h)).convert("RGB"))
            name = f"{os.path.splitext(os.path.basename(src))[0]}_x{x}_y{y}_l{a.level}"
        else:
            g = g or 2.2
            rgb = np.asarray(Image.open(src).convert("RGB"))
            name = os.path.splitext(os.path.basename(src))[0]
        lin = decode(rgb / 255.0, a.curve, g)
        data = np.round(lin * 65535).astype(np.uint16) if a.out == "uint16" else lin.astype(np.float32)
        path = os.path.join(out_dir, f"{name}_linear.tif")
        tifffile.imwrite(path, data, photometric="rgb", compression="zlib", predictor=True if a.out == "uint16" else None,
                         description=f"linear light from {os.path.basename(src)}: curve {a.curve}" + (f", gamma {g}" if a.curve == "gamma" else ""))
        print(f"{src} -> {path}  ({rgb.shape[1]}x{rgb.shape[0]}, {a.curve}" + (f" {g}" if a.curve == "gamma" else "") + f", {a.out})")


if __name__ == "__main__":
    main()
