#!/usr/bin/env python3
"""
Gaussian pyramid of every patch in an HDF5 file, written as individual TIFFs.
Level 0 is the input; output starts at level 1 (half size), then 1/4, 1/8, ...

usage: python h5_pyramid.py IN.h5 OUT_DIR [--levels 3] [--sigma 1.0] [--dtype uint8] [--shard 1/8]

  OUT_DIR/<patch>.tif : one pyramidal TIFF per patch, L1 (H/2 x W/2) first, then L2 (H/4),
  L3 (H/8), ... The original (level 0) is never written. Lower levels have NewSubfileType = 1.
  --layout pages  (default) levels are consecutive top-level directories, like SVS and the
                  files OpenSlide reads; tiffinfo lists every level.
  --layout subifd levels are SubIFDs of the first image (OME-TIFF convention).
  --only N : compute the steps up to level N but write only level N, as a plain
  single-image TIFF (no SubIFDs), e.g. --only 2 -> 300 x 300 for 1200 x 1200 patches.
  All levels are tiled (--tile, default 256; tiles of small levels are padded).
  Read: tifffile.TiffFile(path).pages[k].asarray() (pages layout) or
        tifffile.TiffFile(path).series[0].levels[k].asarray() (subifd layout)  ->  L1, L2, ...
        openslide.OpenSlide(path).read_region(...) also works with the pages layout.

Each reduce step: Gaussian blur (--sigma, in pixels of the finer level) then 2x2 averaging,
so level-n pixel (i, j) is centred on full-resolution position (2^n * i + (2^n - 1) / 2, ...),
the same grid as common image resizers and identical for every input file.
Filtering is done in linear light by default (--encoded to filter the sRGB-encoded values
instead); intermediate levels stay float, each level is quantized once on output.

Effective blur at level n, in full-resolution pixels, with s = sqrt(sigma^2 + 0.25) per step:
  sigma_n = s * sqrt((4^n - 1) / 3)   ->  sigma 1.0: L1 1.1, L2 2.5, L3 5.1, L4 10.3 px
It is stored in each TIFF's ImageDescription (first line 'h5_pyramid v1', then JSON; see read_meta).

Input: datasets of shape (H, W, 3) uint8 (flat names such as "0", "1" or tile_<l>_<x>_<y>).
Output: RGB, deflate-compressed with a predictor (horizontal for integers, floating-point for
floats). --dtype: uint8 (default; sRGB-encoded 0..255), uint16 (0..65535), float16 or float32
(0..1). --linear-out stores linear-light values instead of sRGB-encoded (use a float type).
Requires numpy, scipy, h5py, tifffile (+ imagecodecs for the float predictor -> smaller float files).
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import tifffile
from scipy import ndimage


def dec(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def enc(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def reduce2(a, sigma):
    """Blur each channel, then average 2x2 blocks (odd trailing row/column dropped)."""
    if sigma > 0:
        a = np.stack([ndimage.gaussian_filter(a[..., k], sigma, mode="reflect") for k in range(a.shape[-1])], -1)
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2, -1).mean((1, 3))


def effective_sigma(sigma, n):
    s = np.sqrt(sigma ** 2 + 0.25)
    return float(s * np.sqrt((4 ** n - 1) / 3))


def patch_ids(h5):
    return [k for k, v in h5.items() if isinstance(v, h5py.Dataset) and v.ndim == 3 and v.shape[-1] == 3]


def sort_key(k):
    if k.isdigit():
        return (0, int(k), "")
    parts = k.split("_")[1:]
    if parts and all(p.isdigit() for p in parts):
        return (1, tuple(int(p) for p in parts), "")
    return (2, (), k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5"); ap.add_argument("out")
    ap.add_argument("--levels", type=int, default=3, help="number of levels to write, starting at L1 (default 3)")
    ap.add_argument("--only", type=int, default=0, metavar="N", help="write only level N (single-image TIFF)")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma per reduce step, px of the finer level")
    ap.add_argument("--encoded", action="store_true", help="filter sRGB-encoded values instead of linear light")
    ap.add_argument("--dtype", choices=["uint8", "uint16", "float16", "float32"], default="uint8")
    ap.add_argument("--linear-out", action="store_true", help="store linear light instead of sRGB-encoded values")
    ap.add_argument("--tile", type=int, default=256, help="tile size (multiple of 16)")
    ap.add_argument("--layout", choices=["pages", "subifd"], default="pages",
                    help="levels as top-level directories (default) or as SubIFDs")
    ap.add_argument("--shard", default="1/1", help="i/n: every n-th patch starting at i (parallel runs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    i, n = (int(v) for v in args.shard.split("/"))
    os.makedirs(args.out, exist_ok=True)
    if args.linear_out and args.dtype.startswith("uint"):
        print("note: linear light in integers loses precision in the darks; prefer float16/float32", file=sys.stderr)
    t0 = time.time()
    with h5py.File(args.h5, "r") as f:
        ids = sorted(patch_ids(f), key=sort_key)[i - 1::n]
        for k, pid in enumerate(ids, 1):
            path = os.path.join(args.out, f"{pid}.tif")
            if not args.overwrite and os.path.exists(path):
                continue
            a = f[pid][()].astype(np.float32) / 255.0
            a = a if args.encoded else dec(a).astype(np.float32)
            levels = []                                  # level 0 (original) is not stored
            for lv in range(1, (args.only or args.levels) + 1):
                a = reduce2(a, args.sigma)
                if min(a.shape[:2]) < 1:
                    break
                if not args.only or lv == args.only:
                    levels.append((lv, a))
            if not levels:
                print(f"{pid}: too small for the requested level", file=sys.stderr); continue
            meta = dict(source=os.path.basename(args.h5), patch=pid, dtype=args.dtype,
                        values=("linear light" if args.linear_out else "sRGB-encoded") + ", 0..1 scaled to dtype range",
                        filtered_in="sRGB-encoded" if args.encoded else "linear light", sigma_step=args.sigma,
                        levels=[dict(level=lv, scale=1 / 2 ** lv, shape=list(x.shape[:2]),
                                     sigma_full_res_px=round(effective_sigma(args.sigma, lv), 3),
                                     grid=f"pixel (i,j) centred at full-res ({2 ** lv}*i + {(2 ** lv - 1) / 2}, {2 ** lv}*j + {(2 ** lv - 1) / 2})")
                                for lv, x in levels])
            tmp = path + ".part"
            with tifffile.TiffWriter(tmp) as tw:
                for j, (lv, x) in enumerate(levels):
                    if args.linear_out == (not args.encoded):        # stored values already in the requested space
                        v = x
                    else:
                        v = dec(x) if args.linear_out else enc(x)
                    v = to_dtype(v, args.dtype)
                    kw = dict(photometric="rgb", compression="zlib", metadata=None,
                              predictor=(3 if HAVE_IMAGECODECS else None) if v.dtype.kind == "f" else 2)
                    kw["tile"] = (args.tile, args.tile)
                    if j == 0:
                        sub = (len(levels) - 1) or None if args.layout == "subifd" else None
                        tw.write(v, subifds=sub, description=DESC_PREFIX + json.dumps(meta), **kw)
                    else:
                        tw.write(v, subfiletype=1, **kw)
            os.replace(tmp, path)
            if k % 100 == 0 or k == len(ids):
                print(f"{os.path.basename(args.h5)}: {k}/{len(ids)} patches ({k / (time.time() - t0):.1f}/s)", flush=True)


DESC_PREFIX = "h5_pyramid v1\n"     # then JSON; the prefix keeps readers from mistaking it for tifffile's own JSON metadata
try:
    import imagecodecs  # noqa: F401  (enables the floating-point predictor -> smaller float files)
    HAVE_IMAGECODECS = True
except ImportError:
    HAVE_IMAGECODECS = False


def read_meta(path):
    """Metadata written by this script: json.loads of the ImageDescription after the first line."""
    with tifffile.TiffFile(path) as t:
        return json.loads(t.pages[0].description.split("\n", 1)[1])


def to_dtype(v, dtype):
    v = np.clip(v, 0, 1)
    if dtype == "uint8":
        return np.round(v * 255).astype(np.uint8)
    if dtype == "uint16":
        return np.round(v * 65535).astype(np.uint16)
    return v.astype(dtype)

if __name__ == "__main__":
    main()
