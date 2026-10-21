#!/usr/bin/env python3
"""
Rescale a whole-slide image to a target pixel size and apply a device-link ICC profile, in one
chunked pass with rasterio. Writes a tiled BigTIFF with overviews; memory use is constant.

usage:
  python svs_scale_icc.py slide.svs out.tif --icc src_to_ref_devicelink.icc --target 0.25
  python svs_scale_icc.py slide.svs out.tif --target 0.25                  # scaling only
  python svs_scale_icc.py slide.svs out.tif --icc link.icc --mpp 0.263089  # MPP not in the file

- Source: first image of the file (level 0 of an SVS), read through GDAL. Pixel size from the
  Aperio description ("MPP = ..."), or --mpp.
- Scaling: rasterio WarpedVRT (the GDAL warper, same as gdalwarp -tr), --resampling cubic by
  default; output grid anchored at the top-left corner of the source, size rounded down to whole
  output pixels (no padded edge). Skipped when --target is not given.
- Colour: the device link's table evaluated once for all 256^3 RGB values with tetrahedral
  interpolation (identical to LittleCMS without precalculation, tificc -c0), then a table lookup
  per pixel. Skipped when --icc is not given.
- Output: BigTIFF, tiled (--tile 512), ZSTD level 1 + horizontal predictor by default
  (--compress deflate|zstd|lzw|jpeg|none), resolution tags (pixels per cm) and a description, no
  GeoTIFF tags, no embedded ICC profile; overviews 2..32 (average), --world writes a .wld file.
- Parallel: --workers threads read/warp/convert blocks (each with its own dataset handles); GDAL
  compresses with all cores.

Requires rasterio, numpy; apply_icc_patch.py next to this script (for --icc).
"""
import argparse
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import warnings

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

FAKE_CRS = "EPSG:3857"   # any projected CRS: only used so the warper has consistent (micrometre) units


def read_mpp(src):
    desc = src.tags().get("TIFFTAG_IMAGEDESCRIPTION", "") or src.descriptions[0] or ""
    m = re.search(r"MPP\s*=\s*([0-9.]+)", desc)
    return float(m.group(1)) if m else None


def lut_table(icc_path):
    """256^3 x 3 uint8 table from a device link (tetrahedral, as LittleCMS -c0)."""
    import apply_icc_patch as AP
    cin, clut, cout = AP.read_devicelink(icc_path)
    table = np.empty((256 ** 3, 3), np.uint8)
    gg, bb = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    for r in range(256):
        x = np.stack([np.full(65536, r), gg.ravel(), bb.ravel()], 1) / 255.0
        y = AP.curves(AP.tetrahedral(clut, AP.curves(x, cin)), cout)
        table[r * 65536:(r + 1) * 65536] = np.clip(np.round(y * 255), 0, 255)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--icc", help="device-link profile (RGB -> RGB); omit for scaling only")
    ap.add_argument("--target", type=float, help="output pixel size, um/px (omit: keep size)")
    ap.add_argument("--mpp", type=float, help="source pixel size, um/px (default: from the Aperio description)")
    ap.add_argument("--resampling", default="cubic", choices=["nearest", "bilinear", "cubic", "lanczos", "average"])
    ap.add_argument("--compress", default="zstd", choices=["zstd", "deflate", "lzw", "jpeg", "none"])
    ap.add_argument("--level", type=int, default=1, help="ZSTD/DEFLATE level, or JPEG quality (default 1; JPEG: 95)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="threads for read/warp/convert")
    ap.add_argument("--no-overviews", action="store_true")
    ap.add_argument("--world", action="store_true", help="also write <dst>.wld (pixel size, centre of pixel (0,0) at 0)")
    args = ap.parse_args()
    t0 = time.time()

    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_PAM_ENABLED="NO"):
        with rasterio.open(args.src) as src:
            if src.count < 3 or src.dtypes[0] != "uint8":
                sys.exit(f"expected 8-bit RGB, got {src.count} bands {src.dtypes[0]}")
            mpp = args.mpp or read_mpp(src)
            if args.target and not mpp:
                sys.exit("no MPP found in the file; give --mpp")
            W0, H0 = src.width, src.height
        scale = (mpp / args.target) if args.target else 1.0
        W, H = (int(np.floor(W0 * scale)), int(np.floor(H0 * scale))) if args.target else (W0, H0)
        out_mpp = args.target or mpp
        print(f"source {W0} x {H0} px @ {mpp} um/px -> output {W} x {H} px @ {out_mpp} um/px", flush=True)

        table = None
        if args.icc:
            t1 = time.time(); table = lut_table(args.icc); print(f"colour table from {os.path.basename(args.icc)} ({time.time() - t1:.1f} s)", flush=True)

        comp = {"zstd": ("ZSTD", {"ZSTD_LEVEL": args.level, "PREDICTOR": 2}), "deflate": ("DEFLATE", {"ZLEVEL": args.level, "PREDICTOR": 2}),
                "lzw": ("LZW", {"PREDICTOR": 2}), "jpeg": ("JPEG", {"JPEG_QUALITY": args.level if args.level > 1 else 95, "PHOTOMETRIC": "RGB"}),
                "none": ("NONE", {})}[args.compress]
        profile = dict(driver="GTiff", width=W, height=H, count=3, dtype="uint8", tiled=True, blockxsize=args.tile, blockysize=args.tile,
                       compress=comp[0], BIGTIFF="YES", PROFILE="BASELINE", NUM_THREADS="ALL_CPUS", **comp[1])
        if args.compress != "jpeg":
            profile["photometric"] = "RGB"

        local = threading.local()
        resampling = getattr(Resampling, args.resampling)

        def reader():
            if not hasattr(local, "ds"):
                local.ds = rasterio.open(args.src)
                if args.target:
                    local.vrt = WarpedVRT(local.ds, src_crs=FAKE_CRS, src_transform=Affine.scale(mpp, mpp), crs=FAKE_CRS,
                                          transform=Affine.scale(out_mpp, out_mpp), width=W, height=H, resampling=resampling,
                                          warp_mem_limit=256)
                else:
                    local.vrt = local.ds
            return local.vrt

        def work(win):
            a = reader().read(indexes=[1, 2, 3], window=win)                          # (3, h, w) uint8
            if table is not None:
                key = (a[0].astype(np.uint32) << 16) | (a[1].astype(np.uint32) << 8) | a[2]
                a = np.moveaxis(table[key], -1, 0)
            return win, a

        wins = [Window(c, r, min(args.tile, W - c), min(args.tile, H - r)) for r in range(0, H, args.tile) for c in range(0, W, args.tile)]
        tmp = args.dst + ".part"
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
            pxcm = 1e4 / out_mpp
            dst.update_tags(TIFFTAG_XRESOLUTION=f"{pxcm:.6f}", TIFFTAG_YRESOLUTION=f"{pxcm:.6f}", TIFFTAG_RESOLUTIONUNIT="3",
                            TIFFTAG_IMAGEDESCRIPTION=f"{os.path.basename(args.src)}: {mpp} -> {out_mpp} um/px ({args.resampling})"
                                                     + (f"; colour: {os.path.basename(args.icc)}" if args.icc else ""))
            done = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for win, a in ex.map(work, wins):                                     # in order; GDAL writes sequentially
                    dst.write(a, window=win)
                    done += 1
                    if done % 500 == 0 or done == len(wins):
                        print(f"\r  {done}/{len(wins)} tiles ({time.time() - t0:.0f} s)", end="", flush=True)
            print()
            if not args.no_overviews:
                f, levels = 2, []
                while min(W, H) // f >= args.tile // 2:
                    levels.append(f); f *= 2
                if levels:
                    dst.build_overviews(levels, Resampling.average)
                    print(f"overviews {levels}", flush=True)
        os.replace(tmp, args.dst)
        if args.world:
            with open(os.path.splitext(args.dst)[0] + ".wld", "w") as w:
                w.write(f"{out_mpp}\n0\n0\n{-out_mpp}\n0\n0\n")
    print(f"wrote {args.dst} ({os.path.getsize(args.dst) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
