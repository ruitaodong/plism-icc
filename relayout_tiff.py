#!/usr/bin/env python3
"""
Convert pyramidal TIFFs between the two layouts, losslessly (pixels copied, recompressed with
the same codec, predictor and tile size; ImageDescription kept).

  pages  : levels as consecutive top-level directories (OpenSlide/SVS style; tiffinfo lists all)
  subifd : levels as SubIFDs of the first image (OME-TIFF style)

usage:
  python relayout_tiff.py pyr_dir --out pyr_pages            # convert every *.tif into pyr_pages/
  python relayout_tiff.py pyr_dir/*.tif --inplace            # rewrite in place (via a temp file)
  python relayout_tiff.py pyr_dir --to subifd --out pyr_ome  # the other direction

Files already in the requested layout are copied unchanged (or skipped with --inplace).
libtiff's tiffcp is not an option: it drops SubIFDs and keeps only the first image.
Requires tifffile (+ imagecodecs for float predictors).
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import tifffile

COMPRESSION = {1: None, 8: "zlib", 32946: "zlib", 5: "lzw", 7: "jpeg", 50000: "zstd", 34925: "lzma", 50001: "webp"}


def read_levels(path):
    """Return (levels, description, write kwargs, current layout); levels = list of arrays, largest first."""
    with tifffile.TiffFile(path) as t:
        p0 = t.pages[0]
        if p0.subifds:
            layout = "subifd"
            levels = [p0.asarray()]
            for off in p0.subifds:
                t.filehandle.seek(off)
                levels.append(tifffile.TiffPage(t, index=0).asarray())
        else:
            layout = "pages"
            levels = [p.asarray() for p in t.pages]
        desc = p0.description
        comp = COMPRESSION.get(int(p0.compression), None)
        if int(p0.compression) not in COMPRESSION:
            print(f"  note: {path}: compression {p0.compression} -> writing zlib", file=sys.stderr)
            comp = "zlib"
        kw = dict(photometric=p0.photometric.name.lower() if p0.photometric is not None else None,
                  compression=comp, metadata=None,
                  tile=(p0.tilelength, p0.tilewidth) if p0.is_tiled else None)
        pred = int(p0.predictor) if comp in ("zlib", "lzw", "zstd", "lzma") else 1
        kw["predictor"] = pred if pred in (2, 3) else None
    return levels, desc, kw, layout


def write_levels(path, levels, desc, kw, layout):
    sizes = [lv.shape[:2] for lv in levels]
    if any(a[0] < b[0] or a[1] < b[1] for a, b in zip(sizes, sizes[1:])):
        raise ValueError(f"levels are not ordered largest first: {sizes}")
    with tifffile.TiffWriter(path) as tw:
        for j, lv in enumerate(levels):
            k = dict(kw)
            if k["tile"] is None and min(lv.shape[:2]) >= 16:
                k.pop("tile")
            if j == 0:
                tw.write(lv, subifds=(len(levels) - 1 or None) if layout == "subifd" else None,
                         description=desc or None, **k)
            else:
                tw.write(lv, subfiletype=1, **k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="TIFF files and/or directories (*.tif, *.tiff)")
    ap.add_argument("--to", choices=["pages", "subifd"], default="pages")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="output directory")
    g.add_argument("--inplace", action="store_true", help="replace the input files")
    ap.add_argument("--no-verify", action="store_true", help="skip re-reading the output to compare pixels")
    args = ap.parse_args()

    files = []
    for x in args.inputs:
        files += sorted(glob.glob(os.path.join(x, "*.tif")) + glob.glob(os.path.join(x, "*.tiff"))) if os.path.isdir(x) else [x]
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    done = same = 0
    for f in files:
        levels, desc, kw, layout = read_levels(f)
        dst = f if args.inplace else os.path.join(args.out, os.path.basename(f))
        if layout == args.to:
            same += 1
            if not args.inplace and os.path.abspath(dst) != os.path.abspath(f):
                shutil.copy2(f, dst)
            continue
        tmp = dst + ".part"
        write_levels(tmp, levels, desc, kw, args.to)
        if not args.no_verify:
            back, _, _, lay = read_levels(tmp)
            if lay != args.to or len(back) != len(levels) or not all(np.array_equal(a, b) for a, b in zip(levels, back)):
                os.remove(tmp)
                sys.exit(f"verification failed for {f}; input left unchanged")
        os.replace(tmp, dst)
        done += 1
    print(f"{done} converted to '{args.to}', {same} already in that layout ({len(files)} files)")


if __name__ == "__main__":
    main()
