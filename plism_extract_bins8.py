#!/usr/bin/env python3
"""
Extract 8x8 block-averaged tiles from PLISM .h5 slides, one output file per (stain, scanner),
indexed by patch id.

Input  : ./plism-dataset/<STAIN>_<SCANNER>[_to_GMH_S60].tif.h5
         Either flat per-tile datasets  tile_<level>_<x>_<y>  -> (224, 224, 3) uint8
         or a single (N, 224, 224, 3) dataset (patch id = row index).
Output : <out>/<STAIN>/<STAIN>_<SCANNER>_b8.h5
           sums       (N, 28, 28, 3) uint16  exact 8x8 block SUMS (mean = sums / 64, lossless)
           patch_id   (N,)  str             e.g. "tile_16_100_127", same order in every file
           coords     (N, 3) int32          (level, x, y) parsed from patch_id (-1 if not parseable)
           attrs: stain, scanner, source_file, block, tile_size, n_patches
         <out>/manifest.json   counts per file + patch ids missing vs. the stain's P slide

Rows are sorted by (level, x, y), so row i is the same patch id in every file that
has the same id set; use patch_id to join anyway.

Usage
  python plism_extract_bins8.py                                  # all stains, all 7 scanners
  python plism_extract_bins8.py --stains HRH --scanners GT450 P   # subset
  python plism_extract_bins8.py --max-tiles 4000                  # evenly spaced subset of ids
Already-finished outputs are skipped, so an interrupted run can just be restarted.

Size: ~77 MB raw per slide for all 16,278 patches (uint16), usually ~35-50 MB with gzip.

Load:
  import h5py, numpy as np
  with h5py.File("out/HRH/HRH_GT450_b8.h5") as f:
      ids = f["patch_id"].asstr()[:]; mean = f["sums"][:] / (64 * 255.0)   # float RGB in [0,1]
"""
import argparse
import json
import os
import re
import sys
import time

import h5py
import numpy as np

SCANNERS = ["P", "AT2", "GT450", "S210", "S360", "S60", "SQ"]
NAME_RE = re.compile(r"^(?P<stain>[A-Za-z0-9]+)_(?P<scanner>[A-Za-z0-9]+)(_to_[A-Za-z0-9_]+)?\.tif\.h5$")
TILE_RE = re.compile(r"^tile_(\d+)_(\d+)_(\d+)$")


def list_slides(data_dir):
    idx = {}
    for n in sorted(os.listdir(data_dir)):
        m = NAME_RE.match(n)
        if m:
            idx[(m["stain"], m["scanner"])] = os.path.join(data_dir, n)
    return idx


def coords_of(pid):
    m = TILE_RE.match(pid)
    return tuple(int(v) for v in m.groups()) if m else (-1, -1, -1)


def sort_key(pid):
    c = coords_of(pid)
    return (c, pid) if c[0] >= 0 else ((1 << 30, 0, 0), pid)


def open_tiles(h5):
    """Return (sorted patch ids, getter(list_of_ids) -> (n, H, W, 3) uint8)."""
    flat = [k for k, v in h5.items() if isinstance(v, h5py.Dataset) and v.ndim == 3 and v.shape[-1] == 3]
    if flat:
        ids = sorted(flat, key=sort_key)
        return ids, lambda sel: np.stack([h5[k][()] for k in sel])
    stacks = []
    h5.visititems(lambda k, o: stacks.append(o) if isinstance(o, h5py.Dataset) and o.ndim == 4 and o.shape[-1] == 3 else None)
    if not stacks:
        raise RuntimeError(f"no tiles found; top-level keys: {list(h5.keys())[:10]}")
    ds = max(stacks, key=lambda d: d.shape[0])
    ids = [f"row_{i:06d}" for i in range(ds.shape[0])]
    return ids, lambda sel: ds[sorted(int(s[4:]) for s in sel)]


def block_sums(tiles, b):
    n, h, w, c = tiles.shape
    if h % b or w % b:
        raise ValueError(f"tile size {h}x{w} not divisible by block {b}")
    return tiles.reshape(n, h // b, b, w // b, b, c).sum(axis=(2, 4), dtype=np.uint32).astype(np.uint16)


def choose(ids, max_tiles):
    if not max_tiles or max_tiles >= len(ids):
        return ids
    keep = np.unique(np.linspace(0, len(ids) - 1, max_tiles).round().astype(int))
    return [ids[i] for i in keep]


def extract_one(src_path, out_path, stain, scanner, block, batch, wanted_ids=None):
    tmp = out_path + ".part"
    with h5py.File(src_path, "r") as h5:
        ids, get = open_tiles(h5)
        if wanted_ids is not None:                     # keep the same id set/order as the P slide
            have = set(ids)
            ids = [i for i in wanted_ids if i in have]
        first = get(ids[:1])
        tile = first.shape[1]
        g = tile // block
        with h5py.File(tmp, "w") as out:
            ds = out.create_dataset("sums", (len(ids), g, g, 3), dtype="uint16",
                                    chunks=(min(64, len(ids)), g, g, 3), compression="gzip",
                                    compression_opts=4, shuffle=True)
            out.create_dataset("patch_id", data=np.array(ids, dtype=h5py.string_dtype()))
            out.create_dataset("coords", data=np.array([coords_of(i) for i in ids], dtype=np.int32))
            out.attrs.update(stain=stain, scanner=scanner, source_file=os.path.basename(src_path),
                             block=block, tile_size=tile, n_patches=len(ids),
                             note="sums of uint8 RGB over block x block pixels; mean = sums / block**2")
            t0 = time.time()
            for s in range(0, len(ids), batch):
                sel = ids[s:s + batch]
                ds[s:s + len(sel)] = block_sums(get(sel), block)
                done = s + len(sel)
                rate = done / max(time.time() - t0, 1e-6)
                print(f"\r    {done}/{len(ids)}  ({rate:.0f} tiles/s)", end="", flush=True)
            print()
    os.replace(tmp, out_path)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./plism-dataset")
    ap.add_argument("--out", default="./plism_b8")
    ap.add_argument("--stains", nargs="*", help="default: all found")
    ap.add_argument("--scanners", nargs="*", default=SCANNERS)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--max-tiles", type=int, default=0, help="evenly spaced subset of patch ids (0 = all)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    slides = list_slides(args.data_dir)
    if not slides:
        sys.exit(f"no PLISM .h5 files in {args.data_dir}")
    stains = args.stains or sorted({s for s, _ in slides})
    scanners = ["P"] + [s for s in args.scanners if s != "P"] if "P" in args.scanners else args.scanners
    man_path = os.path.join(args.out, "manifest.json")
    os.makedirs(args.out, exist_ok=True)
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}

    for stain in stains:
        os.makedirs(os.path.join(args.out, stain), exist_ok=True)
        ref_ids = None
        for scanner in scanners:
            key = (stain, scanner)
            if key not in slides:
                print(f"[missing] {stain}_{scanner}"); continue
            out_path = os.path.join(args.out, stain, f"{stain}_{scanner}_b{args.block}.h5")
            if os.path.exists(out_path):
                with h5py.File(out_path, "r") as f:
                    ids = list(f["patch_id"].asstr()[:])
                print(f"[skip] {out_path} exists ({len(ids)} patches)")
            else:
                print(f"[{stain} {scanner}] {os.path.basename(slides[key])}")
                if ref_ids is None:            # first slide of the stain (P if present) defines the ids
                    with h5py.File(slides[key], "r") as h5:
                        ref_ids = choose(open_tiles(h5)[0], args.max_tiles)
                ids = extract_one(slides[key], out_path, stain, scanner, args.block, args.batch, ref_ids)
            if ref_ids is None:
                ref_ids = ids
            missing = sorted(set(ref_ids) - set(ids))
            manifest[f"{stain}_{scanner}"] = {"file": os.path.relpath(out_path, args.out), "n_patches": len(ids),
                                              "missing_vs_ref": len(missing), "missing_examples": missing[:5],
                                              "size_mb": round(os.path.getsize(out_path) / 1e6, 1)}
            json.dump(manifest, open(man_path, "w"), indent=2)
    print(f"done -> {args.out}  (manifest: {man_path})")


if __name__ == "__main__":
    main()
