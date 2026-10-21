#!/usr/bin/env python3
"""4x4 block sums of every tile_* in one PLISM .h5, indexed by patch id.
usage: python extract_b4.py IN.tif.h5 OUT_b4.h5
out:   sums (N, 56, 56, 3) uint16 (mean = sums / 16), patch_id (N,) str, sorted by (level, x, y)"""
import os, sys
import h5py, numpy as np

B = 4
src, dst = sys.argv[1], sys.argv[2]
with h5py.File(src, "r") as f, h5py.File(dst + ".part", "w") as o:
    ids = sorted((k for k in f if k.startswith("tile_")), key=lambda k: tuple(map(int, k.split("_")[1:])))
    h, w, _ = f[ids[0]].shape
    d = o.create_dataset("sums", (len(ids), h // B, w // B, 3), "uint16",
                         chunks=(1, h // B, w // B, 3), compression="gzip", shuffle=True)
    for i, k in enumerate(ids):
        d[i] = f[k][()].reshape(h // B, B, w // B, B, 3).sum((1, 3), dtype=np.uint32)
        if i % 2000 == 0:
            print(f"{os.path.basename(src)} {i}/{len(ids)}", flush=True)
    o["patch_id"] = np.array(ids, dtype=h5py.string_dtype())
    o.attrs.update(block=B, source=os.path.basename(src))
os.replace(dst + ".part", dst)
print(f"{os.path.basename(src)} done: {len(ids)} patches -> {dst}")
