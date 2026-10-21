#!/usr/bin/env python3

import sys, random, h5py

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
    names = sorted(k for k in f if k.startswith("tile_"))
    for k in sorted(random.Random(seed).sample(names, n)):
        d = g.create_dataset(k, data=f[k][()], compression="gzip", compression_opts=9)
        d.attrs.update(f[k].attrs)
    g.attrs.update(f.attrs)
