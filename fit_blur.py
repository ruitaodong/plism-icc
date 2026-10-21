#!/usr/bin/env python3
"""
Fit a SRC -> TARGET color mapping from pre-registered patch pairs (no alignment step) and write
it as ICC profiles + .cube.

  python fit_blur.py SRC.h5 TGT.h5 --out gt450_to_p [--sigma 3] [--robust 2.5]
  # parallel extraction (8 shards), then one merge + fit:
  for i in $(seq 1 8); do python fit_blur.py SRC.h5 TGT.h5 --shard $i/8 --save-bins bins_$i.npz --no-fit & done; wait
  python fit_blur.py --from-bins bins_*.npz --out gt450_to_p --name GT450 --robust 3

Method
  1. For every patch present in both files (same name, same shape; (H, W, 3) uint8 RGB):
     blur BOTH images with the same Gaussian (--sigma px, float) in linear light, so focus
     differences between the scans matter less; take pixel pairs every --stride px.
  2. Accumulate a joint histogram of (source color, target color) pairs: --bits per channel on
     each side. Each joint bin keeps its count and the exact mean source and target colors.
     The histogram is a lossless-enough summary for refitting, including robust reweighting,
     so it can be saved (--save-bins) and refit later without the images (--from-bins).
  3. Fit affine + 3D lattice (plism_icc.fit_lut, --lut-size, --lam). Robust fitting: --iters
     rounds of Tukey reweighting on the joint bins; --robust c sets the strength:
        weight = (1 - (r / (c * s))^2)^2 for r < c * s, else 0;  s = 1.4826 * median residual
        c = 4.685 standard (~5% efficiency loss on clean data), 2.5-3 aggressive, 0 = off.
     Extreme colors that are rare in the blurred data are extrapolated by the lattice
     (affine + smoothly decaying residual).
  4. Write <out>_devicelink.icc (RGB -> RGB), <out>_input.icc (RGB -> Lab, target treated as sRGB),
     <out>.cube and <out>_lut.npy.

Requires plism_icc.py next to this script; numpy, scipy, h5py.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plism_icc as PI   # noqa: E402

FIT_BITS = 7   # source-color bins used for the lattice fit (as in plism_icc)


# ----------------------------------------------------------------------------- image side
def dec(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def enc(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def blur(rgb_u8, sigma):
    """Gaussian blur in linear light; returns encoded RGB float32 in [0, 1]."""
    c = rgb_u8.astype(np.float32) / 255.0
    if sigma <= 0:
        return c
    lin = dec(c).astype(np.float32)
    out = np.stack([ndimage.gaussian_filter(lin[..., k], sigma, mode="reflect") for k in range(3)], -1)
    return enc(out).astype(np.float32)


def patch_ids(h5):
    return {k for k, v in h5.items() if isinstance(v, h5py.Dataset) and v.ndim == 3 and v.shape[-1] == 3}


def sort_key(k):
    if k.isdigit():
        return (0, int(k), "")
    parts = k.split("_")[1:]
    if parts and all(p.isdigit() for p in parts):
        return (1, tuple(int(p) for p in parts), "")
    return (2, (), k)


# ----------------------------------------------------------------------------- joint histogram
def joint_bins(x, y, bits):
    """Joint (source, target) histogram: keys, counts, sum_x, sum_y (float64)."""
    n = 1 << bits
    qx = np.clip((x * n).astype(np.int64), 0, n - 1)
    qy = np.clip((y * n).astype(np.int64), 0, n - 1)
    key = (((qx[:, 0] << bits) | qx[:, 1]) << bits | qx[:, 2]) << (3 * bits) | \
          (((qy[:, 0] << bits) | qy[:, 1]) << bits | qy[:, 2])
    u, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sx = np.stack([np.bincount(inv, x[:, i]) for i in range(3)], 1)
    sy = np.stack([np.bincount(inv, y[:, i]) for i in range(3)], 1)
    return u, cnt, sx, sy


def reduce_bins(parts):
    """Merge partial joint histograms (sums and counts add exactly)."""
    if len(parts) == 1:
        return parts[0]
    keys = np.concatenate([p[0] for p in parts])
    u, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, np.concatenate([p[1] for p in parts]))
    sx = np.stack([np.bincount(inv, np.concatenate([p[2][:, i] for p in parts])) for i in range(3)], 1)
    sy = np.stack([np.bincount(inv, np.concatenate([p[3][:, i] for p in parts])) for i in range(3)], 1)
    return u, cnt, sx, sy


def save_bins(path, b, meta):
    u, cnt, sx, sy = b
    np.savez_compressed(path, keys=u, counts=cnt, mean_src=(sx / cnt[:, None]).astype(np.float32),
                        mean_tgt=(sy / cnt[:, None]).astype(np.float32), **{f"meta_{k}": v for k, v in meta.items()})


def load_bins(paths):
    parts, meta = [], None
    for p in paths:
        z = np.load(p)
        m = {k[5:]: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files if k.startswith("meta_")}
        if meta is None:
            meta = m
        elif any(str(meta.get(k)) != str(m.get(k)) for k in ("sigma", "stride", "bits")):
            sys.exit(f"{p}: sigma/stride/bits differ from {paths[0]}; cannot merge")
        c = z["counts"].astype(np.float64)
        parts.append((z["keys"], c, z["mean_src"].astype(np.float64) * c[:, None], z["mean_tgt"].astype(np.float64) * c[:, None]))
    return reduce_bins(parts), meta


# ----------------------------------------------------------------------------- fitting
def src_bins(x, w):
    """Aggregate weighted samples into FIT_BITS source-color bins for the lattice fit."""
    n = 1 << FIT_BITS
    q = np.clip((x * n).astype(np.int64), 0, n - 1)
    key = (q[:, 0] << (2 * FIT_BITS)) | (q[:, 1] << FIT_BITS) | q[:, 2]
    return key


def fit(xm, ym, cnt, w, lut_size, lam, key, penalty="slope", lam0=0.0):
    ww = cnt * w
    u, inv = np.unique(key, return_inverse=True)
    c = np.bincount(inv, ww)
    keep = c > 0
    sx = np.stack([np.bincount(inv, ww * xm[:, i]) for i in range(3)], 1)[keep] / c[keep, None]
    sy = np.stack([np.bincount(inv, ww * ym[:, i]) for i in range(3)], 1)[keep] / c[keep, None]
    return PI.fit_lut(sx, sy, c[keep], g=lut_size, lam=lam, penalty=penalty, lam0=lam0)


def apply_chunked(lut, x, chunk=2_000_000):
    return np.concatenate([PI.apply_lut(lut, x[i:i + chunk]) for i in range(0, len(x), chunk)])


def robust_fit(b, lut_size, lam, c_tukey, iters, penalty="slope", lam0=0.0):
    u, cnt, sx, sy = b
    xm, ym = sx / cnt[:, None], sy / cnt[:, None]
    key = src_bins(xm, None)
    w = np.ones(len(cnt))
    lut = fit(xm, ym, cnt, w, lut_size, lam, key, penalty, lam0)
    hist = []
    for it in range(iters if c_tukey > 0 else 0):
        r = np.linalg.norm(apply_chunked(lut, xm) - ym, axis=1)
        o = np.argsort(r); cw = np.cumsum(cnt[o]); med = r[o][np.searchsorted(cw, cw[-1] / 2)]   # count-weighted median
        s = 1.4826 * med
        uu = r / (c_tukey * s)
        w = np.where(uu < 1, (1 - uu ** 2) ** 2, 0.0)
        lut = fit(xm, ym, cnt, w, lut_size, lam, key, penalty, lam0)
        hist.append((s * 255, float((cnt * (w == 0)).sum() / cnt.sum()), float((cnt * w).sum() / cnt.sum())))
        print(f"  robust round {it + 1}: scale {s * 255:.2f} levels, rejected {100 * hist[-1][1]:.1f}% of pairs, "
              f"mean weight {hist[-1][2]:.2f}", flush=True)
    return lut, w, hist


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?"); ap.add_argument("tgt", nargs="?")
    ap.add_argument("--out", help="output prefix: <out>_devicelink.icc, <out>_input.icc, <out>.cube, <out>_lut.npy")
    ap.add_argument("--name", default="SRC", help="source device name used in the ICC descriptions")
    ap.add_argument("--sigma", type=float, default=3.0, help="common Gaussian blur, px (float; 0 = none)")
    ap.add_argument("--stride", type=int, default=0, help="pair sampling step, px (default: round(sigma), >= 1)")
    ap.add_argument("--bits", type=int, default=7, help="joint-histogram bits per channel per side (default 7; 6 = smaller file, ~0.3 level mean LUT difference)")
    ap.add_argument("--robust", type=float, default=2.5, help="Tukey constant c (lower = stronger; 4.685 standard; 0 = off)")
    ap.add_argument("--iters", type=int, default=4, help="robust reweighting rounds")
    ap.add_argument("--lam", type=float, default=3e-4, help="lattice smoothness")
    ap.add_argument("--lut-size", type=int, default=33, help="lattice nodes per axis")
    ap.add_argument("--penalty", choices=["slope", "curvature"], default="slope", help="lattice penalty (see plism_icc.fit_lut)")
    ap.add_argument("--lam0", type=float, default=0.0, help="ridge toward the affine base (needed for --penalty curvature, e.g. 1e-4)")
    ap.add_argument("--save-bins", help="write the joint histogram to this .npz")
    ap.add_argument("--from-bins", nargs="+", help="skip images; merge these .npz histograms and fit")
    ap.add_argument("--shard", default="1/1", help="i/n: process every n-th patch starting at i (parallel extraction)")
    ap.add_argument("--no-fit", action="store_true", help="only build (and save) the histogram")
    args = ap.parse_args()

    t0 = time.time()
    if args.from_bins:
        b, meta = load_bins(args.from_bins)
        print(f"loaded {len(args.from_bins)} histogram file(s): {len(b[0])} joint bins, {b[1].sum():.0f} pairs "
              f"(sigma {meta.get('sigma')}, stride {meta.get('stride')}, bits {meta.get('bits')})", flush=True)
    else:
        if not (args.src and args.tgt):
            ap.error("give SRC.h5 and TGT.h5, or --from-bins")
        stride = args.stride or max(1, int(round(args.sigma)))
        i, n = (int(v) for v in args.shard.split("/"))
        fs, ft = h5py.File(args.src, "r"), h5py.File(args.tgt, "r")
        ids = sorted(patch_ids(fs) & patch_ids(ft), key=sort_key)[i - 1::n]
        parts, done = [], 0
        for k, pid in enumerate(ids, 1):
            a, t = fs[pid][()], ft[pid][()]
            if a.shape != t.shape:
                print(f"skip {pid}: shapes {a.shape} vs {t.shape}", file=sys.stderr); continue
            x = blur(a, args.sigma)[::stride, ::stride].reshape(-1, 3)
            y = blur(t, args.sigma)[::stride, ::stride].reshape(-1, 3)
            parts.append(joint_bins(x, y, args.bits)); done += 1
            if len(parts) >= 64:
                parts = [reduce_bins(parts)]
            if k % 50 == 0 or k == len(ids):
                print(f"  {k}/{len(ids)} patches ({k / (time.time() - t0):.1f}/s)", flush=True)
        b = reduce_bins(parts)
        meta = dict(sigma=args.sigma, stride=stride, bits=args.bits, src=os.path.basename(args.src),
                    tgt=os.path.basename(args.tgt), patches=done, shard=args.shard)
        print(f"{done} patch pairs -> {len(b[0])} joint bins, {b[1].sum():.0f} pairs", flush=True)
        if args.save_bins:
            save_bins(args.save_bins, b, meta)
            print(f"saved {args.save_bins} ({os.path.getsize(args.save_bins) / 1e6:.1f} MB)")
    if args.no_fit:
        return
    if not args.out:
        ap.error("--out is required for fitting")

    lut, w, hist = robust_fit(b, args.lut_size, args.lam, args.robust, args.iters, args.penalty, args.lam0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out + "_lut.npy", lut)
    PI.write_devicelink(args.out + "_devicelink.icc", lut, args.name)
    PI.write_input_profile(args.out + "_input.icc", lut, args.name)
    PI.write_cube(args.out + ".cube", lut, f"{args.name} to target (sigma {meta.get('sigma')})")

    print(f"wrote {args.out}_devicelink.icc, {args.out}_input.icc, {args.out}.cube, {args.out}_lut.npy  ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()
