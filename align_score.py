#!/usr/bin/env python3
"""
Alignment score per patch: one scanner's slide vs the P slide of the same stain.
Works for any patch size (e.g. 224x224 or 1200x1200).

usage: python align_score.py SRC.h5 P.h5 OUT.csv [--max-shift 128] [--grid 6] [--limit N]

Score = Pearson correlation of CIE L* (lightness, sRGB assumed) over the overlapping pixels.

Per patch (OUT.csv):
  patch_id       dataset name (flat tile_* datasets; patches present in both files)
  score_before   correlation as stored (no shift)
  dy, dx         best integer shift in px: SRC[y, x] matches P[y - dy, x - dx]
                 (move P by (+dy, +dx) to overlay it on SRC)
  score_after    correlation after the best shift
  overlap        fraction of the patch still overlapping after the shift
  std_src, std_p L* standard deviation; < 2 means a flat patch (scores meaningless)
  tissue         fraction of P pixels that are tissue (min RGB < 204)
  keep           1 if tissue > 0.3, min(std) > 2, score_after >= 0.6, overlap >= 0.6

With --grid N, also OUT.blocks.csv: the patch is cut into N x N blocks and each block's shift is
searched within +-block_search px of the patch shift (the offset drifts across large patches):
  patch_id, by, bx, y0, x0, h, w, score_before, dy, dx, score_after, std_src, std_p, used
  (used = fraction of the block matched; border blocks are trimmed to stay inside P)

Blocks: FFT template matching (normalized cross-correlation) of each block in a window of P.
Search: coarse masked-NCC over all shifts up to --max-shift on a downsampled pair (FFT), then
exhaustive refinement at each finer level, ending at full resolution.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

MIN_OVERLAP = 0.4
M_Y = np.array([0.2225045, 0.7168786, 0.0606169], np.float32)   # sRGB (D50) -> Y


def lightness(rgb_u8):
    c = rgb_u8.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = lin @ M_Y
    return np.where(y > 216 / 24389, 116 * np.cbrt(y) - 16, 24389 / 27 * y).astype(np.float32)


def downsample(a, f):
    if f == 1:
        return a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def crops(F, M, dy, dx):
    """Overlapping views: F[y, x] <-> M[y - dy, x - dx]."""
    H, W = F.shape
    return (F[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)],
            M[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)])


def pearson(X, Y):
    if X.size < 16:
        return -1.0
    X = X - X.mean(); Y = Y - Y.mean()
    d = np.sqrt(float((X * X).sum()) * float((Y * Y).sum()))
    return float((X * Y).sum()) / d if d > 0 else -1.0


def masked_ncc_search(F, M, max_shift, min_overlap=MIN_OVERLAP):
    """Best shift over |dy|,|dx| <= max_shift with overlap >= min_overlap (FFT, Padfield 2012)."""
    from scipy import fft as sf
    H, W = F.shape
    S = (2 * H, 2 * W)
    one = np.ones((H, W), np.float32)
    f1 = sf.rfft2(one, S); c1 = np.conj(f1)
    XF = sf.rfft2(np.stack([F, F * F]), S); XM = sf.rfft2(np.stack([M, M * M]), S)
    N = sf.irfft2(f1 * c1, S)
    sF, sFF = sf.irfft2(XF * c1, S)
    sM, sMM = sf.irfft2(f1 * np.conj(XM), S)
    sFM = sf.irfft2(XF[0] * np.conj(XM[0]), S)
    Nc = np.maximum(N, 1)
    r = (sFM - sF * sM / Nc) / np.sqrt(np.maximum(sFF - sF ** 2 / Nc, 1e-9) * np.maximum(sMM - sM ** 2 / Nc, 1e-9))
    dy = np.fft.fftfreq(2 * H, 1 / (2 * H)).astype(int)[:, None]     # 0..H-1, -H..-1
    dx = np.fft.fftfreq(2 * W, 1 / (2 * W)).astype(int)[None, :]
    ok = (N >= min_overlap * H * W) & (np.abs(dy) <= max_shift) & (np.abs(dx) <= max_shift)
    r = np.where(ok, r, -1)
    iy, ix = np.unravel_index(r.argmax(), r.shape)
    return int(dy[iy, 0]), int(dx[0, ix])


def local_search(F, M, cy, cx, rad, score_at):
    best = (-2.0, cy, cx)
    for dy in range(cy - rad, cy + rad + 1):
        for dx in range(cx - rad, cx + rad + 1):
            s = score_at(dy, dx)
            if s > best[0]:
                best = (s, dy, dx)
    return best


def align_patch(F, M, max_shift):
    """Coarse-to-fine global shift for a whole patch. Returns (score, dy, dx)."""
    H, W = F.shape
    f = 1
    while min(H, W) // (f * 2) >= 96:          # coarse level keeps >= ~96 px per side
        f *= 2
    cy, cx = masked_ncc_search(downsample(F, f), downsample(M, f), max(1, max_shift // f))
    minpx = MIN_OVERLAP * H * W
    while True:
        Fl, Ml = downsample(F, f), downsample(M, f)
        def score_at(dy, dx, Fl=Fl, Ml=Ml, f=f):
            if abs(dy) * f > max_shift or abs(dx) * f > max_shift:
                return -1.0
            X, Y = crops(Fl, Ml, dy, dx)
            return pearson(X, Y) if X.size * f * f >= minpx else -1.0
        s, cy, cx = local_search(Fl, Ml, cy, cx, 1 if f > 1 else 2, score_at)
        if f == 1:
            return s, cy, cx
        f //= 2; cy *= 2; cx *= 2


def align_blocks(F, M, n, dy0, dx0, rad):
    """Per-block shift within +-rad px of the patch shift: FFT template matching of each SRC block
    inside a window of P (normalized cross-correlation = Pearson over the template).
    Blocks at the patch border are trimmed to the part that stays inside P for every searched
    shift; 'used' is the fraction of the block that was matched."""
    from skimage.feature import match_template
    H, W = F.shape
    ys = np.linspace(0, H, n + 1).astype(int); xs = np.linspace(0, W, n + 1).astype(int)
    out = []
    for by in range(n):
        for bx in range(n):
            y0, y1, x0, x1 = ys[by], ys[by + 1], xs[bx], xs[bx + 1]
            Fb = F[y0:y1, x0:x1]
            before = pearson(Fb, M[y0:y1, x0:x1])
            # SRC rows y match P rows y - dy with dy in [dy0-rad, dy0+rad]; keep rows valid for all of them
            ty0, ty1 = max(y0, dy0 + rad), min(y1, H + dy0 - rad)
            tx0, tx1 = max(x0, dx0 + rad), min(x1, W + dx0 - rad)
            used = max(ty1 - ty0, 0) * max(tx1 - tx0, 0) / Fb.size
            if ty1 - ty0 < 16 or tx1 - tx0 < 16:
                out.append((by, bx, y0, x0, y1 - y0, x1 - x0, before, dy0, dx0, -1.0, float(Fb.std()), 0.0, used))
                continue
            T = F[ty0:ty1, tx0:tx1]
            wy0, wx0 = ty0 - dy0 - rad, tx0 - dx0 - rad
            r = match_template(M[wy0:ty1 - dy0 + rad, wx0:tx1 - dx0 + rad], T)   # r[i, j]: T at (wy0+i, wx0+j)
            i, j = np.unravel_index(r.argmax(), r.shape)
            dy, dx = int(ty0 - (wy0 + i)), int(tx0 - (wx0 + j))
            Yb = M[ty0 - dy:ty1 - dy, tx0 - dx:tx1 - dx]
            out.append((by, bx, y0, x0, y1 - y0, x1 - x0, before, dy, dx, float(r[i, j]),
                        float(T.std()), float(Yb.std()), used))
    return out


def tile_ids(h5):
    return {k for k, v in h5.items() if isinstance(v, h5py.Dataset) and v.ndim == 3 and v.shape[-1] == 3}


def sort_key(k):
    if k.isdigit():                                   # plain numeric names: "0", "1", ...
        return (0, int(k))
    parts = k.split("_")[1:]                          # tile_<level>_<x>_<y>
    if parts and all(p.isdigit() for p in parts):
        return (1,) + tuple(int(p) for p in parts)
    return (2, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("p"); ap.add_argument("out")
    ap.add_argument("--max-shift", type=int, default=128, help="largest shift searched, px (default 128)")
    ap.add_argument("--grid", type=int, default=0, help="also score N x N sub-blocks (0 = off)")
    ap.add_argument("--block-search", type=int, default=24, help="block search radius around patch shift, px")
    ap.add_argument("--limit", type=int, default=0, help="only the first N patches (testing)")
    args = ap.parse_args()

    with h5py.File(args.src, "r") as fs, h5py.File(args.p, "r") as fp:
        a, b = tile_ids(fs), tile_ids(fp)
        ids = sorted(a & b, key=sort_key)
        if len(ids) != len(a) or len(ids) != len(b):
            print(f"note: {len(a)} src / {len(b)} P patches, {len(ids)} in common", file=sys.stderr)
        if args.limit:
            ids = ids[:args.limit]
        tmp, btmp = args.out + ".part", args.out[:-4] + ".blocks.csv.part" if args.out.endswith(".csv") else args.out + ".blocks.part"
        out = open(tmp, "w")
        out.write(f"# src={os.path.basename(args.src)} p={os.path.basename(args.p)} max_shift={args.max_shift} min_overlap={MIN_OVERLAP}\n")
        out.write("patch_id,score_before,dy,dx,score_after,overlap,std_src,std_p,tissue,keep\n")
        bout = None
        if args.grid:
            bout = open(btmp, "w")
            bout.write("patch_id,by,bx,y0,x0,h,w,score_before,dy,dx,score_after,std_src,std_p,used\n")
        t0 = time.time()
        for i, k in enumerate(ids, 1):
            g, p = fs[k][()], fp[k][()]
            if g.shape != p.shape:
                print(f"skip {k}: shapes {g.shape} vs {p.shape}", file=sys.stderr); continue
            F, M = lightness(g), lightness(p)
            H, W = F.shape
            before = pearson(F, M)
            after, dy, dx = align_patch(F, M, args.max_shift)
            ov = (H - abs(dy)) * (W - abs(dx)) / (H * W)
            ss, sp = float(F.std()), float(M.std())
            tissue = float((p.min(axis=-1) < 204).mean())
            keep = int(tissue > 0.3 and min(ss, sp) > 2.0 and after >= 0.6 and ov >= 0.6)
            out.write(f"{k},{before:.4f},{dy},{dx},{after:.4f},{ov:.3f},{ss:.2f},{sp:.2f},{tissue:.3f},{keep}\n")
            if bout:
                for r in align_blocks(F, M, args.grid, dy, dx, args.block_search):
                    bout.write(f"{k},{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]:.4f},{r[7]},{r[8]},{r[9]:.4f},{r[10]:.2f},{r[11]:.2f},{r[12]:.2f}\n")
            if i % 200 == 0 or i == len(ids):
                print(f"{os.path.basename(args.src)}: {i}/{len(ids)} ({i / (time.time() - t0):.1f}/s)", flush=True)
        out.close(); os.replace(tmp, args.out)
        if bout:
            bout.close(); os.replace(btmp, btmp[:-5])


if __name__ == "__main__":
    main()
