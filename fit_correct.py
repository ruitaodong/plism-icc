#!/usr/bin/env python3
"""
Fit a SRC -> P color mapping from registered patch pairs, write it as ICC profiles, and correct
the SRC patches.

usage: python fit_correct.py SRC.h5 P.h5 --out DIR [--block 4] [--grid 6] [--robust] [--figure 5]

Inputs : two HDF5 files with the same patch names, each patch (H, W, 3) uint8 RGB
         (e.g. GT450.h5 and P.h5 with patches "0".."4", or PLISM tile_* datasets).
Steps  :
  1. Register: per patch a global shift, then per block (grid x grid) a local shift
     (align_score.py: masked NCC on L*, sub-block template matching).
  2. Keep blocks with score_after >= --min-score; flat blocks (glass, L* std < 2) are kept at the
     patch shift, since alignment does not matter where there is no structure.
  3. Average registered pairs in --block x --block pixel blocks, accumulate color bins.
  4. Fit affine + 33^3 lattice (plism_icc.fit_lut, lambda --lam). --robust adds 3 rounds of Tukey
     reweighting (extra passes over the data).
  5. Write DIR/<SRC>_to_P_devicelink.icc, <SRC>_as_P_input.icc, .cube, .npy.
  6. Correct every SRC patch with the device link (same tetrahedral evaluation as LittleCMS)
     -> DIR/<SRC>_corrected_to_P.h5, and report before/after on the registered blocks
     -> DIR/metrics.csv (pixel dE00, 50 px region dE00, contrast ratio vs P).
  7. --figure N: grid image of the first N patches (whole patch + full-res center crop).

Requires plism_icc.py and align_score.py next to this script; numpy, scipy, scikit-image, h5py, pillow.
"""
import argparse
import csv
import os
import struct
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align_score as AS   # noqa: E402
import plism_icc as PI     # noqa: E402

QB = 7  # color bin resolution (bits per channel), as in plism_icc


# ----------------------------------------------------------------------------- registration
def register(src, p, grid, max_shift, block_search, min_score):
    """Return the patch shift and the registered blocks as (ya, yb, xa, xb, dy, dx):
    SRC[ya:yb, xa:xb] matches P[ya-dy:yb-dy, xa-dx:xb-dx]. Only coordinates are kept (low memory)."""
    F, M = AS.lightness(src), AS.lightness(p)
    H, W = F.shape
    _, pdy, pdx = AS.align_patch(F, M, max_shift)
    out = []
    for (by, bx, y0, x0, h, w, before, dy, dx, after, s_src, s_p, used) in AS.align_blocks(F, M, grid, pdy, pdx, block_search):
        flat = min(s_src, s_p) < 2.0
        if flat:
            dy, dx = pdy, pdx
        elif after < min_score:
            continue
        ya, yb = max(y0, dy), min(y0 + h, H + dy)
        xa, xb = max(x0, dx), min(x0 + w, W + dx)
        if yb - ya < 8 or xb - xa < 8:
            continue
        out.append((ya, yb, xa, xb, dy, dx))
    return (pdy, pdx), out


def crops(src, p, blocks):
    """Registered (src_crop, p_crop) pairs for the block coordinates from register()."""
    return [(src[ya:yb, xa:xb], p[ya - dy:yb - dy, xa - dx:xb - dx]) for ya, yb, xa, xb, dy, dx in blocks]


def block_means(a, B):
    h, w = (a.shape[0] // B) * B, (a.shape[1] // B) * B
    a = a[:h, :w].astype(np.float32) / 255.0
    return a.reshape(h // B, B, w // B, B, 3).mean((1, 3)).reshape(-1, 3)


def weighted_bins(x, y, w):
    """Sparse weighted bin sums (keys, weight, sum_x, sum_y), mergeable with plism_icc.merge_bins."""
    q = np.clip((x * (1 << QB)).astype(np.int64), 0, (1 << QB) - 1)
    key = (q[:, 0] << (2 * QB)) | (q[:, 1] << QB) | q[:, 2]
    u, inv = np.unique(key, return_inverse=True)
    c = np.bincount(inv, w)
    sx = np.stack([np.bincount(inv, w * x[:, i]) for i in range(3)], 1)
    sy = np.stack([np.bincount(inv, w * y[:, i]) for i in range(3)], 1)
    keep = c > 0
    return u[keep], c[keep], sx[keep], sy[keep]


# ----------------------------------------------------------------------------- device link application
def read_lut16(icc_path):
    data = open(icc_path, "rb").read()
    n = struct.unpack(">I", data[128:132])[0]
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", data[132 + 12 * i:144 + 12 * i])
        if sig == b"A2B0":
            tag = data[off:off + size]
            g = tag[10]
            n_in, n_out = struct.unpack(">HH", tag[48:52])
            a = np.frombuffer(tag[52:], dtype=">u2").astype(np.float64) / 65535.0
            cin = a[:3 * n_in].reshape(3, n_in)
            clut = a[3 * n_in:3 * n_in + g ** 3 * 3].reshape(g, g, g, 3)
            cout = a[3 * n_in + g ** 3 * 3:3 * n_in + g ** 3 * 3 + 3 * n_out].reshape(3, n_out)
            return cin, clut, cout
    raise ValueError("no A2B0 tag")


def table_from_devicelink(icc_path):
    """Full 256^3 RGB lookup table, tetrahedral interpolation (what LittleCMS does)."""
    cin, clut, cout = read_lut16(icc_path)
    g = clut.shape[0]; flat = clut.reshape(-1, 3); stride = np.array([g * g, g, 1])
    curve = lambda x, c: np.stack([np.interp(x[:, k], np.linspace(0, 1, c.shape[1]), c[k]) for k in range(3)], 1)
    table = np.empty((256 ** 3, 3), np.uint8)
    gg, bb = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    for r in range(256):
        x = curve(np.stack([np.full(65536, r), gg.ravel(), bb.ravel()], 1) / 255.0, cin)
        p = np.clip(x, 0, 1) * (g - 1); i0 = np.minimum(np.floor(p).astype(np.int64), g - 2); f = p - i0
        order = np.argsort(-f, axis=1); fs = np.take_along_axis(f, order, axis=1)
        idx = i0 @ stride; y = (1 - fs[:, 0:1]) * flat[idx]
        for k in range(3):
            idx = idx + stride[order[:, k]]
            y += (fs[:, k:k + 1] - (fs[:, k + 1:k + 2] if k < 2 else 0)) * flat[idx]
        table[r * 65536:(r + 1) * 65536] = np.clip(np.round(curve(y, cout) * 255), 0, 255)
    return table


def apply_table(table, rgb):
    key = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    return table[key]


# ----------------------------------------------------------------------------- evaluation
def metrics(pairs_src, pairs_cor, pairs_p, region=50):
    px_b, px_a, rg_b, rg_a, ct_b, ct_a = [], [], [], [], [], []
    L = lambda a: PI.srgb_to_lab_d50(a.reshape(-1, 3))[:, 0]
    for g, c, q in zip(pairs_src, pairs_cor, pairs_p):
        g, c, q = g / 255.0, c / 255.0, q / 255.0
        px_b.append(PI.delta_e00(g, q).ravel()); px_a.append(PI.delta_e00(c, q).ravel())
        for y in range(0, g.shape[0] - region + 1, region):
            for x in range(0, g.shape[1] - region + 1, region):
                sl = (slice(y, y + region), slice(x, x + region))
                qm = q[sl].reshape(-1, 3).mean(0)
                rg_b.append(PI.delta_e00(g[sl].reshape(-1, 3).mean(0), qm))
                rg_a.append(PI.delta_e00(c[sl].reshape(-1, 3).mean(0), qm))
                sq = L(q[sl]).std()
                if sq > 3:
                    ct_b.append(L(g[sl]).std() / sq); ct_a.append(L(c[sl]).std() / sq)
    cat = np.concatenate
    return dict(pixel_dE00_before=cat(px_b).mean(), pixel_dE00_after=cat(px_a).mean(),
                region_dE00_before=np.mean(rg_b), region_dE00_after=np.mean(rg_a),
                contrast_before=np.median(ct_b) if ct_b else np.nan, contrast_after=np.median(ct_a) if ct_a else np.nan)


def figure(path, names, fs, fc, fp, rows, T=330, Z=330):
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 13); fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except OSError:
        f = fb = ImageFont.load_default()
    PAD, GAP, LAB, HEAD = 16, 6, 22, 30
    W = PAD * 2 + 3 * T + 3 * Z + 5 * GAP + 14; RH = LAB + T + 26
    img = Image.new("RGB", (W, HEAD + len(names) * RH + 10), "white"); d = ImageDraw.Draw(img)
    cols = ["SRC original", "SRC -> P (corrected)", "P reference"]
    d.text((PAD, 8), "Left: whole patch (downscaled). Right: center crop at full resolution.", fill="#333", font=f)
    for n, pid in enumerate(names):
        y = HEAD + n * RH; m = rows[pid]; ims = [fs[pid][()], fc[pid][()], fp[pid][()]]
        d.text((PAD, y), f"{pid}: pixel dE00 {m['pixel_dE00_before']:.1f} -> {m['pixel_dE00_after']:.1f}   region dE00 "
               f"{m['region_dE00_before']:.1f} -> {m['region_dE00_after']:.2f}   contrast vs P {m['contrast_before']:.2f} -> {m['contrast_after']:.2f}",
               fill="black", font=fb)
        H, W0 = ims[0].shape[:2]; cy, cx = (H - Z) // 2, (W0 - Z) // 2
        for k, a in enumerate(ims):
            img.paste(Image.fromarray(a).resize((T, T), Image.LANCZOS), (PAD + k * (T + GAP), y + LAB))
            img.paste(Image.fromarray(a[cy:cy + Z, cx:cx + Z]), (PAD + 3 * (T + GAP) + 14 + k * (Z + GAP), y + LAB))
            if n == len(names) - 1:
                d.text((PAD + k * (T + GAP), y + LAB + T + 4), cols[k], fill="black", font=f)
                d.text((PAD + 3 * (T + GAP) + 14 + k * (Z + GAP), y + LAB + Z + 4), cols[k] + " (crop)", fill="black", font=f)
    img.save(path, optimize=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("p")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="source scanner name for file names (default: SRC file stem)")
    ap.add_argument("--block", type=int, default=4, help="averaging block for fitting, px (default 4)")
    ap.add_argument("--grid", type=int, default=6, help="registration blocks per side (default 6)")
    ap.add_argument("--min-score", type=float, default=0.6, help="drop blocks scoring below this after alignment")
    ap.add_argument("--max-shift", type=int, default=128); ap.add_argument("--block-search", type=int, default=24)
    ap.add_argument("--lam", type=float, default=3e-4); ap.add_argument("--robust", action="store_true")
    ap.add_argument("--no-apply", action="store_true", help="only fit and write profiles")
    ap.add_argument("--figure", type=int, default=0, help="grid figure of the first N patches")
    args = ap.parse_args()
    name = args.name or os.path.splitext(os.path.basename(args.src))[0]
    os.makedirs(args.out, exist_ok=True)

    fs, fp = h5py.File(args.src, "r"), h5py.File(args.p, "r")
    ids = sorted(AS.tile_ids(fs) & AS.tile_ids(fp), key=AS.sort_key)
    print(f"{len(ids)} patch pairs", flush=True)

    # 1-3: register and accumulate bins (keep registered crops only if we will evaluate)
    parts, reg = [], {}
    for i, pid in enumerate(ids, 1):
        src, p = fs[pid][()], fp[pid][()]
        if src.shape != p.shape:
            print(f"skip {pid}: shapes differ"); continue
        shift, blocks = register(src, p, args.grid, args.max_shift, args.block_search, args.min_score)
        reg[pid] = blocks
        for g, q in crops(src, p, blocks):
            x, y = block_means(g, args.block), block_means(q, args.block)
            parts.append(weighted_bins(x, y, np.ones(len(x))))
        print(f"  [{i}/{len(ids)}] {pid}: shift {shift}, {len(blocks)}/{args.grid ** 2} blocks kept", flush=True)
    x, y, cnt = PI.merge_bins(parts)

    # 4: fit (+ optional robust passes)
    lut = PI.fit_lut(x, y, cnt, lam=args.lam)
    for it in range(3 if args.robust else 0):
        def residual_pass():                       # streams over the data: block means + residuals
            for pid, blocks in reg.items():
                for g, q in crops(fs[pid][()], fp[pid][()], blocks):
                    xb, yb = block_means(g, args.block), block_means(q, args.block)
                    yield xb, yb, np.linalg.norm(PI.apply_lut(lut, xb) - yb, axis=1)
        # robust scale from a subsample of residuals, then reweighted bins in a second stream
        sample = [r[::max(1, len(r) // 200)] for _, _, r in residual_pass()]
        s = 1.4826 * np.median(np.concatenate(sample))
        parts = [weighted_bins(xb, yb, np.clip(1 - (r / (4.685 * s)) ** 2, 0, None) ** 2) for xb, yb, r in residual_pass()]
        lut = PI.fit_lut(*PI.merge_bins(parts), lam=args.lam)
        print(f"  robust pass {it + 1}: scale {s * 255:.2f} levels", flush=True)

    # 5: profiles
    base = os.path.join(args.out, f"{name}_to_P")
    np.save(base + "_lut.npy", lut)
    PI.write_devicelink(base + "_devicelink.icc", lut, name)
    PI.write_input_profile(os.path.join(args.out, f"{name}_as_P_input.icc"), lut, name)
    PI.write_cube(base + ".cube", lut, f"{name} to P")
    print(f"wrote {base}_devicelink.icc, {name}_as_P_input.icc, .cube, .npy")
    if args.no_apply:
        return

    # 6: correct + evaluate
    table = table_from_devicelink(base + "_devicelink.icc")
    cor_path = os.path.join(args.out, f"{name}_corrected_to_P.h5")
    rows = {}
    with h5py.File(cor_path, "w") as fc:
        for pid in reg:
            src, p = fs[pid][()], fp[pid][()]
            fc.create_dataset(pid, data=apply_table(table, src), compression="gzip",
                              chunks=(min(256, src.shape[0]), min(256, src.shape[1]), 3))
            pc = crops(src, p, reg[pid])
            # corrected crops = table applied to the same registered SRC crops
            rows[pid] = metrics([g for g, _ in pc], [apply_table(table, g) for g, _ in pc], [q for _, q in pc])
    keys = list(next(iter(rows.values())).keys())
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patch_id"] + keys)
        for pid, m in rows.items():
            w.writerow([pid] + [f"{m[k]:.4f}" for k in keys])
        w.writerow(["mean"] + [f"{np.nanmean([m[k] for m in rows.values()]):.4f}" for k in keys])
    mean = {k: np.nanmean([m[k] for m in rows.values()]) for k in keys}
    print(f"pixel dE00 {mean['pixel_dE00_before']:.2f} -> {mean['pixel_dE00_after']:.2f} | region dE00 "
          f"{mean['region_dE00_before']:.2f} -> {mean['region_dE00_after']:.2f} | contrast vs P "
          f"{mean['contrast_before']:.2f} -> {mean['contrast_after']:.2f}   (metrics.csv, {cor_path})")

    # 7: figure
    if args.figure:
        with h5py.File(cor_path, "r") as fc:
            figure(os.path.join(args.out, f"{name}_to_P_correction_grid.png"), list(reg)[:args.figure], fs, fc, fp, rows)


if __name__ == "__main__":
    main()
