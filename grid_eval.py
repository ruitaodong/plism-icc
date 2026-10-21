#!/usr/bin/env python3
"""
Score every hold-out patch of one source scanner against the reference after applying its device
link, then draw a grid of N random and N worst patches.

usage:
  python grid_eval.py --src src_pyr/ --ref ref_pyr/ --icc src_to_ref_devicelink.icc \
                      --split src_to_ref_split.csv --out src_vs_ref [--level 0] [--n 5]

Inputs: pyramid TIFFs from h5_pyramid.py (matched by file name), the device link from fit_pyr.py and
its <out>_split.csv (patches marked "holdout" are scored). Applies the profile exactly as LittleCMS
does (fit_correct.table_from_devicelink).
Outputs:
  <out>_patches.csv   per hold-out patch: slide, region_dE00 before/after, pixel_dE00, contrast, dark1
  <out>_grid.png      rows = N random + N worst (by region dE00 after mapping); columns = source,
                      transformed, reference, region dE00 map (transformed vs reference), 1:1 crops
Metrics as in fit_pyr.py: region_dE00 = CIEDE2000 of linear-light region means (--region px),
pixel_dE00 on a random pixel sample, contrast = median ratio of local L* std, dark1 = L* difference
of the darkest 1%. Requires fit_pyr.py, fit_correct.py, plism_icc.py next to this script.
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_correct as FC   # noqa: E402
import fit_pyr as FP       # noqa: E402
import plism_icc as P      # noqa: E402


def dec(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def enc(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def region_means(img, R):
    x = img / 255.0
    H, W = (x.shape[0] // R) * R, (x.shape[1] // R) * R
    m = enc(dec(x[:H, :W]).reshape(H // R, R, W // R, R, 3).mean((1, 3)))
    s = P.srgb_to_lab_d50(x[:H, :W])[..., 0].reshape(H // R, R, W // R, R).std((1, 3))
    return m, s


def score(src, mapped, ref, R, rng):
    (sm, _), (mm, ms), (rm, rs) = region_means(src, R), region_means(mapped, R), region_means(ref, R)
    de_map = P.delta_e00(mm.reshape(-1, 3), rm.reshape(-1, 3)).reshape(mm.shape[:2])
    ok = rs > 3
    idx = rng.choice(src.shape[0] * src.shape[1], min(20000, src.shape[0] * src.shape[1]), replace=False)
    a, b = mapped.reshape(-1, 3)[idx] / 255.0, ref.reshape(-1, 3)[idx] / 255.0
    La, Lb = P.srgb_to_lab_d50(a)[:, 0], P.srgb_to_lab_d50(b)[:, 0]
    return dict(region_dE00_before=float(P.delta_e00(sm.reshape(-1, 3), rm.reshape(-1, 3)).mean()),
                region_dE00=float(de_map.mean()), pixel_dE00=float(P.delta_e00(a, b).mean()),
                contrast=float(np.median(ms[ok] / rs[ok])) if ok.any() else float("nan"),
                dark1=float(np.percentile(La, 1) - np.percentile(Lb, 1))), de_map


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--icc", required=True, help="device link from fit_pyr.py")
    ap.add_argument("--split", required=True, help="<out>_split.csv from fit_pyr.py (holdout rows are scored)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=0, help="stored image index to evaluate at (0 = finest)")
    ap.add_argument("--region", type=int, default=24, help="region size, px at that level")
    ap.add_argument("--n", type=int, default=5, help="number of random and of worst patches in the grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-src", default="source"); ap.add_argument("--label-ref", default="reference")
    args = ap.parse_args()

    S, R = FP.list_files(args.src), FP.list_files(args.ref)
    split = {r["patch"]: r for r in csv.DictReader(open(args.split))}
    names = sorted(n for n, r in split.items() if r["set"] == "holdout" and n in S and n in R)
    print(f"{len(names)} hold-out patches with source and reference images")
    if not names:
        sys.exit("no hold-out patches found: check --split and the file names in --src/--ref")
    table = FC.table_from_devicelink(args.icc)
    rng = np.random.default_rng(args.seed)
    rows = []
    for i, n in enumerate(names, 1):
        a, ma = FP.read_level(S[n], args.level); b, mb = FP.read_level(R[n], args.level)
        if a.shape != b.shape:
            print(f"  skip {n}: shapes {a.shape} vs {b.shape}"); continue
        a8 = np.round(FP.prepare(a, ma, 0) * 255).astype(np.uint8); b8 = np.round(FP.prepare(b, mb, 0) * 255).astype(np.uint8)
        m, _ = score(a8, FC.apply_table(table, a8), b8, args.region, rng)
        rows.append(dict(patch=n, slide=split[n].get("slide", ""), **m))
        if i % 200 == 0 or i == len(names):
            print(f"  scored {i}/{len(names)}", flush=True)
    keys = list(rows[0].keys())
    with open(args.out + "_patches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    r_after = np.array([r["region_dE00"] for r in rows])
    print(f"region dE00 after mapping: mean {r_after.mean():.2f}, median {np.median(r_after):.2f}, p95 {np.percentile(r_after, 95):.2f}, max {r_after.max():.2f}")

    worst = list(np.argsort(-r_after)[:args.n])
    rest = [i for i in range(len(rows)) if i not in set(worst)]
    rand = list(np.random.default_rng(args.seed).choice(rest, min(args.n, len(rest)), replace=False))
    pick = [("random", i) for i in rand] + [("worst", i) for i in worst]

    T, Z, PAD, GAP, LAB, HEAD, SEP = 260, 260, 14, 5, 22, 76, 14
    try:
        f13 = ImageFont.truetype("DejaVuSans.ttf", 13); fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
    except OSError:
        f13 = fb = big = ImageFont.load_default()
    cols = [args.label_src, f"{args.label_src} -> {args.label_ref}", args.label_ref, "region dE00 map (0-4)",
            "transformed - crop", "reference - crop"]
    NC = len(cols); W = PAD * 2 + NC * T + (NC - 1) * GAP + SEP; RH = LAB + T + 10
    H = HEAD + 22 + len(pick) * RH + 24 + 30
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    d.text((PAD, 8), f"{args.label_src} -> {args.label_ref}: {args.n} random and {args.n} worst hold-out patches "
           f"(of {len(rows)}; level {args.level})", fill="black", font=big)
    d.text((PAD, 36), f"region dE00 after mapping over all hold-out patches: mean {r_after.mean():.2f}, median {np.median(r_after):.2f}, "
           f"p95 {np.percentile(r_after, 95):.2f}, worst {r_after.max():.2f}", fill="#333", font=f13)
    cx = lambda k: PAD + k * (T + GAP) + (SEP if k >= 4 else 0)
    for k, c in enumerate(cols):
        d.text((cx(k) + 2, HEAD), c, fill="black", font=fb)
    import matplotlib.cm as cm
    y = HEAD + 22
    for j, (kind, i) in enumerate(pick):
        if j == args.n:
            d.line([(PAD, y + 4), (W - PAD, y + 4)], fill="#a00", width=2); y += 14
        r = rows[i]; n = r["patch"]
        a, ma = FP.read_level(S[n], args.level); b, mb = FP.read_level(R[n], args.level)
        a8 = np.round(FP.prepare(a, ma, 0) * 255).astype(np.uint8); b8 = np.round(FP.prepare(b, mb, 0) * 255).astype(np.uint8)
        c8 = FC.apply_table(table, a8)
        _, de_map = score(a8, c8, b8, args.region, np.random.default_rng(0))
        heat = (cm.magma_r(np.clip(de_map / 4, 0, 1))[..., :3] * 255).astype(np.uint8)
        d.text((PAD, y), f"{kind.upper():6s} {n}   region dE00 {r['region_dE00_before']:.1f} -> {r['region_dE00']:.2f} | pixel {r['pixel_dE00']:.1f} | "
               f"contrast {r['contrast']:.2f} | dark1 {r['dark1']:+.1f}", fill="#a00" if kind == "worst" else "black", font=f13)
        ims = [a8, c8, b8]
        for k, im in enumerate(ims):
            img.paste(Image.fromarray(im).resize((T, T), Image.LANCZOS), (cx(k), y + LAB))
        img.paste(Image.fromarray(heat).resize((T, T), Image.NEAREST), (cx(3), y + LAB))
        gray = a8.mean(-1); hh, ww = gray.shape; zz = min(Z, hh, ww)
        best = max(((gray[yy:yy + zz, xx:xx + zz].std(), yy, xx) for yy in range(0, hh - zz + 1, max(1, (hh - zz) // 4 or 1))
                    for xx in range(0, ww - zz + 1, max(1, (ww - zz) // 4 or 1))))
        for k, im in enumerate([c8, b8]):
            img.paste(Image.fromarray(im[best[1]:best[1] + zz, best[2]:best[2] + zz]).resize((Z, Z), Image.NEAREST), (cx(4 + k), y + LAB))
        y += RH
    d.text((PAD, H - 26), "Random patches drawn (seeded) from the hold-out set excluding the worst; worst = highest region dE00 after mapping. "
           "Heat map: region dE00 between transformed and reference (dark = larger).", fill="#555", font=f13)
    img.save(args.out + "_grid.png", optimize=True)
    print(f"wrote {args.out}_patches.csv and {args.out}_grid.png")


if __name__ == "__main__":
    main()
