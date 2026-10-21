#!/usr/bin/env python3
"""
Fit a SRC -> TGT color mapping from paired pyramid TIFFs (h5_pyramid.py output, either layout)
at one pyramid level, with an optional extra Gaussian blur, and tune it on a hold-out set.

  python fit_pyr.py --src gt450_pyr/ --tgt p_pyr/ --level 1 --sigma 0 0.75 1.5 \
                    --robust 0 2.5 --lam 1e-4 3e-4 --holdout 0.15 \
                    --bins-dir bins/ --out gt450_to_p --name GT450

Inputs
  --src / --tgt   directory, glob, or .txt list of TIFFs; pairs are matched by file name.
  --level k       which stored image to fit on: index in the file (h5_pyramid files: 0 = 1/2 size,
                  1 = 1/4 size, ...). The pixels are assumed registered (no alignment step).
  --sigma s ...   extra Gaussian blur in linear light, in pixels OF THAT LEVEL (float; several
                  values = tuning grid). Equivalent full-resolution blur is printed using the
                  pyramid's own blur from the file metadata.
Method (same as fit_blur.py)
  joint (source, target) color histogram per sigma -> affine + 33^3 lattice -> --iters rounds of
  Tukey reweighting with constant c (--robust; lower = stronger, 0 = off). Lattice penalty: --penalty
  slope (default; first differences of the residual, decays to the affine base away from data) or
  curvature (second differences, zero for affine maps -> --lam penalises non-linearity only; give a
  ridge --lam0 such as 1e-4 so unsupported colours fall back to the affine base). Histograms are cached
  in --bins-dir and reused when the same files/level/sigma/stride/bits/split are requested again.
Hold-out (--holdout 0.15)
  Whole SLIDES are kept out of fitting (slide id from the file name, --group-regex, default
  'SL\\d+'), added in a seeded random order until --holdout of the patches is reached; the split
  is written to <out>_split.csv. Or give your own split: --train-list / --test-list (text files,
  one patch name per line; paths/extensions ignored); patches in neither list are not used.
  Every (sigma, robust, lam, lam0) combination is scored on the hold-out images at
  --eval-level (default 0, the finest stored level), unblurred, as the profile will be used:
    region_dE00   colour error of region means (--region px squares, linear-light means);
                  insensitive to focus differences -> the default selection metric
    pixel_dE00    per-pixel error on a random subsample (includes focus/optics differences)
    contrast      median ratio of local L* std (mapped / target), 1 = same local contrast
    dark1, bright99   L* difference of the darkest 1% / brightest 1% (mapped - target)
  The best combination by --select is refit on ALL patches (train + hold-out) unless --no-refit.
Outputs
  <out>_devicelink.icc, <out>_input.icc, <out>.cube, <out>_lut.npy, <out>_tuning.csv, <out>_split.csv
  With --model-plugin (see model_eval.py, model_plugin_template.py): every candidate is also scored
  through your segmentation/classification models on the hold-out tiles (agreement of mapped source
  vs target outputs), with raw and ceiling references in <out>_model_reference.csv.

Requires fit_blur.py and plism_icc.py next to this script; numpy, scipy, tifffile.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_blur as FB    # noqa: E402
import plism_icc as PI   # noqa: E402


# ----------------------------------------------------------------------------- inputs
def list_files(spec):
    if os.path.isdir(spec):
        fs = glob.glob(os.path.join(spec, "*.tif")) + glob.glob(os.path.join(spec, "*.tiff"))
    elif spec.endswith(".txt"):
        fs = [l.strip() for l in open(spec) if l.strip() and not l.startswith("#")]
    else:
        fs = glob.glob(spec)
    return {os.path.splitext(os.path.basename(f))[0]: f for f in fs}


def read_level(path, k):
    """Image k of a pyramid TIFF (top-level pages or SubIFDs) -> (array, metadata dict or {})."""
    with tifffile.TiffFile(path) as t:
        p0 = t.pages[0]
        meta = {}
        d = p0.description or ""
        if d.startswith("h5_pyramid"):
            meta = json.loads(d.split("\n", 1)[1])
        if p0.subifds:
            if k == 0:
                a = p0.asarray()
            else:
                t.filehandle.seek(p0.subifds[k - 1]); a = tifffile.TiffPage(t, index=0).asarray()
        else:
            a = t.pages[k].asarray()
    return a, meta


def to_float(a):
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return np.clip(a.astype(np.float32), 0, 1)


def prepare(a, meta, sigma):
    """Encoded-sRGB float image after an extra Gaussian blur (linear light) of `sigma` px."""
    x = to_float(a)
    linear_stored = str(meta.get("values", "")).startswith("linear")
    if sigma > 0:
        lin = x if linear_stored else FB.dec(x).astype(np.float32)
        lin = np.stack([ndimage.gaussian_filter(lin[..., c], sigma, mode="reflect") for c in range(3)], -1)
        return FB.enc(lin).astype(np.float32)
    return FB.enc(x).astype(np.float32) if linear_stored else x


def slide_of(name, regex):
    if not regex:
        return name
    m = re.search(regex, name)
    return m.group(0) if m else None


def split_groups(names, frac, regex, seed):
    """Hold out whole groups (slides) in a seeded random order until >= frac of the PATCHES are held out."""
    g = {n: slide_of(n, regex) for n in names}
    bad = [n for n in names if g[n] is None]
    if bad:
        sys.exit(f"{len(bad)} file name(s) contain no slide id matching --group-regex '{regex}', e.g. {bad[:3]}\n"
                 f"fix the regex, or use --group-regex none to hold out individual patches")
    size = {}
    for n in names:
        size[g[n]] = size.get(g[n], 0) + 1
    order = sorted(size, key=lambda x: hashlib.md5(f"{seed}:{x}".encode()).hexdigest())
    hold, n_hold = set(), 0
    for x in order:
        if frac <= 0 or n_hold >= frac * len(names) or len(hold) == len(order) - 1:
            break
        hold.add(x); n_hold += size[x]
    return [n for n in names if g[n] not in hold], [n for n in names if g[n] in hold], g, sorted(hold), len(size)


def read_names(path):
    """Patch names from a text file: one per line; paths and extensions are stripped; '#' comments."""
    out = []
    for l in open(path):
        l = l.strip()
        if l and not l.startswith("#"):
            out.append(os.path.splitext(os.path.basename(l))[0])
    return out


def user_split(names, train_file, test_file, regex):
    avail = set(names)
    tr, te = read_names(train_file), read_names(test_file)
    dup_tr, dup_te = len(tr) - len(set(tr)), len(te) - len(set(te))
    tr, te = set(tr), set(te)
    both = tr & te
    miss_tr, miss_te = sorted(tr - avail), sorted(te - avail)
    unused = avail - tr - te
    print(f"user split: train list {len(tr)}, test list {len(te)} patch names")
    for msg, xs in (("in BOTH lists (kept in test, removed from train)", sorted(both)),
                    ("in train list but missing from --src/--tgt", miss_tr), ("in test list but missing from --src/--tgt", miss_te)):
        if xs:
            print(f"  WARNING {len(xs)} {msg}: {xs[:5]}{' ...' if len(xs) > 5 else ''}")
    if dup_tr or dup_te:
        print(f"  note: duplicate lines ignored (train {dup_tr}, test {dup_te})")
    if unused:
        print(f"  note: {len(unused)} available patches are in neither list and are not used")
    train = [n for n in names if n in tr and n not in both]
    hold = [n for n in names if n in te]
    if not train or not hold:
        sys.exit("train or test set is empty after matching the lists to the files")
    slide = {n: (slide_of(n, regex) or "?") for n in names}
    if regex:
        st, sh = {slide[n] for n in train}, {slide[n] for n in hold}
        shared = sorted(st & sh - {"?"})
        print(f"  {len(train)} train patches from {len(st)} slides, {len(hold)} test patches from {len(sh)} slides "
              f"({100 * len(hold) / (len(train) + len(hold)):.1f}% of used patches)")
        if shared:
            print(f"  WARNING {len(shared)} slide(s) appear in both train and test: {shared[:5]}{' ...' if len(shared) > 5 else ''}")
    return train, hold, slide


# ----------------------------------------------------------------------------- histograms
def build_bins(pairs, names, level, sigma, stride, bits):
    parts = []
    for k, n in enumerate(names, 1):
        a, ma = read_level(pairs[n][0], level); b, mb = read_level(pairs[n][1], level)
        if a.shape != b.shape:
            print(f"  skip {n}: shapes {a.shape} vs {b.shape}", file=sys.stderr); continue
        x = prepare(a, ma, sigma)[::stride, ::stride].reshape(-1, 3)
        y = prepare(b, mb, sigma)[::stride, ::stride].reshape(-1, 3)
        parts.append(FB.joint_bins(x, y, bits))
        if len(parts) >= 64:
            parts = [FB.reduce_bins(parts)]
    return FB.reduce_bins(parts)


def cached_bins(args, pairs, names, sigma, tag):
    h = hashlib.md5(("\n".join(sorted(pairs[n][0] + "|" + pairs[n][1] for n in names))).encode()).hexdigest()[:10]
    path = os.path.join(args.bins_dir, f"bins_L{args.level}_s{sigma:g}_st{args.stride}_b{args.bits}_{tag}_{h}.npz") if args.bins_dir else None
    if path and os.path.exists(path):
        b, _ = FB.load_bins([path]); print(f"  sigma {sigma:g} {tag}: cached {os.path.basename(path)}"); return b
    t = time.time(); b = build_bins(pairs, names, args.level, sigma, args.stride, args.bits)
    print(f"  sigma {sigma:g} {tag}: {len(names)} patches -> {len(b[0])} joint bins, {b[1].sum():.0f} pairs ({time.time() - t:.0f}s)")
    if path:
        os.makedirs(args.bins_dir, exist_ok=True)
        FB.save_bins(path, b, dict(sigma=sigma, stride=args.stride, bits=args.bits, level=args.level, split=tag))
    return b


# ----------------------------------------------------------------------------- evaluation
def lut_table(lut):
    """256^3 uint8 lookup table from a lattice LUT (trilinear, as used for fitting)."""
    t = np.empty((256 ** 3, 3), np.uint8)
    gg, bb = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    for r in range(256):
        x = np.stack([np.full(65536, r), gg.ravel(), bb.ravel()], 1) / 255.0
        t[r * 65536:(r + 1) * 65536] = np.clip(np.round(PI.apply_lut(lut, x) * 255), 0, 255)
    return t


class HoldoutSet:
    """Hold-out images at the evaluation level (uint8 encoded), with target statistics precomputed."""

    def __init__(self, pairs, names, level, region, n_px, seed=0):
        self.items = []
        rng = np.random.default_rng(seed)
        for n in names:
            a, ma = read_level(pairs[n][0], level); b, mb = read_level(pairs[n][1], level)
            if a.shape != b.shape:
                continue
            a8 = np.round(prepare(a, ma, 0) * 255).astype(np.uint8); b8 = np.round(prepare(b, mb, 0) * 255).astype(np.uint8)
            H, W = (a8.shape[0] // region) * region, (a8.shape[1] // region) * region
            idx = rng.choice(a8.shape[0] * a8.shape[1], min(n_px, a8.shape[0] * a8.shape[1]), replace=False)
            tm, ts = self._regions(b8, H, W, region)
            self.items.append(dict(src=a8, H=H, W=W, tgt_mean=tm, tgt_std=ts, idx=idx,
                                   tgt_px=b8.reshape(-1, 3)[idx] / 255.0))
        self.region = region
        self.tgt_px = np.concatenate([i["tgt_px"] for i in self.items]) if self.items else np.zeros((0, 3))
        self.tgt_L = PI.srgb_to_lab_d50(self.tgt_px)[:, 0]

    @staticmethod
    def _regions(img8, H, W, R):
        x = img8[:H, :W] / 255.0
        lin = FB.dec(x).reshape(H // R, R, W // R, R, 3).mean((1, 3))
        L = PI.srgb_to_lab_d50(x)[..., 0].reshape(H // R, R, W // R, R).std((1, 3))
        return FB.enc(lin).reshape(-1, 3), L.reshape(-1)

    def score(self, table):
        rm, rt, cs, px = [], [], [], []
        for it in self.items:
            s = it["src"]
            key = (s[..., 0].astype(np.uint32) << 16) | (s[..., 1].astype(np.uint32) << 8) | s[..., 2]
            m = table[key]
            mm, ms = self._regions(m, it["H"], it["W"], self.region)
            rm.append(mm); rt.append(it["tgt_mean"])
            ok = it["tgt_std"] > 3
            cs.append(ms[ok] / it["tgt_std"][ok])
            px.append(m.reshape(-1, 3)[it["idx"]] / 255.0)
        rm, rt, px = np.concatenate(rm), np.concatenate(rt), np.concatenate(px)
        Lp = PI.srgb_to_lab_d50(px)[:, 0]
        return dict(region_dE00=float(PI.delta_e00(rm, rt).mean()), pixel_dE00=float(PI.delta_e00(px, self.tgt_px).mean()),
                    contrast=float(np.median(np.concatenate(cs))) if cs else float("nan"),
                    dark1=float(np.percentile(Lp, 1) - np.percentile(self.tgt_L, 1)),
                    bright99=float(np.percentile(Lp, 99) - np.percentile(self.tgt_L, 99)))


# ----------------------------------------------------------------------------- main
def select_key(row, metric):
    """Smaller is better."""
    import model_eval as M
    v = row[metric]
    if metric in M.HIGHER or metric.startswith("gap_closed_"):
        return -v
    if metric in ("dark1", "bright99"):
        return abs(v)
    if metric == "contrast":
        return abs(1 - v)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--tgt", required=True)
    ap.add_argument("--level", type=int, default=1, help="stored image index to fit on (h5_pyramid: 0 = 1/2, 1 = 1/4)")
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.0], help="extra blur at that level, px (one or more)")
    ap.add_argument("--stride", type=int, default=1, help="pair sampling step at the fitting level")
    ap.add_argument("--bits", type=int, default=7, help="joint-histogram bits per channel per side")
    ap.add_argument("--robust", type=float, nargs="+", default=[2.5], help="Tukey c values (0 = off)")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--lam", type=float, nargs="+", default=[3e-4], help="lattice smoothness values")
    ap.add_argument("--lut-size", type=int, default=33)
    ap.add_argument("--penalty", choices=["slope", "curvature"], default="slope",
                    help="slope (default): first differences of the lattice residual; curvature: second differences, "
                         "penalises non-linearity only (use with --lam0)")
    ap.add_argument("--lam0", type=float, nargs="+", default=[0.0], help="ridge toward the affine base (values; e.g. 1e-4 with curvature)")
    ap.add_argument("--holdout", type=float, default=0.15, help="fraction of patches/groups held out (0 = none)")
    ap.add_argument("--group-regex", default=r"SL\d+",
                    help="slide id in the file name (default 'SL\\d+'); whole slides are held out. 'none' = per patch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-list", help="text file of patch names to fit on (overrides the internal split)")
    ap.add_argument("--test-list", help="text file of hold-out patch names (use with --train-list)")
    ap.add_argument("--eval-level", type=int, default=0, help="stored image index for hold-out scoring (0 = finest)")
    ap.add_argument("--region", type=int, default=24, help="region size for region_dE00/contrast, px at the eval level")
    ap.add_argument("--eval-px", type=int, default=5000, help="pixels sampled per hold-out patch for pixel metrics")
    ap.add_argument("--select", default=None,
                    help="metric used to pick the final settings: region_dE00 (default without a model), pixel_dE00, "
                         "dark1, contrast, or a model metric such as seg_abs_dp, seg_tol_dice, cls_abs_dp, cls_kappa, "
                         "feat_cos, gap_closed_seg_abs_dp (default with --model-plugin: seg_abs_dp, else cls_abs_dp, feat_cos)")
    ap.add_argument("--model-plugin", help="python file with TILE, load(), predict(): score candidates through your models")
    ap.add_argument("--model-level", type=int, default=None,
                    help="stored image index matching the model's magnification (default: --eval-level)")
    ap.add_argument("--no-refit", action="store_true", help="final profile from training patches only")
    ap.add_argument("--bins-dir", help="cache joint histograms here (reused on later runs)")
    ap.add_argument("--out", required=True); ap.add_argument("--name", default="SRC")
    args = ap.parse_args()

    S, T = list_files(args.src), list_files(args.tgt)
    names = sorted(set(S) & set(T))
    if not names:
        sys.exit("no matching file names between --src and --tgt")
    pairs = {n: (S[n], T[n]) for n in names}
    a, meta = read_level(S[names[0]], args.level)
    lv_meta = next((d for d in meta.get("levels", []) if d.get("shape") == list(a.shape[:2])), None)
    f = 1 / lv_meta["scale"] if lv_meta else None
    print(f"{len(names)} pairs | level {args.level}: {a.shape[1]}x{a.shape[0]} {a.dtype}" +
          (f", 1/{f:g} of full res, pyramid blur {lv_meta['sigma_full_res_px']} px full-res" if lv_meta else ""))
    if lv_meta:
        for s in args.sigma:
            print(f"  extra sigma {s:g} px at this level -> total ~{np.hypot(lv_meta['sigma_full_res_px'], s * f):.2f} px at full resolution")
    regex = None if args.group_regex.lower() == "none" else args.group_regex
    if args.train_list or args.test_list:
        if not (args.train_list and args.test_list):
            sys.exit("give both --train-list and --test-list")
        train, hold, slide = user_split(names, args.train_list, args.test_list, regex)
    else:
        train, hold, slide, held, ng = split_groups(names, args.holdout, regex, args.seed)
        unit = "slides" if regex else "patches"
        print(f"split by {unit}: {len(held)} of {ng} {unit} held out = {len(hold)} of {len(names)} patches "
              f"({100 * len(hold) / len(names):.1f}%); train {len(train)} patches from {ng - len(held)} {unit}")
        if regex:
            print("  held-out slides: " + ", ".join(held[:12]) + (" ..." if len(held) > 12 else ""))
    hs, ts = set(hold), set(train)
    with open(args.out + "_split.csv", "w", newline="") as fs:
        wr = csv.writer(fs); wr.writerow(["patch", "slide", "set"])
        for n in names:
            wr.writerow([n, slide[n], "holdout" if n in hs else "train" if n in ts else "unused"])

    H = HoldoutSet(pairs, hold, args.eval_level, args.region, args.eval_px) if hold else None
    ME = None
    if args.model_plugin:
        import model_eval
        if not hold:
            sys.exit("--model-plugin needs a hold-out set")
        ml = args.eval_level if args.model_level is None else args.model_level
        imgs = []
        for n in hold:
            a, ma = read_level(pairs[n][0], ml); b, mb = read_level(pairs[n][1], ml)
            if a.shape == b.shape:
                imgs.append((np.round(prepare(a, ma, 0) * 255).astype(np.uint8), np.round(prepare(b, mb, 0) * 255).astype(np.uint8)))
        ME = model_eval.ModelEval(args.model_plugin, imgs)
        with open(args.out + "_model_reference.csv", "w", newline="") as fr:
            keys = sorted({k for v in ME.ref.values() for k in v})
            wr = csv.writer(fr); wr.writerow(["reference"] + keys)
            for name, v in ME.ref.items():
                wr.writerow([name] + [v.get(k, "") for k in keys])
    rows, best, select = [], None, None
    for s in args.sigma:
        btr = cached_bins(args, pairs, train, s, "train")
        for c in args.robust:
            for lam, lam0 in [(a, b) for a in args.lam for b in args.lam0]:
                print(f"fit sigma {s:g}, robust {c:g}, lam {lam:g}, lam0 {lam0:g} ({args.penalty})", flush=True)
                lut, w, hist = FB.robust_fit(btr, args.lut_size, lam, c, args.iters if c > 0 else 0, args.penalty, lam0)
                row = dict(sigma=s, robust=c, lam=lam, lam0=lam0, penalty=args.penalty, rejected=hist[-1][1] if hist else 0.0)
                if H:
                    table = lut_table(lut)
                    row.update(H.score(table))
                    print("   hold-out: " + "  ".join(f"{k} {row[k]:+.3f}" if k.startswith(("dark", "bright")) else f"{k} {row[k]:.3f}"
                                                    for k in ("region_dE00", "pixel_dE00", "contrast", "dark1", "bright99")), flush=True)
                    if ME:
                        mm = ME.score(table); row.update(mm)
                        print("   model:    " + "  ".join(f"{k} {v:.3f}" for k, v in mm.items() if not k.startswith("gap")), flush=True)
                rows.append(row)
                if select is None:
                    import model_eval as _M
                    select = args.select or next((m for m in _M.DEFAULT_SELECT if m in row), "region_dE00")
                    if H and select not in row:
                        sys.exit(f"--select {select}: not available; choose from {sorted(k for k in row if k not in ('sigma', 'robust', 'lam', 'lam0', 'penalty'))}")
                key = select_key(row, select) if H else 0
                if best is None or key < best[0]:
                    best = (key, row, lut)
    with open(args.out + "_tuning.csv", "w", newline="") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    _, brow, lut = best
    print(f"selected by {select}: sigma {brow['sigma']:g}, robust {brow['robust']:g}, lam {brow['lam']:g}, lam0 {brow['lam0']:g} ({args.penalty})")
    if hold and not args.no_refit:
        print("refitting the selected settings on all patches (train + hold-out)")
        ball = FB.reduce_bins([cached_bins(args, pairs, train, brow["sigma"], "train"),
                               cached_bins(args, pairs, hold, brow["sigma"], "holdout")])
        lut, _, _ = FB.robust_fit(ball, args.lut_size, brow["lam"], brow["robust"], args.iters if brow["robust"] > 0 else 0,
                                  args.penalty, brow["lam0"])
    np.save(args.out + "_lut.npy", lut)
    PI.write_devicelink(args.out + "_devicelink.icc", lut, args.name)
    PI.write_input_profile(args.out + "_input.icc", lut, args.name)
    PI.write_cube(args.out + ".cube", lut, f"{args.name} to target (level {args.level}, sigma {brow['sigma']:g})")
    print(f"wrote {args.out}_devicelink.icc, {args.out}_input.icc, {args.out}.cube, {args.out}_lut.npy, {args.out}_tuning.csv")


if __name__ == "__main__":
    main()
