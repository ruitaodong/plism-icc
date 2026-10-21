#!/usr/bin/env python3
"""
Build color mappings from six PLISM WSI scanners to the Philips scanner ("P"),
and write them as ICC profiles.

Data: owkin/plism-dataset on Hugging Face (91 .h5 files = 13 stains x 7 scanners,
16,278 registered 224x224 tiles each, same tile order in every file). For a fixed
stain, tile i from scanner X and tile i from scanner P show the same tissue, so
every pixel pair is a (source RGB -> target RGB) training sample.

Method
  1. For each stain, read the same tile indices from X and P, block-average
     (default 8x8) to absorb small registration residuals, collect pixel pairs.
  2. Aggregate pairs into 7-bit source RGB bins (mean source, mean target, count).
  3. Fit a weighted global affine map, then a 3D lattice (default 33^3) on the
     residual with a smoothness penalty (lattice regression). Where there is no
     data the lattice falls back smoothly to the affine map.
  4. Evaluate CIEDE2000 on held-out tiles (before vs after mapping).
  5. Write per scanner:
       <X>_to_P_devicelink.icc  ICC v2 device link, RGB(X) -> RGB(P), A2B0 lut16
       <X>_as_P_input.icc       ICC v2 input profile, RGB(X) -> Lab, built so that
                                converting to sRGB reproduces P (P treated as sRGB)
       <X>_to_P.cube            same LUT as .cube (for convenience)

Usage
  pip install numpy scipy scikit-image h5py huggingface_hub pillow
  python plism_icc.py --out icc_out                  # stream from Hugging Face
  python plism_icc.py --local-dir /data/plism --out icc_out   # pre-downloaded .h5
  python plism_icc.py --from-bins bins_dir --out icc_out   # refit from saved *_bins.npz
  Options: --tiles-per-slide 1500 --grid 33 --block 8 --stains GMH GV ...
           --n-workers N   processes (default: all cores)   --save-bins   --solver cg|direct

Parallelism
  Read stage  map-reduce over (stain, tile chunk) tasks. Each worker reads P once plus the
              six sources for its tiles and returns sparse bin sums; sums and counts are
              additive, so merging them gives exactly the sequential result.
  Fit stage   one process per scanner (independent). Each fit solves a 33^3 sparse system
              with Jacobi-preconditioned CG (~0.4 s; --solver direct is ~10 s, same LUT).

Apply a device link, e.g. with LittleCMS:  tificc -l AT2_to_P_devicelink.icc in.tif out.tif
Or tag an AT2 image with AT2_as_P_input.icc and let any color-managed viewer
convert it to sRGB/display.
"""
import argparse
import datetime
import json
import os
import re
import struct
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

TARGET = "P"
SOURCES = ["AT2", "GT450", "S210", "S360", "S60", "SQ"]
REPO = "datasets/owkin/plism-dataset"
NAME_RE = re.compile(r"^(?P<stain>[A-Za-z0-9]+)_(?P<scanner>[A-Za-z0-9]+)(_to_[A-Za-z0-9_]+)?\.tif\.h5$")
QBITS = 7  # source-RGB binning resolution


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
class SlideStore:
    """Opens PLISM .h5 slides either locally or lazily over HTTP (range reads)."""

    def __init__(self, local_dir=None, index=None):
        self.local_dir = local_dir
        if not local_dir:
            from huggingface_hub import HfFileSystem
            self.fs = HfFileSystem()
        if index is not None:          # reuse a listing done by the parent process
            self.index = dict(index)
            return
        if local_dir:
            names = [f for f in os.listdir(local_dir) if f.endswith(".h5")]
        else:
            names = [os.path.basename(p) for p in self.fs.glob(f"{REPO}/*.h5")]
        self.index = {}
        for n in names:
            m = NAME_RE.match(n)
            if m:
                self.index[(m["stain"], m["scanner"])] = n
        if not self.index:
            raise RuntimeError("No PLISM .h5 files found")

    def stains(self):
        return sorted({s for s, _ in self.index})

    def has(self, stain, scanner):
        return (stain, scanner) in self.index

    def read_tiles(self, stain, scanner, idx):
        import h5py
        name = self.index[(stain, scanner)]
        fobj = open(os.path.join(self.local_dir, name), "rb") if self.local_dir \
            else self.fs.open(f"{REPO}/{name}", "rb", block_size=16 * 2**20)
        with fobj, h5py.File(fobj, "r") as h5:
            ds = _find_image_dataset(h5)
            return ds[np.sort(idx)]


def _find_image_dataset(h5):
    """Return the (N, H, W, 3) uint8 array inside the file, whatever its key."""
    found = []
    h5.visititems(lambda k, o: found.append(o) if getattr(o, "ndim", 0) == 4 and o.shape[-1] == 3 else None)
    if not found:
        raise RuntimeError(f"No (N,H,W,3) dataset in {h5.filename}; keys: {list(h5.keys())}")
    return max(found, key=lambda d: d.shape[0])


def block_mean(tiles, b):
    n, h, w, c = tiles.shape
    h2, w2 = h // b, w // b
    t = tiles[:, :h2 * b, :w2 * b].astype(np.float32).reshape(n, h2, b, w2, b, c)
    return t.mean(axis=(2, 4)) / 255.0


# ----------------------------------------------------------------------------
# Aggregation of pixel pairs into source-color bins
# ----------------------------------------------------------------------------
class BinAccumulator:
    def __init__(self, qbits=QBITS):
        self.q = qbits
        n = 1 << (3 * qbits)
        self.cnt = np.zeros(n, np.float64)
        self.sx = np.zeros((n, 3), np.float64)
        self.sy = np.zeros((n, 3), np.float64)

    def add(self, x, y):
        x = x.reshape(-1, 3); y = y.reshape(-1, 3)
        qi = np.clip((x * (1 << self.q)).astype(np.int64), 0, (1 << self.q) - 1)
        key = (qi[:, 0] << (2 * self.q)) | (qi[:, 1] << self.q) | qi[:, 2]
        n = self.cnt.size
        self.cnt += np.bincount(key, minlength=n)
        for c in range(3):
            self.sx[:, c] += np.bincount(key, x[:, c], minlength=n)
            self.sy[:, c] += np.bincount(key, y[:, c], minlength=n)

    def samples(self):
        m = self.cnt > 0
        c = self.cnt[m][:, None]
        return self.sx[m] / c, self.sy[m] / c, self.cnt[m]


def bin_pairs(x, y, qbits=QBITS):
    """Sparse bin sums for pixel pairs: (keys, counts, sum_x, sum_y), occupied bins only.
    Sums and counts are additive, so partial results from any split of the data can be
    merged exactly with merge_bins (this is what makes the read stage map-reduce)."""
    x = x.reshape(-1, 3).astype(np.float64); y = y.reshape(-1, 3).astype(np.float64)
    qi = np.clip((x * (1 << qbits)).astype(np.int64), 0, (1 << qbits) - 1)
    key = (qi[:, 0] << (2 * qbits)) | (qi[:, 1] << qbits) | qi[:, 2]
    u, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sx = np.stack([np.bincount(inv, x[:, c]) for c in range(3)], 1)
    sy = np.stack([np.bincount(inv, y[:, c]) for c in range(3)], 1)
    return u, cnt, sx, sy


def merge_bins(parts):
    """Reduce a list of bin_pairs results into (mean_x, mean_y, counts)."""
    keys = np.concatenate([p[0] for p in parts])
    u, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, np.concatenate([p[1] for p in parts]))
    sx = np.stack([np.bincount(inv, np.concatenate([p[2][:, c] for p in parts])) for c in range(3)], 1)
    sy = np.stack([np.bincount(inv, np.concatenate([p[3][:, c] for p in parts])) for c in range(3)], 1)
    return sx / cnt[:, None], sy / cnt[:, None], cnt


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def fit_affine(x, y, w):
    X = np.hstack([x, np.ones((len(x), 1))])
    sw = np.sqrt(w)[:, None]
    A, *_ = np.linalg.lstsq(X * sw, y * sw, rcond=None)
    return A  # (4, 3)


def trilinear_weights(x, g):
    """Sparse (N, g^3) interpolation matrix for points x in [0,1]^3 on a g^3 grid."""
    p = np.clip(x, 0, 1) * (g - 1)
    i0 = np.minimum(np.floor(p).astype(np.int64), g - 2)
    f = p - i0
    rows, cols, vals = [], [], []
    r = np.arange(len(x))
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                wgt = ((f[:, 0] if dr else 1 - f[:, 0]) * (f[:, 1] if dg else 1 - f[:, 1])
                       * (f[:, 2] if db else 1 - f[:, 2]))
                idx = ((i0[:, 0] + dr) * g + (i0[:, 1] + dg)) * g + (i0[:, 2] + db)
                rows.append(r); cols.append(idx); vals.append(wgt)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(x), g ** 3))


def grid_difference_operator(g):
    """First differences along each axis of a g^3 grid (C-order: r slowest, b fastest)."""
    d1 = sp.diags([-np.ones(g - 1), np.ones(g - 1)], [0, 1], shape=(g - 1, g))
    I = sp.identity(g)
    return sp.vstack([sp.kron(sp.kron(d1, I), I), sp.kron(sp.kron(I, d1), I),
                      sp.kron(sp.kron(I, I), d1)]).tocsr()


def d2_operator(g):
    """Second differences of a g^3 grid, pure (rr, gg, bb) and mixed (rg, rb, gb): zero for any affine field."""
    I = sp.identity(g, format="csr")
    d1 = sp.diags([-np.ones(g - 1), np.ones(g - 1)], [0, 1], shape=(g - 1, g))
    d2 = sp.diags([np.ones(g - 2), -2 * np.ones(g - 2), np.ones(g - 2)], [0, 1, 2], shape=(g - 2, g))
    k3 = lambda a, b, c: sp.kron(sp.kron(a, b), c)
    return sp.vstack([k3(d2, I, I), k3(I, d2, I), k3(I, I, d2),
                      np.sqrt(2) * k3(d1, d1, I), np.sqrt(2) * k3(d1, I, d1), np.sqrt(2) * k3(I, d1, d1)]).tocsr()


def fit_lut(x, y, cnt, g=33, lam=3e-4, solver="cg", penalty="slope", lam0=0.0):
    """penalty="slope" (default): lam * first differences of the lattice residual (decays to the affine base
    away from data). penalty="curvature": lam * second differences (pure + mixed), zero for any affine map, so
    lam penalises non-linearity only; needs lam0 > 0 (e.g. 1e-4, ridge on the residual) to pull unsupported
    regions back to the affine base. lam0 can also be used with "slope"."""
    """Returns LUT of shape (g, g, g, 3), values in [0,1], indexed [r, g, b]."""
    w = np.sqrt(cnt)  # dampen dominance of background/glass pixels
    w = w / w.sum()
    A_aff = fit_affine(x, y, w)
    resid = y - np.hstack([x, np.ones((len(x), 1))]) @ A_aff
    B = trilinear_weights(x, g)
    if penalty == "curvature" and lam0 <= 0:
        raise ValueError("penalty='curvature' needs lam0 > 0 (e.g. 1e-4); without it the problem is ill-posed")
    D = grid_difference_operator(g) if penalty == "slope" else d2_operator(g)
    Wm = sp.diags(w)
    M = (B.T @ Wm @ B + lam * (D.T @ D) + (lam0 + 1e-9) * sp.identity(g ** 3)).tocsr()
    rhs = B.T @ (w[:, None] * resid)
    if solver == "direct":       # sparse LU: exact, ~10 s and ~1 GB at g=33
        solve = spla.factorized(M.tocsc())
        L = np.stack([solve(rhs[:, c]) for c in range(3)], axis=1)
    else:                        # Jacobi-preconditioned CG: same result (<1e-4 levels), ~0.4 s
        Pinv = sp.diags(1.0 / M.diagonal())
        L = np.empty((g ** 3, 3))
        for c in range(3):
            L[:, c], info = spla.cg(M, rhs[:, c], rtol=1e-8, maxiter=40000, M=Pinv)
            if info != 0:        # fall back if CG did not converge
                L[:, c] = spla.spsolve(M.tocsc(), rhs[:, c])
    axis = np.linspace(0, 1, g)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    out = np.hstack([grid, np.ones((len(grid), 1))]) @ A_aff + L
    return np.clip(out, 0, 1).reshape(g, g, g, 3)


def apply_lut(lut, img):
    """Trilinear application of a (g,g,g,3) LUT to float RGB in [0,1] of any shape (...,3)."""
    g = lut.shape[0]
    shp = img.shape
    B = trilinear_weights(img.reshape(-1, 3), g)
    return (B @ lut.reshape(-1, 3)).reshape(shp)


# ----------------------------------------------------------------------------
# Color science helpers (sRGB, D50 PCS)
# ----------------------------------------------------------------------------
M_SRGB_D50 = np.array([[0.4360747, 0.3850649, 0.1430804],
                       [0.2225045, 0.7168786, 0.0606169],
                       [0.0139322, 0.0971045, 0.7141733]])
D50 = np.array([0.9642, 1.0, 0.8249])


def srgb_to_lab_d50(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ M_SRGB_D50.T / D50
    e = 216 / 24389
    f = np.where(xyz > e, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def delta_e00(a_rgb, b_rgb):
    from skimage.color import deltaE_ciede2000
    return deltaE_ciede2000(srgb_to_lab_d50(a_rgb), srgb_to_lab_d50(b_rgb))


# ----------------------------------------------------------------------------
# ICC writing (v2.4, lut16Type)
# ----------------------------------------------------------------------------
def s15f16(v):
    return struct.pack(">i", int(round(v * 65536)))


def tag_desc(text):
    a = text.encode("ascii", "replace") + b"\0"
    return (b"desc" + b"\0" * 4 + struct.pack(">I", len(a)) + a
            + struct.pack(">II", 0, 0) + struct.pack(">HB", 0, 0) + b"\0" * 67)


def tag_text(text):
    return b"text" + b"\0" * 4 + text.encode("ascii", "replace") + b"\0"


def tag_xyz(xyz):
    return b"XYZ " + b"\0" * 4 + b"".join(s15f16(v) for v in xyz)


def tag_lut16(clut_u16, n_table=256):
    """clut_u16: (g, g, g, 3) uint16, first input channel slowest."""
    g = clut_u16.shape[0]
    ident = np.round(np.linspace(0, 65535, n_table)).astype(">u2").tobytes()
    matrix = b"".join(s15f16(v) for v in (1, 0, 0, 0, 1, 0, 0, 0, 1))
    return (b"mft2" + b"\0" * 4 + struct.pack(">BBBB", 3, 3, g, 0) + matrix
            + struct.pack(">HH", n_table, n_table) + ident * 3
            + clut_u16.astype(">u2").tobytes() + ident * 3)


def tag_pseq(entries):
    out = b"pseq" + b"\0" * 4 + struct.pack(">I", len(entries))
    for mfg, model in entries:
        out += b"\0" * 4 + b"\0" * 4 + b"\0" * 8 + b"dcam" + tag_desc(mfg) + tag_desc(model)
    return out


def build_icc(dev_class, colorspace, pcs, tags):
    now = datetime.datetime.now(datetime.timezone.utc)
    body, table, offset = b"", b"", 128 + 4 + 12 * len(tags)
    for sig, data in tags:
        pad = (-len(data)) % 4
        table += sig + struct.pack(">II", offset + len(body), len(data))
        body += data + b"\0" * pad
    size = offset + len(body)
    header = (struct.pack(">I", size) + b"\0" * 4 + struct.pack(">I", 0x02400000)
              + dev_class + colorspace + pcs
              + struct.pack(">6H", now.year, now.month, now.day, now.hour, now.minute, now.second)
              + b"acsp" + b"\0" * 4 + struct.pack(">I", 0) + b"\0" * 8 + struct.pack(">Q", 0)
              + struct.pack(">I", 0) + b"".join(s15f16(v) for v in D50) + b"\0" * 4 + b"\0" * 44)
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + body


def write_devicelink(path, lut, src):
    clut = np.round(np.clip(lut, 0, 1) * 65535).astype(np.uint16)
    desc = f"PLISM {src} -> P (Philips) device link"
    data = build_icc(b"link", b"RGB ", b"RGB ", [
        (b"desc", tag_desc(desc)),
        (b"cprt", tag_text("Derived from PLISM (CC BY 4.0), Ochi et al. 2024")),
        (b"A2B0", tag_lut16(clut)),
        (b"pseq", tag_pseq([("PLISM", f"{src} scanner"), ("PLISM", "P scanner (sRGB reference)")])),
    ])
    open(path, "wb").write(data)


def write_input_profile(path, lut, src):
    lab = srgb_to_lab_d50(np.clip(lut, 0, 1))
    enc = np.stack([np.clip(lab[..., 0], 0, 100) * 652.8,               # L: 0..100 -> 0..0xFF00
                    np.clip((lab[..., 1] + 128) * 256, 0, 65535),        # a: 0 -> 0x8000
                    np.clip((lab[..., 2] + 128) * 256, 0, 65535)], -1)
    data = build_icc(b"scnr", b"RGB ", b"Lab ", [
        (b"desc", tag_desc(f"PLISM {src} scanner, characterized to P (Philips) as sRGB")),
        (b"cprt", tag_text("Derived from PLISM (CC BY 4.0), Ochi et al. 2024")),
        (b"wtpt", tag_xyz(D50)),
        (b"A2B0", tag_lut16(np.round(enc).astype(np.uint16))),
    ])
    open(path, "wb").write(data)


def write_cube(path, lut, title):
    g = lut.shape[0]
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {g}\n')
        for b in range(g):          # .cube: red varies fastest
            for gg in range(g):
                for r in range(g):
                    f.write("%.6f %.6f %.6f\n" % tuple(lut[r, gg, b]))


# ----------------------------------------------------------------------------
# Parallel stages
# ----------------------------------------------------------------------------
def _read_task(task):
    """Map step: one (stain, chunk of tiles). Reads P once, then each source scanner.
    Returns sparse bin sums (training tiles) and raw pixel pairs (held-out tiles)."""
    local_dir, index, stain, idx, is_test, sources, block = task
    store = SlideStore(local_dir, index=index)
    tgt = block_mean(store.read_tiles(stain, TARGET, idx), block)
    out = {}
    for src in sources:
        if not store.has(stain, src):
            continue
        x = block_mean(store.read_tiles(stain, src, idx), block)
        out[src] = (bin_pairs(x[~is_test], tgt[~is_test]),
                    x[is_test].reshape(-1, 3), tgt[is_test].reshape(-1, 3))
    return stain, len(idx), out


def _fit_task(task):
    """One scanner: fit, write ICC/.cube/.npy, evaluate on held-out pixels."""
    src, x, y, cnt, tx, ty, out_dir, grid, lam, solver = task
    t0 = time.time()
    lut = fit_lut(x, y, cnt, g=grid, lam=lam, solver=solver)
    t_fit = time.time() - t0
    np.save(os.path.join(out_dir, f"{src}_to_P_lut.npy"), lut)
    write_devicelink(os.path.join(out_dir, f"{src}_to_P_devicelink.icc"), lut, src)
    write_input_profile(os.path.join(out_dir, f"{src}_as_P_input.icc"), lut, src)
    write_cube(os.path.join(out_dir, f"{src}_to_P.cube"), lut, f"PLISM {src} to P")
    rep = {"bins": int(len(x)), "pixels": float(cnt.sum()), "fit_seconds": round(t_fit, 2)}
    if tx is not None and len(tx):
        tissue = ty.min(axis=1) < 0.8  # drop glass/background from the tissue metric
        before, after = delta_e00(tx, ty), delta_e00(apply_lut(lut, tx), ty)
        rep.update({k: float(v) for k, v in {
            "dE00_mean_before": before.mean(), "dE00_mean_after": after.mean(),
            "dE00_tissue_mean_before": before[tissue].mean(),
            "dE00_tissue_mean_after": after[tissue].mean(),
            "dE00_tissue_p95_after": np.percentile(after[tissue], 95)}.items()})
    return src, rep


def _pool(n_workers):
    """Process pool (h5py serializes threads, so threads would not help).
    'spawn' avoids fork issues with fsspec/HTTP sessions; one BLAS thread per worker."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"))


def _run(fn, tasks, n_workers):
    if n_workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            yield fn(t)
    else:
        with _pool(min(n_workers, len(tasks))) as ex:
            yield from ex.map(fn, tasks)   # map keeps task order -> deterministic output


def read_stage(args):
    store = SlideStore(args.local_dir)
    stains = args.stains or store.stains()
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(args.n_tiles_total, args.tiles_per_slide, replace=False))
    is_test = rng.random(len(idx)) < args.holdout

    # split each stain's tiles into chunks so all workers stay busy (P is read once per chunk)
    n_chunks = max(1, -(-args.n_workers // max(1, len(stains)))) if args.n_workers > 1 else 1
    tasks = []
    for stain in stains:
        if not store.has(stain, TARGET):
            print(f"[skip] {stain}: no {TARGET} slide"); continue
        for part in np.array_split(np.arange(len(idx)), n_chunks):
            tasks.append((args.local_dir, store.index, stain, idx[part], is_test[part],
                          args.sources, args.block))
    print(f"[read] {len(tasks)} tasks ({len(stains)} stains x {n_chunks} chunks), "
          f"{args.n_workers} workers", flush=True)

    parts = {s: [] for s in args.sources}
    test = {s: ([], []) for s in args.sources}
    for i, (stain, n, out) in enumerate(_run(_read_task, tasks, args.n_workers), 1):
        for src, (b, tx, ty) in out.items():      # reduce step: just collect
            parts[src].append(b); test[src][0].append(tx); test[src][1].append(ty)
        print(f"  [{i}/{len(tasks)}] {stain}: {n} tiles, {len(out)} scanners", flush=True)

    data = {}
    for src in args.sources:
        if not parts[src]:
            continue
        x, y, cnt = merge_bins(parts[src])
        tx, ty = np.concatenate(test[src][0]), np.concatenate(test[src][1])
        data[src] = (x, y, cnt, tx, ty)
        if args.save_bins:
            np.savez_compressed(os.path.join(args.out, f"{src}_bins.npz"), sources=x, targets=y, counts=cnt)
            np.savez_compressed(os.path.join(args.out, f"{src}_test.npz"),
                                test_sources=tx.astype(np.float32), test_targets=ty.astype(np.float32))
    return data


def load_bins(args):
    data = {}
    for src in args.sources:
        f = os.path.join(args.from_bins, f"{src}_bins.npz")
        if not os.path.exists(f):
            print(f"[skip] {f} not found"); continue
        b = np.load(f)
        tf = os.path.join(args.from_bins, f"{src}_test.npz")
        tx = ty = None
        if os.path.exists(tf):
            t = np.load(tf)
            tx, ty = t["test_sources"].astype(np.float64), t["test_targets"].astype(np.float64)
        data[src] = (b["sources"], b["targets"], b["counts"], tx, ty)
    return data


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", help="directory with PLISM .h5 files (otherwise stream from HF)")
    ap.add_argument("--out", default="icc_out")
    ap.add_argument("--stains", nargs="*", help="subset of stain codes (default: all 13)")
    ap.add_argument("--sources", nargs="*", default=SOURCES)
    ap.add_argument("--tiles-per-slide", type=int, default=1500)
    ap.add_argument("--n-tiles-total", type=int, default=16278)
    ap.add_argument("--block", type=int, default=8, help="block-average size to absorb misregistration")
    ap.add_argument("--grid", type=int, default=33)
    ap.add_argument("--lam", type=float, default=3e-4, help="lattice smoothness")
    ap.add_argument("--solver", choices=["cg", "direct"], default="cg")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=os.cpu_count() or 1,
                    help="processes for reading (stain x chunk) and fitting (one per scanner)")
    ap.add_argument("--save-bins", action="store_true",
                    help="also write <SRC>_bins.npz / <SRC>_test.npz so fitting can be rerun without the data")
    ap.add_argument("--from-bins", metavar="DIR", help="skip reading; fit from saved *_bins.npz (+ *_test.npz)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    data = load_bins(args) if args.from_bins else read_stage(args)
    t_read = time.time() - t0

    tasks = [(src, x, y, cnt, tx, ty, args.out, args.grid, args.lam, args.solver)
             for src, (x, y, cnt, tx, ty) in data.items()]
    print(f"[fit] {len(tasks)} scanners, {min(args.n_workers, len(tasks))} workers, solver={args.solver}", flush=True)
    report = {}
    for src, rep in _run(_fit_task, tasks, args.n_workers):
        report[src] = rep
        msg = f"  {src}: {rep['bins']} bins, fit {rep['fit_seconds']}s"
        if "dE00_tissue_mean_before" in rep:
            msg += f", held-out tissue dE00 {rep['dE00_tissue_mean_before']:.2f} -> {rep['dE00_tissue_mean_after']:.2f}"
        print(msg, flush=True)

    report["_timing"] = {"read_seconds": round(t_read, 1), "fit_seconds": round(time.time() - t0 - t_read, 1)}
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=2)
    print(f"done -> {args.out}  (read {t_read:.1f}s, fit {report['_timing']['fit_seconds']}s)")


if __name__ == "__main__":
    main()
