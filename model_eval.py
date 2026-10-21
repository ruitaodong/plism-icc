#!/usr/bin/env python3
"""
Score colour mappings through your own models (segmentation + classification + features).

Used by fit_pyr.py --model-plugin plugin.py. The plugin is a small Python file you write
(template: model_plugin_template.py):

    TILE  = 512        # input tile size, px (tiles are cut from the hold-out patches)
    BATCH = 16
    def load():                      -> model object
    def predict(model, tiles):       tiles: (N, TILE, TILE, 3) uint8 RGB
        return {"seg":  (N, C, h, w) class probabilities   (optional)
                "cls":  (N, K) class probabilities          (optional)
                "feat": (N, D) feature vectors              (optional)}

For every candidate LUT the hold-out SOURCE tiles are mapped, passed through the model and
compared with the model's outputs on the matching TARGET tiles (same tissue, other scanner):

  segmentation  seg_abs_dp      mean |p_mapped - p_target| per pixel and class               (lower better)
                seg_dice        Dice of argmax masks, mean over present classes              (higher better)
                seg_tol_dice    same with a +-2 px boundary tolerance                        (higher better)
                seg_area_mae    mean abs. difference of class area fractions per tile, %    (lower better)
  classification cls_agree      argmax agreement                                             (higher better)
                cls_kappa       Cohen's kappa of argmax labels                              (higher better)
                cls_abs_dp      mean |p_mapped - p_target|                                  (lower better)
                cls_wass        mean 1-D Wasserstein distance of per-class score distributions (lower better)
  features      feat_cos        cosine similarity of matched tiles                          (higher better)
                feat_top1       cross-scanner retrieval: own pair is the nearest target tile (higher better)

Two reference rows are reported once:
  raw      source tiles without any mapping (what the mapping should improve on)
  ceiling  target vs. target shifted by 1 px and blurred (sigma 1, linear light): an estimate of the
           best agreement possible given residual misregistration and optics differences.
  gap_closed_<metric> = (mapped - raw) / (ceiling - raw) is added for the main metrics
  (0 = no better than raw, 1 = as good as the ceiling estimate; >1 is possible, the ceiling is an estimate).
  Soft Dice of two probability maps is deliberately not used: it is below 1 even for identical but
  uncertain maps, so it rewards confidence rather than agreement.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy import ndimage

HIGHER = {"seg_dice", "seg_tol_dice", "cls_agree", "cls_kappa", "feat_cos", "feat_top1"}
LOWER = {"seg_abs_dp", "seg_area_mae", "cls_abs_dp", "cls_wass"}
MAIN = ["seg_abs_dp", "seg_tol_dice", "cls_abs_dp", "cls_kappa", "feat_cos", "feat_top1"]
DEFAULT_SELECT = ("seg_abs_dp", "cls_abs_dp", "feat_cos")


def load_plugin(path):
    spec = importlib.util.spec_from_file_location("model_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("load", "predict"):
        if not hasattr(mod, fn):
            sys.exit(f"{path}: plugin must define {fn}()")
    return mod


def _dec(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _enc(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def degrade(tile_u8, shift=1, sigma=1.0):
    """Target tile shifted by `shift` px (reflect) and blurred in linear light: the 'ceiling' reference."""
    lin = _dec(tile_u8 / 255.0)
    lin = np.roll(lin, (shift, shift), axis=(0, 1))
    lin = np.stack([ndimage.gaussian_filter(lin[..., k], sigma) for k in range(3)], -1)
    return np.round(_enc(lin) * 255).astype(np.uint8)


def cut_tiles(img, T):
    """Non-overlapping T x T tiles, centred in the image (none if the image is smaller than T)."""
    H, W = img.shape[:2]
    ny, nx = H // T, W // T
    oy, ox = (H - ny * T) // 2, (W - nx * T) // 2
    return [img[oy + i * T:oy + (i + 1) * T, ox + j * T:ox + (j + 1) * T] for i in range(ny) for j in range(nx)]


class ModelEval:
    def __init__(self, plugin_path, holdout_images):
        """holdout_images: list of (src_uint8, tgt_uint8) at the model's resolution."""
        self.P = load_plugin(plugin_path)
        self.T = int(getattr(self.P, "TILE", 512))
        self.batch = int(getattr(self.P, "BATCH", 16))
        self.model = self.P.load()
        src, tgt = [], []
        for a, b in holdout_images:
            ta, tb = cut_tiles(a, self.T), cut_tiles(b, self.T)
            src += ta; tgt += tb
        if not src:
            sys.exit(f"model evaluation: hold-out images are smaller than TILE={self.T}")
        self.src = np.stack(src); self.tgt = np.stack(tgt)
        print(f"model evaluation: {len(self.src)} hold-out tiles of {self.T}x{self.T}", flush=True)
        self.out_tgt = self._predict(self.tgt)
        self.ref = {"raw": self._metrics(self._predict(self.src)),
                    "ceiling": self._metrics(self._predict(np.stack([degrade(t) for t in self.tgt])))}
        for k, v in self.ref.items():
            print(f"  {k:8s}: " + "  ".join(f"{m} {x:.3f}" for m, x in v.items()), flush=True)

    def _predict(self, tiles):
        outs = {}
        for i in range(0, len(tiles), self.batch):
            o = self.P.predict(self.model, tiles[i:i + self.batch])
            for k, v in o.items():
                if v is not None:
                    outs.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        return {k: np.concatenate(v) for k, v in outs.items()}

    def score(self, table):
        s = self.src
        key = (s[..., 0].astype(np.uint32) << 16) | (s[..., 1].astype(np.uint32) << 8) | s[..., 2]
        m = self._metrics(self._predict(table[key]))
        for k in MAIN:
            if k in m and k in self.ref["raw"]:
                d = self.ref["ceiling"][k] - self.ref["raw"][k]
                m[f"gap_closed_{k}"] = (m[k] - self.ref["raw"][k]) / d if abs(d) > 1e-9 else float("nan")
        return m

    # ------------------------------------------------------------------ metrics vs target outputs
    def _metrics(self, o):
        t, r = self.out_tgt, {}
        if "seg" in o and "seg" in t:
            p, q = o["seg"], t["seg"]
            if p.shape != q.shape:
                sys.exit(f"seg shapes differ: {p.shape} vs {q.shape}")
            C = p.shape[1]
            present = q.reshape(q.shape[0], C, -1).mean(-1).sum(0) > 1e-3 * q.shape[0]
            r["seg_abs_dp"] = float(np.abs(p - q).mean())
            ap, aq = p.argmax(1), q.argmax(1)
            dice = [2 * ((ap == c) & (aq == c)).sum() / max((ap == c).sum() + (aq == c).sum(), 1) for c in np.where(present)[0]]
            r["seg_dice"] = float(np.mean(dice)) if dice else float("nan")
            st = np.ones((5, 5), bool)                                   # +-2 px tolerance
            tol = []
            for c in np.where(present)[0]:
                A, Bm = ap == c, aq == c
                if A.sum() == 0 and Bm.sum() == 0:
                    continue
                dA = np.stack([ndimage.binary_dilation(x, st) for x in A]); dB = np.stack([ndimage.binary_dilation(x, st) for x in Bm])
                prec = (A & dB).sum() / max(A.sum(), 1); rec = (Bm & dA).sum() / max(Bm.sum(), 1)
                tol.append(2 * prec * rec / max(prec + rec, 1e-9))
            r["seg_tol_dice"] = float(np.mean(tol)) if tol else float("nan")
            fa = np.stack([(ap == c).mean((1, 2)) for c in range(C)], 1); fb = np.stack([(aq == c).mean((1, 2)) for c in range(C)], 1)
            r["seg_area_mae"] = float(100 * np.abs(fa - fb)[:, present].mean())
        if "cls" in o and "cls" in t:
            p, q = o["cls"], t["cls"]
            a, b = p.argmax(1), q.argmax(1)
            r["cls_agree"] = float((a == b).mean())
            K = p.shape[1]; cm = np.zeros((K, K)); np.add.at(cm, (a, b), 1); n = cm.sum()
            pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2; po = np.trace(cm) / n
            r["cls_kappa"] = float((po - pe) / (1 - pe)) if pe < 1 else 1.0
            r["cls_abs_dp"] = float(np.abs(p - q).mean())
            r["cls_wass"] = float(np.mean([np.abs(np.sort(p[:, k]) - np.sort(q[:, k])).mean() for k in range(K)]))
        if "feat" in o and "feat" in t:
            f = o["feat"] / np.maximum(np.linalg.norm(o["feat"], axis=1, keepdims=True), 1e-9)
            g = t["feat"] / np.maximum(np.linalg.norm(t["feat"], axis=1, keepdims=True), 1e-9)
            r["feat_cos"] = float((f * g).sum(1).mean())
            r["feat_top1"] = float(((f @ g.T).argmax(1) == np.arange(len(f))).mean())
        return r
