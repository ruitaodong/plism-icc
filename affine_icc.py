#!/usr/bin/env python3
"""
Write an affine colour mapping in linear light as ICC profiles.

  out = encode( clip( M @ [decode(x), 1] ) ),   decode(x) = x ** gamma_in,  encode(y) = y ** (1 / gamma_out)

M is a 3 x 4 matrix (3 x 3 + offset) acting on linear-light RGB in 0..1. gamma_in = 1.0 means the input
is already linear (no de-gamma).

usage:
  python affine_icc.py matrix.txt out_prefix [--gamma-in 2.2] [--gamma-out 2.2] [--name SRC]
writes
  <out_prefix>_devicelink.icc   RGB -> RGB device link            (tificc -c0 -l ...)
  <out_prefix>_input.icc        RGB -> XYZ input profile          (vips icc_transform ... --input-profile)
  <out_prefix>_output.icc       matching output profile (sRGB primaries, D50, gamma_out), use it as the target
matrix.txt: 3 rows of 4 numbers (R', G', B' from R, G, B, 1), '#' comments allowed.

Exactness: the input curves remove the gamma; the 33^3 grid holds the affine map in linear light unclipped
(stored with headroom [-0.5, 1.5], so interpolation reproduces the linear function exactly); the output
curves clip to 0..1 and encode. The input profile clips in XYZ. Use tificc -c0 (or LittleCMS without
optimisation): the default precalculation resamples the transform and adds small errors. The input/output pair is a mathematical construction (sRGB-primaries XYZ as the intermediate):
use the two only together, never embed them in images.
Requires numpy; plism_icc.py next to this script.
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plism_icc as PI  # noqa: E402

M_SRGB_D50 = np.array([[0.4360747, 0.3850649, 0.1430804],   # linear sRGB primaries -> XYZ, D50 adapted (columns sum to D50)
                       [0.2225045, 0.7168786, 0.0606169],
                       [0.0139322, 0.0971045, 0.7141733]])
G, NT = 33, 4096
LO, HI = -0.5, 1.5        # headroom: the grid stores the unclipped linear output mapped from [LO, HI] to [0, 1]


def mft2(curves_in, clut01, curves_out):
    """lut16 (mft2) tag: input curves (3 x n), CLUT (G^3 x 3, values 0..1), output curves (3 x n)."""
    q = lambda a: np.clip(np.round(np.asarray(a) * 65535), 0, 65535).astype(">u2").tobytes()
    head = b"mft2" + b"\0" * 4 + bytes([3, 3, G, 0]) + b"".join(struct.pack(">i", v) for v in (65536, 0, 0, 0, 65536, 0, 0, 0, 65536))
    head += struct.pack(">HH", curves_in.shape[1], curves_out.shape[1])
    return head + q(curves_in) + q(clut01) + q(curves_out)


def curv_table(f):
    t = np.clip(f(np.linspace(0, 1, NT)), 0, 1)
    return b"curv" + b"\0" * 4 + struct.pack(">I", NT) + np.round(t * 65535).astype(">u2").tobytes()


def build(M, gamma_in, gamma_out, prefix, name):
    axis = np.linspace(0, 1, G)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)        # linear-light inputs at the nodes
    lin_raw = np.hstack([grid, np.ones((len(grid), 1))]) @ M.T                               # unclipped: interpolates exactly
    t = np.linspace(0, 1, NT)
    cin = np.stack([t ** gamma_in] * 3)                                                       # de-gamma
    note = PI.tag_text(f"affine in linear light, gamma in {gamma_in:g}, gamma out {gamma_out:g}; use the input/output pair only together")
    # device link: RGB -> RGB
    link = mft2(cin, np.clip((lin_raw - LO) / (HI - LO), 0, 1),
                np.stack([np.clip(LO + t * (HI - LO), 0, 1) ** (1 / gamma_out)] * 3))       # clip only here, then encode
    open(prefix + "_devicelink.icc", "wb").write(PI.build_icc(b"link", b"RGB ", b"RGB ", [
        (b"desc", PI.tag_desc(f"{name}: affine device link")), (b"cprt", note), (b"A2B0", link)]))
    # input profile: RGB -> XYZ (lut16 XYZ encoding: 1.0 = 0x8000)
    xyz = np.clip(lin_raw @ M_SRGB_D50.T, 0, 65535 / 32768)                                 # clip in XYZ, not per RGB channel
    a2b = mft2(cin, xyz * 32768 / 65535, np.stack([t] * 3))
    open(prefix + "_input.icc", "wb").write(PI.build_icc(b"scnr", b"RGB ", b"XYZ ", [
        (b"desc", PI.tag_desc(f"{name}: affine input half (pair with _output)")), (b"cprt", note),
        (b"wtpt", PI.tag_xyz(PI.D50)), (b"A2B0", a2b), (b"A2B1", a2b), (b"A2B2", a2b)]))
    # output profile: sRGB primaries, D50, tone curve y = x ** gamma_out (device value x -> linear y)
    trc = curv_table(lambda x: x ** gamma_out)
    open(prefix + "_output.icc", "wb").write(PI.build_icc(b"mntr", b"RGB ", b"XYZ ", [
        (b"desc", PI.tag_desc(f"{name}: output half, sRGB primaries, gamma {gamma_out:g}")), (b"cprt", note),
        (b"wtpt", PI.tag_xyz(PI.D50)), (b"rXYZ", PI.tag_xyz(M_SRGB_D50[:, 0])), (b"gXYZ", PI.tag_xyz(M_SRGB_D50[:, 1])),
        (b"bXYZ", PI.tag_xyz(M_SRGB_D50[:, 2])), (b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix"); ap.add_argument("out_prefix")
    ap.add_argument("--gamma-in", type=float, default=2.2); ap.add_argument("--gamma-out", type=float, default=2.2)
    ap.add_argument("--name", default="SRC")
    a = ap.parse_args()
    M = np.loadtxt(a.matrix, comments="#").reshape(3, 4)
    build(M, a.gamma_in, a.gamma_out, a.out_prefix, a.name)
    print(f"wrote {a.out_prefix}_devicelink.icc, _input.icc, _output.icc (gamma in {a.gamma_in:g}, out {a.gamma_out:g})")


if __name__ == "__main__":
    main()
