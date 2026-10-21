#!/usr/bin/env python3
"""
Split an RGB->RGB device link into an input profile + output profile pair that reproduces it in
two-profile colour management (libvips icc_transform, Pillow ImageCms, LittleCMS transicc
without -l, ...), for tools that cannot use device links.

usage: python decompose_devicelink.py src_to_tgt_devicelink.icc out_prefix [--name SRC]
  writes <out_prefix>_in.icc   input profile  (class scnr, RGB -> XYZ, lut16 A2B0/1/2)
         <out_prefix>_out.icc  output profile (class mntr, RGB <-> XYZ, matrix + linear TRC)
  use:   vips icc_transform in.tif out.tif <out_prefix>_out.icc --input-profile <out_prefix>_in.icc --intent relative

How: the device link maps source RGB s to target RGB t = L(s). Choose an output profile that is
LINEAR in the target values, XYZ = M t (sRGB/D50 primaries, gamma 1.0, white = D50). The input
profile's table is then In(s) = M L(s) at every grid node. Because M is linear, interpolating In
equals M times interpolating L, so  Out^-1(In(s)) = L(s)  up to 16-bit encoding precision, with the
same interpolation as the device link. (Splitting through sRGB/Lab instead interpolates in a
non-linear space and costs up to ~16 levels at extreme colours.)

The pair is a mathematical construction: the PCS values are not real colorimetry of either scanner.
Use it only as a pair to apply the mapping; do not embed either profile in images or use it for
display. Requires numpy; plism_icc.py and apply_icc_patch.py next to this script.
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_icc_patch as AP   # noqa: E402
import plism_icc as PI         # noqa: E402

M = np.array([[0.4360747, 0.3850649, 0.1430804],      # linear RGB (sRGB primaries) -> XYZ, D50-adapted;
              [0.2225045, 0.7168786, 0.0606169],      # columns sum to the D50 white, so RGB (1,1,1) -> D50
              [0.0139322, 0.0971045, 0.7141733]])


def curv_identity():
    return b"curv" + b"\0" * 4 + struct.pack(">I", 0)        # 0 entries = identity (gamma 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("devicelink"); ap.add_argument("out_prefix"); ap.add_argument("--name", default="SRC")
    args = ap.parse_args()

    cin, clut, cout = AP.read_devicelink(args.devicelink)
    g = clut.shape[0]
    axis = np.linspace(0, 1, g)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    # device link evaluated exactly at the grid nodes (input/output curves included)
    t = AP.curves(AP.tetrahedral(clut, AP.curves(grid, cin)), cout)
    xyz = np.clip(t, 0, 1) @ M.T                                                  # In(s) = M L(s)
    enc = np.clip(np.round(xyz * 32768), 0, 65535).astype(np.uint16).reshape(g, g, g, 3)   # lut16 XYZ: 1.0 = 0x8000
    a2b = PI.tag_lut16(enc)
    cprt = PI.tag_text(f"Derived from {os.path.basename(args.devicelink)}; use only as an in/out pair")
    in_icc = PI.build_icc(b"scnr", b"RGB ", b"XYZ ", [
        (b"desc", PI.tag_desc(f"{args.name}: device-link input half (pair with _out)")), (b"cprt", cprt),
        (b"wtpt", PI.tag_xyz(PI.D50)), (b"A2B0", a2b), (b"A2B1", a2b), (b"A2B2", a2b)])
    out_icc = PI.build_icc(b"mntr", b"RGB ", b"XYZ ", [
        (b"desc", PI.tag_desc(f"{args.name}: device-link output half (linear target RGB)")), (b"cprt", cprt),
        (b"wtpt", PI.tag_xyz(PI.D50)),
        (b"rXYZ", PI.tag_xyz(M[:, 0])), (b"gXYZ", PI.tag_xyz(M[:, 1])), (b"bXYZ", PI.tag_xyz(M[:, 2])),
        (b"rTRC", curv_identity()), (b"gTRC", curv_identity()), (b"bTRC", curv_identity())])
    open(args.out_prefix + "_in.icc", "wb").write(in_icc)
    open(args.out_prefix + "_out.icc", "wb").write(out_icc)
    print(f"wrote {args.out_prefix}_in.icc ({len(in_icc):,} B, {g}^3 grid) and {args.out_prefix}_out.icc ({len(out_icc):,} B)")


if __name__ == "__main__":
    main()
