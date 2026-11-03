#!/usr/bin/env python3
"""Convert whole-slide image pixels to CIE XYZ (ICC PCS: D50, white Y = 1).

  Philips UFS (iSyntax, no filter): pixels are already sRGB  -> decode sRGB, sRGB->XYZ matrix
  Aperio GT450:                    pixels are scanner RGB    -> scanner ICC profile (A2B lut16)

Both functions return float arrays (..., 3) in the same space, so the results are
directly comparable (glass / display white -> D50 = 0.9642, 1.0, 0.8249).

Usage:
    python wsi_xyz.py ufs  philips_tile.png
    python wsi_xyz.py gt450 gt450_tile.png --icc GT450_v2.icc
"""
import argparse
import struct

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

D50 = np.array([0.9642, 1.0, 0.8249])

# Linear sRGB -> XYZ, Bradford-adapted D65 -> D50 (the matrix in the standard sRGB ICC profile)
SRGB_TO_XYZ_D50 = np.array([[0.4360747, 0.3850649, 0.1430804],
                            [0.2225045, 0.7168786, 0.0606169],
                            [0.0139322, 0.0971045, 0.7141733]])
# Linear sRGB -> XYZ with D65 white (plain CIE colorimetry, if you prefer D65)
SRGB_TO_XYZ_D65 = np.array([[0.4124, 0.3576, 0.1805],
                            [0.2126, 0.7152, 0.0722],
                            [0.0193, 0.1192, 0.9505]])


def srgb_decode(v):
    """sRGB code values (0-1) -> linear light (0-1)."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def ufs_to_xyz(rgb8, white="D50"):
    """Philips UFS pixels (uint8 RGB, already sRGB) -> XYZ."""
    lin = srgb_decode(np.asarray(rgb8, dtype=np.float64) / 255.0)
    M = SRGB_TO_XYZ_D50 if white == "D50" else SRGB_TO_XYZ_D65
    return lin @ M.T


class ICCLut16:
    """Minimal reader for an RGB->XYZ ICC profile whose A2B tag is lut16Type ('mft2'),
    as in the GT450 Segue profiles. Applies input curves, 3D grid (trilinear), output curves."""

    def __init__(self, path, tag="A2B1"):  # A2B1 = media-relative colorimetric (same data as A2B0 in the Segue profiles)
        d = open(path, "rb").read()
        if d[20:24] != b"XYZ ":
            raise ValueError("profile PCS is not XYZ")
        n = struct.unpack(">I", d[128:132])[0]
        tags = {}
        for i in range(n):
            sig, off, ln = struct.unpack(">4sII", d[132 + 12 * i:144 + 12 * i])
            tags[sig.decode()] = (off, ln)
        if tag not in tags:
            tag = "A2B0"
        off, _ = tags[tag]
        t = d[off:]
        if t[:4] != b"mft2":
            raise ValueError(f"{tag} is {t[:4]!r}, expected lut16 'mft2'")
        nin, nout, g = t[8], t[9], t[10]
        nI, nO = struct.unpack(">HH", t[48:52])
        p = 52
        u16 = lambda k: np.frombuffer(t, dtype=">u2", count=k, offset=p).astype(np.float64) / 65535.0
        self.inp = u16(nin * nI).reshape(nin, nI); p += nin * nI * 2
        clut = u16(g ** nin * nout).reshape((g,) * nin + (nout,)); p += g ** nin * nout * 2
        self.out = u16(nout * nO).reshape(nout, nO)
        axis = np.linspace(0, 1, g)
        self.interp = RegularGridInterpolator((axis,) * nin, clut)
        # lut16 XYZ encoding: 1.0 is stored as 0x8000, i.e. value * 65535/32768
        self.xyz_scale = 65535.0 / 32768.0

    @staticmethod
    def _curves(x, tables):
        return np.stack([np.interp(x[..., c], np.linspace(0, 1, tables.shape[1]), tables[c])
                         for c in range(x.shape[-1])], axis=-1)

    def __call__(self, rgb01):
        shape = rgb01.shape
        x = self._curves(rgb01.reshape(-1, 3), self.inp)
        y = self._curves(self.interp(x), self.out)
        return (y * self.xyz_scale).reshape(shape)


def gt450_to_xyz(rgb8, icc_path="GT450_v2.icc"):
    """Aperio GT450 pixels (uint8 RGB, scanner RGB) -> XYZ via the scanner's ICC profile."""
    return ICCLut16(icc_path)(np.asarray(rgb8, dtype=np.float64) / 255.0)


def xyz_to_lab(xyz, white=D50):
    t = xyz / white
    f = np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scanner", choices=["ufs", "gt450"])
    ap.add_argument("image")
    ap.add_argument("--icc", default="GT450_v2.icc", help="GT450 profile (default: %(default)s)")
    ap.add_argument("-o", "--out", help="save XYZ as .npy (float32, H x W x 3)")
    a = ap.parse_args()

    rgb = np.asarray(Image.open(a.image).convert("RGB"))
    xyz = ufs_to_xyz(rgb) if a.scanner == "ufs" else gt450_to_xyz(rgb, a.icc)
    lab = xyz_to_lab(xyz)
    print(f"{a.image}: mean XYZ {xyz.reshape(-1, 3).mean(0).round(4)}  mean Lab {lab.reshape(-1, 3).mean(0).round(1)}")
    if a.out:
        np.save(a.out, xyz.astype(np.float32))
        print("saved", a.out)
