#!/usr/bin/env python3
"""
Apply an RGB->RGB device-link ICC profile (lut16, as written by plism_icc / fit_pyr) to one patch.

usage: python apply_icc_patch.py gt450_to_p_devicelink.icc input.png output.png

Same result as LittleCMS (tetrahedral interpolation of the 33^3 table). Needs numpy and pillow.
"""
import struct
import sys

import numpy as np
from PIL import Image


def read_devicelink(path):
    data = open(path, "rb").read()
    n = struct.unpack(">I", data[128:132])[0]
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", data[132 + 12 * i:144 + 12 * i])
        if sig == b"A2B0":
            tag = data[off:off + size]
            if tag[:4] != b"mft2":
                sys.exit("A2B0 is not a lut16 table")
            g = tag[10]
            n_in, n_out = struct.unpack(">HH", tag[48:52])
            a = np.frombuffer(tag[52:], dtype=">u2").astype(np.float64) / 65535.0
            curves_in = a[:3 * n_in].reshape(3, n_in)
            clut = a[3 * n_in:3 * n_in + g ** 3 * 3].reshape(g, g, g, 3)          # indexed [r, g, b]
            curves_out = a[3 * n_in + g ** 3 * 3:3 * n_in + g ** 3 * 3 + 3 * n_out].reshape(3, n_out)
            return curves_in, clut, curves_out
    sys.exit("no A2B0 tag found")


def curves(x, c):
    t = np.linspace(0, 1, c.shape[1])
    return np.stack([np.interp(x[:, k], t, c[k]) for k in range(3)], 1)


def tetrahedral(clut, x):
    g = clut.shape[0]
    p = np.clip(x, 0, 1) * (g - 1)
    i0 = np.minimum(np.floor(p).astype(np.int64), g - 2)
    f = p - i0
    order = np.argsort(-f, axis=1)                       # largest fraction first
    fs = np.take_along_axis(f, order, axis=1)
    flat, stride = clut.reshape(-1, 3), np.array([g * g, g, 1])
    idx = i0 @ stride
    out = (1 - fs[:, :1]) * flat[idx]
    for k in range(3):
        idx = idx + stride[order[:, k]]
        out += (fs[:, k:k + 1] - (fs[:, k + 1:k + 2] if k < 2 else 0)) * flat[idx]
    return out


def apply(icc, rgb_u8):
    cin, clut, cout = read_devicelink(icc)
    x = rgb_u8.reshape(-1, 3) / 255.0
    y = curves(tetrahedral(clut, curves(x, cin)), cout)
    return np.clip(np.round(y * 255), 0, 255).astype(np.uint8).reshape(rgb_u8.shape)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    icc, src, dst = sys.argv[1:]
    img = Image.open(src)
    rgb = np.asarray(img.convert("RGB"))
    out = apply(icc, rgb)
    if img.mode == "RGBA":                               # keep alpha unchanged
        out = np.dstack([out, np.asarray(img)[..., 3]])
    Image.fromarray(out).save(dst)
    print(f"{src} -> {dst} ({rgb.shape[1]}x{rgb.shape[0]})")
