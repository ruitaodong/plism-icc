#!/usr/bin/env python3
"""
Apply or embed PLISM ICC profiles in rasters with rasterio.

GDAL/rasterio read and write ICC profiles but never apply them to pixels, so this
script does the color conversion itself:

  apply  Convert pixels, block by block, and write a new tiled GeoTIFF.
         - Device link (<X>_to_P_devicelink.icc): its lut16 table is parsed and
           evaluated with tetrahedral interpolation (the method LittleCMS uses).
         - Input profile (<X>_as_P_input.icc): converted to sRGB with LittleCMS
           via Pillow's ImageCms.
         Either way a full 256^3 lookup table is built once (~50 MB), so large
         whole-slide images are processed at memory-copy speed.
  tag    Copy pixels unchanged and embed an input profile, so color-managed
         viewers show the image as it would look on scanner P.
  info   Print the ICC profile embedded in a raster, if any.

Examples
  python apply_icc_rasterio.py apply AT2_slide.tif AT2_to_P_devicelink.icc out_P.tif
  python apply_icc_rasterio.py apply AT2_slide.tif AT2_as_P_input.icc out_P.tif
  python apply_icc_rasterio.py tag   AT2_slide.tif AT2_as_P_input.icc out_tagged.tif
  python apply_icc_rasterio.py info  out_tagged.tif

Requirements: pip install rasterio numpy pillow
Input must be 8-bit with at least 3 bands (bands 1-3 = RGB); extra bands such as
alpha are copied through unchanged. Georeferencing and nodata are preserved.
"""
import argparse
import base64
import io
import struct
import sys

import numpy as np
import rasterio
from rasterio.enums import ColorInterp


# ----------------------------------------------------------------------------
# ICC parsing
# ----------------------------------------------------------------------------
def read_icc(path):
    data = open(path, "rb").read()
    if data[36:40] != b"acsp":
        raise ValueError(f"{path} is not an ICC profile")
    header = {"class": data[12:16], "colorspace": data[16:20], "pcs": data[20:24]}
    n = struct.unpack(">I", data[128:132])[0]
    tags = {}
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", data[132 + 12 * i: 144 + 12 * i])
        tags[sig] = data[off:off + size]
    return data, header, tags


def parse_lut16(tag):
    """Parse an lut16Type ('mft2') or lut8Type ('mft1') A2B0 tag."""
    kind = tag[:4]
    nin, nout, g = tag[8], tag[9], tag[10]
    if nin != 3 or nout != 3:
        raise ValueError(f"expected a 3-in/3-out table, got {nin}-in/{nout}-out")
    if kind == b"mft2":
        n_in, n_out = struct.unpack(">HH", tag[48:52])
        a = np.frombuffer(tag[52:], dtype=">u2").astype(np.float64) / 65535.0
    elif kind == b"mft1":
        n_in = n_out = 256
        a = np.frombuffer(tag[48:], dtype=np.uint8).astype(np.float64) / 255.0
    else:
        raise ValueError(f"A2B0 type {kind!r} not supported (only lut16/lut8)")
    p = 0
    curves_in = a[p:p + 3 * n_in].reshape(3, n_in); p += 3 * n_in
    clut = a[p:p + g ** 3 * 3].reshape(g, g, g, 3); p += g ** 3 * 3
    curves_out = a[p:p + 3 * n_out].reshape(3, n_out)
    return curves_in, clut, curves_out


def apply_curves(x, curves):
    n = curves.shape[1]
    grid = np.linspace(0, 1, n)
    return np.stack([np.interp(x[:, c], grid, curves[c]) for c in range(3)], axis=1)


def tetrahedral(clut, x):
    """Tetrahedral interpolation of a (g,g,g,3) table at points x (N,3) in [0,1]."""
    g = clut.shape[0]
    p = np.clip(x, 0, 1) * (g - 1)
    i0 = np.minimum(np.floor(p).astype(np.int64), g - 2)
    f = p - i0
    order = np.argsort(-f, axis=1)                      # largest fraction first
    fs = np.take_along_axis(f, order, axis=1)
    flat = clut.reshape(-1, 3)
    stride = np.array([g * g, g, 1])
    idx = i0 @ stride
    out = (1 - fs[:, 0:1]) * flat[idx]
    for k in range(3):
        idx = idx + stride[order[:, k]]
        w = fs[:, k:k + 1] - (fs[:, k + 1:k + 2] if k < 2 else 0)
        out += w * flat[idx]
    return out


# ----------------------------------------------------------------------------
# 256^3 lookup table
# ----------------------------------------------------------------------------
def all_colors_chunk(r):
    gg, bb = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    return np.stack([np.full(65536, r), gg.ravel(), bb.ravel()], axis=1)


def table_from_devicelink(tags):
    curves_in, clut, curves_out = parse_lut16(tags[b"A2B0"])
    table = np.empty((256 ** 3, 3), np.uint8)
    for r in range(256):
        x = apply_curves(all_colors_chunk(r) / 255.0, curves_in)
        y = apply_curves(tetrahedral(clut, x), curves_out)
        table[r * 65536:(r + 1) * 65536] = np.clip(np.round(y * 255), 0, 255)
    return table


def table_from_input_profile(path):
    from PIL import Image, ImageCms
    xform = ImageCms.buildTransform(ImageCms.ImageCmsProfile(path), ImageCms.createProfile("sRGB"),
                                    "RGB", "RGB", renderingIntent=ImageCms.Intent.PERCEPTUAL)
    cube = np.arange(256 ** 3, dtype=np.uint32)
    rgb = np.stack([(cube >> 16) & 255, (cube >> 8) & 255, cube & 255], -1).astype(np.uint8)
    img = Image.fromarray(rgb.reshape(4096, 4096, 3))
    return np.asarray(ImageCms.applyTransform(img, xform)).reshape(-1, 3)


def build_table(icc_path):
    _, header, tags = read_icc(icc_path)
    if header["class"] == b"link":
        if header["colorspace"] != b"RGB " or header["pcs"] != b"RGB ":
            raise ValueError("device link must be RGB -> RGB")
        print("device link: evaluating lut16 with tetrahedral interpolation", file=sys.stderr)
        return table_from_devicelink(tags)
    if header["colorspace"] != b"RGB ":
        raise ValueError("profile must have an RGB device color space")
    print("input profile: converting to sRGB with LittleCMS (Pillow)", file=sys.stderr)
    return table_from_input_profile(icc_path)


def apply_table(table, rgb):
    """rgb: (3, H, W) uint8 -> (3, H, W) uint8."""
    key = (rgb[0].astype(np.uint32) << 16) | (rgb[1].astype(np.uint32) << 8) | rgb[2]
    return np.moveaxis(table[key], -1, 0)


# ----------------------------------------------------------------------------
# rasterio I/O
# ----------------------------------------------------------------------------
def output_profile(src, compress, extra=None):
    prof = src.profile.copy()
    prof.update(driver="GTiff", tiled=True, blockxsize=512, blockysize=512,
                compress=compress, BIGTIFF="IF_SAFER")
    prof.pop("photometric", None)
    if src.count == 3 and compress.lower() == "jpeg":
        prof["photometric"] = "YCBCR"
    elif src.count >= 3:
        prof["photometric"] = "RGB"
    if extra:
        prof.update(extra)
    return prof


def check_input(src):
    if src.count < 3 or any(dt != "uint8" for dt in src.dtypes[:3]):
        raise SystemExit("input must have >= 3 uint8 bands (RGB)")
    embedded = src.tags(ns="COLOR_PROFILE").get("SOURCE_ICC_PROFILE")
    if embedded:
        print("note: input already embeds an ICC profile. The PLISM mappings were fit on raw "
              "pixel values, so only use this if your pixels are also raw (not already converted).",
              file=sys.stderr)


def copy_band_meta(src, dst):
    ci = list(src.colorinterp)
    ci[:3] = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
    dst.colorinterp = ci
    dst.update_tags(**src.tags())


def cmd_apply(args):
    table = build_table(args.icc)
    with rasterio.open(args.input) as src:
        check_input(src)
        with rasterio.open(args.output, "w", **output_profile(src, args.compress)) as dst:
            copy_band_meta(src, dst)
            windows = [w for _, w in dst.block_windows(1)]
            for i, win in enumerate(windows, 1):
                data = src.read(window=win)
                data[:3] = apply_table(table, data[:3])
                dst.write(data, window=win)
                if i % 200 == 0 or i == len(windows):
                    print(f"\r{i}/{len(windows)} blocks", end="", file=sys.stderr)
    print(f"\nwrote {args.output}", file=sys.stderr)


def cmd_tag(args):
    raw, header, _ = read_icc(args.icc)
    if header["class"] == b"link":
        raise SystemExit("a device link cannot be embedded; use the *_as_P_input.icc profile")
    b64 = base64.b64encode(raw).decode("ascii")
    with rasterio.open(args.input) as src:
        check_input(src)
        prof = output_profile(src, args.compress, {"SOURCE_ICC_PROFILE": b64})
        with rasterio.open(args.output, "w", **prof) as dst:
            copy_band_meta(src, dst)
            for _, win in dst.block_windows(1):
                dst.write(src.read(window=win), window=win)
    print(f"wrote {args.output} with embedded {args.icc}", file=sys.stderr)


def cmd_info(args):
    with rasterio.open(args.input) as src:
        b64 = src.tags(ns="COLOR_PROFILE").get("SOURCE_ICC_PROFILE")
    if not b64:
        print("no embedded ICC profile"); return
    raw = base64.b64decode(b64)
    from PIL import ImageCms
    prof = ImageCms.ImageCmsProfile(io.BytesIO(raw))
    print(f"embedded ICC profile: {len(raw)} bytes")
    print("description:", ImageCms.getProfileDescription(prof).strip())
    if args.extract:
        open(args.extract, "wb").write(raw)
        print("saved to", args.extract)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("apply", cmd_apply), ("tag", cmd_tag)):
        p = sub.add_parser(name)
        p.add_argument("input"); p.add_argument("icc"); p.add_argument("output")
        p.add_argument("--compress", default="deflate", help="deflate (default), lzw, jpeg, zstd, none")
        p.set_defaults(func=fn)
    p = sub.add_parser("info")
    p.add_argument("input"); p.add_argument("--extract", help="save the embedded profile to this path")
    p.set_defaults(func=cmd_info)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
