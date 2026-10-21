#!/usr/bin/env python3
"""
Convert TIFFs with an RGB->RGB device-link ICC profile, keeping the file structure.

usage:
  python icc_tiff.py src_to_ref_devicelink.icc in.tif out.tif
  python icc_tiff.py src_to_ref_devicelink.icc in_dir/ out_dir/        # every *.tif / *.tiff / *.svs
  python icc_tiff.py link.icc slide.svs slide_as_ref.svs --compression jpeg --quality 95

- Every image in the file is converted: single images, pyramids stored as top-level pages, and
  pyramids stored as SubIFDs (h5_pyramid.py / relayout_tiff.py layouts). Files without SubIFDs
  (e.g. Aperio SVS) are converted page by page, each page keeping its own size, tiling,
  description, NewSubfileType and resolution, so thumbnail, label and macro images survive and
  OpenSlide still recognises the file. SVS label and macro photos (description contains "label" /
  "macro") are copied unchanged unless --convert-associated. JPEG (RGB or YCbCr) and Aperio JPEG
  2000 tiles are decoded with tifffile/imagecodecs.
- Output compression: the input codec by default (JPEG at --quality, default 95; Aperio JPEG 2000
  is written as JPEG), or --compression deflate|lzw|jpeg|none. JPEG output is lossy (written
  without chroma subsampling unless --chroma-subsampling): use deflate if pixels must be exact.
- Tile size, compression, predictor, layout and the ImageDescription are kept; for h5_pyramid
  files the applied profile is recorded in the metadata ("icc_applied").
- uint8 and uint16 RGB, or float RGB in 0..1 (sRGB-encoded, or linear light if the h5_pyramid
  metadata says so). Pixels are converted with tetrahedral interpolation of the profile's table,
  matching LittleCMS without precalculation (tificc -c0 -l ...).
Requires numpy, tifffile (+ imagecodecs for float predictors); apply_icc_patch.py, relayout_tiff.py
next to this script.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_icc_patch as AP   # noqa: E402
import relayout_tiff as RT     # noqa: E402

PREFIX = "h5_pyramid"


def dec(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def enc(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


class DeviceLink:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self.cin, self.clut, self.cout = AP.read_devicelink(path)

    def map01(self, x):
        """(N, 3) encoded RGB in 0..1 -> (N, 3) in 0..1."""
        return AP.curves(AP.tetrahedral(self.clut, AP.curves(x, self.cin)), self.cout)

    def apply(self, img, linear=False):
        if img.ndim != 3 or img.shape[-1] < 3:
            raise ValueError(f"expected RGB(A) image, got shape {img.shape}")
        rgb, extra = img[..., :3], img[..., 3:]
        if img.dtype == np.uint8:
            x, scale = rgb.reshape(-1, 3) / 255.0, 255
        elif img.dtype == np.uint16:
            x, scale = rgb.reshape(-1, 3) / 65535.0, 65535
        elif img.dtype.kind == "f":
            x, scale = np.clip(rgb.reshape(-1, 3).astype(np.float64), 0, 1), None
            if linear:
                x = enc(x)
        else:
            raise ValueError(f"unsupported dtype {img.dtype}")
        y = np.concatenate([self.map01(x[i:i + 2_000_000]) for i in range(0, len(x), 2_000_000)])
        if scale is None:
            out = (dec(y) if linear else y).astype(img.dtype).reshape(rgb.shape)
        else:
            out = np.clip(np.round(y * scale), 0, scale).astype(img.dtype).reshape(rgb.shape)
        return np.concatenate([out, extra], -1) if extra.size else out


def convert(link, src, dst):
    levels, desc, kw, layout = RT.read_levels(src)
    linear = False
    if desc and desc.startswith(PREFIX):
        head, js = desc.split("\n", 1)
        meta = json.loads(js)
        linear = str(meta.get("values", "")).startswith("linear")
        meta["icc_applied"] = link.name
        desc = head + "\n" + json.dumps(meta)
    out = [link.apply(lv, linear) for lv in levels]
    tmp = dst + ".part"
    RT.write_levels(tmp, out, desc, kw, layout)
    os.replace(tmp, dst)
    return [lv.shape for lv in out], layout


COMP = {1: None, 5: "lzw", 7: "jpeg", 8: "zlib", 32946: "zlib", 50000: "zstd", 33003: "jpeg", 33005: "jpeg", 34712: "jpeg"}


def convert_pages(link, src, dst, compression=None, quality=95, associated=False, subsampling=(1, 1)):
    """Convert every top-level page independently, keeping its own tags (SVS-safe)."""
    import tifffile
    with tifffile.TiffFile(src) as t:
        pages = list(t.pages)
        tmp = dst + ".part"
        with tifffile.TiffWriter(tmp, bigtiff=os.path.getsize(src) > 2 ** 31) as tw:
            for i, p in enumerate(pages):
                a = p.asarray()
                desc = p.description
                assoc = bool(desc) and any(w in desc.lower() for w in ("label", "macro"))      # SVS label / macro photos
                rgb = a.ndim == 3 and a.shape[-1] in (3, 4) and (associated or not assoc)
                if i == 0 and desc and desc.startswith(PREFIX):
                    head, js = desc.split("\n", 1); meta = json.loads(js); meta["icc_applied"] = link.name
                    desc = head + "\n" + json.dumps(meta)
                out = link.apply(a) if rgb else a
                comp = compression if compression else COMP.get(int(p.compression), "zlib")
                if comp == "none":
                    comp = None
                kw = dict(compression=comp, metadata=None, description=desc or None,
                          subfiletype=int(p.tags["NewSubfileType"].value) if "NewSubfileType" in p.tags else 0)
                if comp == "jpeg":
                    kw["compressionargs"] = {"level": quality}
                    if a.ndim == 3 and a.shape[-1] == 3:
                        kw["subsampling"] = subsampling
                elif comp in ("zlib", "lzw", "zstd") and out.dtype.kind in "ui":
                    kw["predictor"] = 2
                if rgb:
                    kw["photometric"] = "rgb"
                if p.is_tiled:
                    kw["tile"] = (p.tilelength, p.tilewidth)
                if "XResolution" in p.tags and "YResolution" in p.tags:
                    kw["resolution"] = (p.tags["XResolution"].value, p.tags["YResolution"].value)
                    if "ResolutionUnit" in p.tags:
                        kw["resolutionunit"] = int(p.tags["ResolutionUnit"].value)
                tw.write(out, **kw)
    os.replace(tmp, dst)
    return len(pages)


def convert_any(link, src, dst, compression=None, quality=95, associated=False, subsampling=(1, 1)):
    import tifffile
    with tifffile.TiffFile(src) as t:
        has_subifds = bool(t.pages[0].subifds)
    if has_subifds:
        return convert(link, src, dst)[0].__len__()
    return convert_pages(link, src, dst, compression, quality, associated, subsampling)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("icc"); ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--compression", choices=["deflate", "lzw", "jpeg", "none"], default=None,
                    help="output compression (default: same codec as each input page)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality for JPEG output (default 95)")
    ap.add_argument("--chroma-subsampling", action="store_true", help="JPEG with 2x2 chroma subsampling (smaller, less colour-accurate)")
    ap.add_argument("--convert-associated", action="store_true", help="also convert SVS label/macro photos (default: copied unchanged)")
    a = ap.parse_args()
    icc, src, dst = a.icc, a.src, a.dst
    comp = {"deflate": "zlib"}.get(a.compression, a.compression)
    link = DeviceLink(icc)
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        files = sorted(f for f in os.listdir(src) if f.lower().endswith((".tif", ".tiff", ".svs")))
        for i, f in enumerate(files, 1):
            convert_any(link, os.path.join(src, f), os.path.join(dst, f), comp, a.quality, a.convert_associated, (2, 2) if a.chroma_subsampling else (1, 1))
            if i % 100 == 0 or i == len(files):
                print(f"{i}/{len(files)} converted", flush=True)
    else:
        n = convert_any(link, src, dst, comp, a.quality, a.convert_associated, (2, 2) if a.chroma_subsampling else (1, 1))
        print(f"{src} -> {dst}: {n} image(s) converted")


if __name__ == "__main__":
    main()
