#!/usr/bin/env python3
"""Apply the PLISM-sm XYZ bridge (GT450 -> Philips UFS look) with one ICC input profile, using Pillow (LittleCMS).

    GT450 pixels --[optional gamma encode: v ** (1/gamma)]--[input profile]--> sRGB (P look)

The input profile (profiles/plism_sm_xyz/plism_gt450_to_p_xyz3x3_input.icc) contains the GT450
Segue_v2 profile followed by the PLISM 3x3 in XYZ, and expects gamma-2.2-encoded GT450 pixels, as
stored in PLISM's GT450 slides (SVS header "Gamma = 2.2").

--gamma (default 1.0 = disabled): set to 2.2 if your GT450 pixels are linear rather than
gamma-2.2 encoded. Each pixel value v (0..1) then becomes v ** (1/2.2) before the profile.

Usage
  python apply_xyz_bridge.py tile.png out.png                        # gamma-2.2 GT450 data (as in PLISM)
  python apply_xyz_bridge.py tile.png out.png --gamma 2.2            # linear GT450 data
  python apply_xyz_bridge.py tiles_dir/ out_dir/ --gamma 2.2         # batch
  python apply_xyz_bridge.py slide.svs out.png --region X Y W H [--gamma 2.2]   # SVS region via OpenSlide
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms

PROFILE = Path(__file__).resolve().parent / "profiles" / "plism_sm_xyz" / "plism_gt450_to_p_xyz3x3_input.icc"
_transform = None


def bridge_transform(profile=None):
    """GT450 RGB -> sRGB through the bridge profile. NOOPTIMIZE: LittleCMS's default 8-bit
    precalculation is too coarse for this lookup table (up to ~26 of 255 levels off in dark colours)."""
    global _transform
    if _transform is None:
        _transform = ImageCms.buildTransform(
            ImageCms.ImageCmsProfile(str(profile or PROFILE)), ImageCms.createProfile("sRGB"), "RGB", "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, flags=ImageCms.Flags.NOOPTIMIZE)
    return _transform


def gamma_encode(img: Image.Image, gamma: float) -> Image.Image:
    """v -> v ** (1/gamma) on 8-bit values (lookup table). gamma = 1 leaves the image unchanged."""
    if gamma == 1:
        return img
    lut = np.round(255 * (np.arange(256) / 255.0) ** (1.0 / gamma)).astype(np.uint8)
    return Image.fromarray(lut[np.asarray(img)])


def to_ufs(img: Image.Image, gamma: float = 1.0) -> Image.Image:
    img = gamma_encode(img.convert("RGB"), gamma)
    out = ImageCms.applyTransform(img, bridge_transform())
    out.info.pop("icc_profile", None)                     # save untagged sRGB, like UFS output
    return out


def read_svs_region(path, x, y, w, h):
    import openslide
    img = openslide.OpenSlide(path).read_region((x, y), 0, (w, h)).convert("RGB")   # raw stored pixels
    img.info.pop("icc_profile", None)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="gamma-encode the input first: 1 = off (default), 2.2 = linear GT450 data")
    ap.add_argument("--icc", default=str(PROFILE), help="bridge input profile (default: %(default)s)")
    ap.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="level-0 region of an .svs")
    a = ap.parse_args()
    src, dst = Path(a.input), Path(a.output)
    bridge_transform(a.icc)

    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg")):
            to_ufs(Image.open(f), a.gamma).save(dst / (f.stem + ".png"))
            print(f.name)
        return
    img = read_svs_region(str(src), *a.region) if a.region else Image.open(src)
    to_ufs(img, a.gamma).save(dst)
    print(f"{src} -> {dst}  (gamma {a.gamma:g})")


if __name__ == "__main__":
    main()
