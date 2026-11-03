#!/usr/bin/env python3
"""Re-crop the five PLISM-sm HRH prostate fields from the original slides with OpenSlide.

Boxes are level-0 pixels (top-left x, y, square size w) found by matching each PLISM PNG
against the slide. The PLISM PNGs are on the S60 grid (~0.219 um/px), so a 512-px PNG covers
~420-430 px on the GT450 (0.262 um/px) and ~430-450 px on the UFS (0.25 um/px).
The P slide (HRH_P.tiff) is mounted 180 degrees rotated relative to GT450 -> rotate the crop.

Usage:  python crop_hrh_prostate.py HRH_GT450.svs HRH_P.tiff outdir/
"""
import sys, os, openslide
from PIL import Image

BOXES = {   # coordinate: (GT450 x, y, w), (P x, y, w)
    "19154_31834_22060_35597": ((23000, 34179, 415), (93436, 43552, 433)),
    "19220_28123_22105_31717": ((23066, 31060, 428), (93356, 46820, 451)),
    "19242_30177_22191_33763": ((23133, 32786, 417), (93292, 45014, 440)),
    "19351_27926_22205_31512": ((23157, 30891, 425), (93262, 47001, 451)),
    "21385_33740_24254_37426": ((24862, 35701, 425), (91434, 41933, 448)),
}

def crop(slide, x, y, w, rotate=0):
    im = slide.read_region((x, y), 0, (w, w)).convert("RGB")   # raw stored pixels, no ICC applied
    im = im.rotate(rotate) if rotate else im
    im.info.pop("icc_profile", None)   # OpenSlide attaches the slide's ICC (13 MB Segue_v2 for GT450); save untagged like PLISM
    return im

if __name__ == "__main__":
    gt, p, out = openslide.OpenSlide(sys.argv[1]), openslide.OpenSlide(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    for c, ((gx, gy, gw), (px, py, pw)) in BOXES.items():
        g_img, p_img = crop(gt, gx, gy, gw), crop(p, px, py, pw, rotate=180)
        g_img.save(f"{out}/HRH_GT450_{c}_native.png"); p_img.save(f"{out}/HRH_P_{c}_native.png")
        g_img.resize((512, 512), Image.BICUBIC).save(f"{out}/HRH_GT450_{c}_512.png")   # PLISM grid
        p_img.resize((512, 512), Image.BICUBIC).save(f"{out}/HRH_P_{c}_512.png")
        print(c, "done")
