# PLISM-wsi and the original slides: GT450 and P

Notes on the PLISM-wsi tiles and on two original PLISM slides per scanner (Ochi et al., Sci Data
11, 330, 2024; Figshare+ doi:10.25452/figshare.plus.c.6773925; CC BY 4.0). All numbers come from
PLISM data only.

Release [`PLISM_wsi`](https://github.com/ruitaodong/plism-icc/releases/tag/PLISM_wsi):
`PLISM_wsi_en.csv`, `HRH_GT450.tar.gz`, `HRH_P.tar.gz` (tiles), and the original slides
`HRH_GT450.svs`, `HR_GT450.svs`, `HRH_P.tiff`, `HR_P.tiff`, `HRH_AT2.svs`.

## 1. The tile set

- **CSV.** 310,947 rows (13 stains x 7 scanners); columns `stain, device, tissue_type, coordinate,
  path`. `HRH` has 3,417 tiles per scanner, with the same coordinates for all seven scanners.
- **Coordinates.** `x_y` is the top-left of a 1,024 x 1,024 tile on a regular grid (step 1,024) in
  the reference frame of the `S60` scan. The PNG is the central 512 x 512 crop.
- **Pixel size.** The PNGs (PLISM-wsi and PLISM-sm alike) are on the `S60` grid, about
  **0.219 um/px**: a 512 px PNG covers about 112 um, i.e. about 415-443 GT450 pixels (0.262 um/px)
  and 433-461 P pixels (0.25 um/px). Found by registering the tiles to the original slides (scale
  0.836 relative to the GT450 level 0).
- **Pixel values are the stored slide pixels.** Crops read with OpenSlide at level 0 and resampled to
  the PNG grid match the PNGs to within about 1 of 255 per channel on average, for both scanners
  (`figures/plism_wsi_openslide_crop_vs_png.png`). PLISM applied no color management: GT450 tiles
  are raw GT450 RGB (the embedded Segue profile is not applied), P tiles are the stored UFS sRGB.
- **PLISM-sm vs PLISM-wsi.** Different field selections (smartphone fields vs a regular grid) from
  the same scans.

## 2. Locating tiles in the original slides

- `HRH_AT2.svs` and `HRH_GT450.svs` are related by a similarity transform (low-resolution feature
  registration: scale 0.966 = ratio of the two pixel sizes, rotation 0.05 deg).
- `HRH_P.tiff` is mounted **180 deg rotated** relative to `HRH_GT450.svs` (scale 1.059, rotation
  -179.94 deg): rotate P crops by 180 deg to match the tiles.
- A single affine from the CSV coordinates (S60 frame) to the GT450 slide is only approximate
  (residuals of a few hundred pixels: PLISM's registration is non-rigid). Predict with the affine,
  then refine each tile by template matching at the S60 scale (0.836 for GT450, about 0.876 for P).
- `analysis/plism_wsi/crop_hrh_prostate.py` crops five `HRH` prostate fields from both slides with
  the boxes found this way (`hrh_prostate_boxes.json` for PLISM-sm fields,
  `hrh_prostate_wsi_tile_boxes.json` for the nearest PLISM-wsi tiles). OpenSlide attaches the
  slide's embedded ICC profile to every region it reads (13 MB for GT450); the script removes it so
  the crops are untagged like the PNGs.

## 3. Registration of GT450 vs P tiles

| | PLISM-wsi `HRH` (421 tissue tiles) | PLISM-sm `HRH` (239) |
|---|---|---|
| Global shift, median / p90 | 0 / 2 px | 0 / 0 px |
| Tiles shifted by more than 5 px | 3% | 0% |
| Worst 128 px sub-tile shift, median / p90 | 3 / 6 px | 1 / 6 px |
| Tiles with a sub-tile off by more than 10 px | 1% | 2% |
| Lightness correlation, median | 0.81 | 0.93 |
| Correlation below 0.6 / 0.4 | 22% / 10% | 6% / 1% |

Low correlations in PLISM-wsi are mostly not misregistration: they come from low-texture tiles
(mounting medium, smooth stroma), different contrast between the scanners' renderings of the same
structures, and focus differences; a few tiles have genuine local warps of a few tens of pixels.
Residual shifts of 3-6 px (about 1 um) barely change region means; per-pixel comparisons are
sensitive to them.

## 4. Slide headers

**GT450 (`HRH_GT450.svs`, `HR_GT450.svs`)**, Aperio SVS ImageDescription:

```
Aperio Leica Biosystems GT450 v1.0.1
108994x81609 [0,0,108994x81609] (256x256) JPEG/YCC Q=91|AppMag = 40|
CalibrationAverageBlue = 215.34|CalibrationAverageGreen = 214.38|CalibrationAverageRed = 214.90|
Date = 05/10/2022|Exposure Scale = 0.000001|Exposure Time = 8|Filtered = 3|Focus Offset = -0.5|
Gamma = 2.2|Left = 35.15|MPP = 0.262120|Rack = 5|ScanScope ID = SS12016|Slide = 13|
StripeWidth = 4096|Time = 12:38:28|Time Zone = GMT+0900|Top = 22.22
```

- Embedded ICC profile (TIFF tag 34675): `Segue_v2` (ICC v4.0, RGB -> XYZ, lut16 with a 103^3 grid,
  gray axis close to gamma 2.2, glass -> D50 white). Stored, not applied to the pixels.
- `Gamma = 2.2` describes the stored pixel encoding; the Segue profile expects it.
- `Filtered = 3` is consistent with in-scanner sharpening (section 5). Field meanings other than
  size, magnification, MPP and dates are not publicly documented by Leica; the interpretations here
  are inferred.
- The two slides differ only in slide-specific values (position, size, white-calibration averages
  about 1.5% lower for `HR`).

**P (`HRH_P.tiff`)**, Philips TIFF export (`Philips DP v1.0`), XML in ImageDescription:

- Converted from iSyntax by Philips' exporter: `tiff-useBigTIFF=0-useRgb=0-levels=...-processing=0-q80`.
- JPEG quality 80, YCbCr 4:2:0, 512 x 512 tiles, 10 levels, 0.25 um/px.
- No ICC profile in the TIFF (the iSyntax colour-management block is not carried over).
- Scanner calibration recorded minutes before the scan.

**Philips UFS color pipeline.** The UFS converts its linear sensor data on the scanner with its own
matrix-only ICC profile and absolute colorimetric rendering to sRGB (Shrestha & Hulsken, J. Med.
Imaging 1(2):027501, 2014). The profile stored with an iSyntax file (DICOM ICC Profile (0028,2000))
documents that conversion; it does not describe the stored pixels, which are already sRGB. Do not
apply it again. P renders clear glass at about L* 90-92, slightly lavender; clipping to the sRGB
gamut is negligible for H&E (0.03% of PLISM P pixels at 255, none at 0).

## 5. Sharpness

Comparing amplitude spectra of 10 `HRH` fields read at native resolution (about 300 um squares):

| Detail scale | P / GT450 detail contrast | Equivalent extra blur on P |
|---|---|---|
| > 3 um (< 0.3 cycles/um) | 0.9-1.0 | about 0 |
| 1-2 um (0.5-0.9 cycles/um) | 0.71-0.74 | Gaussian sigma about 0.2-0.25 um |
| < 0.8 um (> 1.3 cycles/um) | 1.0-1.3 | P has more (noise, compression texture) |

GT450 is crisper at the nuclear scale, consistent with its in-scanner filter (`Filtered = 3`); the
difference is not a simple Gaussian. In the PLISM PNGs it disappears (single-Gaussian fit: no
difference), because both scanners were resampled to the `S60` grid.
