# PLISM-sm: GT450 -> P findings

Analysis of the PLISM-sm subset (Ochi et al., Sci Data 11, 330, 2024; Figshare+
doi:10.25452/figshare.plus.c.6773925; CC BY 4.0), restricted to the Leica Aperio GT450 and the
Philips UltraFast Scanner (P). All numbers below come from PLISM data only.

## 1. The subset

Source list: `PLISM_sm_en.csv` (57,902 rows).

- **Groups.** 57,902 images = 4,454 groups x 13 devices (7 WSI scanners: GT450, AT2, S360, S60,
  S210, SQ, P; 6 smartphones). A group is one field of view in one stain (key: stain +
  coordinate); every group has all 13 devices and a single tissue type.
- **Same section within a group.** One stain is one glass slide, so the 7 scanner images of a
  group show the same physical section: valid pixel pairs for any scanner pair. Different stains
  are different serial sections and must not be paired at pixel level.
- **Coordinates.** A box `x1_y1_x2_y2`, described as AT2 WSI coordinates; median 2,743 x 3,497 px
  (the smartphone microscope field). The PNGs are 512 x 512 RGB crops after alignment (about
  325 KB each). In `HRH_AT2.svs` the PNG content is not at the box position: locating it in the
  original slides needed a search well beyond the box (see `PLISM_wsi.md`).
- **Groups per stain.** GV 437, LMH 436, GVH 420, MY 394, KR 393, GMH 381, LM 317, KRH 301,
  GIVH 297, GIV 297, HR 268, GM 258, HRH 255.
- **Tissues.** All 46 types, named `NN_name`; 6-130 groups per tissue (median 106). 534 of 598
  tissue x stain combinations are present; only 19 tissues occur in all 13 stains. Sparse:
  `40_collagenous_fiber` (2 stains, 6 groups), `01_penis` (5 stains), `09_follicle` (8 stains).
  Some names contain typos (`13_major_salvary_gland`, `25_ceberal_cortex`, `20_broncus`,
  `34_colon_adenocarcionma`): join on the number prefix.

Release [`PLISM_sm`](https://github.com/ruitaodong/plism-icc/releases/tag/PLISM_sm):
`PLISM_sm-GT450.tar.bz2` (1.2 GB) and `PLISM_sm-P.tar.bz2` (1.15 GB), 8,908 PNGs (4,454 per
scanner), all decoding and matching the CSV.

## 2. Registration (GT450 vs P)

40 random groups per stain (519 pairs), lightness correlation, best integer shift, and an
affine fit to 4 x 4 sub-block shifts.

| | PLISM-sm | Owkin 224 px tiles (release `300`) |
|---|---|---|
| Correlation as stored | **0.93** | 0.03 |
| After best shift | 0.94 | about 0.75 |
| Pairs with r > 0.6 | **97%** (92-100% per stain) | about 77% (42-98% per stain) |
| Shift needed | **median 1 px**, p90 3.6 px | median about 25 px, p90 about 60 px |
| Rotation / scale within a patch | +0.01 deg / < 0.2% | similar |
| Local residual | about 2 px | < 1 px after shifting |

Per stain (correlation before / after / share r > 0.6): GIV 0.92 / 0.93 / 100%, GIVH 0.92 /
0.94 / 97.5%, GM 0.91 / 0.92 / 95%, GMH 0.87 / 0.88 / 92.3%, GV 0.92 / 0.95 / 97.5%, GVH 0.92 /
0.95 / 95%, HR 0.96 / 0.96 / 97.5%, HRH 0.93 / 0.93 / 95%, KR 0.89 / 0.93 / 97.5%, KRH 0.94 /
0.95 / 97.5%, LM 0.92 / 0.93 / 97.5%, LMH 0.95 / 0.96 / 92.5%, MY 0.95 / 0.95 / 100%.

The pale stains that registered worst in the Owkin tiles (MY, LMH, LM) are as good as the
others here. Per-patch alignment is only a safety net.

## 3. Pixel scale

The PNGs are on the `S60` pixel grid, about **0.219 um/px**: a 512 px patch covers about 112 um,
i.e. about 415-443 GT450 pixels and 433-461 P pixels at native resolution. Measured by registering
PLISM tiles to the original GT450 and P slides (see `PLISM_wsi.md`); this replaces an earlier
estimate of 0.3-0.4 um/px from nucleus sizes. The PNG pixel values are the stored slide pixels
(GT450 raw RGB without its embedded profile, P stored sRGB), resampled once onto that grid.

## 4. GT450 -> P fit with a 15% hold-out

**Setup.**
- Pairs: for every group, an integer shift by phase correlation (|shift| <= 24 px; 310 groups
  needed more than 3 px), then 8 x 8 block means in linear light (64 x 64 blocks per group).
  Median correlation after alignment 0.944; the 188 groups below 0.6 were kept in training so
  that the robust weighting can deal with them.
- Split: 15% of groups held out at random, stratified by stain (seed 0): 3,785 training and 669
  held-out groups (all 13 stains, 45 tissues).
- Fit: joint (GT450, P) histogram with 7 bits per channel (4.63 M bins), affine + 33^3 lattice,
  Tukey reweighting for 4 rounds.
- Evaluation on held-out groups: region dE00 (64 px regions, linear-light means), contrast
  (median ratio of local L* spread), darkest / brightest 1% L* difference.

**Robust strength** (Tukey c, smaller = stronger; lambda 3e-4 unless noted):

| c | lambda | Rejected | Region dE00 | p95 | Contrast | Dark 1% | Bright 1% |
|---|---|---|---|---|---|---|---|
| (no mapping) | | | 8.80 | 10.32 | 1.05 | +7.6 | +8.2 |
| off | 3e-4 | 0% | 0.877 | 1.78 | 0.974 | +0.99 | -0.73 |
| 4.685 | 3e-4 | 2.7% | 0.779 | 1.53 | 1.010 | +0.30 | -0.37 |
| 3 | 3e-4 | 6.2% | 0.762 | 1.50 | 1.016 | +0.15 | -0.31 |
| 2.5 | 3e-4 | 8.5% | 0.757 | 1.49 | 1.018 | +0.11 | -0.30 |
| 2 | 3e-4 | 12.3% | 0.751 | 1.48 | 1.020 | +0.14 | -0.28 |
| 1.5 | 3e-4 | 19.2% | 0.750 | 1.49 | 1.019 | +0.32 | -0.26 |
| 1 | 3e-4 | 33.3% | 0.758 | 1.54 | 1.015 | +0.77 | -0.24 |
| **2** | **1e-4** | 12.5% | **0.748** | **1.48** | 1.017 | **-0.03** | -0.39 |
| 2 | 1e-3 | 12.1% | 0.760 | 1.51 | 1.024 | +0.50 | -0.10 |

- **Robust weighting is a reliable gain.** c = 2 (lambda 1e-4) beats the non-robust fit in 83%
  of held-out groups, by 0.129 dE00 on average (95% bootstrap interval 0.116-0.140), and fixes
  the tails (darkest 1%: +0.99 -> -0.03 L*).
- **The optimum is broad.** c = 1.5 and 2 cannot be told apart (difference 0.001, interval
  -0.003 to +0.005); c = 2.5 is slightly worse (0.008, interval 0.006-0.011); c = 1 rejects a
  third of the data and gets worse again.
- **Smoothness matters little.** lambda 1e-4 is marginally better than 3e-4; 1e-3 is worse.

**Per stain** (held-out region dE00: no mapping / non-robust / robust c = 2):

| Stain | n | Raw | Non-robust | Robust |
|---|---|---|---|---|
| GV | 66 | 8.97 | 1.24 | 1.07 |
| LM | 48 | 7.87 | 1.18 | 0.95 |
| HR | 40 | 8.78 | 1.11 | 0.91 |
| GIV | 45 | 8.75 | 1.04 | 0.85 |
| HRH | 38 | 8.92 | 0.84 | 0.78 |
| GMH | 57 | 9.42 | 0.87 | 0.77 |
| GM | 39 | 7.54 | 0.93 | 0.77 |
| KR | 59 | 7.99 | 0.85 | 0.75 |
| GVH | 63 | 9.52 | 0.76 | 0.70 |
| GIVH | 45 | 9.51 | 0.66 | 0.62 |
| LMH | 65 | 9.42 | 0.63 | 0.56 |
| KRH | 45 | 9.08 | 0.68 | 0.55 |
| MY | 59 | 8.14 | 0.70 | 0.50 |

Darker, low-contrast stains (GV, LM, HR, GIV) remain hardest and gain the most from robust
weighting.

**Per tissue.** Most tissues reach 0.6-0.9 with robust weighting. Outliers: `20_broncus` 1.38
(1.43 without robust weighting, so hardly improved) and `21_cartilage` 1.00 (1.24): cartilage
matrix has colors unlike other tissue, and bronchus mixes cartilage, mucus and epithelium.
Best: `25_ceberal_cortex` 0.49, `30_myxofibroma` 0.56, `43_stomach` 0.59, `31_NEC` 0.60.

## 5. Final profile

Refit on all 4,454 groups (5.20 M joint bins) with c = 2, lambda 1e-4; 12.4% of pairs
down-weighted to zero. Checked as a real device link on full-resolution held-out PNG pairs:
region dE00 9.87 -> 0.64 (six groups).

Files in `profiles/plism_sm_lut/`:

| File | Content |
|---|---|
| `gt450_to_p_plism_sm_devicelink.icc` | ICC v2.4 device link, GT450 RGB -> P RGB |
| `gt450_to_p_plism_sm_input.icc` | ICC v2.4 input profile (P treated as sRGB) |
| `gt450_to_p_plism_sm.cube`, `gt450_to_p_plism_sm_lut.npy` | same LUT as `.cube` and (33, 33, 33, 3) array |
| `tuning_holdout15.csv` | every tested setting with all metrics |
| `holdout15_per_group.csv` | each held-out group: stain, tissue, coordinate, dE00 raw / non-robust / robust |
| `analysis/plism_sm/sm_prep_chunk.py`, `analysis/plism_sm/sm_fit.py` | pair preparation (chunked) and fit / evaluation |

## 6. Profile-based alternative: GT450 vendor profile + 3x3 in XYZ

The same data, fitted after converting each side with fixed, known steps:

    GT450 RGB --[Segue_v2, the GT450's embedded ICC]--> XYZ --[M, 3x3]--> XYZ <--[sRGB decode]-- P RGB

| Region dE00 (64 px regions) | Result |
|---|---|
| No mapping | 8.78 |
| Vendor profile only (no M) | 5.58 |
| Segue_v2 -> XYZ -> 3x3, random 15% hold-out | **0.744** (median 0.63) |
| Leave-one-stain-out, mean / worst (`HR`) | 0.79 / 1.60 |
| Segue_v2 -> XYZ -> 3x4 (offset) | 0.704 |
| Raw GT450 (gamma 2.2 decoded) -> 3x3 / 3x4 | 0.924 / 0.708 |

- The vendor profile brings GT450 most of the way in saturation (C* 10.8 -> 22.7, P 21.4) and leaves
  a lightness offset (P renders glass at about L* 90, slightly lavender) and a hue rotation of about
  11 deg (GT450 bluer), which M removes.
- What remains after the profile is linear in light (an extra fitted exponent comes out at 1.013),
  so 9 numbers suffice, as accurate as the 33^3 LUT above, with sensible behaviour for white, gray
  and saturated colors.
- An `HRH`-only fit gives nearly the same matrix (0.49 on held-out `HRH`): the stain does not change
  the scanner relation.
- Files: `profiles/plism_sm_xyz/`, fit script `analysis/plism_sm/sm_xyz_fit.py`, apply script
  `apply_xyz_bridge.py`, examples `figures/plism_sm_xyz_holdout_5_examples.png`.

## 7. Caveats and next steps

- The profile maps PLISM's own GT450 and Philips UFS units; scanner relationships are
  unit-specific, so it is a test bed, not a profile for other scanners.
- The random hold-out shares tissue cores with training (neighbouring fields of the same core),
  so it measures unseen fields, not unseen tissue. **Next: leave-one-tissue-out** with the same
  settings, preferably on the 19 tissues present in all 13 stains.
- PLISM-sm patches were resampled once onto the `S60` grid (mild interpolation blur); a fit on these
  may differ slightly from one on native WSI pixels. Native-resolution GT450 is crisper than P at the
  nuclear scale (equivalent Gaussian sigma about 0.2 um, see `PLISM_wsi.md`); the resampling hides it.
