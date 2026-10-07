# plism-icc

Scanner-to-scanner color calibration for digital pathology, built on the PLISM dataset.

This repository fits 3D lookup tables that map the colors of six whole-slide scanners to a
reference scanner (Philips UltraFast Scanner, `P`) and writes them as ICC profiles. It also
contains tools to extract training data from PLISM, score the alignment of every tile pair, and
apply the profiles to rasters.

> Research use only. The profiles are not validated for clinical or diagnostic use.

## Data origin

All images come from PLISM, created by Ochi, Komura, Onoyama and Ishikawa at the University of
Tokyo. The data reaches this repository through three steps:

| Step | What | Where | License |
|------|------|-------|---------|
| 1. Original dataset | PLISM: 46 tissue types, 13 H&E staining conditions, 7 WSI scanners + 6 smartphones | Figshare+, [doi:10.25452/figshare.plus.c.6773925](https://doi.org/10.25452/figshare.plus.c.6773925) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 2. Registered tiles | 91 slides (13 stains x 7 scanners), 16,278 224x224 tiles each, registered to the `GMH_S60` slide with Elastix | Hugging Face, [owkin/plism-dataset](https://huggingface.co/datasets/owkin/plism-dataset) (Owkin) | CC BY 4.0 |
| 3. Trimmed subset (this repo) | 3,000 tile pairs of stain `HRH`, scanners `GT450` and `P` | [Release v1](https://github.com/ruitaodong/plism-icc/releases/tag/v1): `HRH_GT450_3000.h5`, `HRH_P_3000.h5` | CC BY 4.0 (inherited) |

The release files keep the layout of step 2: one dataset per tile, named
`tile_<level>_<x>_<y>`, each a `(224, 224, 3)` uint8 RGB array, gzip-compressed. Both files
contain the same 3,000 patch ids.

Licensing, attribution and the list of changes are in [license.md](license.md).

### Scanners

| Abbrev. | Vendor | Device | Role here |
|---------|--------|--------|-----------|
| `S360` | Hamamatsu | NanoZoomer-S360 C13220-01 | source |
| `S210` | Hamamatsu | NanoZoomer-S210 C13239-01 | source |
| `SQ` | Hamamatsu | NanoZoomer-SQ C13140-D03 | source |
| `S60` | Hamamatsu | NanoZoomer-S60 C13210-01 | source |
| `AT2` | Leica | Aperio AT2 | source |
| `GT450` | Leica | Aperio GT450 | source |
| `P` | Philips | UltraFast Scanner | target |

The smartphone subset of PLISM (PLISM-sm) is not used.

### Staining conditions

| Abbrev. | Hematoxylin | Hematoxylin (min) | Eosin (min) | Dehydrations |
|---------|-------------|-------------------|-------------|--------------|
| `GIVH` | Gill IV | 0.5 | 15 | 0 |
| `GIV` | Gill IV | overnight* | 15 | 3 |
| `GMH` | GM | 2 | 15 | 0 |
| `GM` | New Type G | 5 | 15 | 3 |
| `GVH` | Gill V | 5 | 15 | 0 |
| `GV` | Gill V | 60 | 15 | 3 |
| `MY` | Mayer | 3 | 3 | 0 |
| `HRH` | Harris | 2 | 15 | 0 |
| `HR` | Harris | overnight* | 15 | 0 |
| `KRH` | Carrazi | 5 | 15 | 0 |
| `KR` | Carrazi | 60 | 15 | 3 |
| `LMH` | Lillie-Mayer | 2 | 15 | 0 |
| `LM` | Lillie-Mayer | 2 | 15 | 4 |

\* About 24 hours. Tables follow the PLISM project page, https://p024eb.github.io/ (figures:
[devices](https://p024eb.github.io/images/graph/counterpart.png),
[staining](https://p024eb.github.io/images/graph/stain_condition.png)). The figures are linked,
not copied into this repository.

## Repository contents

| File | Purpose |
|------|---------|
| `plism_icc.py` | Main pipeline: read tile pairs, bin colors, fit affine + 33^3 lattice per scanner, write ICC profiles, `.cube` and `.npy` LUTs, evaluate on held-out tiles. Parallel (`--n-workers`); can refit from saved bins (`--from-bins`). |
| `apply_icc_rasterio.py` | Apply a device link or input profile to a raster with rasterio (`apply`), embed a profile (`tag`), or inspect one (`info`). Output matches LittleCMS. |
| `extract_b4.py` | Extract exact 4x4 block sums from one slide into HDF5, indexed by patch id. |
| `run_extract_b4.sh` | Runs `extract_b4.py` for all 91 slides in parallel; one line per slide, comment out to skip. |
| `align_score.py` | Per-patch alignment score of one scanner's slide against `P` of the same stain (masked NCC on L*, best shift, overlap, texture, tissue, keep flag) to CSV. |
| `run_align_score.sh` | Runs `align_score.py` for all 13 stains x 6 scanners in parallel. |

Requirements: `numpy scipy scikit-image h5py huggingface_hub pillow rasterio`.

## Typical workflow

```bash
# 1. score alignment of every tile pair (all stains x scanners vs P)
MAXJ=16 bash run_align_score.sh                 # -> plism_align/<STAIN>_<SCANNER>_vs_P.csv

# 2. extract 4x4 block sums for fitting
MAXJ=16 bash run_extract_b4.sh                  # -> plism_b4/<STAIN>_<SCANNER>_b4.h5

# 3. fit profiles
python plism_icc.py --local-dir plism-dataset --out icc_out

# 4. apply to an image
python apply_icc_rasterio.py apply slide_GT450.tif icc_out/GT450_to_P_devicelink.icc slide_as_P.tif
```

### Outputs per source scanner

| File | Description |
|------|-------------|
| `<SCANNER>_to_P_devicelink.icc` | ICC v2.4 device link, RGB(source) -> RGB(P), A2B0 lut16 with 33^3 grid and 256-entry curves |
| `<SCANNER>_as_P_input.icc` | ICC v2.4 input profile, RGB(source) -> Lab, built so that converting to sRGB reproduces P (P treated as sRGB) |
| `<SCANNER>_to_P.cube` | Same LUT in `.cube` format |
| `<SCANNER>_to_P_lut.npy` | Same LUT as a NumPy array, shape (33, 33, 33, 3) |
| `report.json` | CIEDE2000 before/after on held-out tiles |

## Findings so far

- **The registered tiles are not pixel-aligned between scanners.** For `HRH`, `P` is offset from
  `GT450` by about (-20, -21) px at the median, varying smoothly across the slide (18 to 52 px,
  10th to 90th percentile). There is no slide-wide rotation or scale (fitted scale error
  -0.017%, rotation +0.005 deg). Some tiles show local rotation (up to about 2 deg) or scale (up
  to about 6%).
- **Fitting on misaligned pairs flattens contrast.** Pairing pixels that show different tissue
  pulls dark colors toward the middle. On `GT450 -> P`, contrast of the mapped tiles was about
  0.6 of P's when fitted on unaligned pairs, versus about 0.9 after per-tile alignment and
  rejection (separate experiments, different evaluation sets).
- **Rejecting badly aligned tiles is cheap and helps.** Alignment costs about 36 ms per tile.
  Dropping tiles with alignment score `r < 0.6` (21% of `HRH` tiles) slightly improved every
  metric and removed no colors from the training set.
- **Block size matters.** 8x8 averaging removes the darkest colors (1st percentile L* 55 instead
  of 47 at full resolution). After alignment, 4x4 blocks keep most of the dark tail.

Numbers come from the `HRH` subset and earlier `GT450` experiments; full-dataset results will
follow.

## Citation

If you use this repository or its data, please cite the original PLISM work:

```bibtex
@article{ochi2024plism,
  title   = {Registered multi-device/staining histology image dataset for domain-agnostic machine learning models},
  author  = {Ochi, Mieko and Komura, Daisuke and Onoyama, Takumi and others},
  journal = {Scientific Data},
  volume  = {11},
  pages   = {330},
  year    = {2024},
  doi     = {10.1038/s41597-024-03122-5}
}

@misc{ochi2023plismdata,
  title     = {Pathology Images of Scanners and Mobilephones (PLISM) Dataset},
  author    = {Ochi, Mieko and Komura, Daisuke and Onoyama, Takumi and Ishikawa, Shumpei},
  publisher = {Figshare+},
  year      = {2023},
  doi       = {10.25452/figshare.plus.c.6773925}
}
```

For the registered tiles, also cite Owkin's work as listed on the
[owkin/plism-dataset](https://huggingface.co/datasets/owkin/plism-dataset) page (arXiv:2501.16239).

## License

Data and derived data: CC BY 4.0, with attribution to the PLISM authors and Owkin. Code: to be decided. See [license.md](license.md) for details, the changes made to the data, and the attribution text for releases.
