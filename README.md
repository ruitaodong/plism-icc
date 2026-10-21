# plism-icc

Scanner-to-scanner color calibration for digital pathology, developed on the PLISM dataset.

The tools fit a 3D lookup table (affine + 33^3 lattice) that maps the colors of a source scanner
to a target scanner from registered image pairs, and write it as ICC profiles (device link and
input profile) and as a `.cube` LUT. The repository covers the whole path: alignment scoring,
pyramid building, robust fitting with slide-level hold-out tuning, optional scoring through your
own models, and applying the profiles to rasters.

> Research use only. The profiles are not validated for clinical or diagnostic use.

## Data origin

All data published with this repository come from PLISM, created by Ochi, Komura, Onoyama and
Ishikawa (University of Tokyo):

| Step | What | Where | License |
|------|------|-------|---------|
| 1. Original dataset | PLISM: 46 tissue types, 13 H&E staining conditions, 7 WSI scanners + 6 smartphones | Figshare+, [doi:10.25452/figshare.plus.c.6773925](https://doi.org/10.25452/figshare.plus.c.6773925) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 2. Registered tiles | 91 slides (13 stains x 7 scanners), 16,278 tiles of 224x224 px each, registered to the `GMH_S60` slide with Elastix | Hugging Face, [owkin/plism-dataset](https://huggingface.co/datasets/owkin/plism-dataset) (Owkin) | CC BY 4.0 |
| 3a. Subset (this repo) | 3,000 tile pairs of stain `HRH`, scanners `GT450` and `P` | [Release v1](https://github.com/ruitaodong/plism-icc/releases/tag/v1) | CC BY 4.0 (inherited) |
| 3b. Subset (this repo) | 300 matched tiles for each of the 13 stains, scanners `GT450` and `P` (26 files, same tile ids in all) | [Release 300](https://github.com/ruitaodong/plism-icc/releases/tag/300) | CC BY 4.0 (inherited) |

Release files keep the layout of step 2: one dataset per tile, named `tile_<level>_<x>_<y>`,
each a `(224, 224, 3)` uint8 RGB array, gzip-compressed. Licensing, attribution and the list of
changes are in [license.md](license.md).

### Scanners

| Abbrev. | Vendor | Device |
|---------|--------|--------|
| `S360` | Hamamatsu | NanoZoomer-S360 C13220-01 |
| `S210` | Hamamatsu | NanoZoomer-S210 C13239-01 |
| `SQ` | Hamamatsu | NanoZoomer-SQ C13140-D03 |
| `S60` | Hamamatsu | NanoZoomer-S60 C13210-01 |
| `AT2` | Leica | Aperio AT2 |
| `GT450` | Leica | Aperio GT450 |
| `P` | Philips | UltraFast Scanner |

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

## How it works

1. **Pairs.** Source and target images of the same tissue, stored as HDF5 files with one dataset
   per patch and the same patch names on both sides. Pairs must be registered; small residual
   shifts are handled by the blur in step 3 or measured with `align_score.py`.
2. **Pyramids.** `h5_pyramid.py` builds a Gaussian pyramid per patch (filtering in linear
   light), so the fit can run at a reduced resolution with a known, documented blur.
3. **Fit.** `fit_pyr.py` takes both pyramid sets at one level, applies an optional extra Gaussian
   blur to both sides (so focus and small registration differences matter less), accumulates a
   joint (source, target) color histogram, and fits an affine + 33^3 lattice with Tukey
   reweighting against outliers (dust, folds, pen, residual misregistration).
4. **Tune.** Candidate settings (extra blur, robust strength, lattice smoothness) are scored on
   held-out slides, never held-out pixels or patches of a training slide. The best setting is
   refit on all data.
5. **Write.** ICC v2.4 device link (RGB -> RGB), ICC v2.4 input profile (RGB -> Lab, target
   treated as sRGB), `.cube` and `.npy`.
6. **Apply.** `apply_icc_patch.py` (one image), `icc_tiff.py` (TIFF files and folders, keeping
   pyramids, tiling and tags) or `apply_icc_rasterio.py` (rasterio); all evaluate the device
   link's table with tetrahedral interpolation, as LittleCMS does without transform
   precalculation. Alternatively, embed the input profile so color-managed viewers convert the
   image.
7. **Check.** `grid_eval.py` scores every hold-out patch and draws random and worst cases.

## Repository contents

| File | Purpose |
|------|---------|
| `h5_pyramid.py` | Patch HDF5 -> pyramidal TIFF per patch (L1 = 1/2, L2 = 1/4, ...; level 0 is not stored). Gaussian + 2x2 reduce in linear light; uint8/uint16/float16/float32; top-level pages (default, OpenSlide-readable) or SubIFDs; `--only N` writes a single level. Metadata (equivalent blur, sampling grid) in the ImageDescription. |
| `fit_pyr.py` | Main fitting tool: source/target pyramid lists at `--level`, extra `--sigma` grid, `--robust` and `--lam` grids, lattice penalty (`--penalty slope` default, or `curvature` with a ridge `--lam0`), slide-level hold-out (`--group-regex`, or your own `--train-list` / `--test-list`), cached histograms, tuning table, optional `--model-plugin`, profile output. |
| `model_eval.py` | Scores candidate profiles through your own segmentation/classification models (agreement of outputs on mapped source vs target tiles, with raw and ceiling references). Used by `fit_pyr.py --model-plugin`. |
| `model_plugin_template.py` | Template plugin (PyTorch, ONNX variant in comments): `TILE`, `load()`, `predict()` returning probabilities and optional features. |
| `fit_blur.py` | Same fit directly from two patch HDF5 files, without pyramids or tuning. Saves/merges joint histograms (`--save-bins`, `--from-bins`, `--shard i/n`) for parallel extraction and refits. |
| `fit_correct.py` | Registration-based variant: per-patch and per-block alignment, block averaging, fit, correction of every source patch, before/after metrics and a comparison figure. |
| `align_score.py`, `run_align_score.sh` | Alignment score per patch (L* correlation before shift, best shift, after shift, overlap, texture, tissue) and optionally per block (`--grid N`); the shell script runs every stain x scanner pair in parallel. |
| `apply_icc_patch.py` | Apply a device link to one image (PNG/TIFF/...); minimal, numpy + Pillow only. Also importable: `apply(icc, rgb_uint8)`. |
| `icc_tiff.py` | Apply a device link to TIFFs or whole folders: every level of a pyramid (top-level pages or SubIFDs), tiling, compression, predictor and description kept; page-by-page for files such as SVS (label/macro left unchanged); 8/16-bit and float; `--compression deflate|lzw|jpeg|none`, `--quality`. Loads one image at a time into memory, so it is meant for patches and pyramids, not full-resolution whole slides. |
| `apply_icc_rasterio.py` | Apply a device link or input profile to a raster (`apply`), embed an input profile (`tag`), or inspect one (`info`). Identical to LittleCMS output. |
| `grid_eval.py`, `run_grid_eval.sh` | Per-patch hold-out evaluation of a fitted device link against the reference (region/pixel dE00, contrast, dark tail) to CSV, and a grid of N random and N worst patches with an error map; the shell script runs several source scanners in parallel. |
| `relayout_tiff.py` | Lossless conversion of pyramidal TIFFs between top-level pages and SubIFDs (libtiff `tiffcp` drops SubIFDs). |
| `plism_icc.py` | Core library (lattice fit, ICC/`.cube` writers, color utilities) and the original PLISM pipeline (`--local-dir` or streaming from Hugging Face). |
| `extract_b4.py`, `run_extract_b4.sh`, `plism_extract_bins8.py`, `run_hrh.sh` | Earlier extraction helpers for PLISM block sums; kept for reproducibility. |

Requirements: `numpy scipy scikit-image h5py tifffile pillow` (+ `imagecodecs` for float TIFF
predictors, `rasterio` for `apply_icc_rasterio.py`, `huggingface_hub` for streaming PLISM).

## Typical workflow

```bash
# 1. pyramids for both scanners (one TIFF per patch)
python h5_pyramid.py SRC.h5 src_pyr --levels 3
python h5_pyramid.py TGT.h5 tgt_pyr --levels 3

# 2. fit and tune with a slide-level hold-out (your own lists, or --holdout 0.15 --group-regex ...)
python fit_pyr.py --src src_pyr --tgt tgt_pyr --level 1 \
                  --sigma 0 0.4 0.75 --robust 0 2.5 --lam 1e-4 3e-4 \
                  --train-list train.txt --test-list test.txt \
                  --bins-dir bins --out src_to_tgt --name SRC

# 3. check the hold-out patches: per-patch metrics + grid of 5 random and 5 worst
python grid_eval.py --src src_pyr --ref tgt_pyr --icc src_to_tgt_devicelink.icc \
                    --split src_to_tgt_split.csv --out src_vs_tgt --n 5

# 4. apply (images read the same way as the training data)
python apply_icc_patch.py src_to_tgt_devicelink.icc patch.png patch_as_tgt.png
python icc_tiff.py src_to_tgt_devicelink.icc src_pyr/ src_as_tgt_pyr/
python apply_icc_rasterio.py apply input.tif src_to_tgt_devicelink.icc output.tif
```

Outputs of `fit_pyr.py`:

| File | Description |
|------|-------------|
| `<out>_devicelink.icc` | ICC v2.4 device link, RGB(source) -> RGB(target), A2B0 lut16 with 33^3 grid and 256-entry curves |
| `<out>_input.icc` | ICC v2.4 input profile, RGB(source) -> Lab; converting to sRGB reproduces the target |
| `<out>.cube`, `<out>_lut.npy` | Same LUT as `.cube` and as a (33, 33, 33, 3) NumPy array |
| `<out>_tuning.csv` | Hold-out metrics for every candidate setting |
| `<out>_split.csv` | Train / hold-out assignment of every patch |
| `<out>_model_reference.csv` | Raw and ceiling references (with `--model-plugin`) |

### Evaluation metrics

- `region_dE00`: CIEDE2000 between region means (linear light) of mapped source and target;
  insensitive to focus and small misregistration. Default selection metric.
- `pixel_dE00`: per-pixel error on a sample; includes optics, focus and registration differences
  that no color mapping can remove.
- `contrast`: median ratio of local L* spread, mapped / target (1 = same local contrast).
- `dark1`, `bright99`: L* difference of the darkest 1% and brightest 1% of pixels.
- With a model plugin: probability-map agreement, boundary-tolerant Dice, class-area error,
  classification agreement and kappa, feature cosine similarity and retrieval, each also as the
  fraction of the gap closed between "raw" and an estimated "ceiling".

## Findings on PLISM

**Registration.** The registered tiles are not pixel-aligned between scanners. For `HRH`, `P` is
offset from `GT450` by a median of about (-20, -21) px (18 to 52 px, 10th-90th percentile),
varying smoothly across the slide; there is no slide-wide rotation or scale. Fitting on
misaligned pairs flattens contrast (dark colors pulled toward the middle); per-tile alignment
and rejecting tiles with an aligned L* correlation below about 0.6 fix this at negligible cost
(about 36 ms per tile, no colors lost). Robust reweighting does not compensate for dense
misalignment, but helps once pairs are aligned.

**Scanner shift across stains.** With 300 matched tiles per stain, `P` is darker than `GT450` by
7.4 to 8.8 L* and more chromatic by +6.9 to +12.7 C* for every one of the 13 stains (tile-mean
dE00 7.8 to 9.8); the chroma boost is larger for conditions without dehydration. The stains
differ far more from each other (median GT450 L* from 41 for `HR` to 91 for `MY`) than the
scanners do.

**Generalization to unseen stains (leave-one-stain-out, region dE00 on the held-out stain):**

| Model | Mean of 13 stains | Worst stain (`HR`) |
|-------|-------------------|--------------------|
| Affine (3x4) | 1.34 | 4.27 |
| Root-polynomial, 2nd order | 1.34 | 4.72 |
| **33^3 LUT, lambda 3e-4** | **1.23** | **3.54** |
| 33^3 LUT, lambda 3e-3 / 3e-2 | 1.30 / 1.32 | 4.01 / 4.24 |
| 33^3 LUT, lambda 3e-4, stain included in training | 1.02 | 1.70 |

An unseen stain costs little when its colors lie within the training range, and a lot when they
do not (`HR`, the very dark overnight Harris condition). Simpler or smoother models do not
generalize better. Robustness comes from the color coverage of the training data. If the stain
is known at application time, stain-specific profiles reduce the error further (held-out tiles:
0.87 stain-specific vs 1.04 pooled vs 1.26 without that stain).

**Affine vs full LUT (stains seen in training).** Fit on 80% of the 300 matched tiles of all 13
stains, evaluated on the other 20% (region dE00): no mapping 8.95, affine 1.17, 33^3 LUT 1.04.
The affine part does almost all of the work; the LUT's gain is concentrated in dark and dense
staining (`HR` 2.58 -> 1.77, `LM` 1.69 -> 1.32, `GIV` 1.29 -> 0.94), while for `GIVH` and `GMH`
the affine fit is marginally better (one pooled LUT bends slightly toward the dark stains).

**Penalising non-linearity.** A curvature penalty (second differences of the lattice, zero for
any affine map, plus a small ridge toward the affine base) makes the profile more affine but does
not generalise better to unseen stains: leave-one-stain-out region dE00 1.25 / 1.30 / 1.34 for
curvature weights 1e-4 / 1e-3 / 1e-2, against 1.23 for the default slope penalty. The non-linear
part is signal (tone-curve bends at the dark and bright ends), not noise. The curvature penalty
needs the ridge: without it, linear trends far from the data are undetermined.

**Training resolution.** Share of held-out native-resolution PLISM pixels whose colors never occur
in training data downsampled by:

| Scanner | 1x | 2x | 4x | 8x |
|---------|----|----|----|----|
| GT450 | 0.00% | 0.05% | 2.8% | 16.6% |
| P | 0.01% | 0.07% | 1.6% | 10.2% |

Fit no more than 2x below the resolution at which the profile will be applied; beyond that, the
darkest and most saturated pixel colors fall outside the data-supported part of the LUT.

## Practical guidance

- **A profile is specific to the scanner units and the image reading path.** The relationship
  between two scanner models can differ between units, sites and processing pipelines, so do not
  apply a profile fitted on PLISM's scanners to other units of the same models. Read images the
  same way for fitting and for application (same reader/exporter, no extra color management
  step on one side); a different export path can shift colors by several dE00.
- **Focus differences are random and bias pixel-level fits toward lower contrast.** Apply the
  same Gaussian blur to both sides before pairing (in linear light, larger than the expected
  defocus and registration error), or average over blocks of a few micrometres. Blurring only
  the sharper image, to match the other, removes the target's extreme colors and makes the fit
  worse.
- **Use region-level metrics** to judge the color mapping and keep pixel-level metrics for the
  optics; validate on held-out slides, and for robustness on held-out groups (organ, staining
  batch), judging by the worst group.
- **Cover the color range.** Include deliberately dark and pale slides and several organs and
  staining batches, scanned on both scanners. At application time, report the share of pixels
  outside the training color coverage and flag slides above a few percent.
- **Work on raw pixels on both sides.** Slide readers such as OpenSlide return the stored pixels
  and do not apply a slide's embedded ICC profile (OpenSlide 4 only exposes it); vendor SDKs may
  apply theirs only on request. Vendor profiles describe different kinds of data (display-encoded
  RGB, or linear sensor data at a calibration exposure) and, in our tests, did not bring scanners
  closer together. Fit and apply on raw pixels read the same way, and do not carry the source
  slide's embedded profile into converted files, or color-managed viewers will convert twice.
- **Apply device links exactly.** Use tetrahedral (or trilinear) interpolation of the table:
  coarse precalculated transforms in color-management engines can shift colors by several
  levels. When reading the `.cube` file yourself, note that red varies fastest: a plain
  `reshape(33, 33, 33, 3)` gives `[b, g, r]` and swaps red and blue; `<out>_lut.npy` is already
  `[r, g, b]`.
- **ICC is static.** Remaining sharpness and noise differences between scanners need a separate
  spatial step or training-time augmentation.

## Published results (PLISM only)

| Folder | Content |
|---|---|
| `profiles/plism_sm_lut/` | PLISM-sm GT450 -> P 33^3 LUT profile (robust c = 2, lambda 1e-4), tuning and 15% hold-out tables |
| `profiles/plism_sm_affine/` | the same mapping as a linear-light affine (12 numbers) with exact ICC files |
| `analysis/plism_sm/` | pair preparation and fit / evaluation scripts for PLISM-sm |
| `lists/plism_sm/` | PLISM-sm GT450/P file lists and manifests |
| `figures/` | PLISM figures (registration, hold-out examples, affine vs LUT) |

See `PLISM_sm.md` for the PLISM-sm findings.

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

Data and derived data: CC BY 4.0, with attribution to the PLISM authors and Owkin. Code: to be
decided. See [license.md](license.md) for details, the changes made to the data, and the
attribution text for releases.
