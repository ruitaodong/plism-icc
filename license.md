# Licenses

This file covers the licenses of the data, derived data and code in
[plism-icc](README.md), and how this repository complies with them.

## Summary

- **Data and derived data** (release files, block sums, alignment scores, LUTs, ICC profiles,
  figures): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with attribution to
  Ochi et al. (2024) and Owkin as described below.
- **Code**: **TODO: choose a code license (e.g. MIT or Apache-2.0) and add a `LICENSE` file.**
  Code licenses do not apply to the data.

## Data sources and their licenses

| Source | Where | License |
|--------|-------|---------|
| PLISM (Ochi, Komura, Onoyama, Ishikawa; University of Tokyo) | Figshare+, [doi:10.25452/figshare.plus.c.6773925](https://doi.org/10.25452/figshare.plus.c.6773925) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Registered tiles (Owkin) | Hugging Face, [owkin/plism-dataset](https://huggingface.co/datasets/owkin/plism-dataset) | CC BY 4.0 |
| Trimmed subset (this repository) | [Release v1](https://github.com/ruitaodong/plism-icc/releases/tag/v1) | CC BY 4.0 (inherited) |

The PLISM figures referenced in the README are linked from the PLISM project page
(https://p024eb.github.io/), not copied into this repository.

## How we respect the license

PLISM and Owkin's registered tiles are both released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
CC BY 4.0 allows copying, redistribution and adaptation, including commercially, under these
conditions. This is how this repository meets each one:

| CC BY 4.0 condition | What we do |
|---------------------|------------|
| **Give appropriate credit** | The original authors, the dataset DOI and the paper are credited in the README, in this file, in the release notes (see the attribution block below), and in the copyright tag of every ICC profile we write (`Derived from PLISM (CC BY 4.0), Ochi et al. 2024`). Owkin is credited for the registered tiles. |
| **Provide a link to the license** | Linked in the README, in this file and in the release notes. |
| **Indicate if changes were made** | Listed under "Changes made" below. |
| **No additional restrictions** | The redistributed data stays under CC BY 4.0. We add no terms, access controls or technical measures that would restrict what the license allows. |
| **No implied endorsement** | This project is independent. It is not affiliated with or endorsed by the PLISM authors, the University of Tokyo, Owkin, or any scanner vendor. |

### Changes made

To the redistributed tiles (release v1):

- Subset only: one stain (`HRH`), two scanners (`GT450`, `P`), 3,000 of the 16,278 tile
  positions, the same patch ids in both files.
- Repackaged into one HDF5 file per scanner, with gzip compression.
- Pixel values are not modified. Tiles are copied as published by Owkin.
- Selection of the 3,000 tiles: **TODO: describe the selection rule (e.g. random with seed, or
  first N ids)**.

Derived data produced by the tools in this repository (block sums, alignment scores, fitted
LUTs, ICC profiles, `.cube` files, comparison figures) are adaptations of PLISM. We release them
under CC BY 4.0 as well, with the same attribution.

### Attribution block for releases

Every release that contains PLISM data or derived data carries this text in its notes:

```text
Contains data from PLISM (Pathology Images of Scanners and Mobilephones),
Ochi M., Komura D., Onoyama T., Ishikawa S., Figshare+ (2023),
https://doi.org/10.25452/figshare.plus.c.6773925, and Ochi et al., Sci Data 11, 330 (2024).
Registered tiles from owkin/plism-dataset (Owkin), https://huggingface.co/datasets/owkin/plism-dataset.
Licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
Changes: subset of stains/scanners/tiles, repackaged into HDF5; see license.md "Changes made".
Not endorsed by the PLISM authors or Owkin.
```

### Third-party components

- Elastix (Apache 2.0) was used by Owkin to register the tiles. It is not used or redistributed
  here.
- Scanner and vendor names are trademarks of their owners and are used only to identify devices.

### If you reuse the data

Keep the attribution, link to CC BY 4.0, and cite the PLISM paper and dataset (see
[Citation for attribution](#citation-for-attribution)). If you use the registered tiles, also cite Owkin's work.

## Citation for attribution

Cite the PLISM paper and dataset (BibTeX in the [README](README.md#citation)):

- Ochi, M., Komura, D., Onoyama, T. et al. Registered multi-device/staining histology image
  dataset for domain-agnostic machine learning models. *Scientific Data* 11, 330 (2024).
  https://doi.org/10.1038/s41597-024-03122-5
- Ochi, M., Komura, D., Onoyama, T., Ishikawa, S. Pathology Images of Scanners and Mobilephones
  (PLISM) Dataset. Figshare+ (2023). https://doi.org/10.25452/figshare.plus.c.6773925

For the registered tiles, also cite Owkin's work as listed on the
[owkin/plism-dataset](https://huggingface.co/datasets/owkin/plism-dataset) page (arXiv:2501.16239).
