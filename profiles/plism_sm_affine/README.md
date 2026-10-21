# PLISM-sm GT450 -> P, affine in linear light

    out = clip( M @ [ in ** 2.2 , 1 ] , 0, 1 ) ** (1 / 2.2)

`plism_gt450_to_p_affine.txt` holds M (3 x 4; rows R', G', B' from R, G, B, 1), fitted on region means
(64 px) of all 4,454 PLISM-sm GT450/P groups. Held-out check (fit on 85%, test on 15%): region dE00 0.72
(33^3 LUT: 0.76). Written with `affine_icc.py`.

- `plism_gt450_to_p_affine_devicelink.icc`: RGB -> RGB device link, `tificc -c0 -l ... in.tif out.tif`
- `plism_gt450_to_p_affine_input.icc` + `plism_gt450_to_p_affine_output.icc`: the same mapping as an
  input/output pair for tools without device-link support (e.g. `vips icc_transform in.tif out.tif[strip]
  plism_gt450_to_p_affine_output.icc --input-profile plism_gt450_to_p_affine_input.icc --intent relative`).
  A mathematical construction: use the two only together, never embed them in images.

Accuracy against the formula: `tificc -c0` max 1 level; LittleCMS default precalculation (tificc without
-c0, Pillow default, libvips) up to about 20 levels, mean 0.1-0.4. The profile maps PLISM's own GT450 and
Philips units; scanner relationships are unit-specific.
