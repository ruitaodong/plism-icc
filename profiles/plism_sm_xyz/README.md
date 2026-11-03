# PLISM-sm GT450 -> P, one 3x3 in CIE XYZ after the GT450 vendor profile

    XYZ_GT450 = Segue_v2(GT450 RGB)                 # the GT450's embedded ICC profile (lut16, 103^3)
    XYZ_P     = M @ XYZ_GT450                       # M: 3x3, plism_gt450_to_p_xyz3x3.txt
    P RGB     = sRGB_encode( XYZ_P -> linear sRGB ) # P pixels are stored as sRGB

Both scanner sides are converted with fixed, known steps (vendor profile on the GT450 side, the sRGB
definition on the P side); only the 9 numbers of M are fitted. M was fitted on the 4,148 PLISM-sm
groups whose aligned lightness correlation is at least 0.6, minimising CIEDE2000 with a soft-L1 loss
(`analysis/plism_sm/sm_xyz_fit.py`).

| Check (region dE00, 64 px regions) | Result |
|---|---|
| No mapping | 8.78 |
| Vendor profile only (Segue_v2 -> XYZ -> sRGB, no M) | 5.58 |
| Random 15% hold-out (by stain) | **0.744** (median 0.63, p95 1.58) |
| Leave-one-stain-out, mean / worst (`HR`) | 0.79 / 1.60 |
| Same with an offset (3x4) | 0.704 |
| `HRH` only, fit and test on `HRH` | 0.49 |

- **The difference that remains after the vendor profile is linear in light.** Allowing an extra
  exponent, XYZ_P = M @ XYZ_GT450^g, gives g = 1.013; an offset adds little (0.744 -> 0.704). On raw
  GT450 RGB (decoded as gamma 2.2) a 3x3 alone reaches only 0.924 and needs the offset (0.708), as in
  `../plism_sm_affine/`. With the vendor profile in front, 9 numbers suffice.
- **M behaves sensibly outside H&E**, because the 13 stains spread the colors: GT450 white maps to
  Lab (91.7, 4.3, -4.8), close to how P renders clear glass (about L* 90, slightly lavender); gray
  stays near neutral; sRGB primaries move by 7-17 dE00 but stay plausible. An H&E-only fit from a
  few slides does not (it constrains only one color direction).
- Accuracy is the same as the data-driven 33^3 LUT (`../plism_sm_lut/`, 0.748 on its own 15%
  hold-out) and the linear-light affine (`../plism_sm_affine/`, 0.72).

## Files

| File | Content |
|---|---|
| `plism_gt450_to_p_xyz3x3.txt` | M (column form, XYZ D50, white Y = 1) |
| `plism_gt450_to_p_xyz3x3_input.icc` | ICC v4.3 input profile, GT450 RGB -> XYZ (Segue_v2 then M, 65^3). Converting to sRGB gives the P look. |
| `plism_gt450_to_p_xyz3x3_devicelink.icc` | ICC v4.3 device link, GT450 RGB -> P sRGB (65^3) |

Both expect **gamma-2.2-encoded GT450 pixels**, as stored in PLISM's GT450 slides (SVS header
`Gamma = 2.2`; OpenSlide returns these stored pixels). If your GT450 pixels are linear rather than
gamma-2.2 encoded, encode them first (`apply_xyz_bridge.py --gamma 2.2`).

```bash
python apply_xyz_bridge.py tile.png out.png                    # Pillow / LittleCMS, input profile -> sRGB
python apply_xyz_bridge.py tile.png out.png --gamma 2.2        # linear GT450 pixels
tificc -c0 -l plism_gt450_to_p_xyz3x3_devicelink.icc in.tif out.tif
```

Switch off LittleCMS's precalculation (`-c0`, or `flags=ImageCms.Flags.NOOPTIMIZE` in Pillow): the
default 8-bit optimisation is close on average but up to about 26 of 255 levels off in some dark
colors with these tables.

**Scope.** M relates PLISM's own GT450 and Philips UFS units. GT450 units embed different Segue
profile versions; this bridge is defined with Segue_v2, the profile of PLISM's GT450, and should be
validated on paired slides before it is used for other units or for colors far from H&E (IHC/DAB,
special stains, ink). The input profile and device link contain a table derived from Leica's
Segue_v2 profile; check redistribution terms before publishing them outside this repository.
