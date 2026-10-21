#!/usr/bin/env bash
# SVS -> resample to TARGET um/px (lossless) -> device-link colour conversion (tificc, exact) -> BigTIFF with overviews
# usage: bash svs_warp_icc.sh slide.svs devicelink.icc out.tif [target_um_per_px]
set -euo pipefail
IN=$1; ICC=$2; OUT=$3; T=${4:-0.25}
TMP=$(mktemp -d "${OUT%/*}/.svs_warp_icc.XXXXXX" 2>/dev/null || mktemp -d); trap 'rm -rf "$TMP"' EXIT
Z=(-co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 -co COMPRESS=ZSTD -co ZSTD_LEVEL=1 -co PREDICTOR=2 -co NUM_THREADS=ALL_CPUS)

# 1. source pixel size (Aperio description) -> world file for the full-resolution image
MPP=$(set +o pipefail; gdalinfo "$IN" 2>/dev/null | awk 'match($0, /MPP = [0-9.]+/) {print substr($0, RSTART + 6, RLENGTH - 6); exit}')
[ -n "$MPP" ] || { echo "no MPP in $IN" >&2; exit 1; }
printf '%s\n0\n0\n-%s\n0\n0\n' "$MPP" "$MPP" > "${IN%.*}.wld"
echo "MPP $MPP -> $T um/px"

# 2. resample, lossless (BigTIFF allowed: this is the large intermediate)
gdalwarp -q -overwrite --config GDAL_PAM_ENABLED NO -ovr NONE -tr "$T" "$T" -r cubic -multi -wo NUM_THREADS=ALL_CPUS \
  -co PROFILE=BASELINE -co BIGTIFF=IF_SAFER "${Z[@]}" "$IN" "$TMP/warp.tif"
PXCM=$(awk -v t="$T" 'BEGIN{printf "%.6f", 1e4/t}')
tiffset -s 282 "$PXCM" "$TMP/warp.tif"; tiffset -s 283 "$PXCM" "$TMP/warp.tif"; tiffset -s 296 3 "$TMP/warp.tif"
tiffset -s 270 "resampled from MPP $MPP to $T um/px; colour: $(basename "$ICC")" "$TMP/warp.tif"

# 3. colour conversion, exact (-c0); tificc keeps tiling, ZSTD + predictor, tags; writes classic TIFF (< 4 GB)
tificc -c0 -l "$ICC" "$TMP/warp.tif" "$TMP/icc.tif" > /dev/null 2>&1 || {
  echo "tificc failed (output > 4 GB classic TIFF limit?) - size of the lossless input: $(du -h "$TMP/warp.tif" | cut -f1)" >&2; exit 1; }

# 4. repack as BigTIFF (fast, lossless) and add overviews
gdal_translate -q --config GDAL_PAM_ENABLED NO -co BIGTIFF=YES "${Z[@]}" "$TMP/icc.tif" "$OUT"
gdaladdo -q -r average --config GDAL_NUM_THREADS ALL_CPUS --config ZSTD_LEVEL_OVERVIEW 1 "$OUT" 2 4 8 16 32
printf '%s\n0\n0\n-%s\n0\n0\n' "$T" "$T" > "${OUT%.*}.wld"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) and ${OUT%.*}.wld"
