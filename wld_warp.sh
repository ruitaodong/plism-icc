#!/usr/bin/env bash
# usage: bash wld_warp.sh slide.svs slide_0p25.tif [target_um_per_px]
set -euo pipefail
IN=$1; OUT=$2; T=${3:-0.25}

# 1. MPP of the full-resolution image from the Aperio description (gdalinfo opens the first directory = level 0)
#    (pipefail off here: awk stops at the first match, which ends gdalinfo early; the embedded ICC makes its output long)
MPP=$(set +o pipefail; gdalinfo "$IN" 2>/dev/null | awk 'match($0, /MPP = [0-9.]+/) {print substr($0, RSTART + 6, RLENGTH - 6); exit}')
[ -n "$MPP" ] || { echo "no 'MPP = ...' found in the gdalinfo metadata of $IN" >&2; exit 1; }
echo "source MPP $MPP um/px -> target $T um/px"

# 2. input world file (um units): pixel size, rotation x2, -pixel size, centre of top-left pixel
awk -v m="$MPP" 'BEGIN{printf "%.10f\n0\n0\n%.10f\n%.10f\n%.10f\n", m, -m, m/2, -m/2}' > "${IN%.*}.wld"

# 3. warp to the target pixel size; GDAL writes the output world file (.tfw) -> renamed to .wld
gdalwarp -q -overwrite --config GDAL_PAM_ENABLED NO -tr "$T" "$T" -r cubic -multi -wo NUM_THREADS=ALL_CPUS \
  -co PROFILE=BASELINE -co TFW=YES -co TILED=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 \
  -co COMPRESS=JPEG -co JPEG_QUALITY=95 -co PHOTOMETRIC=RGB -co BIGTIFF=IF_SAFER "$IN" "$OUT"
mv "${OUT%.*}.tfw" "${OUT%.*}.wld"

gdaladdo -q -r average --config COMPRESS_OVERVIEW JPEG --config JPEG_QUALITY_OVERVIEW 95 --config PHOTOMETRIC_OVERVIEW RGB "$OUT" 2 4 8 16 32
echo "wrote $OUT and ${OUT%.*}.wld"
