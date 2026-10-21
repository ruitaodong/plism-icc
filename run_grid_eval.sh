#!/usr/bin/env bash
# Random-N / worst-N hold-out grids for several source scanners against one reference scanner.
# Edit the variables; expects <src>_pyr/ pyramids and fit_pyr.py outputs <src>_to_<ref>_devicelink.icc / _split.csv.
set -u
REF=ref                              # reference scanner name (folder ${REF}_pyr)
SOURCES="scanner_a scanner_b"        # source scanner names (folders <name>_pyr)
FIT=outputs                          # folder with the fit_pyr.py outputs
mkdir -p grids
for s in $SOURCES; do
  python grid_eval.py --src ${s}_pyr --ref ${REF}_pyr --icc $FIT/${s}_to_${REF}_devicelink.icc \
                      --split $FIT/${s}_to_${REF}_split.csv --out grids/${s}_vs_${REF} \
                      --label-src ${s} --label-ref ${REF} --level 0 --n 5 > grids/${s}.log 2>&1 &
done
wait; tail -n 2 grids/*.log
