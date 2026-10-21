#!/usr/bin/env bash
# Alignment scores (align_score.py) for every stain x source scanner, each against P of the same stain.
# Comment out any "run" line to skip it. Finished CSVs are skipped automatically.
#   bash run_align_score.sh
#   MAXJ=16 bash run_align_score.sh           # concurrent jobs (default: number of cores)
#   PYTHON=~/venv/bin/python bash run_align_score.sh
#   ARGS="--grid 6" bash run_align_score.sh     # extra options for align_score.py (e.g. per-block scores)
set -u
DATA=./plism-dataset
OUT=./plism_align
PY="$(dirname "$0")/align_score.py"
PYTHON=${PYTHON:-python3}
MAXJ=${MAXJ:-$(nproc)}
ARGS=${ARGS:-}
mkdir -p "$OUT/logs"

slide() {  # slide STAIN SCANNER -> path or empty
  local f
  for f in "$DATA/${1}_${2}.tif.h5" "$DATA/${1}_${2}"_to_*.tif.h5; do [ -e "$f" ] && echo "$f" && return; done
}

run() {  # run STAIN SCANNER
  local src p dst="$OUT/${1}_${2}_vs_P.csv"
  src=$(slide "$1" "$2"); p=$(slide "$1" P)
  if [ -z "$src" ] || [ -z "$p" ]; then echo "[missing] $1 $2 (src='$src' P='$p')"; return; fi
  if [ -e "$dst" ]; then echo "[skip] $dst exists"; return; fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
  echo "[start] $1 $2"
  "$PYTHON" "$PY" "$src" "$p" "$dst" $ARGS > "$OUT/logs/${1}_${2}.log" 2>&1 \
    && echo "[done] $1 $2" || echo "[FAILED] $1 $2 (see $OUT/logs/${1}_${2}.log)" &
}

# --- GIVH ---
run GIVH AT2
run GIVH GT450
run GIVH S210
run GIVH S360
run GIVH S60
run GIVH SQ
# --- GIV ---
run GIV  AT2
run GIV  GT450
run GIV  S210
run GIV  S360
run GIV  S60
run GIV  SQ
# --- GMH ---
run GMH  AT2
run GMH  GT450
run GMH  S210
run GMH  S360
run GMH  S60
run GMH  SQ
# --- GM ---
run GM   AT2
run GM   GT450
run GM   S210
run GM   S360
run GM   S60
run GM   SQ
# --- GVH ---
run GVH  AT2
run GVH  GT450
run GVH  S210
run GVH  S360
run GVH  S60
run GVH  SQ
# --- GV ---
run GV   AT2
run GV   GT450
run GV   S210
run GV   S360
run GV   S60
run GV   SQ
# --- MY ---
run MY   AT2
run MY   GT450
run MY   S210
run MY   S360
run MY   S60
run MY   SQ
# --- HRH ---
run HRH  AT2
run HRH  GT450
run HRH  S210
run HRH  S360
run HRH  S60
run HRH  SQ
# --- HR ---
run HR   AT2
run HR   GT450
run HR   S210
run HR   S360
run HR   S60
run HR   SQ
# --- KRH ---
run KRH  AT2
run KRH  GT450
run KRH  S210
run KRH  S360
run KRH  S60
run KRH  SQ
# --- KR ---
run KR   AT2
run KR   GT450
run KR   S210
run KR   S360
run KR   S60
run KR   SQ
# --- LMH ---
run LMH  AT2
run LMH  GT450
run LMH  S210
run LMH  S360
run LMH  S60
run LMH  SQ
# --- LM ---
run LM   AT2
run LM   GT450
run LM   S210
run LM   S360
run LM   S60
run LM   SQ

wait
echo "all jobs finished -> $OUT ($(ls -1 "$OUT"/*_vs_P.csv 2>/dev/null | wc -l) CSV files)"
