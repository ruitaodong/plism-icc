#!/usr/bin/env bash
# Extract 4x4 block sums for every PLISM slide, in parallel.
# Comment out any "run" line to skip it. Finished outputs are skipped automatically.
#   bash run_extract_b4.sh            # MAXJ = number of cores
#   MAXJ=8 bash run_extract_b4.sh     # limit concurrent jobs (disk-bound, so 8-16 is often plenty)
#   PYTHON=python bash run_extract_b4.sh   # choose interpreter (needs h5py, numpy)
set -u
DATA=./plism-dataset
OUT=./plism_b4
PY="$(dirname "$0")/extract_b4.py"
PYTHON=${PYTHON:-python3}   # e.g. PYTHON=~/venv/bin/python
MAXJ=${MAXJ:-$(nproc)}
mkdir -p "$OUT/logs"

run() {  # run STAIN SCANNER
  local src="" f
  for f in "$DATA/${1}_${2}.tif.h5" "$DATA/${1}_${2}"_to_*.tif.h5; do [ -e "$f" ] && src="$f"; done
  local dst="$OUT/${1}_${2}_b4.h5"
  if [ -z "$src" ]; then echo "[missing] $1 $2"; return; fi
  if [ -e "$dst" ]; then echo "[skip] $dst exists"; return; fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
  echo "[start] $1 $2"
  "$PYTHON" "$PY" "$src" "$dst" > "$OUT/logs/${1}_${2}.log" 2>&1 \
    && echo "[done] $1 $2" || echo "[FAILED] $1 $2 (see $OUT/logs/${1}_${2}.log)" &
}

# --- GIVH ---
run GIVH P
run GIVH AT2
run GIVH GT450
run GIVH S210
run GIVH S360
run GIVH S60
run GIVH SQ
# --- GIV ---
run GIV  P
run GIV  AT2
run GIV  GT450
run GIV  S210
run GIV  S360
run GIV  S60
run GIV  SQ
# --- GMH ---
run GMH  P
run GMH  AT2
run GMH  GT450
run GMH  S210
run GMH  S360
run GMH  S60
run GMH  SQ
# --- GM ---
run GM   P
run GM   AT2
run GM   GT450
run GM   S210
run GM   S360
run GM   S60
run GM   SQ
# --- GVH ---
run GVH  P
run GVH  AT2
run GVH  GT450
run GVH  S210
run GVH  S360
run GVH  S60
run GVH  SQ
# --- GV ---
run GV   P
run GV   AT2
run GV   GT450
run GV   S210
run GV   S360
run GV   S60
run GV   SQ
# --- MY ---
run MY   P
run MY   AT2
run MY   GT450
run MY   S210
run MY   S360
run MY   S60
run MY   SQ
# --- HRH ---
run HRH  P
run HRH  AT2
run HRH  GT450
run HRH  S210
run HRH  S360
run HRH  S60
run HRH  SQ
# --- HR ---
run HR   P
run HR   AT2
run HR   GT450
run HR   S210
run HR   S360
run HR   S60
run HR   SQ
# --- KRH ---
run KRH  P
run KRH  AT2
run KRH  GT450
run KRH  S210
run KRH  S360
run KRH  S60
run KRH  SQ
# --- KR ---
run KR   P
run KR   AT2
run KR   GT450
run KR   S210
run KR   S360
run KR   S60
run KR   SQ
# --- LMH ---
run LMH  P
run LMH  AT2
run LMH  GT450
run LMH  S210
run LMH  S360
run LMH  S60
run LMH  SQ
# --- LM ---
run LM   P
run LM   AT2
run LM   GT450
run LM   S210
run LM   S360
run LM   S60
run LM   SQ

wait
echo "all jobs finished -> $OUT"
ls -1 "$OUT"/*_b4.h5 2>/dev/null | wc -l | xargs echo "files:"
