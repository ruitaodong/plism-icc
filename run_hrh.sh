#!/usr/bin/env bash
# Build scanner -> P ICC profiles from the HRH stain, using a local copy of
# owkin/plism-dataset under ./plism-dataset (or .h5 files directly in ./).
#
#   ./run_hrh.sh                 # use local files, offer to download missing ones
#   ./run_hrh.sh --tiles-per-slide 300   # extra args are passed to plism_icc.py
set -euo pipefail

STAIN="HRH"
SCANNERS=(P AT2 GT450 S210 S360 S60 SQ)
OUT="./icc_${STAIN}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/plism_icc.py"

[[ -f "$PY_SCRIPT" ]] || { echo "plism_icc.py not found next to this script"; exit 1; }

# --- locate the dataset ------------------------------------------------------
if compgen -G "./plism-dataset/*.h5" >/dev/null; then DATA="./plism-dataset"
elif compgen -G "./*.h5" >/dev/null;             then DATA="."
else                                                  DATA="./plism-dataset"
fi
echo "Dataset dir: $DATA"

# --- check the 7 HRH slides (and catch un-pulled Git LFS pointers) -----------
missing=()
for s in "${SCANNERS[@]}"; do
  f=$(ls "$DATA"/${STAIN}_${s}.tif.h5 "$DATA"/${STAIN}_${s}_to_*.tif.h5 2>/dev/null | head -n1 || true)
  if [[ -z "$f" ]]; then
    missing+=("$s")
  elif [[ $(stat -c %s "$f" 2>/dev/null || stat -f %z "$f") -lt 1000000 ]]; then
    echo "ERROR: $f is a Git LFS pointer, not data. Run: git lfs pull --include \"${STAIN}_*\""
    exit 1
  else
    echo "  ok  $(basename "$f")"
  fi
done

if (( ${#missing[@]} )); then
  echo "Missing ${STAIN} slides for: ${missing[*]} (~2.5 GB each)"
  read -r -p "Download all ${STAIN} files from Hugging Face into $DATA? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || exit 1
  mkdir -p "$DATA"
  if command -v hf >/dev/null; then
    hf download owkin/plism-dataset --repo-type dataset --include "${STAIN}_*" --local-dir "$DATA"
  elif command -v huggingface-cli >/dev/null; then
    huggingface-cli download owkin/plism-dataset --repo-type dataset --include "${STAIN}_*" --local-dir "$DATA"
  else
    echo "Install the CLI first: pip install -U huggingface_hub"; exit 1
  fi
fi

# --- python deps --------------------------------------------------------------
python3 - <<'EOF' || { echo "Run: pip install numpy scipy scikit-image h5py huggingface_hub pillow"; exit 1; }
import numpy, scipy, skimage, h5py, huggingface_hub
EOF

# --- run ----------------------------------------------------------------------
python3 "$PY_SCRIPT" --local-dir "$DATA" --stains "$STAIN" --out "$OUT" "$@"

echo
echo "Held-out results ($OUT/report.json):"
python3 - "$OUT/report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"{'scanner':8s} {'tissue dE00 before':>19s} {'after':>7s} {'p95 after':>10s}")
for s, v in r.items():
    print(f"{s:8s} {v['dE00_tissue_mean_before']:19.2f} {v['dE00_tissue_mean_after']:7.2f} {v['dE00_tissue_p95_after']:10.2f}")
EOF
