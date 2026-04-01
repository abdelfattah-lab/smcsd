#!/bin/bash
set -euo pipefail

# =============================================================================
# Merge all experiment CSV files into a single combined CSV.
#
# Usage:
#   bash merge_results.sh                          # auto-find latest results_* dir
#   bash merge_results.sh results_20260401_120000  # specify a results dir
#   bash merge_results.sh *.csv                    # specify individual CSV files
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Collect input CSV files ---
CSV_FILES=()

if [[ $# -eq 0 ]]; then
  # No args: find the latest results_* directory
  LATEST_DIR=$(ls -dt "${SCRIPT_DIR}"/results_* 2>/dev/null | head -1)
  if [[ -z "$LATEST_DIR" ]]; then
    # Fallback: look for loose CSVs in the script directory
    while IFS= read -r f; do
      CSV_FILES+=("$f")
    done < <(find "$SCRIPT_DIR" -maxdepth 1 -name "*.csv" -type f | sort)
  else
    echo "Auto-detected results directory: $LATEST_DIR"
    while IFS= read -r f; do
      CSV_FILES+=("$f")
    done < <(find "$LATEST_DIR" -maxdepth 1 -name "*.csv" ! -name "all_results.csv" -type f | sort)
  fi
elif [[ $# -eq 1 && -d "$1" ]]; then
  # Single arg that is a directory
  echo "Using results directory: $1"
  while IFS= read -r f; do
    CSV_FILES+=("$f")
  done < <(find "$1" -maxdepth 1 -name "*.csv" ! -name "all_results.csv" -type f | sort)
else
  # Args are individual CSV files
  for f in "$@"; do
    if [[ -f "$f" ]]; then
      CSV_FILES+=("$f")
    else
      echo "WARNING: skipping '$f' (not a file)"
    fi
  done
fi

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No CSV files found to merge."
  exit 1
fi

echo "Found ${#CSV_FILES[@]} CSV file(s) to merge:"
for f in "${CSV_FILES[@]}"; do
  echo "  - $(basename "$f")"
done

# --- Determine output path ---
if [[ $# -le 1 && -d "${LATEST_DIR:-${1:-}}" ]]; then
  OUTDIR="${LATEST_DIR:-$1}"
else
  OUTDIR="$SCRIPT_DIR"
fi
OUTFILE="${OUTDIR}/all_results.csv"

# --- Merge ---
echo "method,gamma,n,tps,b,source" > "$OUTFILE"

for csv in "${CSV_FILES[@]}"; do
  source_name="$(basename "${csv%.csv}")"
  tail -n +2 "$csv" | while IFS= read -r line; do
    # Skip empty lines
    [[ -z "$line" ]] && continue
    echo "${line},${source_name}"
  done >> "$OUTFILE"
done

total_rows=$(tail -n +2 "$OUTFILE" | wc -l)
echo ""
echo "Merged $total_rows rows into: $OUTFILE"
echo ""
echo "--- Preview ---"
column -t -s',' "$OUTFILE" 2>/dev/null || cat "$OUTFILE"
