#!/bin/bash
set -euo pipefail

# =============================================================================
# Collect scattered CSV result files from the current directory (or a given
# search root) into a single folder, then merge them.
#
# Usage:
#   bash collect_csvs.sh                  # search current dir recursively
#   bash collect_csvs.sh /path/to/search  # search a specific directory
#   bash collect_csvs.sh -o results_dir /path  # custom output directory
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Parse arguments ---
OUT_DIR=""
SEARCH_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      SEARCH_ROOT="$1"
      shift
      ;;
  esac
done

SEARCH_ROOT="${SEARCH_ROOT:-.}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/collected_${TIMESTAMP}}"

# --- Find CSV files (exclude already-merged files and collected dirs) ---
echo "Searching for CSV files under: $SEARCH_ROOT"

CSV_FILES=()
while IFS= read -r f; do
  base="$(basename "$f")"
  # Skip merged/collected outputs
  [[ "$base" == "all_results.csv" ]] && continue
  CSV_FILES+=("$f")
done < <(find "$SEARCH_ROOT" -name "*.csv" -type f \
  ! -path "*/collected_*/*" \
  2>/dev/null | sort)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "No CSV files found."
  exit 1
fi

echo "Found ${#CSV_FILES[@]} CSV file(s):"
for f in "${CSV_FILES[@]}"; do
  echo "  - $f"
done

# --- Copy into output directory ---
mkdir -p "$OUT_DIR"

for f in "${CSV_FILES[@]}"; do
  base="$(basename "$f")"
  # Handle name collisions by prepending parent dir
  if [[ -f "${OUT_DIR}/${base}" ]]; then
    parent="$(basename "$(dirname "$f")")"
    dest="${OUT_DIR}/${parent}_${base}"
  else
    dest="${OUT_DIR}/${base}"
  fi
  cp "$f" "$dest"
  echo "  Copied: $f -> $(basename "$dest")"
done

# --- Merge all into a combined CSV ---
COMBINED="${OUT_DIR}/all_results.csv"
echo "method,gamma,n,tps,b,source" > "$COMBINED"

for csv in "${OUT_DIR}"/*.csv; do
  [[ "$(basename "$csv")" == "all_results.csv" ]] && continue
  source_name="$(basename "${csv%.csv}")"
  tail -n +2 "$csv" | while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    echo "${line},${source_name}"
  done >> "$COMBINED"
done

total_rows=$(tail -n +2 "$COMBINED" | wc -l)

echo ""
echo "========================================================"
echo "  Collected ${#CSV_FILES[@]} files -> $OUT_DIR"
echo "  Merged $total_rows rows -> $COMBINED"
echo "========================================================"
echo ""
echo "--- Preview ---"
column -t -s',' "$COMBINED" 2>/dev/null || cat "$COMBINED"
