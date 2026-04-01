#!/bin/bash
set -euo pipefail

# =============================================================================
# One-click runner for TPS benchmark scripts.
#
# Usage:
#   bash run_all.sh                  # run ALL scripts
#   bash run_all.sh sglang           # only SGLang STANDALONE v1
#   bash run_all.sh sglang-v2        # only SGLang STANDALONE v2
#   bash run_all.sh smc              # only SMC
#   bash run_all.sh ssd              # only SSD
#   bash run_all.sh 8b               # only 8B target models
#   bash run_all.sh 70b              # only 70B target models
#   bash run_all.sh sglang 8b        # combine filters (AND logic)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results_${TIMESTAMP}"

# =============================================================================
# Experiment groups — each script is tagged with groups it belongs to
# Format: "script.sh:group1,group2,..."
# =============================================================================
SCRIPT_GROUPS=(
  # --- 8B target ---
  "sglang_sd_1b_8b_triton.sh:sglang,8b"
  "sglang_sd_1b_8b_triton_v2.sh:sglang-v2,8b"
  "sglang_sd_1b_8b_fa3.sh:sglang,8b"
  "sglang_sd_1b_8b_fa3_v2.sh:sglang-v2,8b"
  "smc_1b_8b_triton.sh:smc,8b"
  "smc_1b_8b_fa3.sh:smc,8b"
  "ssd_1b_8b.sh:ssd,8b"
  # --- 70B target ---
  "sglang_sd_1b_70b_triton.sh:sglang,70b"
  "sglang_sd_1b_70b_triton_v2.sh:sglang-v2,70b"
  "sglang_sd_1b_70b_fa3.sh:sglang,70b"
  "sglang_sd_1b_70b_fa3_v2.sh:sglang-v2,70b"
  "smc_1b_70b_triton.sh:smc,70b"
  "smc_1b_70b_fa3.sh:smc,70b"
  "ssd_1b_70b.sh:ssd,70b"
)

# =============================================================================
# Parse filters from command line
# =============================================================================
FILTERS=("$@")

if [[ ${#FILTERS[@]} -eq 0 ]]; then
  echo "No filter specified — running ALL experiments."
fi

# Check if a script matches ALL given filters (AND logic)
matches_filters() {
  local tags="$1"
  for filter in "${FILTERS[@]}"; do
    # Check if the filter appears in the comma-separated tag list
    if [[ ",$tags," != *",$filter,"* ]]; then
      return 1
    fi
  done
  return 0
}

# =============================================================================
# Build the list of scripts to run
# =============================================================================
SCRIPTS=()
for entry in "${SCRIPT_GROUPS[@]}"; do
  script="${entry%%:*}"
  tags="${entry#*:}"
  if [[ ${#FILTERS[@]} -eq 0 ]] || matches_filters "$tags"; then
    SCRIPTS+=("$script")
  fi
done

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  echo "ERROR: No scripts matched filters: ${FILTERS[*]}"
  echo ""
  echo "Available groups:"
  echo "  sglang     — SGLang STANDALONE v1 (triton + fa3)"
  echo "  sglang-v2  — SGLang STANDALONE v2 (triton + fa3, SGLANG_ENABLE_SPEC_V2)"
  echo "  smc        — SMC (triton)"
  echo "  ssd        — SSD backend"
  echo "  8b         — 8B target model only"
  echo "  70b        — 70B target model only"
  echo ""
  echo "Combine filters with AND logic: bash run_all.sh sglang 8b"
  exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "========================================================"
echo "  TPS Benchmark Suite - $(date)"
if [[ ${#FILTERS[@]} -gt 0 ]]; then
  echo "  Filters: ${FILTERS[*]}"
fi
echo "  Scripts to run: ${#SCRIPTS[@]}"
echo "  Results directory: $RESULTS_DIR"
echo "========================================================"

for i in "${!SCRIPTS[@]}"; do
  echo "  $((i+1)). ${SCRIPTS[$i]}"
done

# =============================================================================
# Run selected scripts
# =============================================================================
FAILED=()

for script in "${SCRIPTS[@]}"; do
  script_path="${SCRIPT_DIR}/${script}"
  name="${script%.sh}"
  outfile="${RESULTS_DIR}/${name}.csv"

  echo ""
  echo "--------------------------------------------------------"
  echo "  Running: ${script}  ->  ${name}.csv"
  echo "  Started: $(date)"
  echo "--------------------------------------------------------"

  if bash "$script_path" "$outfile"; then
    echo "  FINISHED: ${script} (success)"
  else
    echo "  FAILED: ${script} (exit code $?)"
    FAILED+=("$script")
  fi
done

# =============================================================================
# Merge all per-script CSVs into one combined file
# =============================================================================
COMBINED="${RESULTS_DIR}/all_results.csv"
echo "method,gamma,n,tps,b,source" > "$COMBINED"

for script in "${SCRIPTS[@]}"; do
  name="${script%.sh}"
  csv="${RESULTS_DIR}/${name}.csv"
  if [[ -f "$csv" ]]; then
    tail -n +2 "$csv" | while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      echo "${line},${name}" >> "$COMBINED"
    done
  fi
done

echo ""
echo "========================================================"
echo "  All benchmarks complete - $(date)"
echo "  Combined results: $COMBINED"
echo "  Individual CSVs:  $RESULTS_DIR/"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo ""
  echo "  WARNING: The following scripts FAILED:"
  for f in "${FAILED[@]}"; do
    echo "    - $f"
  done
fi
echo "========================================================"
echo ""
echo "--- Combined results preview ---"
column -t -s',' "$COMBINED" 2>/dev/null || cat "$COMBINED"
