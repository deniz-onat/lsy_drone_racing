#!/usr/bin/env bash
# Paired-seed comparison sweep for the KaFa v10.* controllers.
#
# Wraps scripts/compare_v10_3.py (the paired-seed harness): every controller flies the
# exact same fixed-seed track sequences, so the numbers are directly comparable. Runs are
# sequential ON PURPOSE -- the acados C-code generation into c_generated_code/ is not safe
# from two processes at once. Each run is invoked under `pixi run` (ACADOS env), so launch
# this script from a plain shell.
#
# Usage:
#   scripts/compare_v10.sh                                   # v10..v10.4 on level2, seeds 42 7 123
#   scripts/compare_v10.sh -c level2_sharp_slalom.toml       # stress the slalom fold
#   scripts/compare_v10.sh -s "42" -n 10                     # quick single-seed pass
#   scripts/compare_v10.sh -c level3.toml KaFa_1500_v10_l3.py   # the level-3 variant on its level
#
# Options:
#   -c CONFIG   config file in config/ (default: level2.toml)
#   -s SEEDS    space-separated seed list (default: "42 7 123")
#   -n N_RUNS   episodes per (controller, seed) (default: 20)
#   -h          this help
#   Positional args override the controller list (files in lsy_drone_racing/control/).
#
# Per-run logs + a results.tsv land in logs/compare_v10/<config>_<timestamp>/, and a
# summary table (per seed + aggregated across seeds) is printed at the end. Compare
# same-seed rows against each other; run-to-run noise is ~+/-2 finishes per 20 runs.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="level2.toml"
SEEDS="42 7 123"
N_RUNS=20

while getopts "c:s:n:h" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    s) SEEDS="$OPTARG" ;;
    n) N_RUNS="$OPTARG" ;;
    h) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option (try -h)" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

if [ "$#" -gt 0 ]; then
  CONTROLLERS=("$@")
else
  # The level2 family. KaFa_1500_v10_l3.py is excluded by default: it is the level-3
  # search variant and belongs on level3.toml (pass it explicitly, see usage).
  CONTROLLERS=(
    KaFa_1500_v10.py
    KaFa_1500_v10_1.py
    KaFa_1500_v10_2.py
    KaFa_1500_v10_3.py
    KaFa_1500_v10_4.py
  )
fi

cd "$REPO_ROOT"
for ctrl in "${CONTROLLERS[@]}"; do
  if [ ! -f "lsy_drone_racing/control/$ctrl" ]; then
    echo "error: controller not found: lsy_drone_racing/control/$ctrl" >&2; exit 2
  fi
done
if [ ! -f "config/$CONFIG" ]; then
  echo "error: config not found: config/$CONFIG" >&2; exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/compare_v10/${CONFIG%.toml}_${STAMP}"
mkdir -p "$OUT_DIR"
TSV="$OUT_DIR/results.tsv"
printf "controller\tseed\tfinished\tn_runs\tavg_s\tbest_s\tworst_s\n" > "$TSV"

n_seeds=$(echo $SEEDS | wc -w | tr -d ' ')
total_jobs=$((${#CONTROLLERS[@]} * n_seeds))
job=0
echo "config=$CONFIG  seeds=[$SEEDS]  n_runs=$N_RUNS  -> $total_jobs sweeps, logs in $OUT_DIR"
echo "(first run per controller code-generates the acados solver -- it takes a while)"

for ctrl in "${CONTROLLERS[@]}"; do
  for seed in $SEEDS; do
    job=$((job + 1))
    name="${ctrl%.py}"
    log="$OUT_DIR/${name}_seed${seed}.log"
    echo
    echo "[$job/$total_jobs] $ctrl  seed=$seed"
    if ! pixi run python scripts/compare_v10_3.py \
        --controller="$ctrl" --config="$CONFIG" --n_runs="$N_RUNS" --seed="$seed" \
        2>&1 | tee "$log"; then
      echo "warning: sweep failed for $ctrl seed=$seed (see $log), continuing" >&2
      printf "%s\t%s\t-\t%s\t-\t-\t-\n" "$name" "$seed" "$N_RUNS" >> "$TSV"
      continue
    fi
    # Summary line: "finish 19/20 (95%) | avg 8.103 s | best 7.762 s | worst 8.582 s"
    # or, with zero finishes, just "finish 0/20".
    grep '^finish ' "$log" | tail -1 | awk -v c="$name" -v s="$seed" '{
      split($2, fr, "/")
      avg = best = worst = "-"
      for (i = 1; i <= NF; i++) {
        if ($i == "avg")   avg   = $(i + 1)
        if ($i == "best")  best  = $(i + 1)
        if ($i == "worst") worst = $(i + 1)
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", c, s, fr[1], fr[2], avg, best, worst
    }' >> "$TSV"
  done
done

echo
echo "=== per-seed results ($CONFIG, n_runs=$N_RUNS) ==="
awk -F'\t' 'NR > 1 {
  printf "%-22s seed %-5s %3s/%-3s  avg %-8s best %-8s worst %s\n", $1, $2, $3, $4, $5, $6, $7
}' "$TSV"

echo
echo "=== aggregated across seeds (avg weighted by finishes; compare same-seed rows for A/B) ==="
awk -F'\t' 'NR > 1 && $3 != "-" {
  fin[$1] += $3; tot[$1] += $4
  if ($5 != "-") { wsum[$1] += $3 * $5; wfin[$1] += $3 }
  if ($6 != "-" && (!(($1) in best) || $6 + 0 < best[$1] + 0))  best[$1] = $6
  if ($7 != "-" && (!(($1) in wrst) || $7 + 0 > wrst[$1] + 0))  wrst[$1] = $7
  if (!seen[$1]++) order[++k] = $1
}
END {
  for (i = 1; i <= k; i++) {
    c = order[i]
    a = (wfin[c] > 0) ? sprintf("%.3f", wsum[c] / wfin[c]) : "-"
    printf "%-22s %3d/%-3d (%3.0f%%)  avg %-8s best %-8s worst %s\n",
      c, fin[c], tot[c], 100 * fin[c] / tot[c], a,
      (c in best) ? best[c] : "-", (c in wrst) ? wrst[c] : "-"
  }
}' "$TSV"

echo
echo "raw table: $TSV"
