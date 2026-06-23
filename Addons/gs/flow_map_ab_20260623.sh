#!/usr/bin/env bash
# v1 flow_map MAPPING-CATCHUP A/B on EndoGSLAM CRCD. base vs flow_map vs uniform-control, n=3,
# on the DYNAMIC snippet (highest depth-residual P99 from the attribution survey; default C2_001).
# Default-off == base. Parity-gated (Inc-0) BEFORE any run. READABLE logs (banners + 20s heartbeat +
# tqdm-suppressed). Resume-safe (.DONE per arm x seed).
#   cd /content/EndoGSLAM && SNIPPETS="C2_001" bash Addons/gs/flow_map_ab_20260623.sh
set -uo pipefail
ENDO=${ENDO:-/content/EndoGSLAM}; CFG=configs/crcd/crcd_base.py
DRIVE=${DRIVE:-/content/drive/MyDrive/Outputs/GS_flowmap_ab_20260623}
LOG="$ENDO/_flowmap_logs"; mkdir -p "$LOG"; cd "$ENDO"
export TQDM_DISABLE=1
export OMP_NUM_THREADS=$(( $(nproc) / ${PARALLEL:-1} > 0 ? $(nproc) / ${PARALLEL:-1} : 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
BASE="FWD_PROP=0 LR_TRANS_MULT=0.2 LR_ROT_MULT=0.2"   # nflr_ref pinned for ALL arms (only differing var = FLOW_MAP)
SNIPPETS=${SNIPPETS:-"C2_001"}                         # the attribution survey's max-depth-P99 snippet
HOLDOUT=${HOLDOUT:-5}
banner(){ echo; echo "==================== $* ($(date +%H:%M:%S)) ===================="; }

banner "STAGE 0  inject flow_map block (idempotent)"
python Addons/gs/inject_flowmap_knobs.py "$ENDO/$CFG" || { echo "FATAL inject"; exit 1; }

banner "STAGE 1  parity gate (Inc-0)"
python Addons/gs/regression/test_flowmap_inc0.py "$ENDO/$CFG" 2>&1 | tee "$LOG/parity.log"
grep -q '>>> INC0 PARITY PASS' "$LOG/parity.log" || { echo "FATAL: parity FAIL -> ABORT (no A/B run)"; exit 1; }

run_one(){   # scene tag "EXTRA_ENV" seed [dynamic_idx.npy]
  local sc="$1" tag="$2" extra="$3" seed="${4:-0}" dynidx="${5:-}"
  local rn="${sc}_${tag}_s${seed}" out="experiments/CRCD_base/${rn}"
  mkdir -p "$out"; [ -f "$out/.DONE" ] && { echo "[$rn] .DONE skip"; return 0; }
  banner "RUN $rn  env:[$BASE $extra]"
  ( env RUN_TAG="$tag" SEED="$seed" SCENE_NUM=0 FM_HEARTBEAT=1 $BASE $extra \
        python scripts/main.py "$CFG" > "$LOG/${rn}.slam.log" 2>&1 ) & local pid=$!
  local t0; t0=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local hb; hb=$(grep -a 'FM_HB' "$LOG/${rn}.slam.log" | tail -1); local el=$(( $(date +%s) - t0 ))
    [ -n "$hb" ] && printf "  [%s] %s · elapsed %dm%02ds\n" "$rn" "${hb#*FM_HB }" $((el/60)) $((el%60))
  done
  wait $pid || { echo "[$rn] !! SLAM FAILED"; tail -8 "$LOG/${rn}.slam.log"; touch "$out/.FAILED"; return 1; }
  banner "EVAL $rn"
  env RUN_TAG="$tag" SEED="$seed" $BASE $extra python scripts/gs_eval.py --config "$CFG" --run "$out" \
        --holdout_every "$HOLDOUT" ${dynidx:+--dynamic_idx "$dynidx"} \
        > "$LOG/${rn}.eval.log" 2>&1 || { echo "[$rn] !! EVAL FAILED"; tail -8 "$LOG/${rn}.eval.log"; touch "$out/.FAILED"; return 1; }
  [ -d /content/drive/MyDrive ] && { d="$DRIVE/$rn"; mkdir -p "$d"; \
     cp "$out"/metrics*.* "$out"/*_6panel.mp4 "$out"/est_c2w_data.txt "$LOG/${rn}".*.log "$d/" 2>/dev/null; }
  touch "$out/.DONE"
  echo "[$rn] DONE -> $(grep -hE 'HEADLINE|HELD-OUT|GUARD|dynPSNR' "$LOG/${rn}.eval.log" 2>/dev/null | tr -s ' \n' ' ')"
}

FM="FM_LAMBDA=${FM_LAMBDA:-1.0} FM_DEADBAND=${FM_DEADBAND:-3.0} FM_DEPTH_DB=${FM_DEPTH_DB:-2.0}"
banner "A/B  base vs flow_map vs uniform-ctrl  n=3  on [$SNIPPETS]  (holdout k=$HOLDOUT)"
for sc in $SNIPPETS; do for s in ${SEEDS:-0 1 2}; do
  run_one "$sc" base    "FM_HOLDOUT_EVERY=$HOLDOUT"                                                      "$s"  ""
  BDY="experiments/CRCD_base/${sc}_base_s${s}/dynamic_idx.npy"   # freeze the DYNAMIC set from base (decoupled)
  run_one "$sc" flowmap "FLOW_MAP=1 $FM FM_HOLDOUT_EVERY=$HOLDOUT"                                       "$s"  "$BDY"
  run_one "$sc" unictrl "FLOW_MAP=1 FM_LAMBDA=${FM_LAMBDA:-1.0} FM_UNIFORM=1 FM_HOLDOUT_EVERY=$HOLDOUT"  "$s"  "$BDY"
done; done

banner "DECISION"
for sc in $SNIPPETS; do python Addons/gs/flowmap_decide.py experiments/CRCD_base "$sc"; done
echo "Panels + metrics_split.json in $DRIVE/<run>/"
