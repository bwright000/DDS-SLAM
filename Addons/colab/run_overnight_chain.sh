#!/bin/bash
# ============================================================================
# Overnight chain wrapper: StereoMIS T0 ONLY -> CRCD 4-snippet batch.
#
# Designed for user sleep window 2026-06-04 -> 2026-06-05 morning.
#
# Flow:
#   1. StereoMIS P2_1 last-4000 SLAM with SKIP_T1=1 (T0_literal only).
#      Uses tarball-first staging (P2_1_staging.tar already on Drive).
#      Uses MoGe-2 last-4000 slice (commit 500ee9a fixes OOM).
#      Stereo anchor calibration Phase 1.6.
#   2. CRCD 4-snippet batch (F_3/007 -> C_1/001 -> C_2/001 -> F_1/002).
#      Uses per-snippet tarballs (user builds in parallel; see commit 163b7b0).
#      Each snippet: preprocess + MoGe + stereo anchor + SLAM + eval.
#
# Sentinel-gated everywhere; if Colab disconnects mid-way you can re-launch
# the same chain command and it'll resume.
#
# Wall estimates:
#   On A100: ~5-6 hr total (StereoMIS T0 ~1.5 hr + CRCD batch ~4 hr)
#   On T4:   ~25-30 hr total (StereoMIS T0 ~8 hr + CRCD batch ~22 hr)
#
# If you wake up before CRCD finishes: that's fine, just keep monitoring.
# If StereoMIS T0 crashes: the chain continues to CRCD anyway (set +e).
# ============================================================================
set +e   # don't exit on errors — we want CRCD to run even if StereoMIS fails
trap '' INT TERM   # don't die on disconnect signals; runbooks have their own resumability

CHAIN_LOG=/content/drive/MyDrive/Outputs/dds_overnight_$(date +%Y%m%d_%H%M).log
mkdir -p "$(dirname "$CHAIN_LOG")"

echo "============================================================" | tee -a "$CHAIN_LOG"
echo "=== overnight chain start  $(date -Iseconds)" | tee -a "$CHAIN_LOG"
echo "=== log: $CHAIN_LOG" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"

cd /content/DDS-SLAM
git fetch origin && git merge --ff-only origin/main 2>&1 | tee -a "$CHAIN_LOG"

# ============================================================================
# STAGE 1 -- StereoMIS T0 only
# ============================================================================
echo "" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"
echo "=== STAGE 1: StereoMIS P2_1 T0_literal (SKIP_T1=1)" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"
T0_START=$(date +%s)
SKIP_T1=1 bash /content/DDS-SLAM/Addons/colab/run_stereomis_p2_1.sh 2>&1 | tee -a "$CHAIN_LOG"
T0_RC=$?
T0_ELAPSED=$(( ($(date +%s) - T0_START) / 60 ))
echo "" | tee -a "$CHAIN_LOG"
echo "=== STAGE 1 done in ${T0_ELAPSED} min (exit $T0_RC) at $(date -Iseconds)" | tee -a "$CHAIN_LOG"

# ============================================================================
# STAGE 2 -- CRCD 4-snippet batch
# ============================================================================
echo "" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"
echo "=== STAGE 2: CRCD 4-snippet batch (F_3/007 -> C_1/001 -> C_2/001 -> F_1/002)" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"
C_START=$(date +%s)
bash /content/DDS-SLAM/Addons/colab/run_crcd_4snippets.sh 2>&1 | tee -a "$CHAIN_LOG"
C_RC=$?
C_ELAPSED=$(( ($(date +%s) - C_START) / 60 ))
echo "" | tee -a "$CHAIN_LOG"
echo "=== STAGE 2 done in ${C_ELAPSED} min (exit $C_RC) at $(date -Iseconds)" | tee -a "$CHAIN_LOG"

# ============================================================================
# WAKE-UP SUMMARY
# ============================================================================
echo "" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"
echo "=== overnight chain DONE  $(date -Iseconds)" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"
echo "" | tee -a "$CHAIN_LOG"
echo "STEREOMIS T0 (${T0_ELAPSED} min, exit $T0_RC):" | tee -a "$CHAIN_LOG"
LATEST_SM=$(ls -d /content/drive/MyDrive/Outputs/dds_stereomis_p2_1_* 2>/dev/null | tail -1)
if [ -n "$LATEST_SM" ] && [ -f "$LATEST_SM/summary.txt" ]; then
  echo "  --- summary.txt ---" | tee -a "$CHAIN_LOG"
  cat "$LATEST_SM/summary.txt" | tee -a "$CHAIN_LOG"
else
  echo "  (no summary; check $LATEST_SM)" | tee -a "$CHAIN_LOG"
fi
echo "" | tee -a "$CHAIN_LOG"
echo "CRCD 4-SNIPPET (${C_ELAPSED} min, exit $C_RC):" | tee -a "$CHAIN_LOG"
LATEST_C=$(ls -d /content/drive/MyDrive/Outputs/dds_crcd_4snippets_* 2>/dev/null | tail -1)
if [ -n "$LATEST_C" ] && [ -f "$LATEST_C/COMBINED_SUMMARY.txt" ]; then
  echo "  --- COMBINED_SUMMARY.txt ---" | tee -a "$CHAIN_LOG"
  cat "$LATEST_C/COMBINED_SUMMARY.txt" | tee -a "$CHAIN_LOG"
else
  echo "  (no combined summary; check $LATEST_C)" | tee -a "$CHAIN_LOG"
fi
echo "" | tee -a "$CHAIN_LOG"
echo "=== Drive output roots:" | tee -a "$CHAIN_LOG"
echo "    StereoMIS: $LATEST_SM" | tee -a "$CHAIN_LOG"
echo "    CRCD     : $LATEST_C" | tee -a "$CHAIN_LOG"
echo "=== chain log: $CHAIN_LOG" | tee -a "$CHAIN_LOG"
