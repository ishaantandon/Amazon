#!/usr/bin/env bash
# v7 (both cross-encoders in stage 2): runs after the v6 chain ends (work/v6.log), then validates.
cd "$(dirname "$0")/.."
until grep -q "\[v6\] end" work/v6.log 2>/dev/null; do sleep 60; done
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
echo "[v7] start $(date)"
if [ -f work/test/ce_base_scores.parquet ]; then
  .venv/Scripts/python tools/ce_ens.py fit write &&
  .venv/Scripts/python student_resource/utils/validate_submission.py \
      --matching work/output_v7_ens/matching_results.tsv --candidate output/candidate_pairs.tsv \
      --test-dir Dataset/test --check-ids
else
  echo "[v7] skipped: v6 test scores missing"
fi
echo "[v7] end $(date)"
