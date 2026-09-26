#!/usr/bin/env bash
# Overnight v6: cross-encoder on multilingual-e5-base (MIT, 278M), then stage 2 + test output.
# Waits for the v5 chain (work/ce_v5.log) to finish so the two never share the GPU.
# Each step is cached/checkpointed, so a crashed step is retried up to 3 times and resumes.
# Outputs: work/ce_base/, work/stage2_ce_base.txt, work/decision2_ce_base.json, work/output_v6_cebase/
cd "$(dirname "$0")/.."
until grep -qE "PASS|FAIL|Traceback" work/ce_v5.log 2>/dev/null; do sleep 30; done
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export CE_TAG=ce_base CE_BASE_MODEL=intfloat/multilingual-e5-base CE_TRAIN_PAIRS=1600000 \
       CE_TRAIN_BATCH=64 CE_LR=3e-5 CE_SCORE_BATCH=512 CE_OUT=output_v6_cebase
retry() { for n in 1 2 3; do "$@" && return 0; echo "[v6] attempt $n failed: $*"; sleep 20; done; return 1; }
echo "[v6] start $(date)"
retry .venv/Scripts/python tools/ce_pilot.py prep train score &&
retry .venv/Scripts/python tools/ce_v5.py fit score write &&
.venv/Scripts/python student_resource/utils/validate_submission.py \
    --matching work/output_v6_cebase/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir Dataset/test --check-ids
echo "[v6] end $(date) exit $?"
