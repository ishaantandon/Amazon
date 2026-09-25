# readme1 — Changes since `readme.md`, findings about the test set, and how to reproduce (team **aid**)

This picks up where [readme.md](readme.md) stops. That file describes the stage-1 pipeline: validation 0.9624, packaged in `aid_submission.zip`, pushed at commit `6f14041`. Everything below happened afterwards, on 2026-09-25 from about 15:00 to 19:00 IST.

> **Current best:** stage 2 trained under **test-like conditions**. The file is `output/matching_results.tsv` (written 18:37:27, validator PASS), with a copy in `work/output_v4_testlike/`. It scores **0.9699 on the test-like validation**, and the expected leaderboard score is **about 0.966**.
> **Goal:** ≥ 0.975 on the leaderboard (the top score is 0.98, and 0.965 is about rank 160).
> **Not yet reflected elsewhere:** `readme.md`, `Documentation_template.md` and `aid_submission.zip` still describe or contain the older stage-1 submission (§9).

---

## 1. Timeline and leaderboard results

| # | Submission (file) | Validation | Test-like validation (§4) | **Leaderboard** |
|---|---|---|---|---|
| v2 | stage 1, threshold 0.3 (`work/output_v2_stage1/`) | 0.9624 | 0.9557 | **0.952** |
| v3 | stage 2 trained on plain data (not backed up, see §9) | 0.9724 | not measured | **0.951** |
| stopgap | stage 1, threshold 0.8 (not backed up, see §9) | 0.9585 | 0.9583 | (not reported) |
| **v4** | **stage 2 trained on test-like data** (`work/output_v4_testlike/`) | — | **0.9699** | **pending, expected about 0.966** |

**The portal upload bug.** The first uploads hung on "Please Wait 0". In DevTools, `create-presigned-url-for-upload-ml-submission` returned 200, but the file upload that should follow never started. The console showed `TypeError: Cannot read properties of null (reading 'filter') at t.getLastSource`. That's a bug in the portal's own JavaScript, and it reproduced in Chrome Incognito and in Edge. Later submissions went through. If it happens again: log out and back in, turn off Edge Tracking Prevention for unstop.com or try another browser, and report it with the console error.

## 2. Error analysis of the stage-1 model

Script: `work/dev/error_analysis.py`. It covers the 152,140 true links of the 43,948 held-out S1 entities, on plain validation:

| Outcome of a true link | Share |
|---|---|
| Found | 92.29% |
| Blocking missed it | 2.27% |
| Another S1 ranked higher | 0.97% |
| Best candidate, but p < threshold | 1.43% |
| Best candidate and p ≥ threshold, but removed by the expected-F0.5 cut | 3.04% |

- There were 1,864 false-positive links: 707 to a record that belongs to another S1, and 1,157 to pure distractors. 149 of the 2,423 singletons received links.
- **Oracle ceilings:**
  - removing all false positives → 0.9736
  - recovering every miss that was in the candidate set → **0.9811**
- **Calibration is good.** Links in the 0.3–0.8 probability band are true about as often as their probability says (0.35 at p ≈ 0.35, 0.66 at p ≈ 0.65). So the expected-F0.5 cut behaves correctly. The weakness is that the model can't separate about 13k genuinely ambiguous pairs, and more discriminative features are the fix.
- **Native scripts aren't worth a fine-tuned model.**
  - In India, native-script names make up 18% of true links. Their recall is 0.897, against 0.914 for Latin-script India names.
  - Closing that gap is worth only about +0.001 macro F0.5.
  - Their blocking miss rate is higher, though (6.9% against 2.7%), so a learned token dictionary for **blocking** remains a cheap option.

## 3. Stage 2: group-aware re-scoring (`src/stage2.py`)

**The idea.** Stage 1 judges each (target, S1) pair on its own. Stage 2 re-scores the uncertain pairs using context from the other targets that confidently belong to the same S1.

**Leak-free training.**
1. `stage2.py oof` trains two stage-1 models, each on 1.5M targets from one half of the targets. The halves are split by `hash(tid, SEED+100) % 2`, a hash independent of the early-stopping split. Each model predicts the other half.
2. The result is an out-of-fold stage-1 probability for **every** training pair (`work/train/p1_oof.parquet`). So every S1's full group is scored honestly.
3. Validation uses the same held-out S1 split as `train.py`.

A bug worth remembering: the first attempt used the same hash seed for the fold split and the early-stopping split. That left one fold with an empty early-stopping set, and LightGBM failed with `num_data > 0`.

**In-play pairs.** Only pairs with stage-1 p ≥ `MIN_P` (0.01) are re-scored: about 9.4M of 82.6M on train and 8.4M of 79.7M on test. All other pairs keep their stage-1 score.

**Stage-2 features** (`FEATURES2`, 22 of them):
- **Stage-1 standing:** `p`, logit, rank within the target, gap to the target's best, best probability among the other candidates.
- **Group strength:** confident members of this S1 excluding self (`g_n_ex`, `g_n9_ex`, `g_sum_ex`); strongest competing S1 group (`alt_g_n`).
- **Sibling agreement:** similarity of the target to up to 6 confident siblings (p ≥ 0.5): address token-set max/mean, name max/mean, numbers, identical address.
- **Name rarity:** how many S1s in the country share the S1's phonetic skeleton or exact name, and the target's skeleton frequency among S1s.
- Also: the target's empty-address and has-number flags, and whether it's from S3.

**Model:** LightGBM, 63 leaves, learning rate 0.05, trained on the in-play pairs of 4M sampled targets. Decisions use the same expected-F0.5 rule, with the threshold re-tuned and saved to `decision2.json`.

**Results on plain validation:**
- stage 1 out-of-fold: 0.9623
- **stage 2: 0.9724** (US 0.9763, India 0.9666); recall +2 points, precision +0.5

**On the leaderboard it scored 0.951, no gain** (explained in §4).

## 4. The main finding: test source 1 is missing about 19% of the entities

**The evidence:**

| | Train | Test |
|---|---|---|
| S2+S3 records per S1 | **4.68** (US 4.67, India 4.68) | **5.76** US, **5.82** India, **5.53** France |
| Share of S2/S3 records with a true S1 | 74% (measured) | about 60% (implied, if links per S1 stay at 3.46) |
| Share of S2/S3 records we link | — | 57–59% (consistent with that) |

**The hypothesis.** The generator creates entities and their S1, S2 and S3 records. When building test, it withholds more S1 records than in train (train's own 26% of unmatched records are probably orphans in the same sense). The withheld entities' S2/S3 records stay behind as **orphaned groups**: realistic, mutually consistent records with no true S1. They get pulled into look-alike S1s.

- A keep rate of 4.68/5.79 ≈ 0.81 means about **19% extra** S1s are dropped in US and India.
- For France the figure is about 4.68/5.53 ≈ 0.85, i.e. about **15%**.

**Why this hurts.**
- An orphan's true S1 was never retrieved, so its best look-alike becomes **rank 1 with no gap**, which the model reads as a strong match.
- Stage 2 made this worse. The orphans' siblings agree with each other, and stage 2 rewards sibling agreement, which is why stage 2 trained on plain data failed on the leaderboard.

**How it was confirmed** (`work/dev/drop_sim2.py`).
1. Remove a random 19% of training S1s (`default_rng(123)`) from the candidate pool **before** re-ranking (`pf_rank`) and recomputing context (`s1_top1_count`, the gaps).
2. Re-score the validation targets and evaluate on the held-out S1s that remain.

A first attempt (`drop_sim.py`) removed the pairs *after* the features were computed. It showed no drop at all, because the stale rank and gap features still marked the orphans as runner-ups. The results of the faithful version, for the stage-1 model trained on plain data:

| Stage-1 model trained on plain data | Plain validation | **Test-like validation (19% dropped)** | Leaderboard |
|---|---|---|---|
| F0.5 at threshold 0.3 | 0.9624 | **0.9557** | **0.952** |
| Singleton accuracy (US / India) | 0.946 / 0.927 | **0.859 / 0.839** | — |
| Best threshold | 0.3 | 0.8 (0.9583) | — |

The test-like validation tracks the leaderboard to within about 0.004. The remaining gap is most likely **France**, which is unlabelled (§8).

## 5. The fix: train and validate under test-like conditions

Code changes in `src/train.py`:
- `DROP_FRAC = 0.19`.
- `dropped_s1(truth_all)`: a fixed set of about 19% of train S1s (`default_rng(123)`).
- `apply_drop(cands, dropped)`: removes those S1s' pairs and recomputes `pf_rank` from `pf_score`.
- `load_train_cands(truth_all)`: raw candidates → `apply_drop` → `add_global_context`. `train.main()`, `stage2.oof()` and `stage2.train()` all use it.
- `val_split` excludes dropped S1s from the held-out set. Their targets still appear, now as distractors.
- Also: `tune()` sweeps thresholds up to 0.9, and `train.py` was refactored into `val_split`, `fit`, `val_context` and `tune(out=…)` so `stage2.py` can reuse them.

Results on the **test-like validation** (35,724 held-out entities):

| Model | Overall | US | India | Threshold |
|---|---|---|---|---|
| Stage 1 trained on plain data | 0.9583 (best threshold) | — | — | 0.8 |
| Stage 1 trained under the test-like regime (out-of-fold) | 0.9605 | 0.9637 | 0.9558 | 0.6 |
| **Stage 2 trained under the test-like regime** | **0.9699** | **0.9736** | **0.9643** | 0.4 |

- Singleton accuracy recovered to 0.946 (US) and 0.937 (India) at the stage-1 level.
- For the **test** predictions, `work/model.txt` is now a copy of `work/stage1_fold0.txt`, a test-like stage-1 model with the same recipe as the out-of-fold models. The stage-2 model is `work/stage2.txt` (copy: `stage2_v4.txt`) with `decision2.json`, threshold 0.4. France uses `UNSEEN_THRESHOLD` = 0.6.
- **Test output v4:** 1,635,437 of 1,732,544 S1s have matches (97,107 empty); 5,844,145 links; validator PASS.

## 6. In progress: stage 2 v2 (`src/stage2_next.py`, not validated yet)

This is a copy of `stage2.py` that writes to separate files (`stage2_next.txt`, `decision2_next.json`, `val_pred2_next.parquet`, `pred2_next.parquet`), so it can't overwrite v4. It adds 22 features, for 44 in total:

- **Pair evidence:** 15 stage-1 features for each in-play pair (`PAIR_EVIDENCE`: `n_tset`, `n_ratio`, `c_jw`, `k_ratio`, `a_tset`, `num_jac`, `num_shared`, `num_first_eq`, `addr_word_jac`, `legal_eq`, `state_eq`, `pf_score`, `pf_gap`, `n_tset_gap`, `a_tset_gap`). They're recomputed with `features.build` on the in-play pairs.
- **Orphan cohesion:** how well the S1's other confident members match the S1 itself (`grp_name_to_s1`, `grp_addr_to_s1`, `grp_num_to_s1`, `grp_name_min`), and the contrast with how well they match this target (`orphan_name`, `orphan_addr`, `self_vs_grp_name`). The rationale: an orphaned group matches itself strongly but its S1 only loosely.

**Status.** Smoke-tested on a test slice. The full training run got through feature building (7.67M rows, about 3.5 minutes) and had just started LightGBM when it was **stopped on request at about 18:50**. To run it: `python src/stage2_next.py train`, which takes about 12 minutes and prints the stage-2 v2 sweep to compare with v4's 0.9699.

## 7. Reproducing the current state (v4)

Run from the repo root with the `.venv` (Python 3.12, `requirements.txt`). Commands are shown for Windows; on Linux/macOS use `.venv/bin/python`.

```powershell
# 0. (once) normalize + block both splits. Unchanged from readme.md; ~35 min per split
.venv\Scripts\python src\run.py --split train --stage prep
.venv\Scripts\python src\run.py --split train --stage block      # also fits work\prefilter.json
.venv\Scripts\python src\run.py --split test  --stage prep
.venv\Scripts\python src\run.py --split test  --stage block

# 1. Out-of-fold stage 1 under the test-like regime (DROP_FRAC=0.19): ~36 min
#    -> work\stage1_fold0.txt, work\stage1_fold1.txt, work\train\p1_oof.parquet
.venv\Scripts\python src\stage2.py oof

# 2. Stage 2 train + test-like validation + threshold tuning: ~8 min
#    -> work\stage2.txt, work\decision2.json, work\decision_oof_stage1.json, work\train\val_pred2.parquet
.venv\Scripts\python src\stage2.py train

# 3. Use the test-like stage-1 model for test predictions (this is what v4 did)
copy work\stage1_fold0.txt work\model.txt
del work\test\pred.parquet work\test\pred2.parquet               # the caches must be rebuilt after a model change

# 4. Test: stage-1 scores (~30 min) -> stage 2 (~3 min) -> output\*.tsv (~8 min)
.venv\Scripts\python src\run.py --split test --stage predict

# 5. Validate
python student_resource\utils\validate_submission.py --matching output\matching_results.tsv `
    --candidate output\candidate_pairs.tsv --test-dir Dataset\test --check-ids
```

**Notes:**
- `run.py --split test --stage predict` applies stage 2 automatically whenever `work/stage2.txt` exists, using `decision2.json`. Otherwise it falls back to stage 1 with `decision.json`.
- `run.py --split train --stage all` now runs prep → block → `train.main()` → stage 2 (out-of-fold, then train). `train.main()` writes its own `model.txt`. To reproduce v4 exactly, step 3 must still copy `stage1_fold0.txt` over it.
- **Caching:** delete `work/test/pred.parquet` and `pred2.parquet` whenever a model changes.
- **Memory:** 16 GB RAM allows only **one heavy job at a time**. Peaks: blocking about 10 GB, stage-2 training about 9 GB. Out-of-memory crashes are silent kills, so run jobs sequentially.
- **Analysis scripts** are in [tools/](tools/) (copied from the git-ignored `work/dev/`). Run them from the repo root, e.g. `.venv\Scripts\python tools\drop_sim2.py 0.19`:
  - `error_analysis.py`: loss breakdown, oracle ceilings, calibration, native-script share
  - `drop_sim2.py` (faithful) and `drop_sim.py` (first, flawed attempt): the dropped-S1 simulation
  - `recall_test.py`, `prefilter_test.py`, `miss_india.py`, `smoke_features.py`: blocking and feature development

**Runtimes** (16 threads):

| Step | Time |
|---|---|
| Out-of-fold stage 1 | 36–46 min |
| Stage-2 train and validation | 7–8 min |
| Stage-2 v2 feature build | 3.5 min |
| Test stage-1 prediction | 30–34 min |
| Test stage 2 | 2–3 min |
| Writing the TSVs | ~8 min |
| Validator with `--check-ids` | ~2 min |

**Artifacts and backups in `work/`:**

| File | What it is |
|---|---|
| `model.txt` | test-like stage-1 model (copy of `stage1_fold0.txt`); used for test |
| `model_v1.txt` | original stage-1 model, trained on plain data (submissions v2, v3, stopgap) |
| `stage1_fold0.txt`, `stage1_fold1.txt` | test-like out-of-fold stage-1 models |
| `stage2.txt` = `stage2_v4.txt` | test-like stage-2 model (v4) |
| `stage2_v1.txt` | stage 2 trained on plain data (v3, leaderboard 0.951) |
| `decision.json` | stage-1 rule, threshold 0.3 (plain) |
| `decision_oof_stage1.json` | 0.6 (test-like stage 1) |
| `decision2.json` = `decision2_v4.json` | 0.4 (test-like stage 2) |
| `train/p1_oof.parquet` | test-like out-of-fold stage-1 scores |
| `train/p1_oof_v1.parquet` | plain-data out-of-fold stage-1 scores |
| `train/val_pred_drop19.parquet` | plain-data model scored in the faithful simulation |
| `test/pred.parquet` | test-like stage-1 test scores |
| `test/pred_v1.parquet` | plain-data stage-1 test scores |
| `test/pred2.parquet` = `pred2_v4.parquet` | v4 stage-2 test scores |
| `test/pred2_v1.parquet` | v3 stage-2 test scores |
| `output_v1_fr080/` | first submission variant (France threshold 0.80) |
| `output_v2_stage1/` | leaderboard 0.952 |
| `output_v4_testlike/` | current best |
| Logs | `train.log`, `test.log`, `stage2.log` (plain stage 2), `v4.log` (test-like run), `drop_sim.log`, `next.log` (stopped v2 run) |

## 8. Speculation about the test set and the challenge

These are beliefs with evidence, not facts.

1. **Test was built by withholding extra S1 entities** (§4): about 19% for US and India, about 15% for France, judging from the 5.53 records per S1. Every modelling and threshold decision should be judged on the **dropped-S1 validation**. Plain validation overstates the leaderboard by about 0.01.
2. **The leaderboard is about test-like validation − 0.004.** The likely cause is France, which has no labels. For the submission that scored 0.952, the test-like validation had US at 0.959 and India at 0.951. Weighting by test S1 counts (India 810k, US 663k, France 259k) puts France at about **0.94**. France's lower drop rate (about 15%) means our threshold of 0.6, tuned for 19%, may be *slightly* conservative there. It's worth trying France at the plain tuned threshold instead.
3. **Train's own unmatched records (26%) are probably also orphans of withheld entities, not random noise.** They're realistic records in sibling groups. The test just has more of them (about 40%). This suggests a target-to-target view (see 5).
4. **The noise generator seems shared across countries.** France shows the same corruptions: domains, accents, legal forms moved, abbreviations, reordering, region replaced by département. That's why the language-neutral features transfer.
5. **The top teams (0.98) probably model the S2/S3 cluster structure directly.** They'd cluster S2↔S3 records into entity groups first, independent of S1, then decide per *cluster* whether an S1 exists. That catches orphans far better than per-record decisions, and it's the natural next step beyond our group features.
6. **Public and private leaderboard splits are random subsets of the same 1.73M S1s.** With hundreds of thousands of entities each, split noise is about ±0.001, so differences above about 0.002 are real.
7. **The metric rewards not merging orphans.** Singletons are about 5.6% of S1 and score all-or-nothing. In the test-like regime, singleton accuracy is the most sensitive quantity: it fell from 0.94 to 0.85 before the fix.

## 9. Known issues and loose ends

- **Stale documents and package:**
  - `readme.md` still describes the stage-1 pipeline and 0.9624.
  - `Documentation_template.md` hasn't been updated with stage 2 or the test-like finding.
  - `aid_submission.zip` contains v2 outputs and older code. Rebuild it after the final choice: `python src/package.py --team aid`. `package.py` includes every `src/*.py`, so `stage2_next.py` would be packaged too. Remove it or finalize it first.
- **Git:** as of the push after this file was written, the code, `tools/`, this file and the v4 `output/matching_results.tsv` are on `main`. `output/candidate_pairs.tsv` (1 GB) must never be pushed normally.
- **Lost outputs:** the v3 output (plain stage 2) and the stopgap output (threshold 0.8) were overwritten without backups. Both can be regenerated from cached predictions:
  - v3: `assign` over `work/test/pred2_v1.parquet` with threshold 0.4, France 0.6
  - stopgap: `assign` over `work/test/pred_v1.parquet` with threshold 0.8
- **Windows line endings:** a Windows git checkout may convert `matching_results.tsv` to CRLF. The validator strips only `\n`, so each row's last ID would get a stray `\r`. Always upload the locally generated file.
- **A process slip:** `taskkill /IM python.exe` at about 18:50 also ended two Python processes that weren't pipeline jobs (probably the VS Code Python extension). Reload the editor window if Python tooling misbehaves.

## 10. Next steps toward 0.975 (test-like validation needs about 0.979)

In order of expected value:

1. **Finish stage 2 v2** (§6). About 12 minutes to validate. If it beats 0.9699, apply it to test (about 15 minutes, since the stage-1 scores are cached).
2. **Per-cluster orphan detection.** Cluster the S2/S3 records of each country among themselves (for example, confident sibling links plus direct S2↔S3 similarity), then add cluster-level features: cluster size, best S1 match of the cluster, and cohesion within the cluster against cohesion to the S1. This probably helps precision on orphans the most.
3. **Raise `K_FINAL` from 8 to 12.** The recall ceiling is 98.5% (US) and 96.5% (India). This needs re-blocking both splits and re-running everything downstream: about 3.5 hours.
4. **A France-specific drop rate and threshold.** Try France at the tuned threshold (0.4) instead of 0.6, given its lower implied drop rate. This can only be checked on the leaderboard.
5. **More stage-1 training data** (4M targets instead of 1.5M), plus light LightGBM tuning. Earlier, the plain-data model hit the 1,500-round cap without early stopping.
6. **A learned native-script ↔ Latin token dictionary for blocking in India.** The native-script blocking miss rate is 6.9%, against 2.7% for Latin script.
