# Business Entity Resolution — Amazon ML Challenge 2026 (team **aid**)

For every Source 1 business record, this pipeline finds all Source 2 and Source 3 records that describe the same real-world business. Names are noisy, addresses messy, there are no shared IDs, and the test set adds France, which does not appear in training.

> **Final submission (v10):** public leaderboard **0.98188** macro F0.5. On our test-like validation it scores **0.9865** (US 0.9877, India 0.9848) on 35,724 held-out S1 entities.
> `output/matching_results.tsv` passes the official validator with `--check-ids`: 1,732,544 S1 rows, 1,632,759 with matches.

**Contents:** [1 Problem](#1-the-problem) · [2 Data](#2-what-the-data-looks-like) · [3 Architecture](#3-architecture) · [4 Components](#4-components-and-why-each-exists) · [5 Results](#5-results) · [6 Reproducing](#6-reproducing) · [7 Layout](#7-package-layout) · [8 Compliance](#8-compliance) · [9 Limitations](#9-limitations)

---

## 1. The problem

- **Source 1 (S1)** is the deduplicated reference. **Sources 2 and 3 (S2, S3)** hold noisy copies of those businesses plus unrelated records. Each record has `entity_id`, `business_name`, `business_address` and `country`.
- For each S1 entity we output the list of matching S2/S3 IDs, which may be empty.
- **Metric: macro F0.5**, computed per S1 entity and averaged over all S1 entities. Precision counts twice as much as recall. An entity with matches scores 0 if we predict nothing, and a true singleton scores 1 only if we predict nothing.

## 2. What the data looks like

| Fact | Value | Consequence |
|---|---|---|
| Train | 2.2M S1, 10.3M S2+S3; US 60%, India 40% | Comparing every pair is impossible, so blocking is needed |
| Test | 1,732,544 S1, 9,969,589 S2/S3; **France is 15% of S1 and absent from train** | Features must be language-neutral; France needs a label-free decision rule |
| **Each S2/S3 record matches at most one S1** | 7.64M links, no ID reused | The key structural fact (§3) |
| Singletons | 5.6% of S1 | Recall matters: an entity with matches scores 0 if left empty |
| Matches per S1 | mean 3.5, max 11 | Each prediction is a set |
| S2+S3 records per S1 | train 4.68; test US 5.76, India 5.82, France 5.53 | **Test S1 appears to be missing ~19% of entities (~15% in France)**. Their S2/S3 records remain as look-alike "orphan" groups with no true S1 (§4.2) |

**Noise seen in real ground-truth groups:**
- **Names:**
  - legal forms moved or dropped (`Herter Federal Chesapeake`, `Gajanand Chitfund`)
  - OCR digits (`6eneral`, `W0rldwide`), typos (`Raevn LLC`, `Asdsociataes`), doubled tokens (`Nagaya Nagaya`)
  - handles and domains (`@adxhennessy`, `#elypediatric`), aliases (`Belozeta f/k/a`)
  - native scripts (`ஏஸ் எஸ்டேட்`, `हरियाणा`)
  - unrelated brand names that match only on address (`Lumwex`, `Drexvio`)
- **Addresses:**
  - abbreviations, reordered parts (`OH, Columbus, … Orville Avenue`), missing numbers
  - city variants (`CITY OF MENOMONIE`, `COLUMUS CDP`)
  - French départements in place of regions

## 3. Architecture

**The central idea: flip the problem.** Since each S2/S3 record ("target") belongs to at most one S1, we decide for each target *which single S1 it is, or none*, then group the answers by S1.
- A target can never be shared by two look-alike S1s, which is the costliest F0.5 error.
- It also lets the model compare a candidate with the target's other candidates.

**Test time:**

```
 Test TSVs (S1 1,732,544 · S2/S3 9,969,589 · US, India, France)
   │
 ① Normalize          names → tokens, compact form, phonetic skeleton, legal form;
   │                   addresses → number tokens + street words (per-country abbreviation tables)
 ② Blocking           per country: 5 sparse IDF channels, top 10 each → logistic-regression prefilter
   │                   → top 8 S1 per target (79,748,463 pairs)
 ③ Pair features      41 per pair: name/address similarity, numbers, channel scores, competition
 ④ Stage-1 LightGBM   p1 for every pair
   │                   pairs with p1 < 0.01 keep p1; the 8,391,628 "in-play" pairs go on
 ⑤ Cross-encoder      fine-tuned multilingual-e5-small reads both records' raw "name | address" → ce
 ⑥ Stage-2 features   35: group/sibling agreement, name rarity, ce, address crowding, name IDF
 ⑦ Stage-2 LightGBM   p2
 ⑧ Rescue lane        (US, India) targets whose best p2 < 0.7 → dense e5 top-5 S1s that blocking missed
   │                   → cross-encoder → rescue LightGBM → merged with p2
 ⑨ Assignment         best S1 per target if p ≥ 0.7; per S1, the expected-F0.5-optimal subset (empty competes)
   ▼
 output/matching_results.tsv, output/candidate_pairs.tsv (= top-8 candidates + scored rescue candidates)
```

**Training and validation: a test-like simulation.**

```
 Train TSVs (US, India, with ground truth)
   │
 Ⓐ Drop a fixed random 19% of S1 (seed 123) BEFORE blocking, so train looks like test
 Ⓑ Hold out 2% of S1 (35,724 remain after the drop) and every target that could link to them: validation only
 ④ Stage-1 scores out-of-fold: two models on hash-split halves of the targets
 Ⓒ In-play training pairs split in two halves by hash:
     half A (~1.2M pairs)     → fine-tune the cross-encoder ⑤
     half B (1,162,980 pairs) → fit stage 2 ⑦ and the rescue model ⑧; thresholds tuned on Ⓑ
```

## 4. Components and why each exists

### 4.1 Candidate generation (① ②)

- **Normalization** (`normalize.py`, `prep.py`):
  - Lowercase, transliterate to ASCII (`anyascii`), and strip URLs, TLDs, handles and alias markers.
  - Fix OCR digits inside words, and remove legal forms (kept separately for a legal-agreement feature).
  - Build a **phonetic skeleton**: a consonant key tuned on the observed transliteration noise.
  - Split addresses into number tokens and street words.
  - Country only selects an abbreviation table, so unseen countries fall back to generic rules.
- **Blocking** (`blocking.py`) runs per country, with five sparse channels:

  | Channel | Keys | Catches |
  |---|---|---|
  | `addr_key` | house number × street word | rebrands |
  | `mix_key` | phonetic name token × address word or number | generic names |
  | `name_ngram` | character 4-grams of the compact name | typos, glued names |
  | `name_key` | name tokens and token pairs | near-exact names |
  | `phon_key` | phonetic tokens | transliterations |

  - The union recall is 98.9% (US) and 96.7% (India).
  - A logistic regression over the channel scores and two fuzzy similarities keeps the top 8 per target, retaining 99.75% of the union's true pairs. A plain sum of the scores would lose 9–13%.

### 4.2 The test-like simulation (Ⓐ–Ⓒ)

- **What the data showed:** the S2+S3-per-S1 counts (§2) imply test S1 lacks ~19% of entities. For their orphaned records, the best look-alike S1 ranks first with no competitor.
- **Why it mattered:** a stage-2 model trained on complete data improved plain validation by +0.010 but lost 0.001 on the leaderboard.
- **The fix:** the cross-encoder, stage 2 and the rescue model are trained and tuned with 19% of S1 removed *before* blocking (`faithful_sim.py`). The stage-1 models use the earlier version, which removes the same S1s *after* blocking (`train.apply_drop`). That earlier version left 77% of targets with fewer than 8 candidates, a tell that test (99.99% have exactly 8) never shows, so it no longer serves as the yardstick.
- **How well it tracks the leaderboard:** closely for US and India. The remaining gap is attributed to France, which has no labels.

### 4.3 Scoring (③–⑦)

- **Stage-1 LightGBM** (`features.py`, `train.py`):
  - 41 features: name similarities (plain, compact, phonetic), address similarities and number overlap, channel scores, and competition features (gap to the target's best candidate, candidate count, how many targets rank this S1 first).
  - Country is deliberately **not** a feature, so France is judged on language-neutral evidence.
  - The two out-of-fold models are trained with the 19% drop applied after blocking (§4.2); fold 0 scores test.
- **In-play cut** (p1 ≥ 0.01) keeps 8.4M of 79.7M test pairs. That keeps the cross-encoder affordable (~50 min on an 8 GB GPU) and loses very few true matches.
- **Cross-encoder** (`ce_pilot.py`, `ce_v5.py`):
  - `intfloat/multilingual-e5-small` (MIT, 118M parameters) with a 1-logit head, fine-tuned on half-A pairs.
  - Settings: 9,212 steps, batch 128, lr 5e-5, bf16, max 128 tokens.
  - It reads both records' raw text together, so it can judge rebrands and which word identifies a business.
  - This is the largest single gain: simulation 0.9691 → 0.9849, leaderboard 0.955 → 0.978.
- **Stage-2 LightGBM** (`stage2.py`, `ce_v8.py`, `ce_v9.py`), 63 leaves, lr 0.05, fit on half B. Its 35 features:
  - stage-1 standing and group context: siblings agreeing on address or number, the strength of the S1's group and of any competing group
  - name rarity
  - the cross-encoder score and its standing among the target's candidates
  - address crowding: how many S1s share an address (13% of France S1s share an exact address, against ~5% in US/India)
  - IDF-weighted name overlap

### 4.4 Rescue lane (⑧, `dense_probe.py`, `dense_rescue.py`)

- **Why:** blocking is the recall cap, worst in India (96.7%), where it finds different neighbours than dense embeddings.
- **How:**
  - For targets whose best stage-2 score is below 0.7, off-the-shelf e5-small embeddings retrieve the 5 nearest S1s of the country that blocking did not return.
  - The fine-tuned cross-encoder scores them, and a 12-feature LightGBM (trained on half B) gives the final probability.
  - Those probabilities join the stage-2 scores before assignment, so a rescued S1 wins only if it beats the target's best existing candidate.
- **Retrieval gain:** sampled India targets go from 96.5% of true matches retrieved to 98.7%.
- **Final scope:**
  - The final submission applies the lane to **US and India only**.
  - With it also applied to France, the public score was 0.981712 against 0.98188. France has no labels to tune the lane on.

### 4.5 Decisions (⑨, `assign.py`)

- Each target keeps its highest-probability S1 if p ≥ 0.7, the threshold tuned in the simulation.
- For each S1, the accepted targets are sorted by p, and we keep the prefix that maximizes the plug-in expected F0.5. The empty set competes with expected score ∏(1−p), so singletons need no special case.
- Countries unseen in training get a threshold of at least 0.6 (`UNSEEN_THRESHOLD`); at 0.7 this doesn't bind.

## 5. Results

Test-like simulation (Ⓐ + Ⓑ, 35,724 held-out S1) and the public leaderboard:

| Version | Change | Simulation F0.5 | Leaderboard |
|---|---|---|---|
| v2 | stage-1 pipeline (①–④, ⑨) | 0.9604 | 0.952 |
| v4 | + stage 2, trained under Ⓐ | 0.9688 | 0.955 |
| v5 | + cross-encoder | 0.9849 | 0.978 |
| v9 | + address-crowding and name-IDF features | 0.9857 | 0.979 |
| **v10** | **+ rescue lane (US, India)** | **0.9865** (US 0.9877, India 0.9848) | **0.98188** |

**Final test output:**
- 1,632,759 of 1,732,544 S1 have matches (99,785 empty).
- Links: US 2,241,749, India 2,718,929, France 873,741.
- The candidate file holds the 79.7M top-8 pairs plus the scored US/India rescue candidates. Every match is inside it.

**Alternatives tested for the cross-encoder** (identical stage 2; simulation F0.5):

| Scorer | F0.5 |
|---|---|
| Fine-tuned cross-encoder (used) | **0.9857** |
| Fine-tuned bi-encoder | 0.9837 |
| Off-the-shelf e5 embeddings | 0.9722 |
| Character-3-gram VAE | 0.9719 |
| TF-IDF | 0.9719 |
| No text model | 0.9715 |

## 6. Reproducing

**Environment** (what we used):
- Windows 11, Python 3.12, 16-thread CPU, 16 GB RAM
- NVIDIA RTX 5060 Laptop GPU (8 GB, CUDA 12.8)
- Internet once, to download `intfloat/multilingual-e5-small` from Hugging Face. Afterwards set `HF_HUB_OFFLINE=1`.
- Run one heavy step at a time: blocking peaks at ~10 GB RAM, and out-of-memory kills are silent.

**Paths** are relative to this folder:
- `ER_DATA_DIR`: the folder containing `train/` and `test/`, default `Dataset/`
- `ER_WORK_DIR`: intermediate files, default `work/`
- `ER_OUTPUT_DIR`: default `output/`

Every step caches its outputs and skips work already done.

```powershell
py -V:3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt       # Linux/macOS: .venv/bin/python

# 1. Normalize and block both splits (~35 min each; train blocking also fits work\prefilter.json)
.venv\Scripts\python src\run.py --split train --stage prep
.venv\Scripts\python src\run.py --split train --stage block
.venv\Scripts\python src\run.py --split test  --stage prep
.venv\Scripts\python src\run.py --split test  --stage block

# 2. Stage 1 out-of-fold under the 19% drop, plus an interim stage 2 (~50 min)
#    -> work\stage1_fold0.txt, stage1_fold1.txt, stage2.txt, decision2.json
.venv\Scripts\python src\run.py --split train --stage stage2
copy work\stage1_fold0.txt work\model.txt

# 3. Stage-1 scores on test (~45 min) -> work\test\pred.parquet (also writes an interim output\)
.venv\Scripts\python src\run.py --split test --stage predict

# 4. Test-like simulation: drop 19% of train S1 before blocking, re-score stage 1 (~1.5 h)
.venv\Scripts\python src\faithful_sim.py

# 5. Cross-encoder: fine-tune on half A, score half B + validation (~1 h, GPU)
.venv\Scripts\python src\ce_pilot.py prep train score

# 6. Stage 2 with the cross-encoder, and cross-encoder scores for all in-play test pairs (~1 h, GPU)
.venv\Scripts\python src\ce_v5.py fit score

# 7. Final stage 2 (35 features) -> work\stage2_v9.txt, work\test\pred2_v9.parquet (~15 min)
.venv\Scripts\python src\ce_v9.py fit write

# 8. Rescue lane (~4 h, GPU) -> work\output_v10_usin\ (final) and work\output_v10_dense\ (France included)
.venv\Scripts\python src\dense_rescue.py

# 9. Final output, then the official validator (from the organisers' student_resource\utils\)
copy work\output_v10_usin\matching_results.tsv output\
copy work\output_v10_usin\candidate_pairs.tsv output\
python validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv `
    --test-dir <data>\test --check-ids
```

**Notes:**
- In this package every script is in `src/`. In our repository, steps 4–8 live in `tools/`.
- Seeds are fixed for sampling, splits, LightGBM and torch, so reruns reproduce the results up to multithreading and GPU nondeterminism.

## 7. Package layout

```
src/
  config.py        paths (env-overridable) and constants
  data_io.py       TSV reading/writing
  normalize.py     name/address normalization, legal forms, phonetic skeleton
  prep.py          ① normalize and cache to parquet
  blocking.py      ② 5 retrieval channels, prefilter, top-8 candidates
  features.py      ③ 41 pair features
  train.py         ④ stage-1 LightGBM, 19% drop (DROP_FRAC), held-out validation, threshold tuning
  stage2.py        ⑥⑦ group features, out-of-fold stage 1, stage-2 LightGBM
  assign.py        ⑨ one S1 per target, expected-F0.5 subset per S1
  metrics.py       official macro F0.5
  run.py           CLI for steps 1–3
  package.py       builds aid_submission.zip
  faithful_sim.py  Ⓐ the test-like simulation (drop before blocking)
  ce_pilot.py      ⑤ cross-encoder fine-tuning and scoring (half A / half B split)
  ce_v5.py         ⑤ cross-encoder scores on test
  ce_v8.py         ⑥ address-crowding and cross-encoder-competition features
  ce_v9.py         ⑥⑦ name-IDF features; final stage 2 and its test scores
  dense_probe.py   ⑧ e5 embeddings of S1 records, exact nearest-neighbour search
  dense_rescue.py  ⑧ rescue lane: candidates, scoring, rescue model, final outputs
README.md, requirements.txt
```

## 8. Compliance

- **No external data or services.** Rules and dictionaries come from the provided training data or general language knowledge: abbreviation lists, legal forms, US state codes. No geocoding, registries or lookup APIs are used.
- **Pretrained model:** `intfloat/multilingual-e5-small`, MIT licence, 118M parameters. Its public weights are downloaded once from Hugging Face and fine-tuned only on the provided training data; no data is looked up at run time.
- **Other models:** three LightGBM models (stage 1, stage 2, rescue), all MIT and far below the 8B-parameter limit.
- **Label-free test statistics:** the IDF weights of the blocking channels and of the name-IDF features are fitted on each split's own S1 records, without labels.
- **Dependencies** (pinned in `requirements.txt`, all permissive):
  - numpy, scipy, scikit-learn, polars, pyarrow, sparse-dot-topn, anyascii (BSD/MIT/Apache/ISC)
  - LightGBM, RapidFuzz (MIT)
  - torch (BSD-3)
  - transformers, tokenizers, huggingface_hub, safetensors (Apache-2.0)
- **Country is an open set.** Nothing is one-hot encoded by country, and every test S1 gets a row, France included.

## 9. Limitations

- **France** (15% of test, no labels) is our weakest country. Leaderboard probes put it near 0.94–0.96, against ~0.986 for US+India, assuming the simulation is right for US/India. We could not find a label-free fix:
  - Loosening France's acceptance lost ~0.035 in France's score.
  - Tightening it (keeping only links with p ≥ 0.99) did not help, though that probe could only be read to 3 decimals.
  - Adding rescued links cost ~0.001, and swapping between the v5 and v9 versions of France's rows changed nothing (< 0.0001).
- **Recall is capped by retrieval,** most in India. The rescue lane recovers part of the gap.
- **Decisions are made per target.** Clustering S2/S3 records into entities first, and then deciding per cluster whether an S1 exists, may handle the orphan groups better. It is untested.
