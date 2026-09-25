# Business Entity Resolution — Amazon ML Challenge 2026 (team **aid**)

This pipeline finds, for every Source 1 business record, all the Source 2 and Source 3 records that describe the same real-world business. The inputs are noisy names, messy addresses and no shared IDs, across US, India and France (France appears only in the test set).

> **Validation macro F0.5: 0.9624** (US 0.966, India 0.957), measured on 43,948 held-out S1 entities at full candidate density. Precision is 0.980, recall 0.924.
> **Test submission:** `output/matching_results.tsv` passes the official validator (with `--check-ids`), and the final package is `aid_submission.zip`. The leaderboard score comes from the portal after upload.

**Contents:**
1. [The problem](#1-the-problem)
2. [What the data looks like](#2-what-the-data-actually-looks-like)
3. [The approach](#3-the-approach)
4. [Why this design](#4-why-this-design)
5. [Results](#5-results)
6. [Reproducing](#6-reproducing)
7. [Submitting](#7-submitting)
8. [Repository layout](#8-repository-layout)
9. [Compliance](#9-compliance)
10. [Limitations and next steps](#10-limitations-and-next-steps)

---

## 1. The problem

There are three sources of business records, each with `entity_id`, `business_name`, `business_address` and `country`.

- **Source 1 (S1)** is the deduplicated reference.
- **Sources 2 and 3 (S2, S3)** hold noisy copies of those businesses, plus unrelated distractors. A record's source is given by its ID prefix (`S1-`, `S2-`, `S3-`).

For each S1 entity we must output the list of matching S2/S3 IDs. The list may be empty.

**Metric: macro F0.5.** F0.5 is computed per S1 entity and then averaged over all S1 entities:

```
F0.5 = 1.25 × Precision × Recall / (0.25 × Precision + Recall)
```

Precision counts twice as much as recall, because merging two different businesses is worse than missing a link.
- An entity with matches scores **0** if we predict nothing.
- A true singleton (no matches) scores **1** if we predict nothing, and **0** if we predict anything.

**Deliverables:**
- `output/matching_results.tsv`: the only file scored on the leaderboard.
- `output/candidate_pairs.tsv`: the exact candidate set the final model scored. It is audited for blocking quality.

**Rules:**
- No external data or lookup services (no geocoding, no business registries, no entity-resolution APIs).
- The final model must be MIT or Apache 2.0 licensed, with at most 8B parameters.
- Country is an open set, and every test S1 entity (France included) must get a row.

## 2. What the data actually looks like

These are measured on the full training set before any modelling. They drive every design decision below.

| Fact | Value | Consequence |
|---|---|---|
| Train size | 2.2M S1, 10.3M S2+S3 (US 60%, India 40%) | Every-pair comparison (~2×10¹³ pairs) is impossible, so we need blocking |
| Test size | 1.73M S1, 10.0M S2+S3; **France is 15% of S1** and absent from train | Features must be language-agnostic, and France needs a label-free decision rule |
| **Each S2/S3 record matches at most one S1** | 7.64M links, **no ID reused** | **The key structural fact** (§3) |
| Share of S2/S3 records that match some S1 | ~74% | Most records are real matches; the rest are hard distractors |
| Singletons (S1 with no match) | **only 5.6%** | Recall matters: an entity with matches scores 0 if we predict nothing |
| Matches per S1 | mean 3.5, max 11 | Each prediction is a set, not a single pick |
| Empty addresses in S2/S3 | ~3% | Name-only matching has to work too |

**Noise patterns seen in real ground-truth groups:**

- **Names:**
  - legal suffixes moved, added or dropped: `L.L.C. Herter Federal Chesapeake`, `Private Gajanand Chitfund Limited`
  - OCR-style errors: `6eneral Printing W0rldwide`
  - typos: `Raevn LLC`, `Asdsociataes`
  - doubled tokens: `Nagaya Nagaya Hospitals`
  - handles and domains: `@adxhennessy`, `#elypediatric`, `Elypediatricdentistry.Com`
  - aliases: `Belozeta f/k/a Little Auto Body`
  - **native scripts**: `ஏஸ் எஸ்டேட் பிரைவேட் லிமிடெட்` is *Ace Estate Private Limited*. Some names went English → Indic script → Latin and come out as `vijy teknoloji` for *Vijay Technology*, or `praim kmsltemsi` for *Prime Consultancy*.
  - **completely unrelated brand names** (`Lumwex`, `Drexvio`) that match **only on address**
- **Addresses:**
  - abbreviations (`LN`, `RD`, `R.` = rue, `Bd.`) and state codes against full names
  - reordered components (`OH, Columbus, 5559 Orville Avenue`)
  - missing house numbers, `<NULL>`, `N/A` and empty fields
  - city variants (`CITY OF MENOMONIE`, `COLUMUS CDP`), and French départements in place of regions
  - native-script state names (`हरियाणा`)

## 3. The approach

```
 raw TSVs ──► 0. normalize ──► 1. blocking ──► 2. pair features ──► 3. LightGBM ──► 4. assignment ──► TSVs
             (names, addrs,    (5 sparse       (41 features:        (P(same       (1 S1 per target,
              phonetic key)     channels +      strings, numbers,    business))    expected-F0.5 cut)
                                LR prefilter,   competition context)
                                top-8 per target)
```

### The central idea: flip the problem around
The task is phrased as "for each S1 entity, find its matches". The data says **each S2/S3 record ("target") belongs to at most one S1**. So we solve the equivalent problem: *for each target, which single S1 is it, or is it none?* The per-target picks are then grouped by S1.

- It turns a set-prediction problem into ~10M small **pick-one-or-none** decisions, which a classifier handles well.
- **"At most one S1 per target" is enforced by construction.** Sharing a target between two look-alike S1s is exactly the false-merge error that F0.5 punishes most, and the flip removes it entirely.
- The model can use **competition features**: how this candidate compares with the target's other candidates. "Close, but another S1 fits better" is often the deciding signal.

### Stage 0: Normalization (`normalize.py`, run by `prep.py`)
- **Names:**
  - lowercase, transliterate to ASCII (`anyascii`, covering accents and Indic scripts), strip URLs, TLDs (`.com`, `.c0m`, `.co.in`, `.fr`…), `@`/`#` handles and alias markers (`f/k/a`, `dba`)
  - fix OCR digits inside words (`0→o`, `1→l`, `6→g`, `5→s`…), `&`→`and`, drop punctuation, collapse repeated tokens
  - remove legal forms and honorifics for US, India and France (`llc`, `inc`, `pvt`, `ltd`, `sarl`, `sas`, `sci`, `eurl`, `m/s`, `sri`…). The legal form is kept as a separate value for the legal-agreement feature.
  - output three forms: tokens, **compact** (no spaces, which matches glued domains and handles), and **phonetic skeleton**
- **Phonetic skeleton:** a consonant key tuned on the observed transliteration noise.
  - Rules: `ph→f`, `x→ks`, `sh→s`, `c(e/i)→s`, other `c/ch/q→k`, `j/g→k`, `v/w→b`, `d→t`, `m→n`, vowels and `h`/`y` dropped, doubled letters collapsed.
  - With it, `praim kmsltemsi` and `prime consultancy` both become `prn knsltns`, and `vijy teknoloji` and `Vijay Technology` both become `bk tknlk`.
  - Legal words are also matched by skeleton, so transliterated `piraivet` and `limitet` are recognised as *private* and *limited*.
- **Addresses:**
  - expand abbreviations with a per-country table (US, India, France) and generic rules for any other country; map directions to short forms
  - map US state names to codes, drop filler (`city`, `cdp`, `<null>`, `n/a`, unit/suite/flat words)
  - split each address into **number tokens** (house, unit, zip/PIN; leading zeros stripped) and **street/locality words**

Country only selects an abbreviation table. It is never used as a hard filter, so unseen countries fall back to the generic rules.

### Stage 1: Blocking (`blocking.py`)
Blocking runs **per country**, since matches never cross countries. Each **channel** turns records into sparse IDF-weighted feature vectors, and a multithreaded sparse top-K matrix product (`sparse_dot_topn`) finds each target's nearest S1 records, 10 per channel:

| Channel | Features | Catches | Recall alone, US / India |
|---|---|---|---|
| `addr_key` | house number × street word, street-word bigrams | rebranded names, garbled names | 89.3% / 82.1% |
| `mix_key` | phonetic name token × address word, and × house number | generic names, sparse addresses (`vijy teknoloji` + Ghaziabad + `343`) | 86.2% / 83.4% |
| `name_ngram` | TF-IDF over rare character 4-grams of the compact name | typos, glued names, handles | 72.7% / 58.8% |
| `name_key` | rare name tokens, token bigrams, compact prefix | near-exact names | 68.6% / 56.7% |
| `phon_key` | phonetic tokens, bigrams, whole skeleton | transliterated names | 63.5% / 46.9% |
| **union** (~34 S1 candidates per target) | | | **98.9% / 96.7%** |

Keys shared by more than 200 S1 records, and 4-grams found in more than 0.2% of names, are dropped. They carry almost no identity information, and dropping them keeps the sparse products fast.

**Prefilter.** The union is re-ranked by a **logistic regression** fitted on a 40k-target sample per training country (2.76M pairs). Its inputs are the five channel scores plus two fast RapidFuzz token-set ratios (name, address):

```
score = −12.02 + 1.20·name_ngram + 0.68·phon_key + 2.19·mix_key − 0.62·name_key + 4.50·addr_key
        + 4.73·pf_name + 4.11·pf_addr
```

We keep the **top 8 S1s per target** (`K_FINAL`), which retains **99.75%** of the true pairs the union found. Those 8 per target are exactly what the model scores, and exactly what `candidate_pairs.tsv` reports after regrouping by S1.

### Stage 2: Pair features (`features.py`)
There are **41 features** per (target, S1) pair. String similarities use vectorized RapidFuzz `cpdist`; set overlaps and context features use polars.

- **Name:**
  - `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` on clean names
  - `ratio`, `partial_ratio` and Jaro-Winkler on compact names
  - `ratio` and `token_set_ratio` on phonetic skeletons
  - token overlap count and Jaccard, legal-form agreement, name lengths, token count
- **Address:** `ratio`, `token_set_ratio` and `partial_ratio`; shared numbers and number Jaccard; whether the first number matches; street-word overlap and Jaccard; US state agreement; empty-address and has-number flags
- **Retrieval:** all five channel scores, plus the prefilter score, rank and its two similarity inputs
- **Competition context:**
  - gap to the target's best candidate on prefilter score, name similarity and address similarity
  - number of candidates
  - `s1_top1_count`: how many targets rank this S1 first (real entities attract several records)
- **Source:** S2 or S3, since the two sources have different noise styles (S3 uses more handles and domains)

Country is deliberately **not** a feature. France has no labels, so the model has to rely on language-neutral similarity.

### Stage 3: Model (`train.py`)
- A **LightGBM** binary classifier estimates P(target and S1 are the same business).
- Hyperparameters: `num_leaves=127`, `learning_rate=0.08`, `min_data_in_leaf=100`, feature and bagging fraction 0.8, `lambda_l2=1`, seed 42.
- Training data: **12.0M candidate pairs (1.09M positive)** from 1.5M training targets. A hash-split 10% of those targets is used for early stopping.
- Result: the best iteration was 1,469 of the 1,500-round cap (early stopping never triggered), with holdout log loss **0.0138**.

### Stage 4: Decisions (`assign.py`)
1. Each target keeps only its **highest-probability S1**.
2. The link is accepted if `p ≥ threshold`. The threshold is tuned on validation macro F0.5 and stored in `work/decision.json`: 0.3, with the expected-F0.5 cut on.
3. For each S1, the accepted targets are sorted by p, and we keep the prefix that maximizes the **plug-in expected F0.5** `1.25·TP / (1.25·TP + 0.25·FN + FP)`, with TP, FP and FN estimated from the probabilities. **The empty prediction competes too**, with expected score ∏(1−p). Singletons are therefore handled by the same rule, with no special case.

### France (no training labels)
- French normalization: `r.`→`rue`, `bd`, `av`, `imp`, `st`→`saint`, and legal forms `sarl`/`sas`/`sasu`/`sci`/`eurl`/`snc`/`selarl`.
- The model uses only language-neutral features.
- Countries unseen in training get a **more cautious threshold, 0.6** (`UNSEEN_THRESHOLD` in `run.py`), instead of the tuned 0.3. On validation, macro F0.5 is flat for thresholds from 0.2 to 0.6 (0.9622–0.9624). So 0.6 costs almost nothing if the model is well calibrated on France, and it guards against overconfident false merges if it is not.
- **Label-free sanity checks on test:**
  - France's predictions have the same shape as the training countries' (table below).
  - France's probabilities are very confident. Moving its threshold from 0.3 to 0.8 changes fewer than 2% of its links.
  - A manual spot check of sampled France groups found them almost all correct: domains, accents, `R.`/`Rue`/`Bd.`, `S.A.R.L.`, reordered words, département in place of region. One likely false merge was a generic-name look-alike on a different street.

  | Test predictions | Entities with a match | Links per entity |
  |---|---|---|
  | France | 95.4% | 3.48 |
  | US | 94.5% | 3.30 |
  | India | 94.3% | 3.23 |

## 4. Why this design

**Why flip to per-target decisions?**
- The data guarantees each target has at most one S1.
- Enforcing that removes a whole class of false merges for free, and it matches the metric's precision focus.
- An S1-centric approach would have to decide set membership for up to 11 records at once, with no such guarantee.

**Why several cheap sparse channels rather than one embedding model?**
- The error analysis showed *different* matches fail for *different* reasons.
- Rebranded names are only findable by address. Transliterated names are only findable phonetically. Generic names need name × location keys.
- Each channel is exact, CPU-only and explainable, and their union reaches ~99% (US) and ~97% (India) recall.
- The ablation (§5) shows every channel adds recall the others miss.
- A single dense embedding would blur these distinct signals, cost GPU hours on 22M records, and struggle with numbers such as house numbers, which are among the strongest evidence.

**Why a learned prefilter before the model?**
- The union has ~34 candidates per target, i.e. ~350M pairs.
- A 7-feature logistic regression cuts that to 8 per target (~80M pairs) while losing only ~0.25% of reachable matches. That makes the feature and model stages about 4× cheaper.
- A plain sum of channel scores loses **9–13%** of true pairs at the same cut (US 9.1%, India 12.7%), so learning the weights matters.

**Why LightGBM on hand-built similarity features?**
- Name and address signals interact non-linearly. For example, a weak name with a perfect address and matching house number is a match, while a strong name with no address depends on how generic the name is. Gradient-boosted trees learn these interactions well from millions of labels.
- It is fast, uses no GPU, and is MIT-licensed, far inside the 8B-parameter limit.
- A fine-tuned transformer cross-encoder might add a little on top. Within a 72-hour hackathon it is not the best use of time, and the precision is already 0.98.

**Why expected-F0.5 subset selection?**
- The metric is per entity, so the right number of links depends on that entity's probabilities.
- A single global threshold can't express "stop adding links once the marginal link lowers this entity's expected F0.5".
- The plug-in optimizer does exactly that, and it covers singletons with the same rule.
- Measured gain over the best plain threshold: 0.9624 against 0.9610. It is also robust to the threshold choice (§5).

**Why validate like this?**
- An earlier version of this repo was validated on 2k S1 records against only ~20k random distractors. That scored 0.88, but real retrieval faces ~10M records with thousands of look-alikes, so the number was far too optimistic. That version has been replaced entirely.
- Here, validation uses the **full-density** candidate set:
  - 2% of S1 entities (43,948) are held out.
  - Every target that has a held-out S1 among its candidates, or is truly linked to one (1.53M targets), is excluded from training.
  - Those targets are scored against *all* their candidates, including non-held-out S1s, so the pick-one competition is the real one.
  - Macro F0.5 is computed over the held-out entities with the official formula (`metrics.py`).

## 5. Results

### Blocking

| | US | India |
|---|---|---|
| Recall of the union of 5 channels (20k-target samples, full S1 set) | 98.9% | 96.7% |
| Share of the union's true pairs kept after the top-8 prefilter | 99.75% | 99.75% |
| **Candidate recall ceiling on held-out validation** | **98.5%** | **96.5%** |

| | Train | Test |
|---|---|---|
| Candidate pairs (≤ 8 per target) | 82,552,209 | 79,748,463 |
| Average candidates per S1 entity | 37.4 (validation) | 46.0 |
| S1 entities with no candidates | — | 65 |

**Channel ablation** (union recall with one channel removed):

| Removed | none | `addr_key` | `mix_key` | `name_ngram` | `name_key` | `phon_key` |
|---|---|---|---|---|---|---|
| US | 98.90% | 94.72% | 97.55% | 98.41% | 98.59% | 98.61% |
| India | 96.65% | 89.48% | 92.99% | 94.55% | 94.74% | 94.65% |

### Matching
Held-out validation at full density: 43,948 S1 entities and their 1.53M candidate targets, none of which were used in training.

| Metric | Overall | US | India |
|---|---|---|---|
| **Macro F0.5** | **0.9624** | **0.9659** | **0.9573** |
| Macro precision | 0.9804 | 0.9820 | 0.9781 |
| Macro recall | 0.9243 | 0.9319 | 0.9131 |
| Singleton accuracy | 0.9385 | 0.9460 | 0.9274 |
| Candidate recall ceiling | 0.9771 | 0.9854 | 0.9647 |

**Decision rule sweep** (macro F0.5 overall):

| Threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| Threshold only | 0.9431 | 0.9516 | 0.9567 | 0.9597 | 0.9610 | 0.9604 | 0.9585 |
| + expected-F0.5 cut | 0.9624 | **0.9624** | 0.9623 | 0.9624 | 0.9622 | 0.9610 | 0.9585 |

**Top features by gain:**
- `pf_rank` and `pf_gap`: the candidate's standing among its target's candidates
- `num_jac`: overlap of address numbers
- `pf_score` and `t_has_num`
- `k_ratio`: similarity of the phonetic skeletons
- `pf_addr`, `legal_eq`, `c_jw`, `n_partial`, `num_first_eq`, `state_eq`, `addr_word_jac`, `s1_top1_count`

### Test predictions (submitted)

| | Value |
|---|---|
| S1 rows | 1,732,544 (one per test S1) |
| Rows with matches / empty | 1,638,626 / 93,918 (5.4% predicted singletons; the train rate is 5.6%) |
| Total links | 5,707,868 |
| Thresholds | US 0.3, India 0.3, France 0.6 |
| Official validator (`--check-ids`) | **PASS** |

A backup of the earlier variant (France threshold 0.80, chosen by an acceptance-rate heuristic that was later replaced) is kept in `work/output_v1_fr080/`. It also passes the validator.

## 6. Reproducing

**Requirements:**
- Python 3.12
- ~16 GB RAM (peak ~10 GB during blocking)
- a multi-core CPU (developed on 16 threads); no GPU

```powershell
# Python 3.12 (Windows install manager; on Linux/macOS use your package manager)
py install 3.12
py -V:3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt       # Linux/macOS: .venv/bin/python

# 1. Train split: normalize, block, fit prefilter, train LightGBM, validate, tune the threshold
.venv\Scripts\python src\run.py --split train --stage all

# 2. Test split: normalize, block, predict, write output\matching_results.tsv + output\candidate_pairs.tsv
.venv\Scripts\python src\run.py --split test --stage all

# 3. Validate the format (official script; stdlib only)
python student_resource\utils\validate_submission.py --matching output\matching_results.tsv `
    --candidate output\candidate_pairs.tsv --test-dir Dataset\test --check-ids

# 4. Build the final package
.venv\Scripts\python src\package.py --team aid                 # -> aid_submission.zip
```

- **The train run must come before the test run.** It produces the artifacts the test run uses: `work/prefilter.json` (prefilter weights), `work/model.txt` (LightGBM model) and `work/decision.json` (threshold and rule).
- **Individual stages:** `--stage prep | block | train | predict`. `train` applies to `--split train` and `predict` to `--split test`.
- **Caching:** each stage skips work whose output already exists in `work/`. Delete a file to recompute it. For example, delete `work/test/pred.parquet` after retraining the model.
- **Paths** are relative to the folder that contains `src/`. They can be overridden with environment variables:
  - `ER_DATA_DIR`: folder containing `train/` and `test/`; default `Dataset/`
  - `ER_WORK_DIR`: intermediate files; default `work/`
  - `ER_OUTPUT_DIR`: default `output/`
- **From the submission zip:** run the same commands inside `code/business_entity_resolution/`, with `ER_DATA_DIR` pointing to the challenge data folder. The official validator is not part of the zip; it ships in the organisers' `student_resource/utils/`.
- **Determinism:** fixed seeds (sampling, validation split, LightGBM) make reruns reproducible up to multithreading order.

**Measured runtimes** (16-thread laptop CPU):

| Stage | Train split | Test split |
|---|---|---|
| Normalize (`prep`) | 1.5 min | 1.3 min |
| Blocking (incl. fitting the prefilter on train) | 35 min | 37 min (France 4, India 20, US 13) |
| Training and validation | 15 min | — |
| Prediction and writing the TSVs | — | 34 min + ~8 min |
| **Total** | **~52 min** | **~80 min** |

## 7. Submitting

1. **Leaderboard:** upload **`output/matching_results.tsv`** (96 MB) in the portal. It is the only file that is scored.
2. **Final package:** submit **`aid_submission.zip`** (484 MB, mostly the 1 GB candidate file):

```
aid_submission.zip
├── output/
│   ├── matching_results.tsv        # same file as the leaderboard upload
│   └── candidate_pairs.tsv         # the top-8-per-target candidate set the model scored
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all 11 pipeline modules
│       ├── README.md               # this file
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled-in methodology write-up
```

3. **After any change** to outputs, code or documentation, re-run the validator and `src/package.py` before submitting.

## 8. Repository layout

```
src/
  config.py      paths (env-overridable) and constants: K per channel (10), K_FINAL (8), key-frequency cap (200)
  data_io.py     TSV reading and writing (tab-separated, all strings, empty kept as "", deduplicated ID lists)
  normalize.py   name/address normalization, legal forms, phonetic skeleton
  prep.py        stage 0: normalize and cache to parquet
  blocking.py    stage 1: 5 retrieval channels, LR prefilter, top-8 candidates
  features.py    stage 2: 41 pair features
  train.py       stage 3: LightGBM training, held-out validation, threshold tuning
  assign.py      stage 4: one S1 per target, expected-F0.5 subset per S1
  metrics.py     official macro F0.5 and candidate recall ceiling
  run.py         end-to-end CLI; UNSEEN_THRESHOLD for countries not seen in training
  package.py     builds <team>_submission.zip
Dataset/                     challenge data (train/, test/); not in git (too large)
output/                      matching_results.tsv, candidate_pairs.tsv
work/                        intermediates (git-ignored): normalized parquet, candidates, predictions,
                             prefilter.json, model.txt, decision.json, train.log, test.log
student_resource/            organisers' README, blank documentation template, utils/validate_submission.py
Documentation_template.md    filled-in methodology write-up
requirements.txt             pinned dependencies
aid_submission.zip           final package (git-ignored)
```

## 9. Compliance

- **No external data or services.** Every rule, dictionary and model parameter comes from the provided training data or from general language knowledge (abbreviation lists, legal-form lists, US state codes). No geocoding, registries or APIs are used.
- **Dependencies** (all permissive), pinned in `requirements.txt`: numpy, scipy, scikit-learn, polars, pyarrow (BSD/MIT/Apache); LightGBM (MIT); RapidFuzz (MIT); sparse_dot_topn (Apache 2.0); anyascii (ISC). `unidecode` was deliberately avoided because it is GPL.
- **Model:** LightGBM gradient-boosted trees (MIT), 1,469 trees × 127 leaves, about 4×10⁵ parameters, far inside the 8B limit.
- **Country as an open set.** Nothing is filtered or one-hot encoded by country, and every test S1 entity gets a row, France included.
- **Output rules:** tab-separated, exact headers, one row per S1, no duplicate IDs, only existing S2/S3 IDs, and matches ⊆ candidates. All of this is verified by the official validator.

## 10. Limitations and next steps

**Known limitations:**
- **Recall is the remaining gap:** 0.924 achieved against a 0.977 ceiling, while precision is 0.98. Most lost recall is true links that are in the candidate set but fall below the per-entity cut.
- **Generic names** ("Balaji Investments", "Galaxy Management") with an empty or city-only address. Hundreds of S1 records share them, which causes most blocking misses and most singleton false merges.
- **India trails the US** (0.957 against 0.966) because of transliterated names and sparser addresses.
- **France is unlabelled.** Its threshold is a principled but unvalidated choice (§3).

**Most promising improvements** (in expected-gain order):
1. **Second-stage group model.** Re-score each target with features of the other targets predicted for the same S1, such as shared address or house number with siblings, sibling count and runner-up S1 strength. This rescues rebranded siblings and rejects lone generic-name merges.
2. **A learned native-script ↔ English token dictionary** from the training pairs (`பிரைவேட்`→`private`, `kmsltemsi`→`consultancy`), plus a larger `K_FINAL` (10–12) for India.
3. **Name-frequency features**, i.e. how many S1 records share this name or skeleton, so that generic names need stronger address evidence.
4. **More training targets** (4–5M instead of 1.5M), light hyperparameter tuning, and a separate threshold per country.
