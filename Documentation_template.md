# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** aid
**Team Members:** Ishaan Tandon
**Submission Date:** September 2026

---

## 1. Executive Summary

We flip the task around. Each Source 2/3 record belongs to **at most one** Source 1 entity (verified: 7.64M links, no ID reused), so for every S2/S3 record we pick the single best S1 or none, then group the picks by S1.

- **Candidate generation:** five cheap sparse retrieval channels, re-ranked by a learned prefilter that keeps the top 8 per record.
- **Matching:** a LightGBM classifier over 41 string, number and "competition" features.
- **Decision:** a per-entity expected-F0.5 cut decides how many links each S1 keeps.

On held-out validation at full candidate density, this reaches **macro F0.5 = 0.9624** (precision 0.980, recall 0.924) on 43,948 unseen S1 entities.

---

## 2. Methodology

### 2.1 Problem Analysis

These were measured on the full training data (2.2M S1, 10.3M S2+S3).

**Structure:**
- Every S2/S3 record matches at most one S1.
- ~74% of S2/S3 records match some S1.
- **Only 5.6% of S1 entities are singletons.**
- Matches per S1: mean 3.5, max 11.
- Because an entity with matches scores 0 when predicted empty, recall matters despite the precision-weighted metric.

**Name noise:**
- legal forms moved, added or dropped (`L.L.C. Herter Federal Chesapeake`)
- OCR digits (`6eneral W0rldwide`), typos, doubled tokens
- handles and domains (`@adxhennessy`, `Elypediatricdentistry.Com`), `f/k/a` aliases
- **native-script names** (Tamil, Devanagari), sometimes romanised back to Latin as `vijy teknoloji` (Vijay Technology) or `praim kmsltemsi` (Prime Consultancy)
- **unrelated brand names** (`Lumwex`, `Drexvio`) that match only on address

**Address noise:**
- abbreviations and state codes against full names, reordered components
- missing house numbers, `<NULL>`, empty fields (~3%)
- city variants (`CITY OF …`, `… CDP`), native-script state names

**Test set:** it adds **France** (~15% of test), which has no training labels. The pipeline treats country as an open set.

### 2.2 Solution Strategy

**Approach Type:** Multi-channel sparse blocking → learned prefilter → gradient-boosted pair classifier → constrained, metric-aware assignment.

**Core Innovation:**
1. **Target-centric reformulation.** It enforces the "one S1 per record" constraint by construction, which eliminates a whole class of false merges, and it enables competition features (how a candidate compares with the record's other candidates).
2. **A phonetic consonant skeleton** tuned to English → Indic → Latin transliteration noise.
3. **A per-entity expected-F0.5 subset rule** that handles singletons with no special case.

---

## 3. Candidate Generation (Blocking)

Blocking runs per country. Each channel maps records to sparse IDF-weighted vectors, and a multithreaded sparse top-K product (`sparse_dot_topn`) retrieves each record's nearest S1s.

| Channel | Keys | Top-K | Recall alone (US / India) |
|---|---|---|---|
| `addr_key` | house number × street word, street-word bigrams | 10 | 89.3% / 82.1% |
| `mix_key` | phonetic name token × address word, and × house number | 10 | 86.2% / 83.4% |
| `name_ngram` | TF-IDF of rare char 4-grams of the compact name | 10 | 72.7% / 58.8% |
| `name_key` | rare name tokens, bigrams, compact prefix | 10 | 68.6% / 56.7% |
| `phon_key` | phonetic-skeleton tokens, bigrams, whole skeleton | 10 | 63.5% / 46.9% |
| **Union** (~34 per record) | | | **98.9% / 96.7%** |

- **Blocking keys used:** the address numbers × street words, the phonetic name × location keys, and the name n-grams and tokens listed above. Keys shared by more than 200 S1 records, and n-grams in more than 0.2% of names, are dropped as non-discriminative.
- **Prefilter:** a logistic regression over the five channel scores plus two RapidFuzz token-set ratios (name, address), fitted on a train sample. It keeps the **top 8 S1s per record**, retaining 99.75% of the union's true pairs. A plain sum of channel scores retains only ~89–91% at a comparable cut.
- **Candidate pairs generated:** 82.5M on train (10.3M records × ≤8). `candidate_pairs.tsv` is exactly this final set, re-grouped by S1, i.e. what the model scores.
- **How we ensured true matches were not lost:**
  - Channels were chosen from error analysis of missed pairs. Each one targets a distinct failure mode: rebranded names, transliteration, generic names with sparse addresses.
  - Recall was measured at every step on samples against the full S1 set of each country, so the density is realistic.
  - The held-out candidate recall ceiling is **97.7%** (US 98.5%, India 96.5%).

---

## 4. Matching Model

**Features used** (41 in total, all vectorized with RapidFuzz):
- **Name features:**
  - `ratio`, `token_set_ratio`, `token_sort_ratio` and `partial_ratio` on normalized names
  - `ratio`, `partial_ratio` and Jaro-Winkler on the compact (space-free) name
  - `ratio` and `token_set_ratio` on the phonetic skeleton
  - token overlap count and Jaccard, legal-form agreement, name lengths and token count
- **Address features:** `ratio`, `token_set_ratio` and `partial_ratio`; shared-number count and number Jaccard; whether the first number matches; street-word overlap and Jaccard; state agreement; empty-address and has-number flags
- **Other:**
  - the five blocking channel scores, and the prefilter score and rank
  - **competition context:** gap to the record's best candidate on prefilter score, name similarity and address similarity; number of candidates; how many records rank this S1 first
  - source indicator (S2 or S3)

Country is deliberately not a feature, so that the model transfers to France.

**Normalization** (it feeds all features):
- ASCII transliteration (`anyascii`), URL/TLD/handle stripping, OCR digit fixes inside words, legal-form removal (US, India, France), repeated-token collapse
- per-country address abbreviation tables with a generic fallback, US state codes
- **phonetic skeleton:** `ph→f`, `x→ks`, `sh→s`, `ch/c/q→k`, `j/g→k`, `v/w→b`, `d→t`, `m→n`, vowels dropped, doubled letters collapsed. With it, `praim kmsltemsi` and `prime consultancy` both become `prn knsltns`.

**Model type:**
- LightGBM binary classifier (MIT licence), 127 leaves, learning rate 0.08, 1,469 trees chosen by early stopping; holdout log loss 0.0138.
- Trained on 12.0M candidate pairs (1.09M positives) from 1.5M training records.
- The top features by gain are prefilter rank and gap, number Jaccard, prefilter score, has-number and phonetic-name ratio.

**Threshold selection method:**
1. Each record keeps only its argmax S1.
2. The link is accepted if p ≥ τ.
3. For each S1, we keep the prefix of its accepted records (sorted by p) that maximizes the plug-in expected F0.5, `1.25·TP / (1.25·TP + 0.25·FN + FP)`, with TP, FP and FN estimated from the probabilities. The empty prediction competes with expected score ∏(1−p).
4. τ was swept on validation macro F0.5. Best: **τ = 0.3 with the expected-F0.5 cut** (0.9624, against 0.9610 for the best plain threshold). Results are flat for τ between 0.2 and 0.6, so the choice is robust.

**France (no labels):**
- Countries unseen in training get τ = 0.6, the most cautious value inside the validated flat region (0.2–0.6). It costs almost nothing if the model is calibrated on France and protects precision if it is not.
- **Label-free check on test:** France has 95.4% of entities with a match and 3.48 links per entity. For the US and India the figures are 94.5% / 3.30 and 94.3% / 3.23.
- France's probabilities are confident: moving τ from 0.3 to 0.8 changes fewer than 2% of its links.

---

## 5. Results & Error Analysis

**Validation protocol:**
- 2% of train S1 entities (43,948) are held out.
- Every record that has a held-out S1 among its candidates (1.53M records) is excluded from training.
- Those records are scored against all their candidates, so the pick-one competition is realistic.
- Macro F0.5 is computed over the held-out entities.

| Metric | Overall | US | India |
|---|---|---|---|
| **Macro F0.5** | **0.9624** | 0.9659 | 0.9573 |
| Macro precision | 0.9804 | 0.9820 | 0.9781 |
| Macro recall | 0.9243 | 0.9319 | 0.9131 |
| Singleton accuracy | 0.9385 | 0.9460 | 0.9274 |
| Candidate recall ceiling | 0.9771 | 0.9854 | 0.9647 |

- **F0.5 score (macro):** 0.9624 on validation.
- **Common false positives (wrong merges):** generic names ("Balaji Investments", "Galaxy Management", "Raj Technologies") with empty or city-only addresses. Many S1 entities share them, so a record can be pulled to the wrong look-alike. They also cause most singleton errors.
- **Common false negatives (missed matches):**
  - true pairs outside the candidate set (2.3%), mainly generic names with no address, and transliterated names with sparse addresses
  - in-candidate true links whose probability falls below the per-entity cut. These are the larger share: recall is 0.924 against a 0.977 ceiling.

---

## 6. Conclusion

Reformulating entity resolution around the data's one-S1-per-record structure, plus recall-driven multi-channel blocking and metric-aware assignment, gives 0.962 macro F0.5 at realistic density on CPU only. Precision is already 0.98, so the remaining headroom is recall on in-candidate links and better handling of generic names. One lesson: validating at full candidate density matters. Evaluating an earlier prototype against a small random-distractor sample badly overstated how easy the task is.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
  src/config.py      paths and constants
  src/data_io.py     TSV I/O
  src/normalize.py   name/address normalization and phonetic skeleton
  src/prep.py        stage 0: normalize → parquet
  src/blocking.py    stage 1: 5 channels + LR prefilter → top-8 candidates
  src/features.py    stage 2: pair features
  src/train.py       stage 3: LightGBM, held-out validation, threshold tuning
  src/assign.py      stage 4: argmax per record + expected-F0.5 subset
  src/metrics.py     macro F0.5 and recall ceiling
  src/run.py         end-to-end CLI
  src/package.py     builds the zip
  README.md, requirements.txt
```

To reproduce (Python 3.12; set `ER_DATA_DIR` to the folder containing `train/` and `test/`):

```bash
python -m pip install -r requirements.txt
python src/run.py --split train --stage all   # normalize, block, train + validate
python src/run.py --split test  --stage all   # normalize, block, predict → output/*.tsv
```

On a 16-thread laptop CPU with 16 GB RAM this takes ~52 min for the train split and ~80 min for the test split (peak RAM ~10 GB).

### B. Additional Results

**Decision rule sweep** (validation macro F0.5):

| τ | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| Threshold only | 0.9431 | 0.9516 | 0.9567 | 0.9597 | 0.9610 | 0.9604 | 0.9585 |
| + expected-F0.5 cut | 0.9624 | **0.9624** | 0.9623 | 0.9624 | 0.9622 | 0.9610 | 0.9585 |

**Blocking ablation** (recall on 20k-record samples at full density; union without each channel):

| Removed channel | US | India |
|---|---|---|
| none (all 5) | 98.90% | 96.65% |
| `addr_key` | 94.72% | 89.48% |
| `mix_key` | 97.55% | 92.99% |
| `name_ngram` | 98.41% | 94.55% |
| `name_key` | 98.59% | 94.74% |
| `phon_key` | 98.61% | 94.65% |

**Compliance:**
- No external data, APIs or lookups.
- All dependencies are MIT, BSD, Apache 2.0 or ISC licensed.
- The model is LightGBM (MIT), far below 8B parameters.
