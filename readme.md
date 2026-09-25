# Amazon ML Challenge 2026: Business Entity Resolution

[![Macro F_0.5](https://img.shields.io/badge/Validation%20Macro%20F0.5-0.8825-brightgreen.svg)]()
[![Precision](https://img.shields.io/badge/Macro%20Precision-93.01%25-blue.svg)]()
[![Recall](https://img.shields.io/badge/Macro%20Recall-79.53%25-orange.svg)]()
[![Candidate Recall Ceiling](https://img.shields.io/badge/Recall%20Ceiling-89.91%25-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

High-performance, memory-efficient Machine Learning solution for the **Business Entity Resolution Challenge**. 

This system resolves fragmented, noisy, and unstandardized business identity records across three independent data sources (**Source 1**, **Source 2**, and **Source 3**) into unified real-world entities.

---

## 1. Challenge Overview & Problem Formulation

In commercial platforms, business data arrives from distinct systems without shared primary keys. Given:
* **Source 1 ($S_1$):** Deduplicated reference entities ($2.2\text{M}$ train, $1.73\text{M}$ test).
* **Source 2 ($S_2$) & Source 3 ($S_3$):** Target records containing partial, corrupted, or noisy fragments ($>10\text{M}$ train records each).

The goal is to determine all matching $S_2$ and $S_3$ entities for every $S_1$ reference entity (1-to-many, 1-to-1, or 1-to-0).

### Official Evaluation Metric: Macro $F_{0.5}$
The evaluation emphasizes **Precision** twice as heavily as Recall to penalize false entity merges:

$$\beta = 0.5 \implies F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

* **Singletons:** A Source 1 entity with zero true matches scores **1.0** if predicted empty, and **0.0** on any false merge.
* **Macro-Averaging:** Computed per Source 1 entity and averaged across the entire corpus.

---

## 2. Architecture & Methodology

```
┌────────────────────────────────────────────────────────┐
│             Raw Source Records (S1, S2, S3)            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Stage 1: Preprocessing & Normalization          │
│  - Strict partition by country (US, India, France)     │
│  - Domain name extraction (e.g. corp.com -> corp)     │
│  - Strip legal suffixes (LLC, Inc, Pvt Ltd, GmbH)      │
│  - Normalize road/address abbreviations (Rd, St, Ave)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│       Stage 2: Candidate Blocking & Generation         │
│  - Multi-Key Inverted Index:                           │
│     * Name token prefixes & compact character stems    │
│     * Address street numbers + primary city/road tokens│
│  - Outputs Top-K candidate pairs                       │
│  - Generates: output/candidate_pairs.tsv               │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Stage 3: Precision Scoring & Calibration        │
│  - RapidFuzz C++ string similarity computation:        │
│     * Token Sort Ratio & Levenshtein Ratio (Name)      │
│     * Token Set Ratio & Partial Ratio (Address)        │
│  - Composite weighted scoring with missing-field logic │
│  - High threshold gate (tau = 0.82) tuned for F_0.5   │
│  - Generates: output/matching_results.tsv              │
└────────────────────────────────────────────────────────┘
```

### Stage 1: Preprocessing & Normalization
* **Country Isolation:** Matches are strictly constrained within the same country partition (US $\leftrightarrow$ US, India $\leftrightarrow$ India, France $\leftrightarrow$ France).
* **Entity Name Cleaning:** 
  * Removes legal entity forms (`inc`, `llc`, `pvt ltd`, `corp`, `gmbh`, `sarl`).
  * Normalizes domains and URLs (e.g., `maurewilliamscolombier.com` $\to$ `maure williams colombier`).
  * Punctuation removal and case folding.
* **Address Normalization:** Standardizes street and unit abbreviations (`rd` $\to$ `road`, `st` $\to$ `street`, `ste` $\to$ `suite`).

### Stage 2: High-Recall Candidate Generation (Blocking)
To avoid an impossible $O(N \times M)$ pairwise comparison ($>20\text{ Trillion}$ pairs), we build an inverted index with composite keys:
1. **Name Keys:** Primary token, bigram prefix, and compact character stems.
2. **Address Keys:** Street number combined with road and city tokens.
* Candidates matching multiple blocking keys are prioritized and capped to the top $K \approx 25\text{--}35$ per query.
* Produces `output/candidate_pairs.tsv` to verify candidate recall ceiling.

### Stage 3: High-Precision Scoring & $F_{0.5}$ Tuning
* Candidate pairs are scored using accelerated fuzzy string similarity:
  * Name similarity: $\max(\text{fuzz.ratio}, \text{fuzz.token\_sort\_ratio})$
  * Address similarity: $\max(\text{fuzz.token\_set\_ratio}, \text{fuzz.partial\_ratio})$
* If an address is missing in $S_2$ or $S_3$, the model dynamically switches to strict high-confidence name verification.
* **Decision Threshold ($\tau = 0.82$):** Conservative thresholding optimizes for the $F_{0.5}$ metric, ensuring non-matches fall back to singletons (scoring 1.0).

---

## 3. Benchmark & Validation Results

Evaluated on the ground-truth validation benchmark (`Dataset/val_sample/`) comprising 2,000 reference entities, 6,865 true matches, and 20,000 realistic distractors:

| Metric | Baseline Score | Description |
| :--- | :---: | :--- |
| **Official Macro $F_{0.5}$** | **0.88249** | Overall competition leaderboard metric |
| **Macro Precision** | **93.01%** | Fraction of predicted links that are correct |
| **Macro Recall** | **79.53%** | Fraction of true links successfully identified |
| **Singleton Accuracy** | **95.41%** | Accuracy on entities with zero true matches |
| **Non-Singleton Macro $F_{0.5}$** | **0.8784** | Score strictly on multi-match entities |
| **Candidate Recall Ceiling** | **89.91%** | Upper bound of recall from blocking stage |
| **Avg Candidates / Entity** | **15.4** | Candidate set size fed to the scorer |
| **Inference Latency** | **1.7 sec** | Total execution time on 2,000 queries (CPU) |

---

## 4. Repository Structure

```
Amazon/
├── Dataset/
│   └── val_sample/               # Fast, self-contained validation benchmark
│       ├── val_source1.tsv
│       ├── val_source2.tsv
│       ├── val_source3.tsv
│       └── val_ground_truth.tsv
├── output/
│   ├── matching_results.tsv      # Leaderboard submission file
│   └── candidate_pairs.tsv       # Blocking candidate pairs
├── src/
│   ├── pipeline.py               # End-to-end pipeline (blocking + scoring + TSV generation)
│   ├── metrics.py                # Official Macro F_0.5 metric implementation
│   ├── evaluate.py               # Standalone evaluation CLI
│   └── create_val_split.py       # Ground-truth validation benchmark generator
├── student_resource/
│   └── utils/
│       └── validate_submission.py # Official competition submission validator
├── CHALLENGE_SPEC.md             # Original challenge problem statement
├── README.md                     # Architecture and pipeline documentation
└── requirements.txt              # Pinned Python dependencies
```

---

## 5. Quickstart & Usage

### Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 1. Run the Baseline Pipeline
Run candidate generation, scoring, and immediate ground-truth validation:
```bash
python src/pipeline.py --mode validate
```

### 2. Evaluate Outputs with Custom Files
```bash
python src/evaluate.py \
    --ground-truth Dataset/val_sample/val_ground_truth.tsv \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv
```

### 3. Verify Output Format with Official Validator
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir Dataset/test
```

### 4. Generate Test Submission
To process the full test dataset and output final files:
```bash
python src/pipeline.py --mode test
```

---

## 6. Hardware & Scaling Roadmap

* **Local Machine (RTX 5060, 8GB VRAM, 16GB RAM):**
  * CPU candidate retrieval + RapidFuzz achieves **~1,200 entities/sec** on CPU.
  * Local GPU is utilized for batch dense embeddings (e.g. `all-MiniLM-L6-v2`) or cross-encoder re-ranking.
* **AWS Cloud Scaling:**
  * Only required if holding all 10M dense embedding vectors in RAM simultaneously (requiring 64GB+ RAM instances, e.g. `r6i.2xlarge`) or when fine-tuning an 8B parameter model.