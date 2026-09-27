# readme2 — Current architecture (v9), why each part exists, and the open problems (team **aid**)

This file describes the pipeline behind our best submission (v9, leaderboard **0.979**).
- [readme.md](readme.md) covers stage 1 in detail.
- [readme1.md](readme1.md) has the chronology, logs, reproduction steps and file locations.

Numbers come from `work/*.log`, the data and the output files unless marked otherwise.

## 1. Architecture

### 1.1 Test-time pipeline (v9)

```
 Test TSVs: 1,732,544 S1 · 9,969,589 S2/S3 · US, India, France (France never seen in training)
   │
 ① Normalize           names → tokens, compact form, phonetic skeleton, legal form
   │                    addresses → number tokens + street words (per-country abbreviation tables)
   │
 ② Blocking            per country: 5 sparse IDF channels (address, name × address, name 4-grams,
   │                    name tokens, phonetic), top 10 each → logistic-regression prefilter
   │                    → top 8 S1 per target = 79,748,463 pairs (= candidate_pairs.tsv)
   │
 ③ Pair features       41 per pair: name/address similarity, numbers, channel scores, competition
   │
 ④ Stage-1 LightGBM    p1 for every pair (model.txt = stage1_fold0.txt)
   │
   │   pairs with p1 < 0.01 (71.4M) skip ⑤–⑦ and keep p1
   ▼   pairs with p1 ≥ 0.01: 8,391,628 "in-play"
 ⑤ Cross-encoder       fine-tuned multilingual-e5-small reads the raw "name | address" of both records → ce
   │
 ⑥ Stage-2 features    35 = 22 group/rarity (v3–v4) + ce, ce_logit (v5)
   │                    + 7 address-crowding / CE-competition (v8) + 4 name-IDF (v9)
   │
 ⑦ Stage-2 LightGBM    stage2_v9.txt → p2
   │
 ⑧ Assignment          each target keeps its best S1 if p ≥ 0.7 (France: max(tuned, 0.6));
   │                    per S1, keep the links that maximize expected F0.5 (the empty set competes)
   ▼
 output/matching_results.tsv: 1,630,493 S1 with matches, 5,786,666 links

 Built, not run (v10): targets v9 leaves unlinked → dense e5 top-5 S1s outside the top 8
                       → cross-encoder → rescue LightGBM → merged into ⑧
```

### 1.2 Training and validation

```
 Train TSVs: 2.2M S1 · 10.3M S2/S3 · US, India, with ground truth
   │
 Ⓐ Drop a fixed random 19% of S1 (seed 123) BEFORE blocking, so train looks like test (§2.1)
   │
 ①②③ Normalize, block (channels and prefilter refit on the remaining S1), features: 82.6M pairs
   │
 Ⓑ Hold out 2% of S1 (35,724 left after the drop) and every target that could link to them
   │   → used only for validation and threshold tuning
   │
 ④ Stage-1 scores: two fold models split by hash(tid) % 2; each target is scored by the fold
   │   that never trained on it. (The fold models themselves were trained in the older
   │   drop-after-blocking simulation, 1.5M targets each; see §3.7.)
   │
 Ⓒ In-play training pairs split by hash(tid, SEED+200) % 2
   ├─ half A (~1.2M pairs)     → fine-tune the cross-encoder ⑤
   └─ half B (1,162,980 pairs) → scored by ⑤ → fit stage 2 ⑦ → tune ⑧ on the held-out S1s (Ⓑ)
```

## 2. Why each part exists, and what it changed

**The evolution at a glance.** The simulation column is the faithful simulation (Ⓐ + Ⓑ) unless marked.

| Version | Added | Simulation | Leaderboard |
|---|---|---|---|
| v2 | ①–④ + ⑧: the stage-1 pipeline | 0.9604 (plain validation 0.9624) | 0.952 |
| v3 | ⑥ group features + ⑦, trained on plain data | plain validation 0.9724 | 0.951 |
| v4 | Ⓐ: train and tune with 19% of S1 dropped | 0.9688 | 0.955 |
| v5 | ⑤ cross-encoder + Ⓒ | 0.9849 | 0.978 |
| v8 | 7 address-crowding / CE-competition features | 0.9855 | not submitted alone |
| v9 | 4 name-IDF features | 0.9857 | **0.979** |
| v10 | dense rescue lane | not run | — |

### 2.1 Ⓐ Dropping 19% of S1: the finding behind everything after v3

**Why.** Test S1 appears to be missing about 19% of the entities.
- Test has 5.76 (US), 5.82 (India) and 5.53 (France) S2+S3 records per S1, against 4.68 in train. That fits test S1 lacking ~19% of entities (~15% in France).
- The missing entities' S2/S3 records remain as mutually consistent **orphan groups** with no true S1.
- For an orphan, the best look-alike S1 ranks first with no competitor. A model trained on complete data reads that as a strong match.

**Effect:**
- It explains v3: +0.010 on plain validation but −0.001 on the leaderboard.
- Training and tuning with the drop gave v4: leaderboard 0.955 (+0.004 over v3).
- The first version dropped S1s *after* blocking, which left 77% of targets with fewer than 8 candidates; on test, 99.99% have exactly 8. Dropping *before* blocking (the "faithful" simulation) removed that tell.
- The faithful simulation is now the yardstick for every decision. Its gap to the leaderboard was 0.008 (v2), 0.014 (v4) and ~0.007 (v5); the remainder is attributed to France (§3.1).

### 2.2 The flip: one S1 per target (④, ⑧)

**Why.** In train, each S2/S3 record matches at most one S1 (7.64M links, no ID reused). So the pipeline answers "for each target, which S1, or none?" and then groups the answers by S1.
- A target can never be shared by two look-alike S1s, which is the costliest F0.5 error.
- It also makes *competition* features possible: how a candidate compares with the target's other candidates.

**Effect.** The competition features `pf_rank` and `pf_gap` are stage 1's top features by gain. The same idea recurs as `rank`, `p_gap` and `p_best_other` in stage 2, and as `ce_rank`, `ce_gap` and `ce_best_other` in v8.

### 2.3 ① Normalize

**Why.** The noise is systematic:
- legal forms moved or dropped
- OCR digits (`6eneral`, `W0rldwide`)
- handles and domains
- native scripts
- address abbreviations and reordering
- French départements in place of regions

Country only selects an abbreviation table, so countries unseen in training (France) still get generic rules.

**Effect.** Everything downstream depends on it; it was never ablated on its own.

### 2.4 ② Blocking

**Why:**
- **Five channels:** different matches fail for different reasons. Rebrands share only the address, transliterations only the phonetic form, and generic names need name × location keys.
- **A prefilter:** the channels' union is ~34 candidates per target (~350M pairs).
- **Logistic regression rather than a plain sum:** the sum loses 9–13% of true pairs at the same cut.
- **Top 8:** it keeps 99.75% of the true pairs the union found.

**Effect** (readme.md §5):
- Union recall is 98.9% (US) and 96.7% (India).
- Removing `addr_key` costs 4.2 / 7.2 points, `mix_key` 1.4 / 3.7, and each other channel 0.3–2.1.
- In the faithful simulation, the share of held-out true pairs that survive blocking is 98.6% (US) and 96.7% (India). This is the hard cap on recall (§3.3).

### 2.5 ③④ Stage-1 features and LightGBM

**Why:**
- Name and address evidence interact non-linearly. A weak name with the exact address and house number is a match; a strong but generic name with no address is doubtful. Gradient-boosted trees learn such interactions from millions of labels on CPU.
- Country is deliberately not a feature, so France is judged on language-neutral evidence.

**Effect:**
- v2 scored 0.9624 on plain validation and 0.952 on the leaderboard.
- Retraining with the drop (Ⓐ) raised the old simulation's score from 0.9583 to 0.9605, and singleton accuracy from 0.859/0.839 to 0.946/0.937 (US/India).

### 2.6 The in-play cut (p1 ≥ 0.01)

**Why.** Steps ⑤–⑦ are expensive (the cross-encoder runs at ~3,000 pairs/s), and 89% of pairs are near-certain non-matches. The cut takes 79.7M pairs down to 8.4M, so the cross-encoder scores test in ~50 min.

**Effect.** It loses very few true matches. Of 20k targets per country that v9 left unlinked, the share with a cut candidate (p1 < 0.01) that the cross-encoder rates ≥ 0.99 is 0.2% (France), 0.1% (India) and 0.05% (US).

### 2.7 ⑤ Cross-encoder

**Why:**
- String features can't judge a rebrand or tell which word identifies a business. A text model reading both records together can.
- `multilingual-e5-small` handles native scripts and French, is MIT-licensed with 118M parameters (the rules allow MIT/Apache models up to 8B), and fits the laptop's 8 GB GPU.
- It is fine-tuned on training pairs, not used zero-shot.

**Effect:**
- The largest single gain: simulation 0.9691 → 0.9849 (precision +1.2 points, recall +2.4, singleton accuracy 0.937 → 0.994), and leaderboard 0.955 → 0.978.
- `ce` and `ce_logit` became stage 2's top two features.
- A 2.4× larger model (e5-base) added only +0.0008, so it isn't in the pipeline.

### 2.8 Ⓒ Two-half split (and out-of-fold stage-1 scores)

**Why.** Stage 2 must see scores that behave like test scores.
- If stage 2 were fit on pairs the cross-encoder had trained on, `ce` would look near-perfect there, and stage 2 would over-trust it on test.
- Stage-1 scores are out-of-fold (④) for the same reason.

**Effect.** Not ablated. Its evidence is that the leaderboard rose by 0.023 for v5, more than the simulation's +0.016.

### 2.9 ⑥ Group features (v3/v4)

**Why.** Stage 1 judges each pair alone. The other targets confidently linked to the same S1 add evidence:
- siblings that share the address or house number
- the strength of the S1's group, and of any competing group
- rarity counts (how many S1s share this name or skeleton), so generic names need more proof

**Effect:**
- Plain validation rose from 0.9623 to 0.9724, but the leaderboard stayed at 0.951 (v3). Orphan groups agree among themselves and got rewarded, which led to Ⓐ.
- Trained with the drop: leaderboard 0.955 (v4).

### 2.10 ⑥ Address crowding and cross-encoder competition (v8, 7 features)

**Why:**
- 13.0% of France's S1s share an exact address with another S1, against 4.7% (US) and 5.1% (India). A shared address is weaker evidence there. The new counts are how many S1s share this S1's address or street, and the target's address.
- The cross-encoder score also gets its own competition features: how it compares with the target's other candidates.

**Effect:**
- Simulation 0.9849 → 0.9855.
- On the shared-address subset (1,962 held-out entities, the closest proxy for France), 0.9796 → 0.9813.
- Not submitted alone, because the gain is below the leaderboard's rounding.

### 2.11 ⑥ Name IDF (v9, 4 features)

**Why.** String similarity weights every word equally. In crowded French names the identifying word is the rare one: a shared "sportive" means little, a shared rare surname a lot. The IDF is fitted per country on its own S1 names, with no external data.

**Effect:**
- Simulation +0.0001 (0.9857); shared-address subset 0.9813 → 0.9818.
- Leaderboard 0.978 → 0.979, which the US/India gain explains. France's change is inconclusive (−0.005 to +0.009).

### 2.12 ⑦ Stage-2 LightGBM

**Why.** It is the same model family as stage 1 (63 leaves, learning rate 0.05). It is fit on half B (Ⓒ) and tuned on the held-out S1s (Ⓑ).

**Effect.** The effects of the features it learns from are in §2.9–2.11.

### 2.13 ⑧ Assignment and thresholds

**Why:**
- F0.5 is computed per S1 and punishes a wrong link ~3× more than a miss. For an S1 with 4 true matches, one missed link scores 0.94; one extra wrong link scores 0.83.
- So the right number of links depends on each S1's own probabilities. Letting the empty set compete covers singletons with the same rule.
- The threshold of 0.7 was tuned in the simulation.
- France has no labels, so it gets a floor of 0.6 (`UNSEEN_THRESHOLD`). At v9's 0.7 the floor has no effect.

**Effect:**
- The expected-F0.5 cut added +0.0014 on plain validation (0.9610 → 0.9624), and scores are flat for thresholds from 0.2 to 0.6.
- France's threshold is not the lever. Adding 49,461 rejected look-alike links gave 0.973; keeping only links with p ≥ 0.99 gave 0.978 (unchanged).

### 2.14 v10: dense rescue lane (built, not run)

**Why:**
- Blocking caps recall, worst in India (96.5% of true pairs retrieved).
- The private leaderboard is probably ~50% India.
- Dense e5 embeddings find different neighbours than the sparse channels.

**Effect** (probe on sampled train targets):
- Top 8 plus dense top 10 retrieves 98.7% of India's true pairs (from 96.5%) and 99.2% of US's (from 98.7%).
- The cross-encoder accepts 83% of rescued India true pairs at ≥ 0.9, and at most 0.2% of new false pairs at ≥ 0.5.
- Expected gain +0.002–0.003 on the leaderboard. A run takes ~2.5 h, and it changes the candidate file too.

### 2.15 Tried and left out

| Idea | Result | Why it is not in the pipeline |
|---|---|---|
| e5-base cross-encoder (v6) | +0.0008 in the simulation | test scoring was interrupted by a GPU dropout; the gain is below leaderboard rounding |
| Both cross-encoders in stage 2 (v7) | never run | it waits for v6 |
| `stage2_next.py` (22 more features) | never validated | the run was stopped on request |
| Removing same-address word-swap links in France | at most +0.0053, likely a loss | the class is as common in true train matches |
| A 7–8B cross-encoder | — | 2–5 days to score test on the 8 GB GPU |
| Web lookups | — | banned by the rules |

## 3. Main problems now

1. **France is the gap, and it is unexplained.**
   - France is ~15% of the leaderboard, has no labels, and scores ~0.94 against ~0.984 for US+India, if the simulation is right for US/India. The model's own estimate for France is 0.984, so its errors are confident ones.
   - Ruled out: thresholds, look-alikes at another house number, word swaps, same-address swaps, the stage-1 cut, the top-8 cap and duplicate S1s. Blocking misses are ruled out at CE ≥ 0.99 (France 2.0% vs India 2.3%), but at ≥ 0.5, 9.0% of France's unlinked targets have a plausible new dense candidate, against 2.9% for India.
   - Two explanations remain open: the simulation is optimistic for US/India, or French noise that the cross-encoder never saw in fine-tuning hurts France's recall.
2. **The field is far ahead.** We are at 0.979, rank 272. The top score is 0.991 and 100+ teams are at ≥ 0.987, so the top 50 is out of reach; the realistic final score is 0.979–0.981, plus 0.002–0.003 with v10. A likely structural difference, which is a belief rather than a measurement: we decide per target, while strong teams may first cluster S2/S3 records into entities and then decide per cluster.
3. **Recall is capped by blocking, worst in India.** Blocking retrieves 96.7% of India's true pairs (US 98.6%), and no scorer can recover the rest. v10 is the fix, and it hasn't been run.
4. **Measurement is weak.**
   - The simulation can't see France.
   - The leaderboard shows 3 decimals.
   - The public split isn't proportional by country (≈ US 43 / India 42 / France 15%), so blanking probes can't separate a country's share from its score.
   - 1 submission remained for 2026-09-26.
5. **The submission package is stale** (required work, ~3 h).
   - `aid_submission.zip` holds v2's outputs and the stage-1 code.
   - `package.py` doesn't include `tools/` (the cross-encoder, v8 and v9 code).
   - `requirements.txt` lacks torch and transformers.
   - `readme.md` and `Documentation_template.md` still describe stage 1.
   - Nothing after v4 is pushed.
6. **Rules risk.** e5 satisfies the license and size rule (MIT, 118M). But the rules also say the challenge uses "only the provided training data", and it's unconfirmed that downloading pretrained weights is allowed. Everything from v5 onward depends on it.
7. **Stage 1 predates the faithful simulation.** Its fold models were trained in the older drop-after-blocking simulation, where many targets had fewer than 8 candidates, and `n_cands` is a stage-1 feature. They were never retrained, and the effect is unmeasured.
8. **One laptop, little time.** The GPU has dropped out twice, and 16 GB RAM allows only one heavy job at a time. The hard deadline is 2026-09-27 23:45 IST.
