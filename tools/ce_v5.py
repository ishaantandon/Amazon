"""v5: stage 2 with the cross-encoder score, applied to test.

Reuses the pilot (tools/ce_pilot.py): the cross-encoder in work/ce/model/ (fine-tuned on pool half A)
and its scores for pool half B + the held-out validation targets (work/ce/scores.parquet).

Phases (cached):
  fit   : stage 2 + cross-encoder trained on half B -> work/stage2_ce.txt, work/decision2_ce.json
          (re-validated on the held-out entities of the faithful simulation)
  score : cross-encoder on every in-play test pair (stage-1 p >= MIN_P) -> work/test/ce_scores.parquet
  write : test stage-2 scores -> work/test/pred2_ce.parquet, work/output_v5_ce/matching_results.tsv

  .venv\\Scripts\\python tools\\ce_v5.py [phase ...]      # default: all
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb
import polars as pl

import ce_pilot
from config import WORK_DIR, work_path

# CE_TAG (see ce_pilot.py) selects the cross-encoder run; the default "ce" gives the v5 file names.
TAG = ce_pilot.CE_TAG
STAGE2_CE = WORK_DIR / f"stage2_{TAG}.txt"
DECISION_CE = WORK_DIR / f"decision2_{TAG}.json"
TEST_CE = work_path("test", f"{TAG}_scores.parquet")
PRED2_CE = work_path("test", f"pred2_{TAG}.parquet")
OUT_DIR = WORK_DIR / os.environ.get("CE_OUT", "output_v5_ce")
UNSEEN_THRESHOLD = 0.6


def _ce_logit(df: pl.DataFrame) -> pl.DataFrame:
    c = pl.col("ce").clip(1e-6, 1 - 1e-6)
    return df.with_columns((c.log() - (1 - c).log()).alias("ce_logit"))


def fit_stage2() -> None:
    if STAGE2_CE.exists():
        return
    import stage2
    from data_io import load_ground_truth, load_links
    from faithful_sim import CANDS, P1
    from stage2 import FEATURES2, PARAMS2
    from train import dropped_s1, fit, tune, val_context, val_split
    truth_all = load_ground_truth("train")
    drop = pl.Series(list(dropped_s1(truth_all)))
    links = load_links("train")
    cands = pl.read_parquet(CANDS, columns=["tid", "s1"])
    val_s1, val_t, _ = val_split(cands, links, truth_all)
    ctx = val_context(cands, truth_all, val_s1)
    del cands
    pred = pl.read_parquet(P1)
    orig = stage2._norm
    stage2._norm = lambda split: (lambda s, t: (s.filter(~pl.col("entity_id").is_in(drop.implode())), t))(*orig(split))
    feats = stage2.group_features(pred, "train")
    stage2._norm = orig
    sc = pl.read_parquet(ce_pilot._p("scores.parquet")).select("tid", "s1", "ce", "part")
    feats = _ce_logit(feats.join(sc, on=["tid", "s1"], how="inner"))
    cols = FEATURES2 + ["ce", "ce_logit"]
    tr = feats.filter(pl.col("part") == "B").join(links.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")),
                                                  on=["tid", "s1"], how="left").with_columns(pl.col("y").fill_null(0))
    model = fit(tr.select(cols).to_numpy(), tr["y"].to_numpy(), tr["tid"], cols, PARAMS2, 3000)
    model.save_model(str(STAGE2_CE))
    va = feats.filter(pl.col("part") == "val")
    p2 = va.select("tid", "s1").with_columns(
        pl.Series("p2", model.predict(va.select(cols).to_numpy(), num_threads=0), dtype=pl.Float32))
    vp = pred.filter(pl.col("tid").is_in(val_t.implode()))
    out = vp.join(p2, on=["tid", "s1"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
    tune(out, *ctx, out=DECISION_CE.name)


def score_test() -> None:
    if TEST_CE.exists():
        return
    from stage2 import MIN_P
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    pairs = pl.read_parquet(work_path("test", "pred.parquet")).filter(pl.col("p") >= MIN_P).select("tid", "s1")
    print(f"[score] test in-play pairs {len(pairs):,}", flush=True)
    t_txt, s_txt = ce_pilot._texts("test")
    tok = AutoTokenizer.from_pretrained(ce_pilot._p("model"))
    model = AutoModelForSequenceClassification.from_pretrained(ce_pilot._p("model")).cuda()
    # Chunked and cached, so a crash loses at most one chunk.
    chunk_dir = work_path("test", f"{TAG}_chunks")
    chunk_dir.mkdir(exist_ok=True)
    step = 1_000_000
    for a in range(0, len(pairs), step):
        f = chunk_dir / f"{a // step:03d}.parquet"
        if f.exists():
            continue
        part = pairs.slice(a, step)
        ce = ce_pilot._score(model, tok, t_txt, s_txt, part["tid"].to_list(), part["s1"].to_list())
        part.with_columns(pl.Series("ce", ce)).write_parquet(f)
        print(f"[score] chunk {a // step} saved ({min(a + step, len(pairs)):,}/{len(pairs):,})", flush=True)
        torch.cuda.empty_cache()
    pl.concat([pl.read_parquet(chunk_dir / f"{i:03d}.parquet")
               for i in range((len(pairs) + step - 1) // step)]).write_parquet(TEST_CE)


def write_test() -> None:
    from assign import assign
    from data_io import load_source, write_id_lists
    from stage2 import FEATURES2, group_features
    pred = pl.read_parquet(work_path("test", "pred.parquet"))
    if not PRED2_CE.exists():
        feats = _ce_logit(group_features(pred, "test").join(pl.read_parquet(TEST_CE), on=["tid", "s1"], how="left"))
        n_missing = feats["ce"].null_count()
        assert n_missing == 0, f"{n_missing} in-play test pairs lack a cross-encoder score"
        cols = FEATURES2 + ["ce", "ce_logit"]
        model = lgb.Booster(model_file=str(STAGE2_CE))
        p2 = feats.select("tid", "s1").with_columns(
            pl.Series("p2", model.predict(feats.select(cols).to_numpy(), num_threads=0), dtype=pl.Float32))
        (pred.join(p2, on=["tid", "s1"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
         .write_parquet(PRED2_CE))
    pred2 = pl.read_parquet(PRED2_CE)
    decision = json.loads(DECISION_CE.read_text())
    seen = set(pl.read_parquet(work_path("train", "s1.parquet"), columns=["country"])["country"].unique())
    matches = {}
    for c in sorted(pred2["country"].unique().to_list()):
        th = decision["threshold"] if c in seen else max(decision["threshold"], UNSEEN_THRESHOLD)
        print(f"[write] {c}: threshold {th}", flush=True)
        matches.update(assign(pred2.filter(pl.col("country") == c), th, decision["expected_f"]))
    s1_ids = load_source("test", 1)["entity_id"].to_list()
    write_id_lists(OUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    print(f"[write] {len(s1_ids):,} S1 rows, {len(matches):,} with matches, "
          f"{sum(len(v) for v in matches.values()):,} links -> {OUT_DIR}", flush=True)


PHASES = {"fit": fit_stage2, "score": score_test, "write": write_test}

if __name__ == "__main__":
    for ph in (sys.argv[1:] or list(PHASES)):
        t0 = time.time()
        PHASES[ph]()
        print(f"[{ph}] done ({time.time() - t0:.0f}s)", flush=True)
