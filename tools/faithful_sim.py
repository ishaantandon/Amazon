"""Faithful test-like simulation: drop ~19% of train S1s BEFORE blocking, then score saved models.

The first simulation (train.apply_drop) removes dropped S1s from the finished candidate lists.
That leaves 77% of validation targets with fewer than 8 candidates (test: ~100% have 8), and the
stage-2 rarity counts still include the dropped S1s. Here the dropped S1s never exist: blocking
(IDF fit, channel top-K, prefilter top-8) and rarity counts see only the kept S1s, as on test.

Phases (each cached, so a rerun resumes):
  block : work/train/cands_faithful.parquet                       (~35 min, ~10 GB peak)
  p1    : work/train/p1_faithful.parquet, stage-1 fold models,     (~25 min)
          each target scored by the fold model that never trained on it
  eval  : v2 (model_v1.txt, t=0.3) and v4 (stage-1 folds + stage2_v4.txt, t=0.4) on the
          held-out S1s, to check the simulation against their leaderboard scores (0.952, 0.955)

  .venv\\Scripts\\python tools\\faithful_sim.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lightgbm as lgb
import polars as pl

import stage2
from blocking import PREFILTER_PATH, block_country
from config import SEED, WORK_DIR, work_path
from data_io import load_ground_truth, load_links
from features import add_global_context
from train import dropped_s1, featurize, predict, tune, val_context, val_split

CANDS = work_path("train", "cands_faithful.parquet")
P1 = work_path("train", "p1_faithful.parquet")


def block_faithful(drop: pl.Series) -> None:
    if CANDS.exists():
        return
    pf = json.loads(PREFILTER_PATH.read_text())
    s1_all = pl.read_parquet(work_path("train", "s1.parquet")).filter(~pl.col("entity_id").is_in(drop.implode()))
    t_all = pl.read_parquet(work_path("train", "t.parquet"))
    out = []
    for country in sorted(s1_all["country"].unique().to_list()):
        s1 = s1_all.filter(pl.col("country") == country)
        t = t_all.filter(pl.col("country") == country)
        print(f"[block] faithful {country}: {len(s1):,} S1 x {len(t):,} targets", flush=True)
        out.append(block_country(s1, t, pf))
    pl.concat(out).write_parquet(CANDS)


def load_cands() -> pl.DataFrame:
    return add_global_context(pl.read_parquet(CANDS))


def p1_faithful(cands: pl.DataFrame) -> None:
    if P1.exists():
        return
    t0 = time.time()
    tids = cands["tid"].unique().sort()
    fold = (tids.hash(SEED + 100) % 2).cast(pl.Int8)  # same fold split as stage2.oof()
    parts = []
    for k in (0, 1):
        model = lgb.Booster(model_file=str(WORK_DIR / f"stage1_fold{k}.txt"))
        mine = tids.filter(fold == k)
        for a in range(0, len(mine), 2_000_000):
            parts.append(predict(model, featurize(cands, "train", mine.slice(a, 2_000_000))))
            print(f"[p1] fold {k}: {min(a + 2_000_000, len(mine)):,}/{len(mine):,} ({time.time() - t0:.0f}s)", flush=True)
    pl.concat(parts).write_parquet(P1)


def evaluate(cands: pl.DataFrame, truth_all: dict, drop: pl.Series) -> None:
    links = load_links("train")
    val_s1, val_t, _ = val_split(cands, links, truth_all)
    ctx = val_context(cands, truth_all, val_s1)
    print(f"[eval] held-out S1 {len(val_s1):,}, val targets {len(val_t):,}", flush=True)
    nc = cands.filter(pl.col("tid").is_in(val_t.implode())).group_by("tid").len()["len"]
    print(f"[eval] val targets with <8 candidates: {(nc < 8).mean():.4f}", flush=True)

    v1 = predict(lgb.Booster(model_file=str(WORK_DIR / "model_v1.txt")), featurize(cands, "train", val_t))
    print("[eval] v2 = plain stage 1 (model_v1.txt); LB 0.952 at t=0.3, old simulation 0.9557:")
    tune(v1, *ctx, out="_faithful_v2.json")

    pred = pl.read_parquet(P1)
    vp = pred.filter(pl.col("tid").is_in(val_t.implode()))
    print("[eval] test-like stage 1 (fold models); old simulation 0.9605 at t=0.6:")
    tune(vp, *ctx, out="_faithful_s1.json")

    orig_norm = stage2._norm
    stage2._norm = lambda split: (lambda s, t: (s.filter(~pl.col("entity_id").is_in(drop.implode())), t))(*orig_norm(split))
    feats = stage2.group_features(pred, "train")
    stage2._norm = orig_norm
    v4 = stage2.rescore(vp, feats.filter(pl.col("tid").is_in(val_t.implode())),
                        lgb.Booster(model_file=str(WORK_DIR / "stage2_v4.txt")))
    print("[eval] v4 = stage 2 (stage2_v4.txt); LB 0.955 at t=0.4, old simulation 0.9699:")
    tune(v4, *ctx, out="_faithful_v4.json")
    for f in ("_faithful_v2.json", "_faithful_s1.json", "_faithful_v4.json"):
        (WORK_DIR / f).unlink(missing_ok=True)


def main() -> None:
    t0 = time.time()
    truth_all = load_ground_truth("train")
    drop = pl.Series(list(dropped_s1(truth_all)))
    block_faithful(drop)
    cands = load_cands()
    p1_faithful(cands)
    evaluate(cands, truth_all, drop)
    print(f"[faithful] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
