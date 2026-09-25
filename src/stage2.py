"""Stage 3b: group-aware re-scoring.

Stage 1 (train.py) judges each (target, S1) pair on its own. Here every pair is re-scored with
context from the other targets competing for, or already confidently assigned to, the same S1:

  * group strength: how many other targets confidently pick this S1 (and the runner-up S1)
  * sibling agreement: similarity of this target's name/address/numbers to those siblings
  * name rarity: how many S1 records share the S1's / the target's phonetic skeleton

Leak-free training: stage-1 probabilities for every train target come from 2-fold cross-fitting
(each fold's model never saw that target), so group features see honest scores for all members
of every S1. Held-out validation uses the same S1 split as train.py.

  python src/stage2.py oof      # out-of-fold stage-1 predictions for all train targets
  python src/stage2.py train    # fit stage 2, validate, tune threshold -> decision2.json
  python src/stage2.py test     # apply stage 2 to test stage-1 predictions -> pred2.parquet
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from assign import assign
from config import SEED, WORK_DIR, work_path
from data_io import load_ground_truth, load_links
from features import FEATURES
from metrics import evaluate, fmt
from train import (N_TRAIN_TARGETS, PARAMS, featurize, fit, load_train_cands, predict, tune, val_context,
                   val_split)

MIN_P = 0.01            # pairs below this stage-1 probability keep their stage-1 score
CONF_P = 0.5            # a target "confidently" belongs to its argmax S1 above this
MAX_SIBLINGS = 6
N_STAGE2_TARGETS = 4_000_000
STAGE2_PATH = WORK_DIR / "stage2.txt"
PARAMS2 = dict(PARAMS, num_leaves=63, learning_rate=0.05)


# ---------------------------------------------------------------- stage-1 out-of-fold

def oof() -> None:
    t0 = time.time()
    truth_all = load_ground_truth("train")
    cands = load_train_cands(truth_all)
    links = load_links("train").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    val_s1, val_t, pool = val_split(cands, links, truth_all)
    tids = cands["tid"].unique().sort()
    fold_of = lambda s: (s.hash(SEED + 100) % 2).cast(pl.Int8)  # independent of fit()'s early-stopping hash
    parts = []
    for k in (0, 1):
        tr_pool = pool.filter(fold_of(pool) != k)
        train_t = tr_pool.sample(min(N_TRAIN_TARGETS, len(tr_pool)), seed=SEED + k)
        tr = featurize(cands, "train", train_t).join(links, on=["tid", "s1"], how="left").with_columns(
            pl.col("y").fill_null(0))
        print(f"[oof] fold {k}: train pairs {len(tr):,} ({time.time() - t0:.0f}s)")
        model = fit(tr.select(FEATURES).to_numpy(), tr["y"].to_numpy(), tr["tid"], FEATURES)
        model.save_model(str(WORK_DIR / f"stage1_fold{k}.txt"))
        del tr
        mine = tids.filter(fold_of(tids) == k)
        step = 2_000_000
        for a in range(0, len(mine), step):
            parts.append(predict(model, featurize(cands, "train", mine.slice(a, step))))
            print(f"[oof] fold {k}: predicted {min(a + step, len(mine)):,}/{len(mine):,} ({time.time() - t0:.0f}s)")
    pl.concat(parts).write_parquet(work_path("train", "p1_oof.parquet"))
    print(f"[oof] done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- group features

def _norm(split: str):
    cols = ["entity_id", "country", "name", "skel", "addr", "nums", "addr_empty"]
    return (pl.read_parquet(work_path(split, "s1.parquet"), columns=cols),
            pl.read_parquet(work_path(split, "t.parquet"), columns=cols))


def group_features(pred: pl.DataFrame, split: str) -> pl.DataFrame:
    """pred: tid, s1, country, p (stage 1). Returns features for in-play pairs (p >= MIN_P)."""
    t0 = time.time()
    s1n, tn = _norm(split)
    P = pred.with_columns(pl.col("p").rank("ordinal", descending=True).over("tid").cast(pl.Int8).alias("rank"))
    best = P.filter(pl.col("rank") == 1)
    g = best.group_by("s1").agg(
        (pl.col("p") >= CONF_P).sum().cast(pl.Int32).alias("g_n"),
        (pl.col("p") >= 0.9).sum().cast(pl.Int32).alias("g_n9"),
        pl.col("p").sum().cast(pl.Float32).alias("g_sum"),
    )
    tgt = P.group_by("tid").agg(
        pl.col("p").max().alias("p_top1"),
        pl.col("p").sort(descending=True).get(1, null_on_oob=True).fill_null(0.0).alias("p_top2"),
    )
    Q = (P.filter(pl.col("p") >= MIN_P)
         .join(g, on="s1", how="left").join(tgt, on="tid", how="left")
         .with_columns(pl.col("g_n", "g_n9").fill_null(0), pl.col("g_sum").fill_null(0.0)))
    self_best = pl.col("rank") == 1
    Q = Q.with_columns(
        # Group stats excluding this target's own contribution.
        (pl.col("g_n") - (self_best & (pl.col("p") >= CONF_P)).cast(pl.Int32)).alias("g_n_ex"),
        (pl.col("g_n9") - (self_best & (pl.col("p") >= 0.9)).cast(pl.Int32)).alias("g_n9_ex"),
        (pl.col("g_sum") - pl.when(self_best).then(pl.col("p")).otherwise(0.0)).alias("g_sum_ex"),
        (pl.col("p_top1") - pl.col("p")).alias("p_gap"),
        pl.when(self_best).then(pl.col("p_top2")).otherwise(pl.col("p_top1")).alias("p_best_other"),
        (pl.col("p").clip(1e-6, 1 - 1e-6).log() - (1 - pl.col("p").clip(1e-6, 1 - 1e-6)).log()).alias("logit"),
    ).drop("g_n", "g_n9", "g_sum")
    # Strongest competing S1 group for the same target.
    alt = Q.group_by("tid").agg(pl.col("g_n_ex").sort(descending=True).alias("_gs"))
    Q = Q.join(alt, on="tid").with_columns(
        pl.when(pl.col("_gs").list.first() == pl.col("g_n_ex"))
        .then(pl.col("_gs").list.get(1, null_on_oob=True)).otherwise(pl.col("_gs").list.first())
        .fill_null(0).alias("alt_g_n")
    ).drop("_gs")
    print(f"[group] {len(Q):,} in-play pairs of {len(P):,} ({time.time() - t0:.0f}s)")

    # Sibling agreement: compare the target with up to MAX_SIBLINGS confident members of the S1.
    sib = (best.filter(pl.col("p") >= CONF_P).sort("p", descending=True)
           .group_by("s1").head(MAX_SIBLINGS).select("s1", pl.col("tid").alias("sib")))
    txt = tn.select("entity_id", "name", pl.concat_str("addr", "nums", separator=" ").alias("an"), "nums")
    sims = []
    q_keys = Q.select("tid", "s1")
    step = 3_000_000
    for a in range(0, len(q_keys), step):
        j = (q_keys.slice(a, step).join(sib, on="s1").filter(pl.col("tid") != pl.col("sib"))
             .join(txt, left_on="tid", right_on="entity_id")
             .join(txt, left_on="sib", right_on="entity_id", suffix="_b"))
        if len(j) == 0:
            continue
        an, anb = j["an"].to_list(), j["an_b"].to_list()
        j = j.with_columns(
            pl.Series("sa", process.cpdist(an, anb, scorer=fuzz.token_set_ratio, workers=-1) / 100, dtype=pl.Float32),
            pl.Series("sn", process.cpdist(j["name"].to_list(), j["name_b"].to_list(), scorer=fuzz.token_set_ratio,
                                           workers=-1) / 100, dtype=pl.Float32),
            pl.Series("snum", process.cpdist(j["nums"].to_list(), j["nums_b"].to_list(), scorer=fuzz.token_set_ratio,
                                             workers=-1) / 100, dtype=pl.Float32),
            ((pl.col("an") == pl.col("an_b")) & (pl.col("an").str.strip_chars() != "")).alias("same"),
        )
        sims.append(j.group_by("tid", "s1").agg(
            pl.len().cast(pl.Int16).alias("sib_k"),
            pl.col("sa").max().alias("sib_addr_max"), pl.col("sa").mean().alias("sib_addr_mean"),
            pl.col("sn").max().alias("sib_name_max"), pl.col("sn").mean().alias("sib_name_mean"),
            pl.col("snum").max().alias("sib_num_max"),
            pl.col("same").any().cast(pl.Int8).alias("sib_same_addr"),
        ))
    Q = Q.join(pl.concat(sims), on=["tid", "s1"], how="left").with_columns(pl.col("sib_k").fill_null(0))
    print(f"[group] sibling features done ({time.time() - t0:.0f}s)")

    # Name rarity within the country's S1 records.
    skel_freq = s1n.group_by("country", "skel").len("skel_freq")
    name_freq = s1n.group_by("country", "name").len("s1_name_freq")
    s1f = (s1n.join(skel_freq, on=["country", "skel"]).join(name_freq, on=["country", "name"])
           .select("entity_id", pl.col("skel_freq").alias("s1_skel_freq"), "s1_name_freq"))
    tf = (tn.select("entity_id", "country", "skel", "addr_empty", "nums")
          .join(skel_freq, on=["country", "skel"], how="left")
          .select("entity_id", pl.col("skel_freq").fill_null(0).alias("t_skel_freq"),
                  pl.col("addr_empty").cast(pl.Int8).alias("t_addr_empty"),
                  (pl.col("nums") != "").cast(pl.Int8).alias("t_has_num")))
    Q = (Q.join(s1f, left_on="s1", right_on="entity_id", how="left")
         .join(tf, left_on="tid", right_on="entity_id", how="left")
         .with_columns(pl.col("tid").str.starts_with("S3").cast(pl.Int8).alias("is_s3")))
    print(f"[group] done: {len(Q):,} rows ({time.time() - t0:.0f}s)")
    return Q


FEATURES2 = [
    "p", "logit", "rank", "p_gap", "p_best_other", "g_n_ex", "g_n9_ex", "g_sum_ex", "alt_g_n",
    "sib_k", "sib_addr_max", "sib_addr_mean", "sib_name_max", "sib_name_mean", "sib_num_max", "sib_same_addr",
    "s1_skel_freq", "s1_name_freq", "t_skel_freq", "t_addr_empty", "t_has_num", "is_s3",
]


def rescore(pred: pl.DataFrame, feats: pl.DataFrame, model: lgb.Booster) -> pl.DataFrame:
    """Stage-2 probability for in-play pairs; all other pairs keep their stage-1 score."""
    p2 = feats.select("tid", "s1").with_columns(
        pl.Series("p2", model.predict(feats.select(FEATURES2).to_numpy(), num_threads=0), dtype=pl.Float32))
    return (pred.join(p2, on=["tid", "s1"], how="left")
            .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))


# ---------------------------------------------------------------- train / test

def train() -> None:
    t0 = time.time()
    pred = pl.read_parquet(work_path("train", "p1_oof.parquet"))
    links = load_links("train").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    truth_all = load_ground_truth("train")
    cands = load_train_cands(truth_all).select("tid", "s1")
    val_s1, val_t, pool = val_split(cands, links, truth_all)
    ctx = val_context(cands, truth_all, val_s1)
    del cands

    feats = group_features(pred, "train").join(links, on=["tid", "s1"], how="left").with_columns(
        pl.col("y").fill_null(0))
    train_t = pool.sample(min(N_STAGE2_TARGETS, len(pool)), seed=SEED + 7)
    tr = feats.filter(pl.col("tid").is_in(train_t.implode()))
    print(f"[stage2] train rows {len(tr):,}, positives {tr['y'].sum():,}")
    model = fit(tr.select(FEATURES2).to_numpy(), tr["y"].to_numpy(), tr["tid"], FEATURES2, PARAMS2, 3000)
    model.save_model(str(STAGE2_PATH))
    del tr

    val_pred = pred.filter(pl.col("tid").is_in(val_t.implode()))
    print("[stage2] stage-1 (out-of-fold) baseline on the same held-out entities:")
    tune(val_pred, *ctx, out="decision_oof_stage1.json")
    val2 = rescore(val_pred, feats.filter(pl.col("tid").is_in(val_t.implode())), model)
    val2.write_parquet(work_path("train", "val_pred2.parquet"))
    print("[stage2] stage 2:")
    tune(val2, *ctx, out="decision2.json")
    print(f"[stage2] done in {time.time() - t0:.0f}s")


def test() -> None:
    pred = pl.read_parquet(work_path("test", "pred.parquet"))
    model = lgb.Booster(model_file=str(STAGE2_PATH))
    rescore(pred, group_features(pred, "test"), model).write_parquet(work_path("test", "pred2.parquet"))
    print("[stage2] wrote test pred2.parquet")


if __name__ == "__main__":
    {"oof": oof, "train": train, "test": test}[sys.argv[1]]()
