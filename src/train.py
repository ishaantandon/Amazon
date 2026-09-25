"""Stage 3: train the pair classifier (LightGBM) and validate on held-out S1 entities.

Validation is realistic: a random VAL_FRAC of train S1 entities is held out, and every target
that has any held-out S1 among its candidates is excluded from training. Those targets are
scored with all their candidates (including non-held-out S1s, so the argmax competition is
the real one), and the metric is computed over the held-out S1 entities.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from assign import assign
from config import SEED, WORK_DIR, work_path
from data_io import load_ground_truth, load_links
from features import FEATURES, add_global_context, build
from metrics import evaluate, fmt

VAL_FRAC = 0.02
N_TRAIN_TARGETS = 1_500_000
CHUNK_TARGETS = 500_000
MODEL_PATH = WORK_DIR / "model.txt"
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=SEED, num_threads=0)


def featurize(cands: pl.DataFrame, split: str, tids: pl.Series) -> pl.DataFrame:
    s1 = pl.read_parquet(work_path(split, "s1.parquet"))
    t = pl.read_parquet(work_path(split, "t.parquet"))
    tids = tids.unique().sort()
    out = []
    for a in range(0, len(tids), CHUNK_TARGETS):
        part = cands.filter(pl.col("tid").is_in(tids.slice(a, CHUNK_TARGETS).implode()))
        out.append(build(part, s1, t))
    return pl.concat(out)


def predict(model: lgb.Booster, feats: pl.DataFrame) -> pl.DataFrame:
    p = model.predict(feats.select(FEATURES).to_numpy(), num_threads=0)
    return feats.select("tid", "s1", pl.col("t_country").alias("country")).with_columns(
        pl.Series("p", p, dtype=pl.Float32))


def main() -> None:
    t0 = time.time()
    cands = add_global_context(pl.read_parquet(work_path("train", "cands.parquet")))
    links = load_links("train").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    truth_all = load_ground_truth("train")

    rng = np.random.default_rng(SEED)
    s1_ids = np.array(sorted(truth_all))
    val_s1 = set(s1_ids[rng.random(len(s1_ids)) < VAL_FRAC])
    val_s1_ser = pl.Series(list(val_s1))
    val_t = cands.filter(pl.col("s1").is_in(val_s1_ser.implode()))["tid"].unique()
    # Targets truly linked to a held-out S1 are validation targets even if blocking missed them.
    pool = cands.select("tid").unique().filter(~pl.col("tid").is_in(val_t.implode()))
    pool = pool.join(links.select("tid", "s1"), on="tid", how="left").filter(
        ~pl.col("s1").is_in(val_s1_ser.implode()).fill_null(False))["tid"]
    train_t = pool.sample(min(N_TRAIN_TARGETS, len(pool)), seed=SEED)
    print(f"[train] val S1 {len(val_s1):,}, val targets {len(val_t):,}, train targets {len(train_t):,}")

    tr = featurize(cands, "train", train_t).join(links, on=["tid", "s1"], how="left").with_columns(
        pl.col("y").fill_null(0))
    print(f"[train] train pairs {len(tr):,}, positives {tr['y'].sum():,} ({time.time() - t0:.0f}s)")
    is_es = (tr["tid"].hash(SEED) % 10 == 0).to_numpy()
    X, y = tr.select(FEATURES).to_numpy(), tr["y"].to_numpy()
    dtr = lgb.Dataset(X[~is_es], y[~is_es], feature_name=FEATURES)
    des = lgb.Dataset(X[is_es], y[is_es], reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(str(MODEL_PATH))
    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    print("[train] top features:", ", ".join(f"{f}={g:.0f}" for f, g in imp[:15]))
    del tr, X, dtr, des

    va = featurize(cands, "train", val_t)
    pred = predict(model, va)
    pred.write_parquet(work_path("train", "val_pred.parquet"))
    truth = {s: truth_all[s] for s in val_s1}
    cand_sets = {}
    for s1, tids in cands.filter(pl.col("s1").is_in(val_s1_ser.implode())).group_by("s1").agg("tid").iter_rows():
        cand_sets[s1] = set(tids)
    s1c = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id", "country"])
    country = dict(zip(s1c["entity_id"], s1c["country"]))
    tune(pred, truth, cand_sets, {s: country[s] for s in truth})
    print(f"[train] done in {time.time() - t0:.0f}s")


def tune(pred: pl.DataFrame, truth: dict, cand_sets: dict, country: dict) -> dict:
    results = {}
    for ef in (False, True):
        for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            m = evaluate(truth, _to_sets(assign(pred, th, ef), truth), cand_sets)
            results[(ef, th)] = m
            print(f"  expected_f={ef} t={th}: {fmt(m)}")
    (ef, th), m = max(results.items(), key=lambda kv: kv[1]["macro_f05"])
    print(f"[train] best: expected_f={ef} threshold={th} macro_f05={m['macro_f05']:.5f}")
    for c in sorted(set(country.values())):
        sub = {s: v for s, v in truth.items() if country[s] == c}
        print(f"    {c}: {fmt(evaluate(sub, _to_sets(assign(pred, th, ef), sub), cand_sets))}")
    (WORK_DIR / "decision.json").write_text(json.dumps({"threshold": th, "expected_f": ef}))
    return {"threshold": th, "expected_f": ef}


def _to_sets(links: dict, keep: dict) -> dict:
    return {s: set(v) for s, v in links.items() if s in keep}


if __name__ == "__main__":
    main()
