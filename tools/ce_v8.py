"""v8: v5 (stage 2 + e5-small cross-encoder) plus address-crowding and cross-encoder competition features.

Why: France (no labels) scores ~0.94 on the leaderboard while US/India match the simulation (~0.985).
13% of France S1s share their exact address with another S1 (US/India ~5%), so "same address" is weak
evidence there. These language-neutral features let stage 2 learn, from the shared-address cases that
do exist in US/India, when to trust the address and when the name must decide:
  s1_addr_n / s1_street_n : S1s of the country with the same address+numbers / the same street words
  t_addr_n                : S1s whose address+numbers equal the target's
  same_addr               : target and S1 share address+numbers
  ce_gap, ce_rank, ce_best_other : the pair's cross-encoder score against the target's other in-play candidates

Validation also reports the held-out S1s whose exact address is shared (the closest proxy for France).
Phases: fit (-> work/stage2_v8.txt, decision2_v8.json), write (-> work/output_v8_addr/).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb
import polars as pl

from config import WORK_DIR, work_path

STAGE2 = WORK_DIR / "stage2_v8.txt"
DECISION = WORK_DIR / "decision2_v8.json"
PRED2 = work_path("test", "pred2_v8.parquet")
OUT_DIR = WORK_DIR / "output_v8_addr"
UNSEEN_THRESHOLD = 0.6
EXTRA = ["ce", "ce_logit", "ce_gap", "ce_rank", "ce_best_other",
         "s1_addr_n", "s1_street_n", "t_addr_n", "same_addr"]


def _cols():
    from stage2 import FEATURES2
    return FEATURES2 + EXTRA


def _keys(split: str, s1_keep=None):
    cols = ["entity_id", "country", "addr", "nums"]
    s1 = pl.read_parquet(work_path(split, "s1.parquet"), columns=cols)
    if s1_keep is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(s1_keep.implode()))
    t = pl.read_parquet(work_path(split, "t.parquet"), columns=cols)
    key = lambda: pl.when(pl.col("addr") == "").then(None).otherwise(pl.col("addr") + "|" + pl.col("nums"))
    s1 = s1.with_columns(key().alias("akey"), pl.when(pl.col("addr") == "").then(None).otherwise(pl.col("addr")).alias("skey"))
    t = t.with_columns(key().alias("akey"))
    return s1, t


def add_features(feats: pl.DataFrame, ce: pl.DataFrame, split: str, s1_keep=None) -> pl.DataFrame:
    """feats: stage-2 group features (tid, s1, ...); ce: tid, s1, ce for every in-play pair."""
    s1, t = _keys(split, s1_keep)
    an = s1.filter(pl.col("akey").is_not_null()).group_by("country", "akey").len("s1_addr_n")
    sn = s1.filter(pl.col("skey").is_not_null()).group_by("country", "skey").len("s1_street_n")
    s1f = (s1.join(an, on=["country", "akey"], how="left").join(sn, on=["country", "skey"], how="left")
           .select(pl.col("entity_id").alias("s1"), pl.col("akey").alias("s_akey"),
                   pl.col("s1_addr_n").fill_null(0), pl.col("s1_street_n").fill_null(0)))
    tf = (t.join(an, on=["country", "akey"], how="left")
          .select(pl.col("entity_id").alias("tid"), pl.col("akey").alias("t_akey"),
                  pl.col("s1_addr_n").fill_null(0).alias("t_addr_n")))
    c = pl.col("ce").clip(1e-6, 1 - 1e-6)
    ce = ce.select("tid", "s1", "ce").with_columns(
        (c.log() - (1 - c).log()).alias("ce_logit"),
        (pl.col("ce").max().over("tid") - pl.col("ce")).alias("ce_gap"),
        pl.col("ce").rank("ordinal", descending=True).over("tid").cast(pl.Int16).alias("ce_rank"),
    )
    top2 = ce.group_by("tid").agg(pl.col("ce").sort(descending=True).head(2).alias("_t"))
    ce = ce.join(top2, on="tid").with_columns(
        pl.when(pl.col("ce_rank") == 1).then(pl.col("_t").list.get(1, null_on_oob=True))
        .otherwise(pl.col("_t").list.first()).fill_null(0.0).alias("ce_best_other")).drop("_t")
    out = (feats.join(ce, on=["tid", "s1"], how="inner").join(s1f, on="s1", how="left").join(tf, on="tid", how="left")
           .with_columns((pl.col("s_akey") == pl.col("t_akey")).fill_null(False).cast(pl.Int8).alias("same_addr"))
           .drop("s_akey", "t_akey"))
    return out


def fit_stage2() -> None:
    if STAGE2.exists():
        return
    import stage2
    from assign import assign
    from data_io import load_ground_truth, load_links
    from faithful_sim import CANDS, P1
    from metrics import evaluate, fmt
    from stage2 import PARAMS2
    from train import _to_sets, dropped_s1, fit, tune, val_context, val_split
    truth_all = load_ground_truth("train")
    drop = pl.Series(list(dropped_s1(truth_all)))
    keep = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id"]).filter(
        ~pl.col("entity_id").is_in(drop.implode()))["entity_id"]
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
    sc = pl.read_parquet(WORK_DIR / "ce" / "scores.parquet")
    feats = add_features(feats.join(sc.select("tid", "s1", "part"), on=["tid", "s1"], how="inner"), sc, "train", keep)
    cols = _cols()
    tr = feats.filter(pl.col("part") == "B").join(links.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")),
                                                  on=["tid", "s1"], how="left").with_columns(pl.col("y").fill_null(0))
    va = feats.filter(pl.col("part") == "val")
    vp = pred.filter(pl.col("tid").is_in(val_t.implode()))
    # Held-out S1s whose exact address is shared with another kept S1: the France-like subset.
    s1k, _ = _keys("train", keep)
    shared = set(s1k.filter(pl.col("akey").is_not_null()).filter(pl.len().over("country", "akey") > 1)["entity_id"].to_list())
    truth, cand_sets, country = ctx
    sub = {s: v for s, v in truth.items() if s in shared}
    print(f"[v8] held-out S1s at a shared exact address: {len(sub):,} of {len(truth):,}", flush=True)
    for name, cs in [("v5 features", [c for c in cols if c in stage2.FEATURES2 + ["ce", "ce_logit"]]),
                     ("v8 features", cols)]:
        model = fit(tr.select(cs).to_numpy(), tr["y"].to_numpy(), tr["tid"], cs, PARAMS2, 3000)
        p2 = va.select("tid", "s1").with_columns(
            pl.Series("p2", model.predict(va.select(cs).to_numpy(), num_threads=0), dtype=pl.Float32))
        out = vp.join(p2, on=["tid", "s1"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
        print(f"[v8] {name}:", flush=True)
        dec = tune(out, *ctx, out=DECISION.name)
        print(f"    shared-address subset: {fmt(evaluate(sub, _to_sets(assign(out, dec['threshold'], dec['expected_f']), sub), cand_sets))}",
              flush=True)
    model.save_model(str(STAGE2))  # the last fitted model = v8 features; DECISION holds its tuned rule


def write_test() -> None:
    from assign import assign
    from data_io import load_source, write_id_lists
    from stage2 import group_features
    pred = pl.read_parquet(work_path("test", "pred.parquet"))
    if not PRED2.exists():
        feats = group_features(pred, "test")
        n = len(feats)
        feats = add_features(feats, pl.read_parquet(work_path("test", "ce_scores.parquet")), "test")
        assert len(feats) == n, f"{n - len(feats)} in-play test pairs lack a cross-encoder score"
        cols = _cols()
        model = lgb.Booster(model_file=str(STAGE2))
        p2 = feats.select("tid", "s1").with_columns(
            pl.Series("p2", model.predict(feats.select(cols).to_numpy(), num_threads=0), dtype=pl.Float32))
        (pred.join(p2, on=["tid", "s1"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
         .write_parquet(PRED2))
    pred2 = pl.read_parquet(PRED2)
    decision = json.loads(DECISION.read_text())
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


PHASES = {"fit": fit_stage2, "write": write_test}

if __name__ == "__main__":
    for ph in (sys.argv[1:] or list(PHASES)):
        t0 = time.time()
        PHASES[ph]()
        print(f"[{ph}] done ({time.time() - t0:.0f}s)", flush=True)
