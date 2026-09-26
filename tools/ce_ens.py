"""v7: stage 2 with BOTH cross-encoder scores (e5-small from v5, e5-base from v6) as features.

Both cross-encoders were fine-tuned on pool half A only and scored the same half-B + validation pairs
(work/ce/scores.parquet, work/ce_base/scores.parquet) and the same in-play test pairs
(work/test/ce_scores.parquet, work/test/ce_base_scores.parquet), so stage 2 stays leak-free.

  .venv\\Scripts\\python tools\\ce_ens.py [fit] [write]     # default: both
Outputs: work/stage2_ens.txt, work/decision2_ens.json, work/test/pred2_ens.parquet, work/output_v7_ens/
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

TAGS = ["ce", "ce_base"]
STAGE2 = WORK_DIR / "stage2_ens.txt"
DECISION = WORK_DIR / "decision2_ens.json"
PRED2 = work_path("test", "pred2_ens.parquet")
OUT_DIR = WORK_DIR / "output_v7_ens"
UNSEEN_THRESHOLD = 0.6


def _cols():
    from stage2 import FEATURES2
    return FEATURES2 + [f"{t}_{k}" for t in TAGS for k in ("p", "logit")]


def _add(feats: pl.DataFrame, files: dict) -> pl.DataFrame:
    for t in TAGS:
        c = pl.col(f"{t}_p").clip(1e-6, 1 - 1e-6)
        feats = feats.join(pl.read_parquet(files[t]).select("tid", "s1", pl.col("ce").alias(f"{t}_p")),
                           on=["tid", "s1"], how="inner").with_columns((c.log() - (1 - c).log()).alias(f"{t}_logit"))
    return feats


def fill_small() -> Path:
    """e5-small scores for the half-B pairs v6 sampled (the two runs sampled B differently)."""
    out = WORK_DIR / "ce" / "scores_for_base_pairs.parquet"
    if out.exists():
        return out
    import ce_pilot
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    need = pl.read_parquet(WORK_DIR / "ce_base" / "scores.parquet").select("tid", "s1", "part")
    have = pl.read_parquet(WORK_DIR / "ce" / "scores.parquet").select("tid", "s1", "ce")
    miss = need.join(have, on=["tid", "s1"], how="anti")
    print(f"[ens] scoring {len(miss):,} pairs with e5-small", flush=True)
    t_txt, s_txt = ce_pilot._texts("train")
    mdir = WORK_DIR / "ce" / "model"
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda()
    ce = ce_pilot._score(model, tok, t_txt, s_txt, miss["tid"].to_list(), miss["s1"].to_list())
    del model
    torch.cuda.empty_cache()
    extra = miss.select("tid", "s1").with_columns(pl.Series("ce", ce))
    need.join(pl.concat([have, extra]), on=["tid", "s1"], how="left").write_parquet(out)
    return out


def fit_stage2() -> None:
    if STAGE2.exists():
        return
    small = fill_small()
    import stage2
    from data_io import load_ground_truth, load_links
    from faithful_sim import CANDS, P1
    from stage2 import PARAMS2
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
    part = pl.read_parquet(WORK_DIR / "ce_base" / "scores.parquet").select("tid", "s1", "part")
    feats = _add(feats.join(part, on=["tid", "s1"], how="inner"),
                 {"ce": small, "ce_base": WORK_DIR / "ce_base" / "scores.parquet"})
    cols = _cols()
    tr = feats.filter(pl.col("part") == "B").join(links.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")),
                                                  on=["tid", "s1"], how="left").with_columns(pl.col("y").fill_null(0))
    model = fit(tr.select(cols).to_numpy(), tr["y"].to_numpy(), tr["tid"], cols, PARAMS2, 3000)
    model.save_model(str(STAGE2))
    va = feats.filter(pl.col("part") == "val")
    p2 = va.select("tid", "s1").with_columns(
        pl.Series("p2", model.predict(va.select(cols).to_numpy(), num_threads=0), dtype=pl.Float32))
    vp = pred.filter(pl.col("tid").is_in(val_t.implode()))
    out = vp.join(p2, on=["tid", "s1"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
    print("[ens] stage 2 with both cross-encoders:", flush=True)
    tune(out, *ctx, out=DECISION.name)


def write_test() -> None:
    from assign import assign
    from data_io import load_source, write_id_lists
    from stage2 import group_features
    pred = pl.read_parquet(work_path("test", "pred.parquet"))
    if not PRED2.exists():
        feats = group_features(pred, "test")
        n = len(feats)
        feats = _add(feats, {t: work_path("test", f"{t}_scores.parquet") for t in TAGS})
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
