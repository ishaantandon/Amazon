"""End-to-end CLI.

  python src/run.py --split train --stage all    # prep -> block -> train -> stage2 (+ validate)
  python src/run.py --split test  --stage all    # prep -> block -> predict -> output/*.tsv

Test predictions use the model and decision threshold learned on train. Countries unseen in
training (France) get UNSEEN_THRESHOLD: the top of the threshold range that validation showed
is flat for macro F0.5 (0.2-0.6), i.e. the most precision-cautious setting that costs almost
nothing if the model is calibrated there.
"""
import argparse
import json
import time

import lightgbm as lgb
import polars as pl

from assign import assign, best_links
from blocking import block
from config import OUTPUT_DIR, WORK_DIR, work_path
from data_io import load_source, write_id_lists
from features import add_global_context
from prep import prep
from stage2 import STAGE2_PATH, group_features, rescore
from train import CHUNK_TARGETS, MODEL_PATH, featurize, predict

UNSEEN_THRESHOLD = 0.6


def predict_split(split: str) -> pl.DataFrame:
    out = work_path(split, "pred.parquet")
    if out.exists():
        return pl.read_parquet(out)
    model = lgb.Booster(model_file=str(MODEL_PATH))
    cands = add_global_context(pl.read_parquet(work_path(split, "cands.parquet")))
    tids = cands["tid"].unique().sort()
    parts = []
    t0 = time.time()
    step = CHUNK_TARGETS * 4
    for a in range(0, len(tids), step):
        parts.append(predict(model, featurize(cands, split, tids.slice(a, step))))
        print(f"[predict] {min(a + step, len(tids)):,}/{len(tids):,} targets ({time.time() - t0:.0f}s)")
    pred = pl.concat(parts)
    pred.write_parquet(out)
    return pred


def country_thresholds(pred: pl.DataFrame, base: float, seen: set[str]) -> dict[str, float]:
    best = best_links(pred)
    rate = best.group_by("country").agg((pl.col("p") >= base).mean().alias("r"))
    th = {}
    for c, r in rate.iter_rows():
        th[c] = base if c in seen else max(base, UNSEEN_THRESHOLD)
        print(f"[predict] {c}: accepted rate at base {r:.3f} -> threshold {th[c]:.3f}")
    return th


def write_outputs(split: str) -> None:
    seen = set(pl.read_parquet(work_path("train", "s1.parquet"), columns=["country"])["country"].unique())
    pred = predict_split(split)
    if STAGE2_PATH.exists():
        # Group-aware re-scoring (stage2.py) with its own tuned decision rule.
        p2 = work_path(split, "pred2.parquet")
        if not p2.exists():
            rescore(pred, group_features(pred, split), lgb.Booster(model_file=str(STAGE2_PATH))).write_parquet(p2)
        pred = pl.read_parquet(p2)
        decision = json.loads((WORK_DIR / "decision2.json").read_text())
        print("[predict] using stage-2 scores")
    else:
        decision = json.loads((WORK_DIR / "decision.json").read_text())
    th = country_thresholds(pred, decision["threshold"], seen)
    matches = {}
    for c, t in th.items():
        matches.update(assign(pred.filter(pl.col("country") == c), t, decision["expected_f"]))

    s1_ids = load_source(split, 1)["entity_id"].to_list()
    cands = pl.read_parquet(work_path(split, "cands.parquet"), columns=["tid", "s1"])
    cand_lists = dict(cands.group_by("s1").agg("tid").iter_rows())
    write_id_lists(OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, cand_lists, "candidate_entity_ids")
    write_id_lists(OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    n_links = sum(len(v) for v in matches.values())
    print(f"[output] {len(s1_ids):,} S1 rows, {len(matches):,} with matches, {n_links:,} links, "
          f"{len(cands):,} candidate pairs -> {OUTPUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--stage", choices=["prep", "block", "train", "stage2", "predict", "all"], default="all")
    a = ap.parse_args()
    train_stages = ["prep", "block", "train", "stage2"]
    stages = (train_stages if a.split == "train" else ["prep", "block", "predict"]) if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "prep":
            prep(a.split)
        elif st == "block":
            if not work_path(a.split, "cands.parquet").exists():
                block(a.split)
        elif st == "train":
            from train import main as train_main
            train_main()
        elif st == "stage2":
            import stage2
            if not work_path("train", "p1_oof.parquet").exists():
                stage2.oof()
            stage2.train()
        elif st == "predict":
            write_outputs(a.split)


if __name__ == "__main__":
    main()
