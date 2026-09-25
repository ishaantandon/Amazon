"""End-to-end CLI.

  python src/run.py --split train --stage all    # prep -> block -> train + validate
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
    decision = json.loads((WORK_DIR / "decision.json").read_text())
    seen = set(pl.read_parquet(work_path("train", "s1.parquet"), columns=["country"])["country"].unique())
    pred = predict_split(split)
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
    ap.add_argument("--stage", choices=["prep", "block", "train", "predict", "all"], default="all")
    a = ap.parse_args()
    stages = ["prep", "block", "train" if a.split == "train" else "predict"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "prep":
            prep(a.split)
        elif st == "block":
            if not work_path(a.split, "cands.parquet").exists():
                block(a.split)
        elif st == "train":
            from train import main as train_main
            train_main()
        elif st == "predict":
            write_outputs(a.split)


if __name__ == "__main__":
    main()
