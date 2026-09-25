"""Faithful dropped-S1 simulation: remove D from the candidate pool BEFORE computing rank/gap/context
features, then re-score validation targets with the stage-1 model. Mimics a test set whose source 1
lost some entities while their S2/S3 records stayed as distractors."""
import sys, time, polars as pl, numpy as np, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import work_path, SEED
from data_io import load_ground_truth, load_links
from features import add_global_context
from train import MODEL_PATH, featurize, predict, val_split
from assign import assign
from metrics import evaluate

DROP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.19
t0 = time.time()
truth_all = load_ground_truth("train")
ids = np.array(sorted(truth_all))
D = set(ids[np.random.default_rng(123).random(len(ids)) < DROP])
cands = pl.read_parquet(work_path("train", "cands.parquet"))
links = load_links("train")
val_s1, val_t, _ = val_split(cands, links, truth_all)
# Remaining candidates move up: recompute the prefilter rank without the dropped S1s, over the
# whole pool (so s1_top1_count is right), then keep the validation targets.
c = cands.filter(~pl.col("s1").is_in(list(D))).with_columns(
    pl.col("pf_score").rank("ordinal", descending=True).over("tid").cast(pl.UInt32).alias("pf_rank"))
del cands
c = add_global_context(c).filter(pl.col("tid").is_in(val_t.implode()))
pred = predict(lgb.Booster(model_file=str(MODEL_PATH)), featurize(c, "train", c["tid"]))
pred.write_parquet(work_path("train", f"val_pred_drop{int(DROP * 100)}.parquet"))
truth = {s: truth_all[s] for s in val_s1 if s not in D}
print(f"dropped {DROP:.0%}: {len(truth):,} val entities, {len(pred):,} pairs ({time.time() - t0:.0f}s)")
s1c = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id", "country"])
country = dict(zip(s1c["entity_id"], s1c["country"]))
for ef in (True, False):
    row = []
    for th in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = {s: set(v) for s, v in assign(pred, th, ef).items() if s in truth}
        row.append(f"{th}:{evaluate(truth, m)['macro_f05']:.4f}")
    print(f"  expected_f={ef!s:5s} " + "  ".join(row))
m = {s: set(v) for s, v in assign(pred, 0.3, True).items() if s in truth}
for cc in ("US", "India"):
    sub = {s: v for s, v in truth.items() if country[s] == cc}
    r = evaluate(sub, m)
    print(f"  {cc:5s} (t=0.3 EF): F={r['macro_f05']:.4f} P={r['macro_precision']:.4f} R={r['macro_recall']:.4f} single_acc={r['singleton_accuracy']:.3f} singletons={r['singletons']}")
