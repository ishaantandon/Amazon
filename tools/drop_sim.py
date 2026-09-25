"""Simulate the test regime on validation: remove a fraction of S1 entities from the candidate pool.
Their S2/S3 records stay as distractors (with siblings), exactly as if the S1 had been dropped
from the test source 1 file."""
import sys, json, polars as pl, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import work_path, WORK_DIR, SEED
from data_io import load_ground_truth
from assign import assign
from metrics import evaluate

path = sys.argv[1] if len(sys.argv) > 1 else "val_pred.parquet"
pred = pl.read_parquet(work_path("train", path))
truth_all = load_ground_truth("train")
ids = np.array(sorted(truth_all))
rng = np.random.default_rng(SEED)
val = set(ids[rng.random(len(ids)) < 0.02])
s1c = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id", "country"])
country = dict(zip(s1c["entity_id"], s1c["country"]))
rng2 = np.random.default_rng(123)
for drop in [0.0, 0.19]:
    D = set(ids[rng2.random(len(ids)) < drop]) if drop else set()
    keep_val = [s for s in val if s not in D]
    truth = {s: truth_all[s] for s in keep_val}
    p = pred.filter(~pl.col("s1").is_in(list(D))) if D else pred
    print(f"\n=== dropped {drop:.0%} of S1: {len(truth):,} val entities ===")
    for ef in (True, False):
        row = []
        for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            m = {s: set(v) for s, v in assign(p, th, ef).items() if s in truth}
            row.append(f"{th}:{evaluate(truth, m)['macro_f05']:.4f}")
        print(f"  expected_f={ef!s:5s} " + "  ".join(row))
    m = {s: set(v) for s, v in assign(p, 0.3, True).items() if s in truth}
    for c in ("US", "India"):
        sub = {s: v for s, v in truth.items() if country[s] == c}
        r = evaluate(sub, m)
        print(f"  {c:5s} (t=0.3 EF): F={r['macro_f05']:.4f} P={r['macro_precision']:.4f} R={r['macro_recall']:.4f} single_acc={r['singleton_accuracy']:.3f} singletons={r['singletons']}")
