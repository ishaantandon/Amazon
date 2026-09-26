"""Label-free per-country diagnostic of a cached stage-2 prediction file.

Model-estimated (plug-in) macro F0.5 per country, using the same decision rule as assign():
if the model's own estimate for France is ~0.985 while the leaderboard implies ~0.94, France errors
are confident (miscalibrated); if the estimate is also low, the model knows France is ambiguous.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import polars as pl

from config import work_path

name, th = sys.argv[1], float(sys.argv[2])
pred = pl.scan_parquet(work_path("test", name)).filter(pl.col("p") >= 0.01).collect()
best = pred.sort("p", descending=True).unique("tid", keep="first")
s1 = pl.read_parquet(work_path("test", "s1.parquet"), columns=["entity_id", "country"])
nt = pl.read_parquet(work_path("test", "t.parquet"), columns=["country"]).group_by("country").len("targets")

acc = best.filter(pl.col("p") >= th)
miss = best.filter(pl.col("p") < th).group_by("s1").agg(pl.col("p").sum().alias("miss"))
g = (acc.sort(["s1", "p"], descending=[False, True])
     .join(miss, on="s1", how="left").with_columns(pl.col("miss").fill_null(0.0))
     .with_columns(pl.col("p").cum_sum().over("s1").alias("tp"),
                   pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                   (pl.col("p").sum().over("s1") + pl.col("miss")).alias("tot"))
     .with_columns((1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("tot") - pl.col("tp"))
                                            + (pl.col("k") - pl.col("tp")))).alias("f")))
per = g.group_by("s1").agg(
    pl.col("f").max().alias("fk"),
    ((1 - pl.col("p")).log().sum() - pl.col("miss").first()).exp().alias("f0"),
    pl.col("f").arg_max().alias("kbest"), pl.len().alias("n_acc"),
)
# S1s with no accepted link: predicted empty; expected score = P(no true match) = exp(-miss).
emp = miss.join(per.select("s1"), on="s1", how="anti").with_columns((-pl.col("miss")).exp().alias("ef"))
per = per.with_columns(pl.max_horizontal("fk", "f0").alias("ef"),
                       pl.when(pl.col("f0") >= pl.col("fk")).then(0).otherwise(pl.col("kbest") + 1).alias("kept"))
ef = pl.concat([per.select("s1", pl.col("ef").cast(pl.Float64), pl.col("kept").cast(pl.Int64)),
                emp.select("s1", pl.col("ef").cast(pl.Float64), pl.lit(0, dtype=pl.Int64).alias("kept"))])
ef = s1.rename({"entity_id": "s1"}).join(ef, on="s1", how="left").with_columns(
    pl.col("ef").fill_null(1.0), pl.col("kept").fill_null(0))

band = best.with_columns(pl.col("p").cut([0.1, 0.5, 0.9, 0.99], labels=["<.1", ".1-.5", ".5-.9", ".9-.99", ">=.99"]).alias("b"))
bs = band.group_by("country", "b").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).alias("share"))
out = (ef.group_by("country").agg(
    pl.len().alias("S1"), pl.col("ef").mean().alias("model_expected_F05"),
    (pl.col("kept") == 0).mean().alias("empty_rate"), pl.col("kept").sum().alias("links"))
    .join(nt, on="country")
    .with_columns((pl.col("links") / pl.col("S1")).alias("links_per_S1"),
                  (pl.col("links") / pl.col("targets")).alias("linked_target_frac"),
                  (pl.col("targets") / pl.col("S1")).alias("targets_per_S1"))
    .sort("country"))
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
print(name, "threshold", th)
print(out)
print(bs.pivot("b", index="country", values="share").sort("country"))
acc_c = acc.group_by("country").agg(
    (1 - pl.col("p")).mean().alias("expected_FP_rate_of_accepted"))
print(acc_c.sort("country"))
