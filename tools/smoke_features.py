import sys, numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import work_path
from data_io import load_links
from features import build, add_global_context, FEATURES
from blocking import CHANNELS
from assign import assign
links = load_links("train").head(3000)
rng = np.random.default_rng(0)
# each target: its true S1 plus 3 random S1s from the sample
s1_ids = links["s1"].unique()
rows = []
for tid, s1 in links.iter_rows():
    rows.append((tid, s1)); rows += [(tid, x) for x in rng.choice(s1_ids.to_numpy(), 3)]
pairs = pl.DataFrame(rows, schema=["tid","s1"], orient="row").unique()
pairs = pairs.with_columns([pl.lit(np.float32(0.3)).alias(c) for c in CHANNELS] +
    [pl.lit(np.float32(0.5)).alias("pf_name"), pl.lit(np.float32(0.5)).alias("pf_addr"),
     pl.Series("pf_score", rng.random(len(pairs)).astype(np.float32))]).with_columns(
     pl.col("pf_score").rank("ordinal", descending=True).over("tid").cast(pl.UInt32).alias("pf_rank"))
pairs = add_global_context(pairs)
s1 = pl.scan_parquet(work_path("train","s1.parquet")).filter(pl.col("entity_id").is_in(pairs["s1"].unique().implode())).collect()
t = pl.scan_parquet(work_path("train","t.parquet")).filter(pl.col("entity_id").is_in(pairs["tid"].unique().implode())).collect()
f = build(pairs, s1, t)
print(f.shape, "missing features:", [c for c in FEATURES if c not in f.columns])
print(f.select(FEATURES).describe().filter(pl.col("statistic").is_in(["mean","min","max","null_count"])).transpose(include_header=True).head(45))
f = f.join(links.with_columns(pl.lit(1).alias("y")), on=["tid","s1"], how="left").with_columns(pl.col("y").fill_null(0))
print(f.group_by("y").agg(pl.col("n_tset").mean(), pl.col("a_tset").mean(), pl.col("num_first_eq").mean()))
pred = f.select("tid","s1", pl.col("n_tset").alias("p"))
out = assign(pred, 0.5, True); print("assign ok, S1s with matches:", len(out))
