"""v9: v8 plus rarity-weighted (IDF) name features.

Why: France (no labels, ~0.94 on the leaderboard) links the right number of records but some to the
wrong look-alike S1. Its names follow a template, "<town> <category> <legal>", and many S1s share a
street, so town/address words make siblings look alike; only rarer words tell them apart. Plain string
similarities weigh every word equally. With per-country IDF over S1 names (language-neutral, trainable
on US/India's own generic words such as "traders", "enterprises"):
  name_idf_jac   : IDF-weighted Jaccard of the name token sets
  s_idf_cov      : share of the S1 name's IDF mass also present in the target name
  s_missing_max  : highest IDF among S1 name tokens missing from the target (a rare word not matched)
  t_extra_max    : highest IDF among target name tokens missing from the S1 name

Reuses tools/ce_v8.py (same data, training split and outputs pattern):
  work/stage2_v9.txt, work/decision2_v9.json, work/test/pred2_v9.parquet, work/output_v9_idf/
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import polars as pl

import ce_v8
from config import WORK_DIR, work_path

NAME_FEATS = ["name_idf_jac", "s_idf_cov", "s_missing_max", "t_extra_max"]


def add_name_idf(feats: pl.DataFrame, split: str, s1_keep=None) -> pl.DataFrame:
    s1 = pl.read_parquet(work_path(split, "s1.parquet"), columns=["entity_id", "country", "name"])
    if s1_keep is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(s1_keep.implode()))
    tn = pl.read_parquet(work_path(split, "t.parquet"), columns=["entity_id", "name"])
    tok = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    s1 = s1.with_columns(tok("name").alias("tok"))
    n_c = s1.group_by("country").len("n_c")
    idf = (s1.select("country", "tok").explode("tok").drop_nulls().group_by("country", "tok").len("df")
           .join(n_c, on="country").with_columns((pl.col("n_c") / pl.col("df")).log().alias("idf")).select("country", "tok", "idf"))
    unseen = n_c.with_columns(pl.col("n_c").cast(pl.Float64).log().alias("idf_max")).select("country", "idf_max")

    pairs = feats.select("tid", "s1", "country").unique(["tid", "s1"])
    st = (pairs.join(s1.select(pl.col("entity_id").alias("s1"), "tok"), on="s1").explode("tok").drop_nulls())
    tt = (pairs.join(tn.select(pl.col("entity_id").alias("tid"), tok("name").alias("tok")), on="tid")
          .explode("tok").drop_nulls())
    st = st.join(tt.select("tid", "s1", "tok").with_columns(pl.lit(True).alias("in_o")), on=["tid", "s1", "tok"], how="left")
    tt = tt.join(st.select("tid", "s1", "tok").with_columns(pl.lit(True).alias("in_o")), on=["tid", "s1", "tok"], how="left")

    def weight(d):
        return (d.join(idf, on=["country", "tok"], how="left").join(unseen, on="country", how="left")
                .with_columns(pl.coalesce("idf", "idf_max").alias("w"), pl.col("in_o").fill_null(False)))
    sa = weight(st).group_by("tid", "s1").agg(
        pl.col("w").sum().alias("w_s"), pl.col("w").filter(pl.col("in_o")).sum().alias("w_sh"),
        pl.col("w").filter(~pl.col("in_o")).max().fill_null(0.0).alias("s_missing_max"))
    ta = weight(tt).group_by("tid", "s1").agg(
        pl.col("w").sum().alias("w_t"), pl.col("w").filter(~pl.col("in_o")).max().fill_null(0.0).alias("t_extra_max"))
    f = (pairs.select("tid", "s1").join(sa, on=["tid", "s1"], how="left").join(ta, on=["tid", "s1"], how="left")
         .with_columns(pl.col("w_s", "w_t", "w_sh", "s_missing_max", "t_extra_max").fill_null(0.0))
         .with_columns(
             (pl.col("w_sh") / (pl.col("w_s") + pl.col("w_t") - pl.col("w_sh"))).fill_nan(0.0).alias("name_idf_jac"),
             (pl.col("w_sh") / pl.col("w_s")).fill_nan(0.0).alias("s_idf_cov"))
         .select("tid", "s1", *NAME_FEATS))
    return feats.join(f, on=["tid", "s1"], how="left").with_columns(pl.col(NAME_FEATS).fill_null(0.0))


# Point the v8 pipeline at v9's features and file names.
_v8_add = ce_v8.add_features


def _add_v9(feats, ce, split, s1_keep=None):
    return add_name_idf(_v8_add(feats, ce, split, s1_keep), split, s1_keep)


ce_v8.add_features = _add_v9
ce_v8.EXTRA = ce_v8.EXTRA + NAME_FEATS
ce_v8.STAGE2 = WORK_DIR / "stage2_v9.txt"
ce_v8.DECISION = WORK_DIR / "decision2_v9.json"
ce_v8.PRED2 = work_path("test", "pred2_v9.parquet")
ce_v8.OUT_DIR = WORK_DIR / "output_v9_idf"

if __name__ == "__main__":
    for ph in (sys.argv[1:] or ["fit", "write"]):
        t0 = time.time()
        ce_v8.PHASES[ph]()
        print(f"[{ph}] done ({time.time() - t0:.0f}s)", flush=True)
