"""Where do France's links deviate from what genuine matches look like? (label-free error hunt)

Every pair is put in a bin: name relation x address relation x house-number relation. The bin shares of
v9's accepted links per country are compared with the bin shares of TRUE train pairs (US/India). For
US/India the accepted shares should track the true ones; a bin where France's share is far above the
genuine-match share is where France's wrong links (or a France-only noise style) concentrate.

  .venv\\Scripts\\python tools\\france_bins.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import polars as pl
from rapidfuzz import fuzz, process

from config import work_path
from data_io import load_links

N = 200_000


def bins(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    cols = ["entity_id", "name", "addr", "nums"]
    s1 = pl.read_parquet(work_path(split, "s1.parquet"), columns=cols).rename({c: "s_" + c for c in cols})
    t = pl.read_parquet(work_path(split, "t.parquet"), columns=cols).rename({c: "t_" + c for c in cols})
    d = pairs.join(s1, left_on="s1", right_on="s_entity_id").join(t, left_on="tid", right_on="t_entity_id")
    sim = lambda a, b: process.cpdist(d[a].to_list(), d[b].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    d = d.with_columns(pl.Series("n", sim("t_name", "s_name")), pl.Series("a", sim("t_addr", "s_addr")))
    first = lambda c: pl.col(c).str.split(" ").list.first()
    tn = pl.col("t_nums").str.split(" ")
    sn = pl.col("s_nums").str.split(" ")
    return d.with_columns(
        pl.when(pl.col("t_name") == pl.col("s_name")).then(pl.lit("n=exact"))
        .when(pl.col("n") >= 90).then(pl.lit("n>=90")).when(pl.col("n") >= 70).then(pl.lit("n70-90"))
        .otherwise(pl.lit("n<70")).alias("nb"),
        pl.when(pl.col("t_addr") == "").then(pl.lit("a=empty"))
        .when(pl.col("a") >= 90).then(pl.lit("a>=90")).when(pl.col("a") >= 60).then(pl.lit("a60-90"))
        .otherwise(pl.lit("a<60")).alias("ab"),
        pl.when(pl.col("t_nums") == "").then(pl.lit("#t_none")).when(pl.col("s_nums") == "").then(pl.lit("#s_none"))
        .when(first("t_nums") == first("s_nums")).then(pl.lit("#same"))
        .when(tn.list.set_intersection(sn).list.len() > 0).then(pl.lit("#shared"))
        .otherwise(pl.lit("#diff")).alias("hb"),
    ).select("tid", "s1", pl.concat_str("nb", "ab", "hb", separator=" ").alias("bin"))


def shares(d: pl.DataFrame, name: str) -> pl.DataFrame:
    return d.group_by("bin").len().select("bin", (pl.col("len") / pl.col("len").sum()).alias(name))


def main() -> None:
    links = load_links("train").sample(2 * N, seed=1)
    cty = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    tr = bins(links, "train").join(cty, on="s1")
    m = pl.read_csv(ROOT / "output" / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
    lk = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids").rename({"source1_entity_id": "s1", "matched_entity_ids": "tid"}))
    ct = pl.read_parquet(work_path("test", "s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    lk = lk.join(ct, on="s1")
    lk = pl.concat([lk.filter(pl.col("country") == c).sample(min(N, int((lk["country"] == c).sum())), seed=1)
                    for c in ("US", "India", "France")])
    te = bins(lk.select("tid", "s1"), "test").join(ct, on="s1")

    tab = shares(tr.filter(pl.col("country") == "US"), "true_US")
    for name, d in [("true_IN", tr.filter(pl.col("country") == "India")),
                    ("acc_US", te.filter(pl.col("country") == "US")),
                    ("acc_IN", te.filter(pl.col("country") == "India")),
                    ("acc_FR", te.filter(pl.col("country") == "France"))]:
        tab = tab.join(shares(d, name), on="bin", how="full", coalesce=True)
    tab = tab.fill_null(0.0).with_columns(
        ((pl.col("true_US") + pl.col("true_IN")) / 2).alias("true_avg")).with_columns(
        (pl.col("acc_FR") - pl.col("true_avg")).alias("FR_excess"),
        (pl.col("acc_FR") / pl.col("true_avg").clip(1e-4)).alias("FR_ratio"))
    pl.Config.set_tbl_rows(30)
    pl.Config.set_float_precision(4)
    cols = ["bin", "true_US", "true_IN", "acc_US", "acc_IN", "acc_FR", "FR_excess", "FR_ratio"]
    print("Bins where France's accepted links exceed the genuine-match share (top 20 by excess):")
    print(tab.sort("FR_excess", descending=True).select(cols).head(20))
    print("Bins where France is below the genuine-match share (top 10 deficits):")
    print(tab.sort("FR_excess").select(cols).head(10))
    tab.write_csv(ROOT / "work" / "france_bins.csv")


if __name__ == "__main__":
    main()
