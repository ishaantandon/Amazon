"""How often is a linked pair a 'word swap': each name has a content word (>=3 chars, legal forms already
removed by normalization) with no fuzzy counterpart (ratio < 70) in the other name?
Compared: true train pairs (US/India base rate of this noise), and v5's accepted test links per country."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import polars as pl
from rapidfuzz import fuzz

from config import work_path
from data_io import load_links


def unmatched(a: list, b: list) -> bool:
    return any(len(x) >= 3 and all(fuzz.ratio(x, y) < 70 for y in b) for x in a)


def swap_flags(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    s1 = pl.read_parquet(work_path(split, "s1.parquet"), columns=["entity_id", "name", "addr", "nums"])
    t = pl.read_parquet(work_path(split, "t.parquet"), columns=["entity_id", "name", "addr", "nums"])
    d = (pairs.join(s1.rename({"entity_id": "s1", "name": "sn", "addr": "sa", "nums": "snum"}), on="s1")
         .join(t.rename({"entity_id": "tid", "name": "tn", "addr": "ta", "nums": "tnum"}), on="tid"))
    sw = [unmatched(a.split(), b.split()) and unmatched(b.split(), a.split())
          for a, b in zip(d["sn"].to_list(), d["tn"].to_list())]
    return d.with_columns(pl.Series("swap", sw),
                          ((pl.col("sa") == pl.col("ta")) & (pl.col("snum") == pl.col("tnum")) & (pl.col("sa") != "")).alias("same_addr"))


# 1) true train pairs (sample)
links = load_links("train").sample(400_000, seed=1)
tr = swap_flags(links, "train")
cty = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
tr = tr.join(cty, on="s1")
print("TRUE train pairs: swap rate by country (and among same-address pairs)")
print(tr.group_by("country").agg(pl.len(), pl.col("swap").mean().alias("swap_rate"),
                                 pl.col("swap").filter(pl.col("same_addr")).mean().alias("swap_rate_same_addr")).sort("country"))

# 2) v5 accepted test links (sample per country)
m = pl.read_csv(ROOT / "output" / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
lk = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").rename({"source1_entity_id": "s1", "matched_entity_ids": "tid"}))
ct = pl.read_parquet(work_path("test", "s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
lk = lk.join(ct, on="s1").group_by("country").map_groups(lambda g: g.sample(min(150_000, len(g)), seed=1))
p = pl.scan_parquet(work_path("test", "pred2_ce.parquet")).select("tid", "s1", "p").collect()
te = swap_flags(lk.select("s1", "tid", "country"), "test").join(p, on=["tid", "s1"], how="left")
print("v5 ACCEPTED test links: swap rate by country")
print(te.group_by("country").agg(pl.len(), pl.col("swap").mean().alias("swap_rate"),
                                 pl.col("swap").filter(pl.col("same_addr")).mean().alias("swap_rate_same_addr"),
                                 pl.col("p").filter(pl.col("swap")).median().alias("median_p_of_swaps")).sort("country"))
pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(60)
print(te.filter((pl.col("country") == "France") & pl.col("swap")).select("sn", "tn", "p", "same_addr").sample(25, seed=3))
