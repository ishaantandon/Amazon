"""France-only leaderboard probe on top of v9: remove France links in the 'same address, identity-word swap' class.

A link is in the class when target and S1
  * share the first house number (non-empty) and a near-identical street (address token_set >= 85), and
  * share a name word (ratio >= 85), are not the same name glued/near-identical (compact ratio < 85), and
  * EACH name has a word (>= 3 chars) with no fuzzy counterpart (ratio < 70) in the other,
ignoring French legal/decoration words that the generator adds as noise.
Example: 'WD Patrimoine SARL, 27 Av de l'Ombrie' <- 'WD Amicale Sarl, No. 27 Avenue De L'ombrie'.

Prints the class rate among TRUE train pairs (US/India) and among v9's accepted links per country, and
best/worst-case leaderboard deltas, then writes work/fr_variants/v9_fr_sameaddr_swap/matching_results.tsv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import polars as pl
from rapidfuzz import fuzz, process

from config import WORK_DIR, work_path
from data_io import load_links, load_source, write_id_lists

NEUTRAL = {"fils", "cie", "compagnie", "holding", "international", "france", "associes", "ets",
           "etablissements", "distribution"}
BASE = ROOT / "output" / "matching_results.tsv"  # v9
OUT = WORK_DIR / "fr_variants" / "v9_fr_sameaddr_swap" / "matching_results.tsv"
FR_SHARE = 259_452 / 1_732_544


def in_class(sn: str, tn: str, sc: str, tc: str) -> bool:
    a = [x for x in sn.split() if x not in NEUTRAL]
    b = [y for y in tn.split() if y not in NEUTRAL]
    if not a or not b or fuzz.ratio(sc, tc) >= 85:
        return False
    if not any(fuzz.ratio(x, y) >= 85 for x in a for y in b):
        return False
    lone_a = any(len(x) >= 3 and all(fuzz.ratio(x, y) < 70 for y in b) for x in a)
    lone_b = any(len(y) >= 3 and all(fuzz.ratio(y, x) < 70 for x in a) for y in b)
    return lone_a and lone_b


def flag(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    cols = ["entity_id", "name", "compact", "addr", "nums"]
    s1 = pl.read_parquet(work_path(split, "s1.parquet"), columns=cols).rename({c: "s_" + c for c in cols})
    t = pl.read_parquet(work_path(split, "t.parquet"), columns=cols).rename({c: "t_" + c for c in cols})
    d = pairs.join(s1, left_on="s1", right_on="s_entity_id").join(t, left_on="tid", right_on="t_entity_id")
    first = lambda c: pl.col(c).str.split(" ").list.first()
    d = d.with_columns(
        pl.Series("asim", process.cpdist(d["t_addr"].to_list(), d["s_addr"].to_list(), scorer=fuzz.token_set_ratio,
                                         workers=-1)),
    ).with_columns(((first("t_nums") == first("s_nums")) & (pl.col("t_nums") != "") & (pl.col("asim") >= 85)
                    & (pl.col("s_addr") != "") & (pl.col("t_addr") != "")).alias("same_addr"))
    cls = [bool(s) and in_class(a, b, c, e) for s, a, b, c, e in
           zip(d["same_addr"], d["s_name"], d["t_name"], d["s_compact"], d["t_compact"])]
    return d.with_columns(pl.Series("cls", cls))


def f05(p: float, r: float) -> float:
    return 1.25 * p * r / (0.25 * p + r) if p and r else 0.0


def main() -> None:
    # 1) Base rate among TRUE train pairs: how often real matches look like this.
    tr = flag(load_links("train").sample(400_000, seed=1), "train")
    cty = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    print("TRUE train pairs (US/India): class rate overall / among same-address pairs")
    print(tr.join(cty, on="s1").group_by("country").agg(
        pl.col("cls").mean().alias("class_rate"), pl.col("same_addr").mean().alias("same_addr_rate"),
        pl.col("cls").filter(pl.col("same_addr")).mean().alias("class_rate_in_same_addr")).sort("country"))

    # 2) v9 accepted test links.
    m = pl.read_csv(BASE, separator="\t", infer_schema=False).fill_null("")
    lk = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids").rename({"source1_entity_id": "s1", "matched_entity_ids": "tid"}))
    ct = pl.read_parquet(work_path("test", "s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    te = flag(lk.join(ct, on="s1"), "test")
    print("v9 ACCEPTED test links: class rate overall / among same-address links")
    print(te.group_by("country").agg(
        pl.len().alias("links"), pl.col("cls").sum().alias("class_links"), pl.col("cls").mean().alias("class_rate"),
        pl.col("same_addr").mean().alias("same_addr_rate"),
        pl.col("cls").filter(pl.col("same_addr")).mean().alias("class_rate_in_same_addr")).sort("country"))

    fr = te.filter(pl.col("country") == "France")
    per = fr.group_by("s1").agg(pl.len().alias("n"), pl.col("cls").sum().alias("k")).filter(pl.col("k") > 0)
    gain = loss = 0.0
    for n, k in per.select("n", "k").iter_rows():
        gain += 1 - f05((n - k) / n, 1.0)                       # class links all wrong, rest right
        loss += 1 - (f05(1.0, (n - k) / n) if n > k else 0.0)   # class links all right
    print(f"France: {len(per):,} S1s affected, {int(per['k'].sum()):,} links removed")
    print(f"  if all removed links are wrong: France +{gain / 259_452:.4f} -> LB +{gain / 259_452 * FR_SHARE:.4f}")
    print(f"  if all removed links are right: France -{loss / 259_452:.4f} -> LB -{loss / 259_452 * FR_SHARE:.4f}")
    pl.Config.set_tbl_rows(20); pl.Config.set_fmt_str_lengths(50)
    print(fr.filter(pl.col("cls")).select("s_name", "t_name", "s_addr", "t_addr").sample(min(20, int(per["k"].sum())), seed=2))

    drop = {(s, t) for s, t in fr.filter(pl.col("cls")).select("s1", "tid").iter_rows()}
    rows = {s: [x for x in v.split(",") if x and (s, x) not in drop] for s, v in m.iter_rows()}
    s1_ids = load_source("test", 1)["entity_id"].to_list()
    write_id_lists(OUT, s1_ids, rows, "matched_entity_ids")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
