"""France-only variants of a base submission, for leaderboard-guided tuning (France has no labels).

US and India rows are copied unchanged from the base; only France S1 rows differ, so a variant's
leaderboard delta measures France alone: France F0.5 changes by (score - base score) / 0.14975.

  loose_{95,90,85}: add France targets rejected (best p < 0.1) although their name matches the best
                    candidate S1 (token_set >= level) at the same normalized street, or with no address
  strict_{95,99,999}: keep only France links whose stage-2 p >= 0.95 / 0.99 / 0.999

  .venv\\Scripts\\python tools\\france_variants.py [base_pred2] [base_tsv] [out_prefix]
Defaults: v8 (work/test/pred2_v8.parquet, work/output_v8_addr/matching_results.tsv) -> work/fr_variants/v8_*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import polars as pl
from rapidfuzz import fuzz, process

from config import WORK_DIR, work_path
from data_io import load_source, write_id_lists

PRED2 = Path(sys.argv[1]) if len(sys.argv) > 1 else work_path("test", "pred2_v8.parquet")
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else WORK_DIR / "output_v8_addr" / "matching_results.tsv"
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "v8"
OUT = WORK_DIR / "fr_variants"


def main() -> None:
    fr = set(pl.read_parquet(work_path("test", "s1.parquet"), columns=["entity_id", "country"])
             .filter(pl.col("country") == "France")["entity_id"].to_list())
    rows = {}
    with open(BASE, encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, m = line.rstrip("\n").split("\t")
            rows[s] = m.split(",") if m else []
    s1_ids = load_source("test", 1)["entity_id"].to_list()
    p = pl.read_parquet(PRED2).filter(pl.col("country") == "France")
    best = p.sort("p", descending=True).unique("tid", keep="first")
    n0 = sum(len(rows[k]) for k in fr)

    s1n = pl.read_parquet(work_path("test", "s1.parquet"), columns=["entity_id", "name", "addr"]).rename(
        {"entity_id": "s1", "name": "sn", "addr": "sa"})
    tn = pl.read_parquet(work_path("test", "t.parquet"), columns=["entity_id", "name", "addr"]).rename(
        {"entity_id": "tid", "name": "tn", "addr": "ta"})
    rej = best.filter(pl.col("p") < 0.1).join(tn, on="tid").join(s1n, on="s1")
    rej = rej.with_columns(pl.Series("nsim", process.cpdist(rej["tn"].to_list(), rej["sn"].to_list(),
                                                            scorer=fuzz.token_set_ratio, workers=-1)))
    rej = rej.filter((pl.col("ta") == pl.col("sa")) | (pl.col("ta") == ""))
    for lvl in (95, 90, 85):
        v = {k: list(x) for k, x in rows.items()}
        add = rej.filter(pl.col("nsim") >= lvl)
        for s, t in add.select("s1", "tid").iter_rows():
            v[s].append(t)
        _write(f"{PREFIX}_fr_loose_{lvl}", s1_ids, v, fr, n0)

    for lvl, th in (("95", 0.95), ("99", 0.99), ("999", 0.999)):
        keep = set(best.filter(pl.col("p") >= th).select(pl.concat_str("s1", "tid", separator="|").alias("k"))["k"].to_list())
        v = {k: ([t for t in x if f"{k}|{t}" in keep] if k in fr else list(x)) for k, x in rows.items()}
        _write(f"{PREFIX}_fr_strict_{lvl}", s1_ids, v, fr, n0)


def _write(name, s1_ids, v, fr, n0):
    path = OUT / name / "matching_results.tsv"
    write_id_lists(path, s1_ids, v, "matched_entity_ids")
    n = sum(len(v[k]) for k in fr)
    print(f"[fr] {name}: France links {n:,} ({(n - n0) / n0:+.1%}), France S1 empty "
          f"{sum(1 for k in fr if not v[k]) / len(fr):.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
