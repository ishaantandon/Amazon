"""Print a few France S1 neighbourhoods from v5: the S1, its predicted links (with p), in-play
candidates it did not get, and any France S2/S3 record at the same normalized address+numbers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import polars as pl

from config import work_path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rd = lambda f: pl.scan_csv(ROOT / "Dataset" / "test" / f, separator="\t", quote_char=None, infer_schema=False)
raw_s1 = rd("test_source1.tsv").filter(pl.col("country") == "France").collect()
raw_t = pl.concat([rd("test_source2.tsv"), rd("test_source3.tsv")]).filter(pl.col("country") == "France").collect()
txt = dict(zip(raw_t["entity_id"], (raw_t["business_name"] + "  |  " + raw_t["business_address"].fill_null(""))))
s1txt = dict(zip(raw_s1["entity_id"], (raw_s1["business_name"] + "  |  " + raw_s1["business_address"].fill_null(""))))

pred = pl.scan_parquet(work_path("test", "pred2_ce.parquet")).filter(pl.col("country") == "France").collect()
best = pred.sort("p", descending=True).unique("tid", keep="first").select("tid", pl.col("s1").alias("best_s1"), pl.col("p").alias("best_p"))
m = pl.read_csv(ROOT / "output" / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
matched = dict(zip(m["source1_entity_id"], m["matched_entity_ids"]))

ns1 = pl.read_parquet(work_path("test", "s1.parquet"), columns=["entity_id", "country", "addr", "nums"]).filter(pl.col("country") == "France")
nt = pl.read_parquet(work_path("test", "t.parquet"), columns=["entity_id", "country", "addr", "nums"]).filter(pl.col("country") == "France")
key = (pl.col("addr") + "|" + pl.col("nums")).alias("k")
ns1, nt = ns1.with_columns(key), nt.with_columns(key)

seeds = raw_s1.sample(N, seed=SEED)["entity_id"].to_list()
for s in seeds:
    links = [x for x in matched.get(s, "").split(",") if x]
    print("=" * 110)
    print(f"S1 {s}: {s1txt[s]}")
    cand = pred.filter(pl.col("s1") == s).join(best, on="tid").sort("p", descending=True)
    for tid, _, _, p, bs, bp in cand.iter_rows():
        if p < 0.01 and tid not in links:
            continue
        tag = "LINK " if tid in links else ("elsewhere" if bs != s else "rejected")
        extra = f" -> best {bs} p={bp:.3f}" if bs != s else ""
        print(f"   {tag:9s} p={p:.3f} {tid}: {txt[tid]}{extra}")
    k = ns1.filter(pl.col("entity_id") == s)["k"][0]
    same = nt.filter(pl.col("k") == k)["entity_id"].to_list() if k.split("|")[0] else []
    others = [t for t in same if t not in set(cand["tid"].to_list())]
    n_s1_same = ns1.filter(pl.col("k") == k).height if k.split("|")[0] else 0
    if others:
        print(f"   -- same normalized address, NOT a candidate of this S1 ({n_s1_same} S1s share this address):")
        for t in others[:6]:
            b = best.filter(pl.col("tid") == t)
            bi = f"best {b['best_s1'][0]} p={b['best_p'][0]:.3f}" if len(b) else "no in-play candidate"
            print(f"      {t}: {txt[t]}   [{bi}]")
