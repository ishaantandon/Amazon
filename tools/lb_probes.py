"""Leaderboard probes: re-assign cached test predictions under different thresholds.

Writes work/lb_probes/<name>/matching_results.tsv. The candidate set is unchanged, so each probe
pairs with output/candidate_pairs.tsv for validation. Run from the repo root:

  .venv\\Scripts\\python tools\\lb_probes.py [probe names...]    # default: all probes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import polars as pl

from assign import assign
from config import WORK_DIR, work_path
from data_io import load_source, write_id_lists

# name: {country: (cached prediction file, threshold)}
PROBES = {
    "v4_t07": {c: ("pred2_v4.parquet", 0.7) for c in ("US", "India", "France")},
    "v4_fr04": {c: ("pred2_v4.parquet", 0.4) for c in ("US", "India", "France")},
    "v2_t08": {c: ("pred_v1.parquet", 0.8) for c in ("US", "India", "France")},
    # v4 for US/India, v2's France: isolates whether stage 2 loses its gain on France.
    "v4_frv2": {"US": ("pred2_v4.parquet", 0.4), "India": ("pred2_v4.parquet", 0.4),
                "France": ("pred_v1.parquet", 0.6)},
}


def main() -> None:
    s1_ids = load_source("test", 1)["entity_id"].to_list()
    only = set(sys.argv[1:])
    for name, spec in PROBES.items():
        if only and name not in only:
            continue
        matches = {}
        for c, (src, t) in spec.items():
            pred = pl.scan_parquet(work_path("test", src)).filter(pl.col("country") == c).collect()
            matches.update(assign(pred, t, True))
        out = WORK_DIR / "lb_probes" / name / "matching_results.tsv"
        write_id_lists(out, s1_ids, matches, "matched_entity_ids")
        print(f"[probe] {name}: {len(matches):,} S1 with matches, "
              f"{sum(len(v) for v in matches.values()):,} links -> {out}")


if __name__ == "__main__":
    main()
