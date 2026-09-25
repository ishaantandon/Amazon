"""Stage 0: load raw TSVs, normalize, and cache as parquet (work/<split>/s1.parquet, t.parquet)."""
import time

from config import work_path
from data_io import load_source, load_targets
from normalize import normalize_frame


def prep(split: str) -> None:
    for name, loader in [("s1", lambda: load_source(split, 1)), ("t", lambda: load_targets(split))]:
        out = work_path(split, f"{name}.parquet")
        if out.exists():
            print(f"[prep] {out} exists, skipping")
            continue
        t0 = time.time()
        df = normalize_frame(loader())
        df.write_parquet(out)
        print(f"[prep] {split}/{name}: {len(df):,} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    prep(sys.argv[1] if len(sys.argv) > 1 else "train")
