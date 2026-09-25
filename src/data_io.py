"""TSV loading/writing. Every column is read as a string; empty fields stay ''."""
from pathlib import Path

import polars as pl

from config import source_path, split_dir

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema=False,
        quote_char=None,
        encoding="utf8-lossy",
        missing_utf8_is_empty_string=True,
    )
    return df.with_columns(pl.all().fill_null(""))


def load_source(split: str, src: int) -> pl.DataFrame:
    return read_tsv(source_path(split, src)).select(COLS)


def load_targets(split: str) -> pl.DataFrame:
    """Source 2 and Source 3 stacked; the source is recoverable from the ID prefix."""
    return pl.concat([load_source(split, 2), load_source(split, 3)])


def load_ground_truth(split: str = "train") -> dict[str, set[str]]:
    df = read_tsv(split_dir(split) / f"{split}_ground_truth.tsv")
    return {
        s1: set(m.split(",")) if m else set()
        for s1, m in zip(df["source1_entity_id"], df["matched_entity_ids"])
    }


def load_links(split: str = "train") -> pl.DataFrame:
    """Ground truth as (tid, s1) rows: one row per matched S2/S3 record."""
    df = read_tsv(split_dir(split) / f"{split}_ground_truth.tsv")
    return (
        df.filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "tid"})
        .select("tid", "s1")
    )


def write_id_lists(path: Path, s1_ids: list[str], lists: dict[str, list[str]], col: str) -> None:
    """One row per S1 id; `col` holds a comma-joined, de-duplicated ID list (possibly empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(lists.get(s1, ())))
            f.write(f"{s1}\t{','.join(ids)}\n")
