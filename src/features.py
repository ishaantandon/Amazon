"""Stage 2: pair features for (target, S1 candidate) pairs.

All string similarities use RapidFuzz's vectorized cpdist (C++, all cores). Context features
describe the candidate's standing among the other candidates of the same target, which is
what the "each target belongs to at most one S1" decision hinges on.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from blocking import CHANNELS

FIELDS = ["name", "compact", "skel", "legal", "addr", "nums", "state", "addr_empty", "country"]


def _sim(a: list, b: list, scorer) -> np.ndarray:
    return (process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / 100).astype(np.float32)


def _set_overlap(a: pl.Series, b: pl.Series) -> tuple[pl.Series, pl.Series]:
    """Shared-token count and Jaccard for space-separated token sets."""
    la, lb = a.str.split(" ").list.eval(pl.element().filter(pl.element() != "")), b.str.split(" ").list.eval(
        pl.element().filter(pl.element() != ""))
    inter = la.list.set_intersection(lb).list.len()
    union = la.list.set_union(lb).list.len()
    return inter.cast(pl.Float32), (inter / union.clip(1)).cast(pl.Float32)


def build(pairs: pl.DataFrame, s1: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    """pairs: tid, s1, channel scores, pf_*; s1/t: normalized frames (entity_id + FIELDS)."""
    d = (
        pairs.join(t.select("entity_id", *FIELDS).rename({f: "t_" + f for f in FIELDS}),
                   left_on="tid", right_on="entity_id", how="left")
        .join(s1.select("entity_id", *FIELDS[:-1]).rename({f: "s_" + f for f in FIELDS[:-1]}),
              left_on="s1", right_on="entity_id", how="left")
    )
    tn, sn = d["t_name"].to_list(), d["s_name"].to_list()
    tc, sc = d["t_compact"].to_list(), d["s_compact"].to_list()
    tk, sk = d["t_skel"].to_list(), d["s_skel"].to_list()
    ta, sa = d["t_addr"].to_list(), d["s_addr"].to_list()
    feats = {
        "n_ratio": _sim(tn, sn, fuzz.ratio),
        "n_tset": _sim(tn, sn, fuzz.token_set_ratio),
        "n_tsort": _sim(tn, sn, fuzz.token_sort_ratio),
        "n_partial": _sim(tn, sn, fuzz.partial_ratio),
        "c_ratio": _sim(tc, sc, fuzz.ratio),
        "c_partial": _sim(tc, sc, fuzz.partial_ratio),
        "c_jw": process.cpdist(tc, sc, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32),
        "k_ratio": _sim(tk, sk, fuzz.ratio),
        "k_tset": _sim(tk, sk, fuzz.token_set_ratio),
        "a_ratio": _sim(ta, sa, fuzz.ratio),
        "a_tset": _sim(ta, sa, fuzz.token_set_ratio),
        "a_partial": _sim(ta, sa, fuzz.partial_ratio),
    }
    num_shared, num_jac = _set_overlap(d["t_nums"], d["s_nums"])
    word_shared, word_jac = _set_overlap(d["t_addr"], d["s_addr"])
    ntok_shared, ntok_jac = _set_overlap(d["t_name"], d["s_name"])
    d = d.with_columns(
        [pl.Series(k, v) for k, v in feats.items()]
        + [
            num_shared.alias("num_shared"), num_jac.alias("num_jac"),
            word_shared.alias("addr_word_shared"), word_jac.alias("addr_word_jac"),
            ntok_shared.alias("name_tok_shared"), ntok_jac.alias("name_tok_jac"),
        ]
    ).with_columns(
        (pl.col("t_nums").str.split(" ").list.first() == pl.col("s_nums").str.split(" ").list.first())
        .and_(pl.col("t_nums") != "").cast(pl.Int8).alias("num_first_eq"),
        (pl.col("t_nums") != "").cast(pl.Int8).alias("t_has_num"),
        pl.col("t_addr_empty").cast(pl.Int8).alias("t_addr_empty_i"),
        pl.when((pl.col("t_state") == "") | (pl.col("s_state") == "")).then(-1)
        .otherwise((pl.col("t_state") == pl.col("s_state")).cast(pl.Int32)).cast(pl.Int8).alias("state_eq"),
        pl.when((pl.col("t_legal") == "") | (pl.col("s_legal") == "")).then(-1)
        .otherwise((pl.col("t_legal") == pl.col("s_legal")).cast(pl.Int32)).cast(pl.Int8).alias("legal_eq"),
        pl.col("t_name").str.len_chars().cast(pl.Int16).alias("t_name_len"),
        pl.col("s_name").str.len_chars().cast(pl.Int16).alias("s_name_len"),
        pl.col("t_name").str.count_matches(" ").add(1).cast(pl.Int8).alias("t_name_ntok"),
        pl.col("tid").str.starts_with("S3").cast(pl.Int8).alias("is_s3"),
    )
    # Candidate context within the target: gaps to the best candidate on key signals.
    d = d.with_columns(
        (pl.col("pf_score").max().over("tid") - pl.col("pf_score")).alias("pf_gap"),
        (pl.col("n_tset").max().over("tid") - pl.col("n_tset")).alias("n_tset_gap"),
        (pl.col("a_tset").max().over("tid") - pl.col("a_tset")).alias("a_tset_gap"),
        pl.len().over("tid").cast(pl.Int8).alias("n_cands"),
    )
    return d.select(["tid", "s1", "t_country"] + FEATURES)


def add_global_context(cands: pl.DataFrame) -> pl.DataFrame:
    """Context that needs the whole candidate set, so it is computed before chunking."""
    top1 = cands.filter(pl.col("pf_rank") == 1).group_by("s1").len("s1_top1_count")
    return cands.join(top1, on="s1", how="left").with_columns(
        pl.col("s1_top1_count").fill_null(0).cast(pl.Int32))


FEATURES = list(CHANNELS) + [
    "pf_name", "pf_addr", "pf_score", "pf_rank",
    "n_ratio", "n_tset", "n_tsort", "n_partial", "c_ratio", "c_partial", "c_jw", "k_ratio", "k_tset",
    "a_ratio", "a_tset", "a_partial", "num_shared", "num_jac", "addr_word_shared", "addr_word_jac",
    "name_tok_shared", "name_tok_jac", "num_first_eq", "t_has_num", "t_addr_empty_i", "state_eq",
    "legal_eq", "t_name_len", "s_name_len", "t_name_ntok", "is_s3",
    "pf_gap", "n_tset_gap", "a_tset_gap", "n_cands", "s1_top1_count",
]
