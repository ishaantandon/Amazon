"""Stage 4: turn pair probabilities into per-S1 match lists.

1. Each target (S2/S3 record) belongs to at most one S1: keep only its best-scoring S1.
2. Accept that link if p >= threshold.
3. Per S1, among its accepted targets (sorted by p), keep the prefix that maximizes the
   plug-in expected F0.5; the empty prediction competes with the expected singleton score.
"""
import numpy as np
import polars as pl


def best_links(pred: pl.DataFrame) -> pl.DataFrame:
    """pred: tid, s1, p (+ optional country). Returns one row per target: its argmax S1."""
    return pred.sort("p", descending=True).unique("tid", keep="first", maintain_order=False)


def _expected_prefix(ps: np.ndarray, missing_mass: float) -> int:
    """Best k for sorted-desc probabilities ps under F0.5 = 1.25TP / (1.25TP + 0.25FN + FP)."""
    total = ps.sum() + missing_mass
    best_k, best = 0, float(np.prod(1 - ps)) * np.exp(-missing_mass)  # P(no true match) ≈ singleton score
    tp = 0.0
    for k, p in enumerate(ps, 1):
        tp += p
        fp = k - tp
        fn = total - tp
        f = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
        if f > best:
            best_k, best = k, f
    return best_k


def assign(pred: pl.DataFrame, threshold: float, expected_f: bool = True) -> dict[str, list[str]]:
    best = best_links(pred)
    acc = best.filter(pl.col("p") >= threshold)
    if not expected_f:
        g = acc.group_by("s1").agg(pl.col("tid"))
        return dict(zip(g["s1"], g["tid"]))
    # Probability mass of links to this S1 that were not accepted (lower-probability targets).
    rest = best.filter(pl.col("p") < threshold).group_by("s1").agg(pl.col("p").sum().alias("miss"))
    g = (acc.sort("p", descending=True).group_by("s1", maintain_order=True)
         .agg(pl.col("tid"), pl.col("p")).join(rest, on="s1", how="left"))
    out = {}
    for s1, tids, ps, miss in g.iter_rows():
        k = _expected_prefix(np.asarray(ps), miss or 0.0)
        if k:
            out[s1] = tids[:k]
    return out
