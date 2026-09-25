"""Stage 1: candidate generation.

The problem is flipped: every S2/S3 record ("target") belongs to at most one S1 entity, so
for each target we retrieve its most plausible S1 records within the same country.

Each channel embeds records as sparse IDF-weighted feature vectors and takes a multithreaded
sparse top-K matrix product (sparse_dot_topn) against all S1 vectors of that country:

  name_ngram : TF-IDF over rare character 4-grams of the compact name (typos, glued names, handles)
  phon_key   : consonant-skeleton name tokens / bigrams / whole skeleton (transliterated names)
  mix_key    : skeleton name token x address word, and x house number (generic names, sparse addresses)
  name_key   : rare name tokens, token bigrams, compact-name prefix
  addr_key   : house number x street word, street word bigrams (unrelated brand names)

The union (~34 S1s per target) is then re-ranked by a logistic-regression prefilter over the
channel scores plus two fast string similarities, and the top K_FINAL are kept. That final
set is exactly what the matching model scores, i.e. what candidate_pairs.tsv reports.

Output: work/<split>/cands.parquet with one row per (tid, s1) pair and the channel scores.
"""
import json
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sparse_dot_topn import sp_matmul_topn

from config import (K_ADDR, K_FINAL, K_NAME_TFIDF, K_NAME_TOKEN, MAX_KEY_DF, N_JOBS, WORK_DIR,
                    work_path)

CHUNK = 1_000_000
PREFILTER_PATH = WORK_DIR / "prefilter.json"


def _name_keys(rec: str) -> list[str]:
    toks = rec.split()
    keys = [t for t in toks if len(t) >= 2]
    keys += [a + "_" + b for a, b in zip(toks, toks[1:])]
    compact = rec.replace(" ", "")
    if len(compact) >= 6:
        keys.append("P:" + compact[:6])
    return keys


def _phon_keys(rec: str) -> list[str]:
    toks = rec.split()
    keys = ["F:" + rec.replace(" ", "")]
    keys += [t for t in toks if len(t) >= 2]
    keys += [a + "_" + b for a, b in zip(toks, toks[1:])]
    return keys


def _mix_keys(rec: str) -> list[str]:
    skel, addr, nums = rec.split("|")
    s = [t for t in skel.split() if len(t) >= 2][:3]
    w = [t for t in addr.split() if len(t) >= 3][:8]
    n = nums.split()[:4]
    return [x + "|" + y for x in s for y in w] + [x + "#" + y for x in s for y in n]


def _addr_keys(rec: str) -> list[str]:
    words, _, nums = rec.partition("|")
    w = [t for t in words.split() if len(t) >= 3]
    n = nums.split()[:3]
    keys = [x + ":" + y for x in n for y in w[:4]]
    keys += [a + "_" + b for a, b in zip(w, w[1:])]
    return keys


CHANNELS = {
    "name_ngram": (
        lambda d: (" " + d["compact"] + " ").to_list(),
        dict(analyzer="char", ngram_range=(4, 4), sublinear_tf=True, max_df=0.002, dtype=np.float32),
        K_NAME_TFIDF,
    ),
    "phon_key": (
        lambda d: d["skel"].to_list(),
        dict(analyzer=_phon_keys, max_df=MAX_KEY_DF, dtype=np.float32),
        K_NAME_TOKEN,
    ),
    "mix_key": (
        lambda d: (d["skel"] + "|" + d["addr"] + "|" + d["nums"]).to_list(),
        dict(analyzer=_mix_keys, max_df=MAX_KEY_DF, dtype=np.float32),
        K_NAME_TOKEN,
    ),
    "name_key": (
        lambda d: d["name"].to_list(),
        dict(analyzer=_name_keys, max_df=MAX_KEY_DF, dtype=np.float32),
        K_NAME_TOKEN,
    ),
    "addr_key": (
        lambda d: (d["addr"] + "|" + d["nums"]).to_list(),
        dict(analyzer=_addr_keys, max_df=MAX_KEY_DF, dtype=np.float32),
        K_ADDR,
    ),
}
PREFILTER_FEATS = list(CHANNELS) + ["pf_name", "pf_addr"]


def _retrieve(fitted, t: pl.DataFrame) -> pl.DataFrame:
    """Union of all channels' top-K S1 indices for each target row of `t`."""
    out = None
    for ch, (vec, s_t) in fitted.items():
        text, _, k = CHANNELS[ch]
        m = sp_matmul_topn(vec.transform(text(t)), s_t, top_n=k, threshold=0.05, n_threads=N_JOBS).tocoo()
        part = pl.DataFrame({"ti": m.row.astype(np.int32), "si": m.col.astype(np.int32), ch: m.data.astype(np.float32)})
        out = part if out is None else out.join(part, on=["ti", "si"], how="full", coalesce=True)
    return out.with_columns([pl.col(c).fill_null(0.0) for c in CHANNELS])


def _cheap_sims(c: pl.DataFrame, s1: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    ti, si = c["ti"], c["si"]
    tn, sn = t["name"].gather(ti).to_list(), s1["name"].gather(si).to_list()
    ta = (t["addr"] + " " + t["nums"]).gather(ti).to_list()
    sa = (s1["addr"] + " " + s1["nums"]).gather(si).to_list()
    return c.with_columns(
        pl.Series("pf_name", process.cpdist(tn, sn, scorer=fuzz.token_set_ratio, workers=-1) / 100, dtype=pl.Float32),
        pl.Series("pf_addr", process.cpdist(ta, sa, scorer=fuzz.token_set_ratio, workers=-1) / 100, dtype=pl.Float32),
    )


def _fit(s1: pl.DataFrame):
    fitted = {}
    for ch, (text, params, _) in CHANNELS.items():
        vec = TfidfVectorizer(**params)
        fitted[ch] = (vec, vec.fit_transform(text(s1)).T.tocsr())
    return fitted


def fit_prefilter(sample_per_country: int = 40_000) -> dict:
    """Logistic-regression re-ranker over channel scores, fitted on a train sample."""
    from data_io import load_links

    links = load_links("train").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    s1_all = pl.read_parquet(work_path("train", "s1.parquet"))
    t_all = pl.read_parquet(work_path("train", "t.parquet"))
    frames = []
    for country in sorted(s1_all["country"].unique().to_list()):
        s1 = s1_all.filter(pl.col("country") == country)
        t = t_all.filter(pl.col("country") == country)
        t = t.sample(min(sample_per_country, len(t)), seed=0)
        c = _cheap_sims(_retrieve(_fit(s1), t), s1, t)
        c = c.with_columns(t["entity_id"].gather(c["ti"]).alias("tid"), s1["entity_id"].gather(c["si"]).alias("s1"))
        frames.append(c.join(links, on=["tid", "s1"], how="left").with_columns(pl.col("y").fill_null(0)))
    d = pl.concat(frames)
    lr = LogisticRegression(max_iter=1000).fit(d.select(PREFILTER_FEATS).to_numpy(), d["y"].to_numpy())
    params = {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "features": PREFILTER_FEATS}
    PREFILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFILTER_PATH.write_text(json.dumps(params, indent=1))
    print(f"[block] prefilter fitted on {len(d):,} pairs: {params}")
    return params


def block_country(s1: pl.DataFrame, t: pl.DataFrame, pf: dict) -> pl.DataFrame:
    t0 = time.time()
    fitted = _fit(s1)
    print(f"    fitted channels in {time.time() - t0:.0f}s")
    coef = np.array(pf["coef"], dtype=np.float32)
    out = []
    for a in range(0, len(t), CHUNK):
        tc = t.slice(a, CHUNK)
        c = _cheap_sims(_retrieve(fitted, tc), s1, tc)
        score = c.select(pf["features"]).to_numpy() @ coef + pf["intercept"]
        c = (
            c.with_columns(pl.Series("pf_score", score, dtype=pl.Float32))
            .with_columns(pl.col("pf_score").rank("ordinal", descending=True).over("ti").alias("pf_rank"))
            .filter(pl.col("pf_rank") <= K_FINAL)
        )
        out.append(c.with_columns(
            tc["entity_id"].gather(c["ti"]).alias("tid"),
            s1["entity_id"].gather(c["si"]).alias("s1"),
        ).drop("ti", "si"))
        print(f"    targets {a + len(tc):,}/{len(t):,}  ({time.time() - t0:.0f}s)")
    return pl.concat(out)


def block(split: str) -> pl.DataFrame:
    pf = json.loads(PREFILTER_PATH.read_text()) if PREFILTER_PATH.exists() else fit_prefilter()
    s1_all = pl.read_parquet(work_path(split, "s1.parquet"))
    t_all = pl.read_parquet(work_path(split, "t.parquet"))
    out = []
    for country in sorted(s1_all["country"].unique().to_list()):
        s1 = s1_all.filter(pl.col("country") == country)
        t = t_all.filter(pl.col("country") == country)
        if len(t) == 0:
            continue
        print(f"[block] {split} {country}: {len(s1):,} S1 x {len(t):,} targets")
        out.append(block_country(s1, t, pf))
    cands = pl.concat(out)
    cands.write_parquet(work_path(split, "cands.parquet"))
    print(f"[block] {split}: {len(cands):,} candidate pairs")
    return cands


if __name__ == "__main__":
    import sys
    block(sys.argv[1] if len(sys.argv) > 1 else "train")
