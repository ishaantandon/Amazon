"""Feasibility probe for a dense-retrieval blocking channel (multilingual-e5-small, MIT, off the shelf).

Each record is embedded as "query: <name> | <address>" (raw text, native scripts kept); a target's
dense candidates are its nearest S1s of the same country by cosine similarity.

Phases (S1 embeddings cached in work/dense/, reused by a full integration):
  emb     : embed all S1 of train and test -> work/dense/{split}_s1_emb.npy, {split}_s1_ids.parquet
  recall  : train, with labels. Of the true pairs of sampled targets, how many are in the current
            top-8, in the dense top-K, and in their union? Are the rescued ones accepted by the
            fine-tuned cross-encoder (work/ce/model)?
  france  : test, label-free. For sampled targets that v9 left unlinked, per country:
            dense : new dense top-5 S1s (not in the top-8) that the cross-encoder rates as matches
            gate  : top-8 candidates that stage 1 dropped (p < 0.01) that the cross-encoder rates as matches
            France far above US/India would locate France's hidden loss.

  .venv\\Scripts\\python tools\\dense_probe.py [phase ...]      # default: all
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from config import WORK_DIR, source_path, work_path
from data_io import COLS, load_links, load_source, load_targets

MODEL = os.environ.get("DENSE_MODEL", "intfloat/multilingual-e5-small")
DENSE_DIR = WORK_DIR / "dense"
MAX_LEN = 64
BATCH = 1024
N_SAMPLE = 20_000


def _raw(split: str, sources: tuple, ids=None, cols=COLS) -> pl.DataFrame:
    """Raw records read lazily, keeping only `ids` (all rows when None): full loads of the ~1 GB
    target files alongside other data exhausted RAM once."""
    lf = pl.concat([pl.scan_csv(source_path(split, s), separator="\t", quote_char=None, infer_schema=False,
                                encoding="utf8-lossy", missing_utf8_is_empty_string=True).select(cols)
                    for s in sources])
    if ids is not None:
        lf = lf.filter(pl.col("entity_id").is_in(pl.Series(ids).implode()))
    return lf.collect().with_columns(pl.all().fill_null(""))


def _text(df: pl.DataFrame) -> list[str]:
    return ("query: " + df["business_name"] + " | " + df["business_address"]).to_list()


def _load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    return tok, AutoModel.from_pretrained(MODEL).cuda().eval()


@torch.no_grad()
def embed(tok, model, texts: list[str]) -> np.ndarray:
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])
    t0 = time.time()
    for b, a in enumerate(range(0, len(order), BATCH)):
        idx = order[a:a + BATCH]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True,
                  pad_to_multiple_of=16, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)
        out[idx] = e.half().cpu().numpy()
        if b % 100 == 99:
            torch.cuda.empty_cache()
        if b % 500 == 0:
            print(f"[emb] {a + len(idx):,}/{len(order):,} ({(a + len(idx)) / (time.time() - t0):.0f}/s)", flush=True)
    return out


def s1_embeddings(split: str) -> tuple[np.ndarray, pl.DataFrame]:
    f_emb, f_ids = DENSE_DIR / f"{split}_s1_emb.npy", DENSE_DIR / f"{split}_s1_ids.parquet"
    if not f_emb.exists():
        DENSE_DIR.mkdir(parents=True, exist_ok=True)
        s1 = load_source(split, 1)
        tok, model = _load_model()
        np.save(f_emb, embed(tok, model, _text(s1)))
        s1.select("entity_id", "country").write_parquet(f_ids)
        del model
        torch.cuda.empty_cache()
    return np.load(f_emb, mmap_mode="r"), pl.read_parquet(f_ids)


@torch.no_grad()
def dense_topk(q: np.ndarray, s: np.ndarray, k: int, chunk: int = 256) -> np.ndarray:
    S = torch.from_numpy(np.ascontiguousarray(s)).cuda()
    out = np.empty((len(q), k), dtype=np.int64)
    for a in range(0, len(q), chunk):
        sim = torch.from_numpy(np.ascontiguousarray(q[a:a + chunk])).cuda() @ S.T
        out[a:a + chunk] = torch.topk(sim, k, dim=1).indices.cpu().numpy()
    del S
    torch.cuda.empty_cache()
    return out


def dense_pairs(split: str, targets: pl.DataFrame, k: int) -> pl.DataFrame:
    """targets: entity_id, business_name, business_address, country. Returns tid, s1, dense_rank (1..k)."""
    s_emb, s_ids = s1_embeddings(split)
    tok, model = _load_model()
    t_emb = embed(tok, model, _text(targets))
    del model
    torch.cuda.empty_cache()
    parts = []
    for c in targets["country"].unique().to_list():
        rows = np.flatnonzero((s_ids["country"] == c).to_numpy())
        tr = np.flatnonzero((targets["country"] == c).to_numpy())
        if len(rows) == 0 or len(tr) == 0:
            continue
        nn = dense_topk(t_emb[tr], s_emb[rows], k)
        ids = s_ids["entity_id"].to_numpy()[rows]
        parts.append(pl.DataFrame({
            "tid": np.repeat(targets["entity_id"].to_numpy()[tr], k),
            "s1": ids[nn.ravel()],
            "dense_rank": np.tile(np.arange(1, k + 1, dtype=np.int16), len(tr)),
        }))
    return pl.concat(parts)


def ce_scores(split: str, pairs: pl.DataFrame) -> np.ndarray:
    import ce_pilot
    from transformers import AutoModelForSequenceClassification
    t = _raw(split, (2, 3), pairs["tid"].unique())
    s = _raw(split, (1,), pairs["s1"].unique())
    txt = lambda df: dict(zip(df["entity_id"].to_list(), (df["business_name"] + " | " + df["business_address"]).to_list()))
    mdir = WORK_DIR / "ce" / "model"
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda()
    out = ce_pilot._score(model, tok, txt(t), txt(s), pairs["tid"].to_list(), pairs["s1"].to_list())
    del model
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- phases

def phase_emb() -> None:
    for split in ("train", "test"):
        t0 = time.time()
        s1_embeddings(split)
        print(f"[emb] {split} S1 done ({time.time() - t0:.0f}s)", flush=True)


def phase_recall() -> None:
    links = load_links("train")
    tg = load_targets("train").filter(pl.col("entity_id").is_in(links["tid"].implode()))
    samp = pl.concat([tg.filter(pl.col("country") == c).sample(N_SAMPLE, seed=5) for c in ("US", "India")])
    true = links.join(samp.select(pl.col("entity_id").alias("tid"), "country"), on="tid")
    top8 = pl.scan_parquet(work_path("train", "cands.parquet")).select("tid", "s1").join(
        samp.lazy().select(pl.col("entity_id").alias("tid")), on="tid", how="semi").collect()
    dp = dense_pairs("train", samp, 20)
    t = (true.join(top8.with_columns(pl.lit(True).alias("in8")), on=["tid", "s1"], how="left")
         .join(dp, on=["tid", "s1"], how="left").with_columns(pl.col("in8").fill_null(False)))
    print("[recall] true pairs of sampled train targets (current top-8 vs dense top-K):", flush=True)
    for c in ("US", "India"):
        d = t.filter(pl.col("country") == c)
        row = [f"top8 {d['in8'].mean():.4f}"]
        for k in (5, 10, 20):
            dk = d["dense_rank"].is_not_null() & (d["dense_rank"] <= k)
            row.append(f"dense@{k} {dk.mean():.4f}  union@{k} {(d['in8'] | dk).mean():.4f}")
        miss = d.filter(~pl.col("in8"))
        resc = miss["dense_rank"].is_not_null().sum()
        print(f"  {c}: " + "  |  ".join(row) + f"  | top8 misses {len(miss):,}, dense@20 finds {resc:,}", flush=True)
    # Would the cross-encoder accept the new pairs? Score dense@10 pairs not in the top-8.
    new = dp.filter(pl.col("dense_rank") <= 10).join(top8, on=["tid", "s1"], how="anti")
    new = new.join(links.with_columns(pl.lit(True).alias("y")), on=["tid", "s1"], how="left").with_columns(
        pl.col("y").fill_null(False)).join(samp.select(pl.col("entity_id").alias("tid"), "country"), on="tid")
    new = new.with_columns(pl.Series("ce", ce_scores("train", new)))
    print("[recall] new dense@10 pairs outside the top-8, scored by the cross-encoder:", flush=True)
    print(new.group_by("country", "y").agg(pl.len(), (pl.col("ce") >= 0.5).mean().alias("ce>=0.5"),
                                           (pl.col("ce") >= 0.9).mean().alias("ce>=0.9")).sort("country", "y"), flush=True)


def phase_france() -> None:
    m = pl.read_csv(ROOT / "output" / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
    linked = m.filter(pl.col("matched_entity_ids") != "")["matched_entity_ids"].str.split(",").explode()
    del m
    tg = _raw("test", (2, 3), cols=["entity_id", "country"]).filter(~pl.col("entity_id").is_in(linked.implode()))
    ids = pl.concat([tg.filter(pl.col("country") == c).sample(N_SAMPLE, seed=5) for c in ("US", "India", "France")])
    del tg, linked
    samp = _raw("test", (2, 3), ids["entity_id"])
    ctry = samp.select(pl.col("entity_id").alias("tid"), "country")
    in_samp = pl.col("tid").is_in(ctry["tid"].implode())
    top8 = pl.scan_parquet(work_path("test", "cands.parquet")).select("tid", "s1").filter(in_samp).collect()
    dense = dense_pairs("test", samp, 5).join(top8, on=["tid", "s1"], how="anti").with_columns(pl.lit("dense").alias("kind"))
    p1 = pl.scan_parquet(work_path("test", "pred.parquet")).select("tid", "s1", "p").filter(in_samp).collect()
    gate = (p1.filter(pl.col("p") < 0.01).sort("p", descending=True).group_by("tid").head(3)
            .select("tid", "s1").with_columns(pl.lit("gate").alias("kind")))
    pairs = pl.concat([dense.select("tid", "s1", "kind"), gate]).join(ctry, on="tid")
    pairs = pairs.with_columns(pl.Series("ce", ce_scores("test", pairs)))
    per = pairs.group_by("country", "kind", "tid").agg(pl.col("ce").max().alias("best"))
    print(f"[france] {N_SAMPLE:,} unlinked targets per country; share whose best NEW pair the cross-encoder rates as a match:",
          flush=True)
    print(per.group_by("country", "kind").agg(
        pl.len().alias("targets_with_pairs"),
        ((pl.col("best") >= 0.5).sum() / N_SAMPLE).alias("ce>=0.5"),
        ((pl.col("best") >= 0.9).sum() / N_SAMPLE).alias("ce>=0.9"),
        ((pl.col("best") >= 0.99).sum() / N_SAMPLE).alias("ce>=0.99")).sort("kind", "country"), flush=True)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    pairs.write_parquet(DENSE_DIR / "france_probe_pairs.parquet")


PHASES = {"emb": phase_emb, "recall": phase_recall, "france": phase_france}

if __name__ == "__main__":
    for ph in (sys.argv[1:] or list(PHASES)):
        t0 = time.time()
        PHASES[ph]()
        print(f"[{ph}] done ({time.time() - t0:.0f}s)", flush=True)
