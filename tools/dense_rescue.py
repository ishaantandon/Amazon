"""v10: a dense-retrieval "rescue lane" on top of v9.

v9 leaves ~40% of S2/S3 records unlinked (best stage-2 p < T_OPEN). For those open records, the nearest S1s
by multilingual-e5-small embedding (tools/dense_probe.py) that blocking never retrieved become rescue
candidates. They are scored by the fine-tuned cross-encoder (work/ce/model) and a small LightGBM rescue model
trained in the faithful simulation (pool half B), then merged with v9's scores before the usual
one-S1-per-record assignment and expected-F0.5 cut. Validated on the simulation's held-out S1s against v9.

Phases (each cached under work/dense/, so a crash resumes):
  widen   : adds label-free records to the simulation: validation records whose dense top-K holds a held-out
            S1, and half-B records with no in-play pair (see phase_widen)
  simpred : v9 stage-2 scores for the simulation's half-B and validation records
  dense   : rescue candidates (dense top-K S1s not already candidates) for open records; the simulation
            searches kept S1s only (dropped S1s must not be retrievable)
  gate    : label check on the simulation: which dense ranks hold the true rescues (sizes the ce phase)
  ce      : cross-encoder scores for rescue candidates with dense rank <= RESCUE_MAX_RANK (chunked)
  fit     : rescue model on half B; simulation score of v9 with and without the rescue lane
  write   : test output -> work/output_v10_dense/{matching_results,candidate_pairs}.tsv, and the same with
            France left as in v9 -> work/output_v10_usin/

  .venv\\Scripts\\python tools\\dense_rescue.py [phase[:split,...] ...]     # e.g. dense:test ce:test
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from rapidfuzz import fuzz, process

from config import WORK_DIR, work_path
from data_io import load_ground_truth, load_links, load_source, write_id_lists
import dense_probe as dp

D = WORK_DIR / "dense"
SIM_PRED2 = D / "sim_pred2.parquet"
T_OPEN = 0.7
K = 5
MAX_RANK = int(os.environ.get("RESCUE_MAX_RANK", K))
RESCUE_MODEL = WORK_DIR / "rescue_v10.txt"
DECISION = WORK_DIR / "decision_v10.json"
OUT_DIR = WORK_DIR / "output_v10_dense"
OUT_DIR_USIN = WORK_DIR / "output_v10_usin"
FEATS = ["ce", "ce_logit", "dense_cos", "dense_rank", "pf_name", "pf_addr", "best_p",
         "ce_rank_t", "ce_max_t", "ce_gap_t", "n_resc", "is_s3"]


def _p(name: str) -> Path:
    D.mkdir(parents=True, exist_ok=True)
    return D / name


def _dropped() -> pl.Series:
    from train import dropped_s1
    return pl.Series(list(dropped_s1(load_ground_truth("train"))))


def _cands_path(split: str) -> Path:
    from faithful_sim import CANDS
    return CANDS if split == "train" else work_path("test", "cands.parquet")


# ---------------------------------------------------------------- widen

WIDEN_T = D / "widen_tids.parquet"   # tid, part: records added to the simulation ("val" or "B")
WIDEN_CE = D / "widen_ce.parquet"    # tid, s1, ce: cross-encoder scores of the added validation records' in-play pairs
WIDEN_P1 = float(os.environ.get("WIDEN_P1", 2.0))  # optional: search only records with max p1 below this
# Share of eligible records searched (random, label-free). Each added record touches one held-out S1 and the
# metric averages over S1s, so the widening's effect scales with this share: full effect ~ measured / WIDEN_FRAC.
WIDEN_FRAC = float(os.environ.get("WIDEN_FRAC", 0.2))


def phase_widen() -> None:
    """Make the simulation see the records the rescue lane is for (label-free; runs before simpred).

    Validation: val_split() keeps only records with a held-out S1 among their top 8. A record whose true
    S1 was held out but missed by blocking usually has none, and neither do most records the lane could
    wrongly link to a held-out S1. Add every other record whose dense top-K (kept S1s) holds a held-out S1.
    Training: pool half B lists only records with an in-play pair (p1 >= 0.01), so the rescue model would
    never see records whose top 8 hold nothing plausible, the lane's main case on test. Add half-B records
    without in-play pairs, sampled at the rate half B was sampled.
    Records the cross-encoder trained on (pool half A pairs) or already in half B are never added. The added
    validation records' in-play pairs get cross-encoder scores, so simpred computes v9's p for them."""
    if WIDEN_CE.exists():
        return
    import ce_pilot
    from config import SEED
    from faithful_sim import CANDS, P1
    from train import val_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.time()
    truth_all = load_ground_truth("train")
    cands = pl.read_parquet(CANDS, columns=["tid", "s1"])
    val_s1, val_t, pool = val_split(cands, load_links("train"), truth_all)
    del cands
    b_t = pl.read_parquet(WORK_DIR / "ce" / "scores.parquet", columns=["tid", "part"]).filter(
        pl.col("part") == "B")["tid"].unique()
    a_t = pl.read_parquet(WORK_DIR / "ce" / "train_pairs.parquet", columns=["tid"])["tid"].unique()
    tgt = pl.scan_parquet(P1).group_by("tid").agg(pl.col("p").max().alias("p1_max")).collect()

    half1 = tgt.filter(pl.col("tid").is_in(pool.filter(ce_pilot._half(pool) == 1).implode()))
    in1 = half1.filter(pl.col("p1_max") >= ce_pilot.MIN_P)["tid"]
    rate = len(b_t) / len(in1)
    b_add = half1.filter(pl.col("p1_max") < ce_pilot.MIN_P)["tid"].sample(fraction=rate, seed=SEED + 300)
    print(f"[widen] half B: {len(b_t):,} records with in-play pairs ({rate:.3f} of half 1, "
          f"{len(b_t.filter(~b_t.is_in(in1.implode()))):,} outside it); adding {len(b_add):,} without", flush=True)

    excl = pl.concat([val_t, a_t, b_t, b_add]).unique()
    u = tgt.filter(~pl.col("tid").is_in(excl.implode()) & (pl.col("p1_max") < WIDEN_P1))["tid"]
    print(f"[widen] {len(u):,} eligible records; searching a random {WIDEN_FRAC:.0%}", flush=True)
    u = u.sort().sample(fraction=WIDEN_FRAC, seed=SEED + 400)
    del tgt
    raw = dp._raw("train", (2, 3), u)
    print(f"[widen] searching {len(raw):,} records for held-out S1s in their dense top-{K} ({time.time() - t0:.0f}s)",
          flush=True)
    s_emb, s_ids = dp.s1_embeddings("train")
    kept = ~s_ids["entity_id"].is_in(_dropped().implode())
    is_val = s_ids["entity_id"].is_in(pl.Series(list(val_s1)).implode()).to_numpy()
    tok, model = dp._load_model()
    added = []
    for c in sorted(raw["country"].unique().to_list()):
        rows = np.flatnonzero(((s_ids["country"] == c) & kept).to_numpy())
        rc = raw.filter(pl.col("country") == c)
        if len(rows) == 0:
            continue
        s_c, v_c = np.ascontiguousarray(s_emb[rows]), is_val[rows]
        for a in range(0, len(rc), 1_000_000):
            ch = rc.slice(a, 1_000_000)
            nn, _ = _topk(dp.embed(tok, model, dp._text(ch)), s_c, K)
            added.append(ch["entity_id"].filter(pl.Series(v_c[nn].any(axis=1))))
            print(f"[widen] {c}: {min(a + 1_000_000, len(rc)):,}/{len(rc):,} searched, "
                  f"{sum(len(x) for x in added):,} added so far ({time.time() - t0:.0f}s)", flush=True)
    del model, raw
    torch.cuda.empty_cache()
    w = pl.concat(added).unique()
    pl.concat([pl.DataFrame({"tid": w}).with_columns(pl.lit("val").alias("part")),
               pl.DataFrame({"tid": b_add}).with_columns(pl.lit("B").alias("part"))]).write_parquet(WIDEN_T)

    pairs = (pl.scan_parquet(P1).filter(pl.col("tid").is_in(w.implode()) & (pl.col("p") >= ce_pilot.MIN_P))
             .select("tid", "s1").collect())
    print(f"[widen] {len(w):,} validation records added; scoring their {len(pairs):,} in-play pairs", flush=True)
    txt = lambda df: dict(zip(df["entity_id"].to_list(), (df["business_name"] + " | " + df["business_address"]).to_list()))
    t_txt, s_txt = txt(dp._raw("train", (2, 3), pairs["tid"].unique())), txt(dp._raw("train", (1,), pairs["s1"].unique()))
    mdir = WORK_DIR / "ce" / "model"
    ctok = AutoTokenizer.from_pretrained(mdir)
    cmodel = AutoModelForSequenceClassification.from_pretrained(mdir).cuda()
    ce = ce_pilot._score(cmodel, ctok, t_txt, s_txt, pairs["tid"].to_list(), pairs["s1"].to_list())
    del cmodel
    torch.cuda.empty_cache()
    pairs.with_columns(pl.Series("ce", ce, dtype=pl.Float32)).write_parquet(WIDEN_CE)
    print(f"[widen] done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- simpred

def phase_simpred() -> None:
    if SIM_PRED2.exists():
        return
    assert WIDEN_CE.exists(), "run the widen phase first"
    import ce_v9  # noqa: F401  (patches ce_v8 to v9's features and file names)
    import ce_v8
    import stage2
    from faithful_sim import CANDS, P1
    from train import val_split
    truth_all = load_ground_truth("train")
    drop = _dropped()
    keep = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id"]).filter(
        ~pl.col("entity_id").is_in(drop.implode()))["entity_id"]
    cands = pl.read_parquet(CANDS, columns=["tid", "s1"])
    _, val_t, _ = val_split(cands, load_links("train"), truth_all)
    del cands
    pred = pl.read_parquet(P1)
    orig = stage2._norm
    stage2._norm = lambda split: (lambda s, t: (s.filter(~pl.col("entity_id").is_in(drop.implode())), t))(*orig(split))
    feats = stage2.group_features(pred, "train")
    stage2._norm = orig
    sc = pl.read_parquet(WORK_DIR / "ce" / "scores.parquet", columns=["tid", "s1", "part", "ce"])
    sc = pl.concat([sc, pl.read_parquet(WIDEN_CE).with_columns(pl.lit("val").alias("part")).select(sc.columns)])
    feats = ce_v8.add_features(feats.join(sc.select("tid", "s1", "part"), on=["tid", "s1"], how="inner"), sc, "train", keep)
    model = lgb.Booster(model_file=str(ce_v8.STAGE2))
    p2 = feats.select("tid", "s1").with_columns(
        pl.Series("p2", model.predict(feats.select(ce_v8._cols()).to_numpy(), num_threads=0), dtype=pl.Float32))
    del feats
    wt = pl.read_parquet(WIDEN_T)
    val_all = pl.concat([val_t, wt.filter(pl.col("part") == "val")["tid"]]).unique()
    tids = pl.concat([sc.filter(pl.col("part") == "B")["tid"], wt["tid"], val_t]).unique()
    (pred.filter(pl.col("tid").is_in(tids.implode())).join(p2, on=["tid", "s1"], how="left")
     .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
     .with_columns(pl.when(pl.col("tid").is_in(val_all.implode())).then(pl.lit("val")).otherwise(pl.lit("B")).alias("part"))
     .write_parquet(SIM_PRED2))


# ---------------------------------------------------------------- dense

def open_targets(split: str) -> pl.DataFrame:
    if split == "train":
        p = pl.read_parquet(SIM_PRED2)
        agg = [pl.col("p").max().alias("best_p"), pl.col("country").first(), pl.col("part").first()]
    else:
        p = pl.read_parquet(work_path("test", "pred2_v9.parquet"), columns=["tid", "country", "p"])
        agg = [pl.col("p").max().alias("best_p"), pl.col("country").first()]
    return p.group_by("tid").agg(agg).filter(pl.col("best_p") < T_OPEN)


@torch.no_grad()
def _topk(q: np.ndarray, s: np.ndarray, k: int, chunk: int = 256):
    S = torch.from_numpy(np.ascontiguousarray(s)).cuda()
    idx, val = np.empty((len(q), k), np.int64), np.empty((len(q), k), np.float32)
    for a in range(0, len(q), chunk):
        v, i = torch.topk(torch.from_numpy(np.ascontiguousarray(q[a:a + chunk])).cuda() @ S.T, k, dim=1)
        idx[a:a + chunk], val[a:a + chunk] = i.cpu().numpy(), v.float().cpu().numpy()
    del S
    torch.cuda.empty_cache()
    return idx, val


def _cheap(r: pl.DataFrame, split: str) -> pl.DataFrame:
    """The prefilter's two fast similarities (normalized name / address+numbers) and the source flag."""
    def norm(name, ids):
        return (pl.scan_parquet(work_path(split, name)).select("entity_id", "name", "addr", "nums")
                .filter(pl.col("entity_id").is_in(ids.implode())).collect()
                .select("entity_id", "name", pl.concat_str("addr", "nums", separator=" ").alias("an")))
    t = norm("t.parquet", r["tid"].unique()).rename({"entity_id": "tid", "name": "tn", "an": "ta"})
    s = norm("s1.parquet", r["s1"].unique()).rename({"entity_id": "s1", "name": "sn", "an": "sa"})
    d = r.join(t, on="tid", how="left").join(s, on="s1", how="left").with_columns(
        pl.col("tn", "sn", "ta", "sa").fill_null(""))
    # In 2M-row chunks: ~16M test rows as four Python string lists at once would exhaust 16 GB of RAM.
    sim = lambda c, a, b: process.cpdist(c[a].to_list(), c[b].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100
    step = 2_000_000
    chunks = [d.slice(a, step) for a in range(0, len(d), step)]
    pn = np.concatenate([sim(c, "tn", "sn") for c in chunks] or [np.empty(0)])
    pa = np.concatenate([sim(c, "ta", "sa") for c in chunks] or [np.empty(0)])
    return d.with_columns(pl.Series("pf_name", pn, dtype=pl.Float32),
                          pl.Series("pf_addr", pa, dtype=pl.Float32),
                          pl.col("tid").str.starts_with("S3").cast(pl.Int8).alias("is_s3")).drop("tn", "sn", "ta", "sa")


def phase_dense(splits=("train", "test")) -> None:
    for split in splits:
        out = _p(f"rescue_{split}.parquet")
        if out.exists():
            continue
        t0 = time.time()
        op = open_targets(split)
        s_emb, s_ids = dp.s1_embeddings(split)
        allowed = (~s_ids["entity_id"].is_in(_dropped().implode())) if split == "train" else pl.repeat(True, len(s_ids), eager=True)
        tok, model = dp._load_model()
        parts = []
        for c in sorted(op["country"].unique().to_list()):
            raw = dp._raw(split, (2, 3), op.filter(pl.col("country") == c)["tid"])
            rows = np.flatnonzero(((s_ids["country"] == c) & allowed).to_numpy())
            nn, cs = _topk(dp.embed(tok, model, dp._text(raw)), s_emb[rows], K)
            parts.append(pl.DataFrame({
                "tid": np.repeat(raw["entity_id"].to_numpy(), K),
                "s1": s_ids["entity_id"].to_numpy()[rows][nn.ravel()],
                "dense_cos": cs.ravel(),
                "dense_rank": np.tile(np.arange(1, K + 1, dtype=np.int16), len(raw)),
            }))
            print(f"[dense] {split} {c}: {len(raw):,} open records ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
        r = pl.concat(parts)
        existing = (pl.scan_parquet(_cands_path(split)).select("tid", "s1")
                    .filter(pl.col("tid").is_in(op["tid"].implode())).collect())
        r = r.join(existing, on=["tid", "s1"], how="anti").join(op, on="tid")
        r = _cheap(r, split)
        r.write_parquet(out)
        print(f"[dense] {split}: {len(r):,} rescue candidates for {r['tid'].n_unique():,} records ({time.time() - t0:.0f}s)",
              flush=True)


def phase_gate() -> None:
    r = pl.read_parquet(_p("rescue_train.parquet")).join(
        load_links("train").with_columns(pl.lit(True).alias("y")), on=["tid", "s1"], how="left").with_columns(
        pl.col("y").fill_null(False))
    pos = int(r["y"].sum())
    print(f"[gate] simulation: {len(r):,} rescue candidates, {pos:,} true", flush=True)
    for mr in range(1, K + 1):
        sub = r.filter(pl.col("dense_rank") <= mr)
        print(f"  dense rank <= {mr}: {len(sub):,} pairs ({len(sub) / len(r):.0%}), true kept {sub['y'].sum() / max(pos, 1):.1%}",
              flush=True)
    print(r.group_by("country").agg(pl.len(), pl.col("y").sum().alias("true")).sort("country"), flush=True)
    if _p("rescue_test.parquet").exists():
        t = pl.read_parquet(_p("rescue_test.parquet"), columns=["dense_rank", "country"])
        print("[gate] test rescue candidates by dense rank:", t.group_by("dense_rank").len().sort("dense_rank").rows(),
              flush=True)


# ---------------------------------------------------------------- ce

def phase_ce(splits=("train", "test")) -> None:
    import ce_pilot
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    mdir = WORK_DIR / "ce" / "model"
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda()
    for split in splits:
        out = _p(f"rescue_ce_{split}.parquet")
        if out.exists():
            continue
        r = pl.read_parquet(_p(f"rescue_{split}.parquet"), columns=["tid", "s1", "dense_rank"]).filter(
            pl.col("dense_rank") <= MAX_RANK).select("tid", "s1")
        txt = lambda df: dict(zip(df["entity_id"].to_list(), (df["business_name"] + " | " + df["business_address"]).to_list()))
        t_txt, s_txt = txt(dp._raw(split, (2, 3), r["tid"].unique())), txt(dp._raw(split, (1,), r["s1"].unique()))
        cdir = _p(f"rescue_ce_{split}_r{MAX_RANK}")
        cdir.mkdir(exist_ok=True)
        step = 1_000_000
        n = (len(r) + step - 1) // step
        for i in range(n):
            f = cdir / f"{i:03d}.parquet"
            if f.exists():
                continue
            part = r.slice(i * step, step)
            ce = ce_pilot._score(model, tok, t_txt, s_txt, part["tid"].to_list(), part["s1"].to_list())
            part.with_columns(pl.Series("ce", ce)).write_parquet(f)
            print(f"[ce] {split} chunk {i + 1}/{n} saved", flush=True)
            torch.cuda.empty_cache()
        pl.concat([pl.read_parquet(cdir / f"{i:03d}.parquet") for i in range(n)]).write_parquet(out)


# ---------------------------------------------------------------- fit / write

def features(split: str) -> pl.DataFrame:
    d = pl.read_parquet(_p(f"rescue_{split}.parquet")).join(
        pl.read_parquet(_p(f"rescue_ce_{split}.parquet")), on=["tid", "s1"], how="inner")
    c = pl.col("ce").clip(1e-6, 1 - 1e-6)
    return d.with_columns(
        (c.log() - (1 - c).log()).alias("ce_logit"),
        pl.col("ce").rank("ordinal", descending=True).over("tid").cast(pl.Int16).alias("ce_rank_t"),
        pl.col("ce").max().over("tid").alias("ce_max_t"),
        pl.len().over("tid").cast(pl.Int16).alias("n_resc"),
    ).with_columns((pl.col("ce_max_t") - pl.col("ce")).alias("ce_gap_t"))


def phase_fit() -> None:
    from faithful_sim import CANDS
    from stage2 import PARAMS2
    from train import fit, tune, val_context, val_split
    links = load_links("train")
    d = features("train").join(links.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["tid", "s1"], how="left") \
        .with_columns(pl.col("y").fill_null(0))
    tr, va = d.filter(pl.col("part") == "B"), d.filter(pl.col("part") == "val")
    print(f"[fit] rescue rows: half B {len(tr):,} ({int(tr['y'].sum()):,} true), validation {len(va):,} "
          f"({int(va['y'].sum()):,} true)", flush=True)
    model = fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy(), tr["tid"], FEATS, PARAMS2, 3000)
    model.save_model(str(RESCUE_MODEL))
    truth_all = load_ground_truth("train")
    cands = pl.read_parquet(CANDS, columns=["tid", "s1"])
    val_s1, val_t, _ = val_split(cands, links, truth_all)
    ctx = val_context(cands, truth_all, val_s1)
    del cands
    base = pl.read_parquet(SIM_PRED2).filter(pl.col("part") == "val").select("tid", "s1", "country", "p")
    print("[fit] v9 alone on the simulation (should reproduce 0.9857):", flush=True)
    tune(base, *ctx, out="_v10_base.json")
    (WORK_DIR / "_v10_base.json").unlink(missing_ok=True)
    resc = va.select("tid", "s1", "country").with_columns(
        pl.Series("p", model.predict(va.select(FEATS).to_numpy(), num_threads=0), dtype=pl.Float32))
    print("[fit] v9 + rescue lane, original validation records only (before widening):", flush=True)
    tune(pl.concat([base, resc.filter(pl.col("tid").is_in(val_t.implode()))]), *ctx, out="_v10_orig.json")
    (WORK_DIR / "_v10_orig.json").unlink(missing_ok=True)
    print("[fit] v9 + rescue lane, widened validation (the decision):", flush=True)
    tune(pl.concat([base, resc]), *ctx, out=DECISION.name)


def _write_cands(path: Path, s1_ids: list, extra: pl.DataFrame) -> None:
    """Candidate set = blocking top-8 plus the scored rescue candidates, written without building a Python dict."""
    cands = (pl.concat([pl.scan_parquet(work_path("test", "cands.parquet")).select("tid", "s1").collect(), extra])
             .group_by("s1").agg(pl.col("tid").unique(maintain_order=True).str.join(",").alias("candidate_entity_ids")))
    (pl.DataFrame({"source1_entity_id": s1_ids}).join(cands.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
     .with_columns(pl.col("candidate_entity_ids").fill_null(""))
     .write_csv(path, separator="\t", quote_style="never", line_terminator="\n"))
    print(f"[write] {path} written", flush=True)


def phase_write() -> None:
    """Two outputs: OUT_DIR (rescue lane in every country) and OUT_DIR_USIN (France keeps v9's links: France
    has no labels, and its unlinked records get far more plausible rescue candidates than India's)."""
    from assign import assign
    model = lgb.Booster(model_file=str(RESCUE_MODEL))
    dec = json.loads(DECISION.read_text())
    d = features("test")
    resc = d.select("tid", "s1", "country").with_columns(
        pl.Series("p", model.predict(d.select(FEATS).to_numpy(), num_threads=0), dtype=pl.Float32))
    rkeys = set(zip(resc["s1"].to_list(), resc["tid"].to_list()))
    matches, usin = {}, {}
    for c in ("France", "India", "US"):
        th = dec["threshold"] if c in ("US", "India") else max(dec["threshold"], 0.6)
        pred = pl.scan_parquet(work_path("test", "pred2_v9.parquet")).filter(pl.col("country") == c).select(
            "tid", "s1", "country", "p").collect()
        m = assign(pl.concat([pred, resc.filter(pl.col("country") == c)]), th, dec["expected_f"])
        n_resc = sum(1 for s, v in m.items() for t in v if (s, t) in rkeys)
        print(f"[write] {c}: threshold {th}, {sum(len(v) for v in m.values()):,} links, {n_resc:,} from the rescue lane",
              flush=True)
        matches.update(m)
        if c == "France":
            m0 = assign(pred, th, dec["expected_f"])
            print(f"[write] France without the rescue lane: {sum(len(v) for v in m0.values()):,} links", flush=True)
            usin.update(m0)
        else:
            usin.update(m)
        del pred
    s1_ids = load_source("test", 1)["entity_id"].to_list()
    write_id_lists(OUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    write_id_lists(OUT_DIR_USIN / "matching_results.tsv", s1_ids, usin, "matched_entity_ids")
    print(f"[write] {len(matches):,} / {len(usin):,} S1 with matches -> {OUT_DIR}, {OUT_DIR_USIN}", flush=True)
    extra = d.select("tid", "s1", "country")
    del d, resc, rkeys, matches, usin  # free RAM before the ~96M-row candidate tables
    _write_cands(OUT_DIR / "candidate_pairs.tsv", s1_ids, extra.select("tid", "s1"))
    _write_cands(OUT_DIR_USIN / "candidate_pairs.tsv", s1_ids,
                 extra.filter(pl.col("country") != "France").select("tid", "s1"))


PHASES = {"widen": phase_widen, "simpred": phase_simpred, "dense": phase_dense, "gate": phase_gate, "ce": phase_ce,
          "fit": phase_fit, "write": phase_write}

if __name__ == "__main__":
    # "phase" or "phase:split[,split]" (dense and ce), e.g. dense:test ce:test runs only the test side.
    for arg in (sys.argv[1:] or list(PHASES)):
        ph, _, sp = arg.partition(":")
        t0 = time.time()
        PHASES[ph](*([tuple(sp.split(","))] if sp else []))
        print(f"[{arg}] done ({time.time() - t0:.0f}s)", flush=True)
