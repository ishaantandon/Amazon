"""Cross-encoder pilot: does a fine-tuned multilingual transformer add signal on top of stage 2?

Everything runs in the faithful test-like simulation (tools/faithful_sim.py must have run).
Pool targets are split by hash into two halves so the comparison is leak-free:
  A: the cross-encoder is fine-tuned on in-play pairs (stage-1 p >= MIN_P) of these targets
  B: stage 2 is retrained on these targets, once without and once with the cross-encoder score
Both stage-2 variants are evaluated on the same held-out S1 entities.

Model: intfloat/multilingual-e5-small (MIT, 118M params) with a 1-logit classification head,
reading "name | address" of the S2/S3 record and of the candidate S1 as a text pair (raw text,
so native scripts and accents are kept).

Phases (cached under work/ce/):
  prep  : pair lists            -> work/ce/train_pairs.parquet, work/ce/score_pairs.parquet
  train : fine-tune             -> work/ce/model/
  score : score B + val pairs   -> work/ce/scores.parquet
  eval  : stage 2 A/B on the held-out entities

  .venv\\Scripts\\python tools\\ce_pilot.py [phase ...]     # default: all phases, skipping cached ones
"""
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import SEED, WORK_DIR, work_path
from data_io import load_ground_truth, load_links, load_source, load_targets

# Overridable per run (env), so a second model never overwrites the first: CE_TAG=ce_base -> work/ce_base/
CE_TAG = os.environ.get("CE_TAG", "ce")
CE_DIR = WORK_DIR / CE_TAG
BASE_MODEL = os.environ.get("CE_BASE_MODEL", "intfloat/multilingual-e5-small")
MIN_P = 0.01
N_CE_TRAIN_PAIRS = int(os.environ.get("CE_TRAIN_PAIRS", 1_200_000))
N_S2_TARGETS = 1_500_000
MAX_LEN = 128
TRAIN_BATCH = int(os.environ.get("CE_TRAIN_BATCH", 128))
SCORE_BATCH = int(os.environ.get("CE_SCORE_BATCH", 1024))
LR = float(os.environ.get("CE_LR", 5e-5))
CKPT_EVERY = 2000


def _p(name: str) -> Path:
    CE_DIR.mkdir(parents=True, exist_ok=True)
    return CE_DIR / name


def _half(s: pl.Series) -> pl.Series:
    return (s.hash(SEED + 200) % 2).cast(pl.Int8)


# ---------------------------------------------------------------- prep

def prep() -> None:
    if _p("score_pairs.parquet").exists():
        return
    from train import val_split
    from faithful_sim import CANDS, P1
    truth_all = load_ground_truth("train")
    links = load_links("train")
    cands = pl.read_parquet(CANDS, columns=["tid", "s1"])
    val_s1, val_t, pool = val_split(cands, links, truth_all)
    del cands
    inplay = pl.read_parquet(P1).filter(pl.col("p") >= MIN_P).select("tid", "s1")
    inplay = inplay.join(links.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["tid", "s1"], how="left") \
                   .with_columns(pl.col("y").fill_null(0))
    a_t, b_t = pool.filter(_half(pool) == 0), pool.filter(_half(pool) == 1)

    tr = inplay.filter(pl.col("tid").is_in(a_t.implode()))
    tr = tr.filter(pl.col("tid").is_in(tr["tid"].unique().sample(fraction=1.0, shuffle=True, seed=SEED)
                                       .head(int(N_CE_TRAIN_PAIRS / (len(tr) / tr["tid"].n_unique()))).implode()))
    tr.sample(fraction=1.0, shuffle=True, seed=SEED).write_parquet(_p("train_pairs.parquet"))

    b_s = b_t.sample(min(N_S2_TARGETS, len(b_t)), seed=SEED)
    sc = pl.concat([inplay.filter(pl.col("tid").is_in(b_s.implode())).with_columns(pl.lit("B").alias("part")),
                    inplay.filter(pl.col("tid").is_in(val_t.implode())).with_columns(pl.lit("val").alias("part"))])
    sc.write_parquet(_p("score_pairs.parquet"))
    print(f"[prep] CE train pairs {len(tr):,} (pos {tr['y'].mean():.3f}); to score: "
          f"B {int((sc['part'] == 'B').sum()):,}, val {int((sc['part'] == 'val').sum()):,}", flush=True)


# ---------------------------------------------------------------- text

def _texts(split: str = "train") -> tuple[dict, dict]:
    def txt(df):
        return dict(zip(df["entity_id"].to_list(),
                        (df["business_name"] + " | " + df["business_address"]).to_list()))
    return txt(load_targets(split)), txt(load_source(split, 1))


def _encode(tok, t_txt, s_txt, tids, s1s):
    # Lengths rounded up to a multiple of 16: few distinct shapes, so the CUDA allocator cache stays small.
    return tok([t_txt[t] for t in tids], [s_txt[s] for s in s1s], truncation=True, max_length=MAX_LEN,
               padding=True, pad_to_multiple_of=16, return_tensors="pt")


# ---------------------------------------------------------------- train

def train() -> None:
    out = _p("model")
    if (out / "config.json").exists():
        return
    torch.manual_seed(SEED)
    pairs = pl.read_parquet(_p("train_pairs.parquet"))
    t_txt, s_txt = _texts()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1).cuda()
    n_hold = 20_000
    hold, fit_ = pairs.head(n_hold), pairs.slice(n_hold)
    steps = math.ceil(len(fit_) / TRAIN_BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    tids, s1s, ys = fit_["tid"].to_list(), fit_["s1"].to_list(), fit_["y"].to_numpy()
    # Checkpoint every CKPT_EVERY steps; a crashed run resumes where it stopped.
    ckpt, start = _p("ckpt.pt"), 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        start = st["step"]
        print(f"[train] resumed from step {start:,}", flush=True)
    t0 = time.time()
    model.train()
    for i in range(start, steps):
        sl = slice(i * TRAIN_BATCH, (i + 1) * TRAIN_BATCH)
        try:
            enc = _encode(tok, t_txt, s_txt, tids[sl], s1s[sl]).to("cuda")
            y = torch.tensor(ys[sl], dtype=torch.float32, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), y)
            loss.backward()
        except torch.OutOfMemoryError:
            print(f"[train] out of memory at step {i + 1:,}, batch skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % CKPT_EVERY == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": i + 1}, ckpt)
        if (i + 1) % 500 == 0 or i + 1 == steps:
            print(f"[train] step {i + 1:,}/{steps:,} loss {loss.item():.4f} "
                  f"({(i + 1 - start) * TRAIN_BATCH / (time.time() - t0):.0f} pairs/s)", flush=True)
    p = _score(model, tok, t_txt, s_txt, hold["tid"].to_list(), hold["s1"].to_list())
    y = hold["y"].to_numpy()
    ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    print(f"[train] holdout {n_hold:,} pairs: logloss {ll:.4f}, accuracy {((p >= 0.5) == y).mean():.4f}, "
          f"positive rate {y.mean():.3f}", flush=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    ckpt.unlink(missing_ok=True)


@torch.no_grad()
def _score(model, tok, t_txt, s_txt, tids, s1s) -> np.ndarray:
    model.eval()
    # Length-sorted batches keep padding (and time) low.
    lens = np.array([len(t_txt[t]) + len(s_txt[s]) for t, s in zip(tids, s1s)])
    order = np.argsort(lens)
    out = np.empty(len(tids), dtype=np.float32)
    t0 = time.time()
    def run(idx):
        try:
            enc = _encode(tok, t_txt, s_txt, [tids[i] for i in idx], [s1s[i] for i in idx]).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
            # A transient GPU allocation failure (seen once on test): free the cache, retry in halves.
            if len(idx) == 1 or "memory" not in str(e).lower() and "CUBLAS" not in str(e):
                raise
            print(f"[score] GPU error on a batch of {len(idx)}, retrying in halves: {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()
            run(idx[:len(idx) // 2])
            run(idx[len(idx) // 2:])

    for a in range(0, len(order), SCORE_BATCH):
        idx = order[a:a + SCORE_BATCH]
        run(idx)
        if (a // SCORE_BATCH) % 100 == 99:
            torch.cuda.empty_cache()  # on Windows a full VRAM spills into slow shared memory
        if (a // SCORE_BATCH) % 500 == 0:
            print(f"[score] {a + len(idx):,}/{len(order):,} ({(a + len(idx)) / (time.time() - t0):.0f} pairs/s)",
                  flush=True)
    model.train()
    return out


# ---------------------------------------------------------------- score

def score() -> None:
    if _p("scores.parquet").exists():
        return
    pairs = pl.read_parquet(_p("score_pairs.parquet"))
    t_txt, s_txt = _texts()
    tok = AutoTokenizer.from_pretrained(_p("model"))
    model = AutoModelForSequenceClassification.from_pretrained(_p("model")).cuda()
    ce = _score(model, tok, t_txt, s_txt, pairs["tid"].to_list(), pairs["s1"].to_list())
    pairs.with_columns(pl.Series("ce", ce)).write_parquet(_p("scores.parquet"))


# ---------------------------------------------------------------- eval

def evaluate() -> None:
    import lightgbm as lgb
    import stage2
    from faithful_sim import CANDS, P1
    from stage2 import FEATURES2, PARAMS2, rescore
    from train import dropped_s1, fit, tune, val_context, val_split
    truth_all = load_ground_truth("train")
    drop = pl.Series(list(dropped_s1(truth_all)))
    links = load_links("train")
    cands = pl.read_parquet(CANDS, columns=["tid", "s1"])
    val_s1, val_t, _ = val_split(cands, links, truth_all)
    ctx = val_context(cands, truth_all, val_s1)
    del cands
    pred = pl.read_parquet(P1)
    orig = stage2._norm
    stage2._norm = lambda split: (lambda s, t: (s.filter(~pl.col("entity_id").is_in(drop.implode())), t))(*orig(split))
    feats = stage2.group_features(pred, "train")
    stage2._norm = orig
    sc = pl.read_parquet(_p("scores.parquet"))
    feats = feats.join(sc.select("tid", "s1", "ce", "part"), on=["tid", "s1"], how="inner").with_columns(
        (pl.col("ce").clip(1e-6, 1 - 1e-6).log() - (1 - pl.col("ce").clip(1e-6, 1 - 1e-6)).log()).alias("ce_logit"))
    tr = feats.filter(pl.col("part") == "B").join(links.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")),
                                                  on=["tid", "s1"], how="left").with_columns(pl.col("y").fill_null(0))
    va = feats.filter(pl.col("part") == "val")
    vp = pred.filter(pl.col("tid").is_in(val_t.implode()))
    print(f"[eval] stage-2 train rows {len(tr):,} (pos {tr['y'].mean():.3f}), val in-play rows {len(va):,}", flush=True)
    for name, cols in [("stage 2 without cross-encoder", FEATURES2),
                       ("stage 2 WITH cross-encoder", FEATURES2 + ["ce", "ce_logit"])]:
        model = fit(tr.select(cols).to_numpy(), tr["y"].to_numpy(), tr["tid"], cols, PARAMS2, 3000)
        p2 = va.select("tid", "s1").with_columns(
            pl.Series("p2", model.predict(va.select(cols).to_numpy(), num_threads=0), dtype=pl.Float32))
        out = vp.join(p2, on=["tid", "s1"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
        print(f"[eval] {name}:", flush=True)
        tune(out, *ctx, out="_ce_pilot_decision.json")
    (WORK_DIR / "_ce_pilot_decision.json").unlink(missing_ok=True)


PHASES = {"prep": prep, "train": train, "score": score, "eval": evaluate}

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for ph in (sys.argv[1:] or list(PHASES)):
        t0 = time.time()
        PHASES[ph]()
        print(f"[{ph}] done ({time.time() - t0:.0f}s)", flush=True)
