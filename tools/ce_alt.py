"""Alternatives to the cross-encoder (step 5 in readme2.md), each scored in the exact v9 stage-2 setup.

Every alternative gives one score in [0, 1] to the same 2,355,363 simulation pairs the cross-encoder
scored (work/ce/scores.parquet: pool half B to fit stage 2, 'val' = the held-out S1s' targets). `eval`
fits v9's stage 2 (FEATURES2 + v8 + v9 features, the five 'ce' features computed from that score) on
half B and tunes it on the held-out S1s, as tools/ce_v9.py fit did for v9 (0.98565).

  tfidf  : cosine of char-3-gram TF-IDF vectors of the raw "name | address" (classical, no network)
  vae    : cosine of the latent means of a char-3-gram variational autoencoder (Mult-VAE, ~67M
           parameters) trained on the records' text without labels
  bienc0 : cosine of off-the-shelf multilingual-e5-small embeddings (bi-encoder, zero-shot)
  bienc  : the same bi-encoder fine-tuned on the cross-encoder's own training pairs (pool half A, same
           batch size, learning rate and schedule); score = sigmoid(w * cosine + b)
  eval   : v9's stage 2 with each score, plus 'ce' (the cross-encoder itself) and 'none' (no text model)
  solo   : the fine-tuned bi-encoder alone: no blocking (exact search over every kept S1 of the
           country), no pair features, no LightGBM; its top-8 S1s per target go to the usual assignment

Outputs go to work/ce_alt/ only; v9's files are never touched.
  .venv\\Scripts\\python tools\\ce_alt.py [phase ...]      # default: tfidf vae bienc0 bienc eval solo
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from config import SEED, WORK_DIR, work_path

D = WORK_DIR / "ce_alt"
E5 = "intfloat/multilingual-e5-small"
CE_COLS = ["ce", "ce_logit", "ce_gap", "ce_rank", "ce_best_other"]
_TXT = {}


def _p(name: str) -> Path:
    D.mkdir(parents=True, exist_ok=True)
    return D / name


def _texts() -> tuple[dict, dict]:
    """Raw "name | address" of every train target and S1 (the cross-encoder's input text)."""
    if not _TXT:
        import ce_pilot
        _TXT["t"], _TXT["s"] = ce_pilot._texts()
    return _TXT["t"], _TXT["s"]


def _pairs() -> pl.DataFrame:
    return pl.read_parquet(WORK_DIR / "ce" / "scores.parquet", columns=["tid", "s1", "y", "part"])


def _index(pairs: pl.DataFrame):
    tids, sids = pairs["tid"].unique().to_list(), pairs["s1"].unique().to_list()
    ti, si = {t: i for i, t in enumerate(tids)}, {s: i for i, s in enumerate(sids)}
    return tids, sids, np.array([ti[t] for t in pairs["tid"].to_list()]), np.array([si[s] for s in pairs["s1"].to_list()])


def _save(name: str, pairs: pl.DataFrame, score: np.ndarray) -> None:
    from sklearn.metrics import roc_auc_score
    score = np.asarray(score, dtype=np.float32)
    pairs.select("tid", "s1", "part").with_columns(pl.Series("ce", score)).write_parquet(_p(f"{name}_scores.parquet"))
    v = (pairs["part"] == "val").to_numpy()
    print(f"[{name}] raw-score AUC on the validation pairs: {roc_auc_score(pairs['y'].to_numpy()[v], score[v]):.4f}",
          flush=True)


def _rowdot(A, B, ia, ib, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(ia), dtype=np.float32)
    for a in range(0, len(ia), chunk):
        x, y = A[ia[a:a + chunk]], B[ib[a:a + chunk]]
        out[a:a + chunk] = np.asarray(x.multiply(y).sum(1)).ravel() if hasattr(x, "multiply") else (x * y).sum(1)
    return out


# ---------------------------------------------------------------- tf-idf

def phase_tfidf() -> None:
    if _p("tfidf_scores.parquet").exists():
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    pairs = _pairs()
    t_txt, s_txt = _texts()
    tids, sids, ia, ib = _index(pairs)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, dtype=np.float32)
    X = vec.fit_transform([t_txt[t] for t in tids] + [s_txt[s] for s in sids])  # rows are L2-normalized
    _save("tfidf", pairs, _rowdot(X[:len(tids)], X[len(tids):], ia, ib))


# ---------------------------------------------------------------- variational autoencoder

class MultVAE(torch.nn.Module):
    """Mult-VAE over hashed char 3-grams: bag -> 512 -> 512 -> (mu, logvar) of 128 -> 512 -> softmax over grams."""

    def __init__(self, V: int, H: int = 512, Z: int = 128):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(V, H, mode="sum")
        self.h = torch.nn.Linear(H, H)
        self.mu, self.lv = torch.nn.Linear(H, Z), torch.nn.Linear(H, Z)
        self.d1, self.d2 = torch.nn.Linear(Z, H), torch.nn.Linear(H, V)

    def encode(self, idx, off, w):
        h = torch.tanh(self.h(torch.tanh(self.emb(idx, off, per_sample_weights=w))))
        return self.mu(h), self.lv(h)

    def decode(self, z):
        return self.d2(torch.tanh(self.d1(z)))


def _bag(X, rows):
    sub = X[rows]
    rs = np.repeat(np.arange(len(rows)), np.diff(sub.indptr))
    tot = np.asarray(sub.sum(1)).ravel().astype(np.float32)
    tot[tot == 0] = 1
    cuda = lambda a: torch.from_numpy(a).cuda()
    return (cuda(sub.indices.astype(np.int64)), cuda(sub.indptr[:-1].astype(np.int64)),
            cuda(sub.data.astype(np.float32) / tot[rs]), cuda(rs), sub)


def phase_vae() -> None:
    if _p("vae_scores.parquet").exists():
        return
    from sklearn.feature_extraction.text import HashingVectorizer
    V, B, EPOCHS = 2 ** 16, 1024, 2
    pairs = _pairs()
    tp = pl.read_parquet(WORK_DIR / "ce" / "train_pairs.parquet")
    t_txt, s_txt = _texts()
    tids, sids, ia, ib = _index(pairs)
    extra_t = sorted(set(tp["tid"].to_list()) - set(tids))
    extra_s = sorted(set(tp["s1"].to_list()) - set(sids))
    texts = [t_txt[t] for t in tids] + [s_txt[s] for s in sids] + [t_txt[t] for t in extra_t] + [s_txt[s] for s in extra_s]
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=V, alternate_sign=False, norm=None,
                           dtype=np.float32)
    t0 = time.time()
    X = hv.transform(texts).tocsr()
    print(f"[vae] {X.shape[0]:,} texts hashed ({time.time() - t0:.0f}s)", flush=True)
    torch.manual_seed(SEED)
    model = MultVAE(V).cuda()
    print(f"[vae] parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    steps = EPOCHS * math.ceil(X.shape[0] / B)
    step, t0 = 0, time.time()
    model.train()
    for ep in range(EPOCHS):
        order = rng.permutation(X.shape[0])
        for a in range(0, len(order), B):
            idx, off, w, rs, sub = _bag(X, order[a:a + B])
            mu, lv = model.encode(idx, off, w)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.decode(z)
            logp = F.log_softmax(logits.float(), dim=-1)
            x = torch.zeros(sub.shape[0], V, device="cuda")
            x.index_put_((rs, idx), torch.from_numpy(sub.data.astype(np.float32)).cuda(), accumulate=True)
            rec = -(x * logp).sum(1)
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)
            beta = min(0.2, 0.2 * step / (0.5 * steps))  # KL annealing, as in Mult-VAE
            loss = (rec + beta * kl).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 1000 == 0 or step == steps:
                print(f"[vae] step {step:,}/{steps:,} rec {rec.mean().item():.2f} kl {kl.mean().item():.2f} "
                      f"beta {beta:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    n = len(tids) + len(sids)
    mus = np.empty((n, 128), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, n, 8192):
            idx, off, w, _, _ = _bag(X, np.arange(a, min(a + 8192, n)))
            mus[a:a + 8192] = F.normalize(model.encode(idx, off, w)[0], dim=-1).cpu().numpy()
    torch.save(model.state_dict(), _p("vae.pt"))
    _save("vae", pairs, (_rowdot(mus[:len(tids)], mus[len(tids):], ia, ib) + 1) / 2)


# ---------------------------------------------------------------- bi-encoders

def phase_bienc0() -> None:
    if _p("bienc0_scores.parquet").exists():
        return
    import dense_probe as dp
    s_emb, s_ids = dp.s1_embeddings("train")  # cached: every train S1, "query: name | address"
    pairs = _pairs()
    t_txt, _ = _texts()
    tids, sids, ia, ib = _index(pairs)
    tok, model = dp._load_model()
    t_emb = dp.embed(tok, model, ["query: " + t_txt[t] for t in tids])
    del model
    torch.cuda.empty_cache()
    row = dict(zip(s_ids["entity_id"].to_list(), range(len(s_ids))))
    s_sel = np.asarray(s_emb[np.array([row[s] for s in sids])], dtype=np.float32)
    _save("bienc0", pairs, (_rowdot(t_emb.astype(np.float32), s_sel, ia, ib) + 1) / 2)


def _pool(model, tok, texts: list[str]) -> torch.Tensor:
    enc = tok(texts, truncation=True, max_length=64, padding=True, pad_to_multiple_of=16, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)


def _train_bienc(out: Path) -> None:
    """Same data, batch size, learning rate, warmup and schedule as the cross-encoder (ce_pilot.train)."""
    import ce_pilot
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(SEED)
    pairs = pl.read_parquet(WORK_DIR / "ce" / "train_pairs.parquet")
    t_txt, s_txt = _texts()
    tok = AutoTokenizer.from_pretrained(E5)
    model = AutoModel.from_pretrained(E5).cuda()
    head = torch.nn.Linear(1, 1).cuda()  # logit = w * cosine + b
    with torch.no_grad():
        head.weight.fill_(20.0), head.bias.fill_(-17.0)
    n_hold, bs = 20_000, ce_pilot.TRAIN_BATCH
    hold, fit_ = pairs.head(n_hold), pairs.slice(n_hold)
    steps = math.ceil(len(fit_) / bs)
    opt = torch.optim.AdamW([{"params": model.parameters()}, {"params": head.parameters(), "lr": 1e-2}],
                            lr=ce_pilot.LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    T = ["query: " + t_txt[t] for t in fit_["tid"].to_list()]
    S = ["query: " + s_txt[s] for s in fit_["s1"].to_list()]
    ys = fit_["y"].to_numpy()
    ckpt, start = _p("bienc_ckpt.pt"), 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)  # on CPU: a GPU copy stays resident and spills VRAM
        model.load_state_dict(st["model"]), head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        start = st["step"]
        del st
        torch.cuda.empty_cache()
        print(f"[bienc] resumed from step {start:,}", flush=True)
    model.train()
    t0 = time.time()
    for i in range(start, steps):
        sl = slice(i * bs, (i + 1) * bs)
        y = torch.tensor(ys[sl], dtype=torch.float32, device="cuda")
        cos = (_pool(model, tok, T[sl]) * _pool(model, tok, S[sl])).sum(-1, keepdim=True)
        loss = lossf(head(cos).squeeze(-1), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 2000 == 0:
            torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": i + 1}, ckpt)
        if (i + 1) % 500 == 0 or i + 1 == steps:
            print(f"[bienc] step {i + 1:,}/{steps:,} loss {loss.item():.4f} "
                  f"({(i + 1 - start) * bs / (time.time() - t0):.0f} pairs/s)", flush=True)
    if start < steps:
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": steps}, ckpt)
    model.eval()
    w, b = head.weight.item(), head.bias.item()
    with torch.no_grad():
        p = []
        for a in range(0, n_hold, 512):
            h = hold.slice(a, 512)
            c = (_pool(model, tok, ["query: " + t_txt[t] for t in h["tid"].to_list()]) *
                 _pool(model, tok, ["query: " + s_txt[s] for s in h["s1"].to_list()])).sum(-1)
            p.append(torch.sigmoid(w * c + b).cpu().numpy())
    p, y = np.concatenate(p), hold["y"].to_numpy()
    ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    print(f"[bienc] holdout {n_hold:,} pairs: logloss {ll:.4f}, accuracy {((p >= 0.5) == y).mean():.4f} "
          f"(cross-encoder: 0.0564, 0.9767)", flush=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "head.json").write_text(json.dumps({"w": w, "b": b}))
    ckpt.unlink(missing_ok=True)


def _load_bienc():
    from transformers import AutoModel, AutoTokenizer
    out = _p("bienc_model")
    if not (out / "config.json").exists():
        _train_bienc(out)
    hd = json.loads((out / "head.json").read_text())
    return AutoTokenizer.from_pretrained(out), AutoModel.from_pretrained(out).cuda().eval(), hd["w"], hd["b"]


def phase_bienc() -> None:
    if _p("bienc_scores.parquet").exists():
        return
    import dense_probe as dp
    tok, model, w, b = _load_bienc()
    pairs = _pairs()
    t_txt, s_txt = _texts()
    tids, sids, ia, ib = _index(pairs)
    te = dp.embed(tok, model, ["query: " + t_txt[t] for t in tids]).astype(np.float32)
    se = dp.embed(tok, model, ["query: " + s_txt[s] for s in sids]).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    _save("bienc", pairs, 1 / (1 + np.exp(-(w * _rowdot(te, se, ia, ib) + b))))


# ---------------------------------------------------------------- evaluation

def _ce_cols(ce: pl.DataFrame) -> pl.DataFrame:
    """The five 'ce' features exactly as tools/ce_v8.add_features derives them."""
    c = pl.col("ce").clip(1e-6, 1 - 1e-6)
    ce = ce.select("tid", "s1", "ce").with_columns(
        (c.log() - (1 - c).log()).alias("ce_logit"),
        (pl.col("ce").max().over("tid") - pl.col("ce")).alias("ce_gap"),
        pl.col("ce").rank("ordinal", descending=True).over("tid").cast(pl.Int16).alias("ce_rank"),
    )
    top2 = ce.group_by("tid").agg(pl.col("ce").sort(descending=True).head(2).alias("_t"))
    return ce.join(top2, on="tid").with_columns(
        pl.when(pl.col("ce_rank") == 1).then(pl.col("_t").list.get(1, null_on_oob=True))
        .otherwise(pl.col("_t").list.first()).fill_null(0.0).alias("ce_best_other")).drop("_t")


def _sim_context():
    from data_io import load_ground_truth, load_links
    from faithful_sim import CANDS
    from train import dropped_s1, val_context, val_split
    truth_all = load_ground_truth("train")
    drop = pl.Series(list(dropped_s1(truth_all)))
    keep = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id"]).filter(
        ~pl.col("entity_id").is_in(drop.implode()))["entity_id"]
    links = load_links("train")
    cands = pl.read_parquet(CANDS, columns=["tid", "s1"])
    val_s1, val_t, _ = val_split(cands, links, truth_all)
    ctx = val_context(cands, truth_all, val_s1)
    return drop, keep, links, val_t, ctx


def _summary(name: str, out: pl.DataFrame, dec: dict, ctx, sub: dict) -> dict:
    from assign import assign
    from metrics import evaluate
    from train import _to_sets
    truth, cand_sets, country = ctx
    links = assign(out, dec["threshold"], dec["expected_f"])
    r = {"name": name, **dec}
    for label, t in [("all", truth), ("US", {s: v for s, v in truth.items() if country[s] == "US"}),
                     ("India", {s: v for s, v in truth.items() if country[s] == "India"}), ("shared_addr", sub)]:
        if not t:
            continue
        m = evaluate(t, _to_sets(links, t), cand_sets)
        r[label] = {k: round(float(m[k]), 5) for k in ("macro_f05", "macro_precision", "macro_recall", "singleton_accuracy")}
    print(f"[summary] {json.dumps(r)}", flush=True)
    return r


def phase_eval() -> None:
    import ce_v8
    import ce_v9  # noqa: F401  (patches ce_v8.add_features / EXTRA to v9's feature set)
    import stage2
    from faithful_sim import P1
    from stage2 import PARAMS2
    from train import fit, tune
    D.mkdir(parents=True, exist_ok=True)
    drop, keep, links, val_t, ctx = _sim_context()
    pred = pl.read_parquet(P1)
    orig = stage2._norm
    stage2._norm = lambda split: (lambda s, t: (s.filter(~pl.col("entity_id").is_in(drop.implode())), t))(*orig(split))
    feats = stage2.group_features(pred, "train")
    stage2._norm = orig
    base = pl.read_parquet(WORK_DIR / "ce" / "scores.parquet")
    feats = feats.join(base.select("tid", "s1", "part"), on=["tid", "s1"], how="inner")
    feats = ce_v8.add_features(feats, base.select("tid", "s1", "ce"), "train", keep).drop(CE_COLS)
    feats = feats.join(links.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["tid", "s1"], how="left") \
                 .with_columns(pl.col("y").fill_null(0))
    cols_all = ce_v8._cols()
    vp = pred.filter(pl.col("tid").is_in(val_t.implode()))
    s1k, _ = ce_v8._keys("train", keep)
    shared = set(s1k.filter(pl.col("akey").is_not_null()).filter(pl.len().over("country", "akey") > 1)["entity_id"].to_list())
    sub = {s: v for s, v in ctx[0].items() if s in shared}
    results = []
    for name in ["ce", "none", "tfidf", "vae", "bienc0", "bienc"]:
        if name == "none":
            f, cols = feats, [c for c in cols_all if c not in CE_COLS]
        else:
            src = WORK_DIR / "ce" / "scores.parquet" if name == "ce" else _p(f"{name}_scores.parquet")
            if not src.exists():
                print(f"[eval] {name}: no scores yet, skipped", flush=True)
                continue
            f, cols = feats.join(_ce_cols(pl.read_parquet(src)), on=["tid", "s1"], how="inner"), cols_all
        tr, va = f.filter(pl.col("part") == "B"), f.filter(pl.col("part") == "val")
        print(f"[eval] {name}: stage-2 train rows {len(tr):,}, val rows {len(va):,}", flush=True)
        model = fit(tr.select(cols).to_numpy(), tr["y"].to_numpy(), tr["tid"], cols, PARAMS2, 3000)
        p2 = va.select("tid", "s1").with_columns(
            pl.Series("p2", model.predict(va.select(cols).to_numpy(), num_threads=0), dtype=pl.Float32))
        out = vp.join(p2, on=["tid", "s1"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
        dec = tune(out, *ctx, out=f"ce_alt/decision_{name}.json")
        results.append(_summary(name, out, dec, ctx, sub))
    _p("eval_results.json").write_text(json.dumps(results, indent=1))


def phase_solo() -> None:
    """The fine-tuned bi-encoder on its own: every kept S1 of the country is a candidate (no blocking)."""
    import dense_probe as dp
    drop, keep, links, val_t, ctx = _sim_context()
    tok, model, w, b = _load_bienc()
    t_txt, s_txt = _texts()
    s1c = pl.read_parquet(work_path("train", "s1.parquet"), columns=["entity_id", "country"]).filter(
        pl.col("entity_id").is_in(keep.implode()))
    tc = pl.read_parquet(work_path("train", "t.parquet"), columns=["entity_id", "country"]).filter(
        pl.col("entity_id").is_in(val_t.implode()))
    se = dp.embed(tok, model, ["query: " + s_txt[s] for s in s1c["entity_id"].to_list()])
    te = dp.embed(tok, model, ["query: " + t_txt[t] for t in tc["entity_id"].to_list()])
    del model
    torch.cuda.empty_cache()
    K, parts = 8, []
    for c in tc["country"].unique().to_list():
        rows, tr = np.flatnonzero((s1c["country"] == c).to_numpy()), np.flatnonzero((tc["country"] == c).to_numpy())
        S = torch.from_numpy(np.ascontiguousarray(se[rows])).cuda()
        vals, idxs = [], []
        with torch.no_grad():
            for a in range(0, len(tr), 256):
                sim = torch.from_numpy(np.ascontiguousarray(te[tr[a:a + 256]])).cuda() @ S.T
                v, i = torch.topk(sim.float(), K, dim=1)
                vals.append(v.cpu().numpy()), idxs.append(i.cpu().numpy())
        del S
        torch.cuda.empty_cache()
        v, i = np.concatenate(vals), np.concatenate(idxs)
        parts.append(pl.DataFrame({"tid": np.repeat(tc["entity_id"].to_numpy()[tr], K),
                                   "s1": s1c["entity_id"].to_numpy()[rows][i.ravel()],
                                   "country": c, "p": (1 / (1 + np.exp(-(w * v.ravel() + b)))).astype(np.float32)}))
        print(f"[solo] {c}: {len(tr):,} targets searched against {len(rows):,} S1", flush=True)
    out = pl.concat(parts)
    true = links.join(tc.select(pl.col("entity_id").alias("tid")), on="tid")
    hit = true.join(out.select("tid", "s1"), on=["tid", "s1"], how="semi")
    print(f"[solo] true pairs of validation targets found in the bi-encoder's top {K}: {len(hit) / len(true):.4f}", flush=True)
    from train import tune
    D.mkdir(parents=True, exist_ok=True)
    dec = tune(out, *ctx, out="ce_alt/decision_solo.json")
    _summary("solo", out, dec, ctx, {})


PHASES = {"tfidf": phase_tfidf, "vae": phase_vae, "bienc0": phase_bienc0, "bienc": phase_bienc,
          "eval": phase_eval, "solo": phase_solo}

if __name__ == "__main__":
    for ph in (sys.argv[1:] or list(PHASES)):
        t0 = time.time()
        PHASES[ph]()
        print(f"[{ph}] done ({time.time() - t0:.0f}s)", flush=True)
