"""Official metric: F_0.5 per Source 1 entity, macro-averaged over all Source 1 entities."""


def entity_f05(true: set, pred: set) -> tuple[float, float, float]:
    if not true:
        return (1.0, 1.0, 1.0) if not pred else (0.0, 0.0, 0.0)
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(true & pred)
    p, r = tp / len(pred), tp / len(true)
    f = 1.25 * p * r / (0.25 * p + r) if tp else 0.0
    return p, r, f


def evaluate(truth: dict, pred: dict, cands: dict | None = None) -> dict:
    """truth/pred/cands: {s1_id: set(ids)}; entities are the keys of `truth`."""
    n = len(truth)
    sp = sr = sf = 0.0
    single = single_ok = 0
    ceil_num = ceil_den = n_cands = 0
    for s1, t in truth.items():
        pr = pred.get(s1, set())
        p, r, f = entity_f05(t, pr)
        sp, sr, sf = sp + p, sr + r, sf + f
        if not t:
            single += 1
            single_ok += not pr
        if cands is not None:
            c = cands.get(s1, set())
            n_cands += len(c)
            if t:
                ceil_num += len(t & c) / len(t)
                ceil_den += 1
    out = {
        "macro_f05": sf / n, "macro_precision": sp / n, "macro_recall": sr / n,
        "entities": n, "singletons": single,
        "singleton_accuracy": single_ok / single if single else 1.0,
    }
    if cands is not None:
        out["candidate_recall_ceiling"] = ceil_num / max(ceil_den, 1)
        out["avg_candidates_per_entity"] = n_cands / n
    return out


def fmt(m: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())
