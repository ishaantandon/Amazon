import sys, json, polars as pl, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import work_path, WORK_DIR
from data_io import load_ground_truth, load_links
from assign import assign, best_links
from metrics import evaluate

pred = pl.read_parquet(work_path("train", "val_pred.parquet"))
truth_all = load_ground_truth("train")
rng = np.random.default_rng(42)
ids = np.array(sorted(truth_all))
val = set(ids[rng.random(len(ids)) < 0.02])
truth = {s: truth_all[s] for s in val}
alll = load_links("train")
links = alll.filter(pl.col("s1").is_in(list(val)))
dec = json.loads((WORK_DIR / "decision.json").read_text())
m = assign(pred, dec["threshold"], dec["expected_f"])
pm = {s: set(v) for s, v in m.items() if s in val}
f = lambda x: evaluate(truth, x)["macro_f05"]
print("base", round(f(pm), 5), "val S1", len(val))

best = best_links(pred).select("tid", pl.col("s1").alias("best"), pl.col("p").alias("pbest"))
inc = pred.select("tid", "s1", pl.col("p").alias("ptrue")).with_columns(pl.lit(True).alias("inc"))
L = links.join(inc, on=["tid", "s1"], how="left").join(best, on="tid", how="left").with_columns(pl.col("inc").fill_null(False))
pp = {(t, s) for s, v in pm.items() for t in v}
L = L.with_columns(pl.Series("predicted", [(t, s) in pp for t, s in zip(L["tid"], L["s1"])]))
cat = L.with_columns(
    pl.when(~pl.col("inc")).then(pl.lit("1 blocking miss"))
    .when(pl.col("predicted")).then(pl.lit("0 found"))
    .when(pl.col("best") != pl.col("s1")).then(pl.lit("2 another S1 ranked higher"))
    .when(pl.col("pbest") < dec["threshold"]).then(pl.lit("3 best, but p < threshold"))
    .otherwise(pl.lit("4 best, p >= t, cut by expected-F")).alias("c")
)
tot = len(L)
print("TRUE LINKS of val S1s:", tot)
for c, n in cat.group_by("c").len().sort("c").iter_rows():
    print(f"   {c:40s} {n:8,d}  {n / tot:.2%}")
print("   p of true link, cat 3:", cat.filter(pl.col("c").str.starts_with("3"))["ptrue"].describe().filter(pl.col("statistic").is_in(["mean", "25%", "50%", "75%"])).rows())

tl = set(zip(links["tid"], links["s1"]))
owner = dict(zip(alll["tid"], alll["s1"]))
fp = [(t, s) for s, v in pm.items() for t in v if (t, s) not in tl]
fo = sum(1 for t, s in fp if t in owner)
print(f"FALSE POSITIVE links: {len(fp):,} (belongs to another S1: {fo:,}, pure distractor: {len(fp) - fo:,});  predicted links total {len(pp):,}")
sing = [s for s in val if not truth[s]]
print("singletons", len(sing), "wrongly given links:", sum(1 for s in sing if pm.get(s)))
zero = [s for s in val if truth[s] and not pm.get(s)]
print(f"non-singletons predicted EMPTY (score 0): {len(zero)} = {len(zero) / len(val):.2%} of entities")

nofp = {s: {t for t in v if (t, s) in tl} for s, v in pm.items()}
print("oracle remove all FPs              ->", round(f(nofp), 5))
add = {s: set(pm.get(s, set())) for s in val}
for t, s, c in zip(L["tid"], L["s1"], L["inc"]):
    if c:
        add[s].add(t)
print("oracle recover in-candidate misses ->", round(f(add), 5))
sing_fix = dict(pm)
for s in sing:
    sing_fix.pop(s, None)
print("oracle fix singleton false merges  ->", round(f(sing_fix), 5))
full = {s: set(truth[s]) for s in val}
print("oracle perfect                     ->", round(f(full), 5))

print("\n--- category 4 detail ---")
c4 = cat.filter(pl.col("c").str.starts_with("4"))
print("p of cut true links:", c4["ptrue"].describe().filter(pl.col("statistic").is_in(["min","25%","50%","75%","max"])).rows())
acc = best_links(pred).filter(pl.col("p") >= dec["threshold"])
g = acc.group_by("s1").agg(pl.len().alias("n_acc"), pl.col("p").sort(descending=True).alias("ps"))
ex = c4.select("s1").unique().head(8).join(g, on="s1")
for s, n, ps in ex.iter_rows():
    kept = len(pm.get(s, ()))
    print(s, "accepted", n, "kept", kept, "true", len(truth[s]), "probs", [round(x, 3) for x in ps])

print("\n--- calibration of links the cut removed (argmax, p>=t, not kept) ---")
acc2 = best_links(pred).filter(pl.col("p") >= dec["threshold"]).filter(pl.col("s1").is_in(list(val)))
acc2 = acc2.with_columns(pl.Series("kept", [(t, s) in pp for t, s in zip(acc2["tid"], acc2["s1"])]),
                         pl.Series("true", [(t, s) in tl for t, s in zip(acc2["tid"], acc2["s1"])]))
cut = acc2.filter(~pl.col("kept"))
print("cut links:", len(cut), "true:", cut["true"].sum(), f"precision {cut['true'].mean():.3f}", f"mean p {cut['p'].mean():.3f}")
for lo, hi in [(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.01)]:
    b = cut.filter((pl.col("p")>=lo)&(pl.col("p")<hi)); a = acc2.filter((pl.col("p")>=lo)&(pl.col("p")<hi))
    if len(a): print(f"  p in [{lo},{hi}): all accepted n={len(a):6d} true rate {a['true'].mean():.3f} | cut n={len(b):5d} true rate {b['true'].mean() if len(b) else float('nan'):.3f}")
# rank position: is the cut link rank>=2 within its S1 group?

print("\n--- native-script share of losses ---")
from data_io import load_targets
raw = load_targets("train").select("entity_id", "business_name", "country").filter(pl.col("entity_id").is_in(L["tid"].implode()))
raw = raw.with_columns(pl.col("business_name").str.contains(r"[^\x00-\x7FÀ-ɏ]").alias("native"))
C = cat.join(raw.rename({"entity_id": "tid"}), on="tid", how="left")
for country in ["India", "US"]:
    d = C.filter(pl.col("country") == country)
    lost = d.filter(~pl.col("c").str.starts_with("0"))
    print(f"{country}: true links {len(d):,}, native-script {d['native'].mean():.1%}; lost {len(lost):,}, of which native {lost['native'].mean():.1%}")
    for n in [True, False]:
        dd = d.filter(pl.col("native") == n)
        if len(dd): print(f"   native={n}: recall {dd['c'].str.starts_with('0').mean():.3f}   blocking-miss {dd['c'].str.starts_with('1').mean():.3f}  n={len(dd):,}")
