import sys, time, numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import work_path
from blocking import _channels, _topk
from data_io import load_links
from sklearn.feature_extraction.text import TfidfVectorizer
links = load_links("train")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
for country in ["US","India"]:
    s1 = pl.read_parquet(work_path("train","s1.parquet")).filter(pl.col("country")==country)
    t = pl.read_parquet(work_path("train","t.parquet")).filter(pl.col("country")==country).sample(N, seed=1)
    gl = links.join(t.select(pl.col("entity_id").alias("tid")), on="tid")
    s1idx = dict(zip(s1["entity_id"], range(len(s1)))); tidx = dict(zip(t["entity_id"], range(len(t))))
    truth = set((tidx[a], s1idx[b]) for a,b in zip(gl["tid"], gl["s1"]))
    found = {}
    for name,(text,params,k) in _channels().items():
        t0=time.time(); vec=TfidfVectorizer(**params); S=vec.fit_transform(text(s1)); Q=vec.transform(text(t))
        t1=time.time(); r,c,v=_topk(Q,S,k); found[name]=set(zip(r.tolist(),c.tolist()))
        print(country, name, f"fit {t1-t0:.0f}s topk {time.time()-t1:.1f}s recall {len(truth&found[name])/len(truth):.4f}", flush=True)
    u=set().union(*found.values())
    print(country, "UNION", f"recall {len(truth&u)/len(truth):.4f} pairs/t {len(u)/len(t):.1f}")
    for drop in found:
        u2=set().union(*[v for k2,v in found.items() if k2!=drop]); print("   without", drop, f"{len(truth&u2)/len(truth):.4f}")
