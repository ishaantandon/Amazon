import sys, random, polars as pl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import work_path
from blocking import _channels, _topk
from data_io import load_links
from sklearn.feature_extraction.text import TfidfVectorizer
links = load_links("train"); country="India"
s1 = pl.read_parquet(work_path("train","s1.parquet")).filter(pl.col("country")==country)
t = pl.read_parquet(work_path("train","t.parquet")).filter(pl.col("country")==country).sample(20000, seed=1)
gl = links.join(t.select(pl.col("entity_id").alias("tid")), on="tid")
s1idx = dict(zip(s1["entity_id"], range(len(s1)))); tidx = dict(zip(t["entity_id"], range(len(t))))
truth = set((tidx[a], s1idx[b]) for a,b in zip(gl["tid"], gl["s1"]))
u=set()
for name,(text,params,k) in _channels().items():
    vec=TfidfVectorizer(**params); S=vec.fit_transform(text(s1)); Q=vec.transform(text(t)); r,c,v=_topk(Q,S,k); u|=set(zip(r.tolist(),c.tolist()))
miss=list(truth-u); random.seed(3); random.shuffle(miss)
empty = sum(t.row(ti,named=True)["addr_empty"] for ti,_ in miss)
print("missed", len(miss), "with empty target addr", empty)
for ti,si in miss[:20]:
    a=t.row(ti,named=True); b=s1.row(si,named=True)
    print("T :", a["name"], "|", a["skel"], "|", a["addr"], "|", a["nums"]); print("S1:", b["name"], "|", b["skel"], "|", b["addr"], "|", b["nums"], "\n")
