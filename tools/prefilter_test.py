import sys, numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import work_path
from blocking import block_country
from data_io import load_links
from rapidfuzz import process, fuzz
from sklearn.linear_model import LogisticRegression
links = load_links("train")
for country in ["US","India"]:
    s1 = pl.read_parquet(work_path("train","s1.parquet")).filter(pl.col("country")==country)
    t = pl.read_parquet(work_path("train","t.parquet")).filter(pl.col("country")==country).sample(40000, seed=2)
    c = block_country(s1, t)
    c = c.with_columns(t["entity_id"].gather(c["ti"]).alias("tid"), s1["entity_id"].gather(c["si"]).alias("s1"))
    c = c.join(links.with_columns(pl.lit(1).alias("y")), on=["tid","s1"], how="left").with_columns(pl.col("y").fill_null(0))
    ntrue = links.join(t.select(pl.col("entity_id").alias("tid")), on="tid").height
    tn = t["name"].gather(c["ti"]).to_list(); sn = s1["name"].gather(c["si"]).to_list()
    ta = (t["addr"]+" "+t["nums"]).gather(c["ti"]).to_list(); sa = (s1["addr"]+" "+s1["nums"]).gather(c["si"]).to_list()
    c = c.with_columns(pl.Series("nr", process.cpdist(tn, sn, scorer=fuzz.token_set_ratio, workers=-1)/100),
                       pl.Series("ar", process.cpdist(ta, sa, scorer=fuzz.token_set_ratio, workers=-1)/100))
    ch = ["name_ngram","phon_key","mix_key","name_key","addr_key"]
    X = c.select(ch+["nr","ar"]).to_numpy(); y = c["y"].to_numpy()
    half = c["ti"].to_numpy() % 2 == 0
    lr = LogisticRegression(max_iter=500).fit(X[half], y[half])
    for nm, sc in [("sum", c.select(pl.sum_horizontal(ch)).to_series().to_numpy()), ("lr", lr.decision_function(X))]:
        d = c.with_columns(pl.Series("sc", sc)).filter(~pl.Series(half)).with_columns(pl.col("sc").rank("ordinal", descending=True).over("ti").alias("rk"))
        tot = d["y"].sum()
        print(country, nm, "all", tot, " ".join(f"@{k}:{d.filter(pl.col('rk')<=k)['y'].sum()/tot:.4f}" for k in [3,5,8,10,12,15,20]))
