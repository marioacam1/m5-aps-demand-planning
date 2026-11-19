"""Genera datos sintéticos con el MISMO esquema que exporta el notebook 08 (solo para pruebas locales)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aps_core as aps

def make(out, last_week=277, seed=3):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stores = ["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"]
    depts = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2"]
    w = np.arange(1, last_week + 1)
    season = 1 + 0.15*np.sin(2*np.pi*(w-8)/52)
    bottom = {}
    for s in stores:
        for d in depts:
            base = rng.uniform(40, 900) * (3 if d.startswith("FOODS") else 1)
            bottom[(s, d)] = rng.poisson(base * season * (1 + 0.002*w))
    rows = []
    def add(sid, lvl, y):
        for wi, v in zip(w, y):
            rows.append((sid, lvl, int(wi), float(v), float(v)))
    tot = sum(bottom.values()); add("Total", 0, tot)
    for st in ("CA","TX","WI"):
        add(st, 1, sum(v for (s,d),v in bottom.items() if s.startswith(st)))
    for s in stores: add(s, 2, sum(v for (s2,d),v in bottom.items() if s2==s))
    for c in ("FOODS","HOBBIES","HOUSEHOLD"): add(c, 3, sum(v for (s,d),v in bottom.items() if d.startswith(c)))
    for d in depts: add(d, 4, sum(v for (s,dd),v in bottom.items() if dd==d))
    for s in stores:
        for c in ("FOODS","HOBBIES","HOUSEHOLD"):
            add(f"{s}|{c}", 5, sum(v for (s2,d),v in bottom.items() if s2==s and d.startswith(c)))
    for (s, d), v in bottom.items(): add(f"{s}|{d}", 6, v)
    hh = pd.DataFrame(rows, columns=["series_id","level","week_idx","units","units_clip"])
    hh.to_csv(out/"history_hierarchy.csv.gz", index=False)

    items, meta = [], []
    for i in range(60):
        d = depts[i % 7]; cat = d.split("_")[0]
        iid = f"{d}_{i:03d}_CA_1"
        mu = rng.choice([0.2, 0.8, 3, 12, 40]); start = int(rng.integers(1, 120))
        y = rng.poisson(mu*season)[start-1:]
        for wi, v in zip(w[start-1:], y): items.append((iid, int(wi), float(v), float(v)))
        yy = y[-52:]; adi = 1/max((yy>0).mean(), .02); cv2 = (yy[yy>0].std()/max(yy[yy>0].mean(),1e-9))**2 if (yy>0).any() else 0
        cls = "Smooth" if adi<1.32 and cv2<.49 else "Erratic" if adi<1.32 else "Intermittent" if cv2<.49 else "Lumpy"
        rev = float(yy.sum()*rng.uniform(1,6))
        meta.append(dict(item_id=iid, dept_id=d, cat_id=cat, abc=None, xyz=None, abc_xyz=None, demand_class=cls,
                         n_weeks=len(y), units_52w=float(yy.sum()), revenue_52w=rev, mean_weekly=float(yy.mean()),
                         std_weekly=float(yy.std()), adi=adi, cv2=cv2, cv=float(yy.std()/max(yy.mean(),1e-9))))
    m = pd.DataFrame(meta)
    cum = m["revenue_52w"].sort_values(ascending=False).cumsum()/m["revenue_52w"].sum()
    m["abc"] = m.index.map(lambda i: "A" if cum[i]<=.8 else "B" if cum[i]<=.95 else "C")
    m["xyz"] = pd.cut(m["cv"], [-1,.5,1,1e9], labels=["X","Y","Z"]).astype(str)
    m["abc_xyz"] = m["abc"]+m["xyz"]
    m.to_csv(out/"items_meta.csv", index=False)
    pd.DataFrame(items, columns=["series_id","week_idx","units","units_clip"]).to_csv(out/"history_items.csv.gz", index=False)

    cnt = m.groupby(["abc","xyz","demand_class"], as_index=False).agg(items=("item_id","count"), revenue_52w=("revenue_52w","sum"))
    cnt.insert(0, "store_id", "CA_1"); cnt.to_csv(out/"segmentation_counts.csv", index=False)

    # backtest sintético de la jerarquía (naive + ruido)
    bt = []
    for (sid, lvl), g in hh.groupby(["series_id","level"]):
        y = g.set_index("week_idx")["units"]
        for win, origin in enumerate([200,213,226,239,252]):
            for h in (1,2,3,4,5,6,7,8,9,10,11,12,13):
                t = origin+h
                if t > last_week: continue
                mean = y[origin]*rng.uniform(.93,1.07)
                bt.append((sid,int(lvl),aps.LEVEL_NAMES[int(lvl)],win,origin,t,h,y[t],mean,mean*rng.uniform(.99,1.01),mean*.85,mean*.99,mean*1.15))
    bt = pd.DataFrame(bt, columns=["series_id","level","level_name","window","origin_week","target_week","h","y_true","mean","mean_rec","p10","p50","p90"])
    bt.to_csv(out/"backtest_hierarchy.csv.gz", index=False)

    def acc(by):
        rows=[]
        for k,g in bt.groupby(by):
            y=g.y_true.to_numpy(); r=dict(zip(by,k if isinstance(k,tuple) else (k,)))
            r.update(n=len(g), wape_media=aps.wape(y,g["mean"]), wape_conciliada=aps.wape(y,g.mean_rec), wape_naive=.2, wape_estacional=.25,
                     wape_prom13=.18, fva_vs_mejor_base=.02, sesgo_media=aps.bias(y,g["mean"]), wape_p50=aps.wape(y,g.p50),
                     sesgo_p50=aps.bias(y,g.p50), cobertura_sin_calibrar=.7, cobertura_calibrada=float(((y>=g.p10)&(y<=g.p90)).mean()),
                     bajo_p10=.1, bajo_p50=.5, bajo_p90=.9, pinball_norm=.05)
            rows.append(r)
        return pd.DataFrame(rows)
    acc(["level","level_name"]).to_csv(out/"accuracy_level.csv", index=False)
    b2 = bt.assign(h_bucket=pd.cut(bt.h,[0,4,8,13],labels=["1-4","5-8","9-13"]).astype(str))
    bt_save = bt; bt = b2; acc(["h_bucket"]).to_csv(out/"accuracy_horizon.csv", index=False)
    bt = bt_save; acc(["window"]).to_csv(out/"accuracy_window.csv", index=False)
    cl = acc(["level"]).assign(demand_class="Smooth").drop(columns="level"); cl.to_csv(out/"accuracy_class.csv", index=False)
    cl.assign(abc="A").drop(columns="demand_class").to_csv(out/"accuracy_abc.csv", index=False)
    pd.DataFrame({"level":range(7),"level_name":[aps.LEVEL_NAMES[i] for i in range(7)],"wape_sin_corregir":.1,"wape_con_correccion":.09,
                  "sesgo_sin_corregir":-.03,"sesgo_con_correccion":-.01,"corrige_sesgo":True,"cobertura_solo_ensancha":.78,
                  "cobertura_bilateral":.8,"acorta_intervalo":False,"cobertura_final":.8}).to_csv(out/"calibration_choice.csv", index=False)
    (out/"manifest.json").write_text(json.dumps({"last_week":last_week,"last_week_start":aps.week_start(last_week).isoformat(),
        "last_data_date":"2016-05-22","hierarchy_series":124,"app_items":len(m),"aps_core_version":aps.VERSION,"generated_on":"test"}))

if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth/data")
