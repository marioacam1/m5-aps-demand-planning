"""Reads the data exported by notebook 08 (data/ folder). No Streamlit: can be tested on its own."""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
REQUIRED = ["history_hierarchy.csv.gz", "history_items.csv.gz", "items_meta.csv", "manifest.json"]
CLASS_EN = {"Suave": "Smooth", "Errática": "Erratic", "Erratica": "Erratic", "Intermitente": "Intermittent",
            "Irregular": "Lumpy", "Sin demanda": "No demand"}


def _english(df, col="demand_class"):
    if df is not None and col in df:
        df = df.copy()
        df[col] = df[col].map(lambda v: CLASS_EN.get(v, v))
    return df


def load_all(data_dir=DATA_DIR):
    """Returns a dict with every table. Required files fail with a clear message."""
    data_dir = Path(data_dir)
    missing = [f for f in REQUIRED if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {', '.join(missing)}. Copy the 'data' folder from the export.")
    d = {"manifest": json.loads((data_dir / "manifest.json").read_text())}
    d["hier"] = pd.read_csv(data_dir / "history_hierarchy.csv.gz")
    d["items"] = pd.read_csv(data_dir / "history_items.csv.gz")
    d["items_meta"] = _english(pd.read_csv(data_dir / "items_meta.csv"))
    seg = data_dir / "segmentation_counts.csv"
    d["seg_counts"] = _english(pd.read_csv(seg)) if seg.exists() else None
    for key, name in [("acc_level", "accuracy_level.csv"), ("acc_horizon", "accuracy_horizon.csv"), ("acc_window", "accuracy_window.csv"),
                      ("acc_class", "accuracy_class.csv"), ("acc_abc", "accuracy_abc.csv")]:
        p = data_dir / name
        d[key] = pd.read_csv(p) if p.exists() else None
    d["acc_class"] = _english(d["acc_class"])
    bt = data_dir / "backtest_hierarchy.csv.gz"
    d["backtest"] = pd.read_csv(bt) if bt.exists() else None
    d["last_week"] = int(d["manifest"].get("last_week", d["hier"]["week_idx"].max()))
    d["hier_meta"] = d["hier"].drop_duplicates("series_id")[["series_id", "level"]].sort_values(
        ["level", "series_id"]).reset_index(drop=True)
    d["hier_by_series"] = {k: g for k, g in d["hier"].groupby("series_id")}
    d["items_by_series"] = {k: g for k, g in d["items"].groupby("series_id")}
    return d
