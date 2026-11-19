"""ABC-XYZ classification recomputed over a chosen window. No Streamlit.

ABC: Pareto of revenue (A up to 80% cumulative, B up to 95%, C the rest). XYZ: coefficient of variation of weekly units
(X < 0.25, Y 0.25–0.50, Z > 0.50). Revenue = units in the window × the SKU's price (52-week revenue / 52-week units).
"""
import numpy as np
import pandas as pd

A_CUT, B_CUT = 0.80, 0.95
X_CUT, Y_CUT = 0.25, 0.50
WINDOWS = {"Last 3 months (13 weeks)": 13, "Last 6 months (26 weeks)": 26, "Last 12 months (52 weeks)": 52}

METHODS = {
    "AX": "Exponential smoothing", "BX": "Exponential smoothing", "CX": "Exponential smoothing",
    "AY": "Holt-Winters (trend + seasonality)", "BY": "Holt-Winters (trend + seasonality)", "CY": "Moving average",
    "AZ": "Moving average / Croston", "BZ": "Moving average / Croston", "CZ": "Moving average / Croston",
}


def method_for(segment, demand_class):
    if segment.endswith("Z"):
        return "Croston (SBA)" if demand_class in ("Intermittent", "Lumpy") else "Moving average"
    return METHODS[segment]


def classify(units_by_item, price_by_item, last_week, window, meta=None):
    """units_by_item: {item_id: Series(units indexed by week_idx)}. Returns one row per SKU."""
    weeks = np.arange(last_week - window + 1, last_week + 1)
    rows = []
    for item, s in units_by_item.items():
        u = s.reindex(weeks).fillna(0.0).to_numpy(float)
        mean = u.mean()
        cv = float(u.std(ddof=0) / mean) if mean > 0 else float("inf")
        rows.append({"item_id": item, "units": float(u.sum()), "revenue": float(u.sum() * price_by_item.get(item, 0.0)),
                     "mean_weekly": float(mean), "cv": cv})
    df = pd.DataFrame(rows).sort_values("revenue", ascending=False).reset_index(drop=True)
    total = max(df["revenue"].sum(), 1e-9)
    df["share"] = df["revenue"] / total
    cum_before = df["share"].cumsum() - df["share"]            # an item is A while the cumulative share before it is < 80%
    df["abc"] = np.where(cum_before < A_CUT, "A", np.where(cum_before < B_CUT, "B", "C"))
    df["xyz"] = np.where(df["cv"] < X_CUT, "X", np.where(df["cv"] <= Y_CUT, "Y", "Z"))
    df["segment"] = df["abc"] + df["xyz"]
    if meta is not None:
        df = df.merge(meta[["item_id", "dept_id", "cat_id", "demand_class"]], on="item_id", how="left")
        df["method"] = [method_for(s, c) for s, c in zip(df["segment"], df["demand_class"])]
    return df


def grid_summary(df):
    """3×3 table: SKUs, revenue share per segment."""
    g = df.groupby("segment").agg(items=("item_id", "count"), share=("share", "sum"))
    out = []
    for a in "ABC":
        for x in "XYZ":
            seg = a + x
            out.append({"segment": seg, "abc": a, "xyz": x, "items": int(g["items"].get(seg, 0)), "share": float(g["share"].get(seg, 0.0))})
    return pd.DataFrame(out)
