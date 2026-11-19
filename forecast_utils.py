"""Request building and forecast handling for the endpoint (no Streamlit)."""
import numpy as np
import pandas as pd

import aps_core as aps

HORIZON = aps.HORIZON
MAX_HISTORY = 300
MIN_HISTORY = 26


def build_request(series_id, level, hist, last_week, horizon=HORIZON):
    """Row expected by the endpoint. hist: DataFrame with week_idx and units_clip (cleaned demand).

    Sends the contiguous history ending at `last_week`: it starts at the first week with data and fills
    intermediate weeks without a row with 0 (weeks with no sales).
    """
    h = hist.dropna(subset=["units_clip"]).sort_values("week_idx")
    h = h[h["week_idx"] <= last_week]
    if h.empty:
        raise ValueError(f"{series_id}: has no history.")
    start = max(int(h["week_idx"].min()), last_week - MAX_HISTORY + 1)
    full = h.set_index("week_idx")["units_clip"].reindex(range(start, last_week + 1)).fillna(0.0)
    if len(full) < MIN_HISTORY:
        raise ValueError(f"{series_id}: only {len(full)} weeks of history (minimum {MIN_HISTORY}).")
    return {"series_id": str(series_id), "level": int(level), "last_week": int(last_week),
            "horizon": int(horizon), "history": [round(float(v), 4) for v in full.to_numpy()]}


def build_requests(meta, history_by_series, last_week, horizon=HORIZON):
    """One row per series in `meta` (series_id, level). Returns (rows, skipped)."""
    rows, skipped = [], []
    for sid, lvl in zip(meta["series_id"], meta["level"]):
        try:
            rows.append(build_request(sid, lvl, history_by_series[sid], last_week, horizon))
        except (ValueError, KeyError) as exc:
            skipped.append((sid, str(exc)))
    return rows, skipped


def forecast_frame(forecasts, level_map=None):
    """Endpoint forecasts -> long table: series_id, h, target_week, week_start, mean, p10, p50, p90."""
    parts = []
    for f in forecasts:
        n = len(f["mean"])
        parts.append(pd.DataFrame({
            "series_id": f["series_id"], "h": np.arange(1, n + 1), "target_week": f["target_week"],
            "week_start": pd.to_datetime(f["week_start"]), "mean": f["mean"], "p10": f["p10"],
            "p50": f["p50"], "p90": f["p90"], "status": f["status"], "model_version": f["model_version"]}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if level_map is not None and not out.empty:
        out["level"] = out["series_id"].map(level_map)
    return out


def reconcile_bulk(fc, meta):
    """Reconciles the mean of the 124 hierarchy series and rescales their quantiles.

    fc: forecast_frame table with every series in `meta` and the same horizons.
    Adds mean_rec, p10_rec, p50_rec, p90_rec.
    """
    meta = meta.reset_index(drop=True)
    order = meta["series_id"].tolist()
    piv = fc.pivot_table(index="series_id", columns="h", values="mean", aggfunc="first").reindex(order)
    if piv.isna().any().any():
        missing = piv.index[piv.isna().any(axis=1)].tolist()
        raise ValueError(f"Forecasts are missing for {len(missing)} series, cannot reconcile: {missing[:5]}")
    S = aps.summing_matrix(meta)
    rec = pd.DataFrame(aps.reconcile_wls(piv.to_numpy(), S), index=piv.index, columns=piv.columns)
    rec_long = rec.stack().rename("mean_rec").reset_index()
    out = fc.merge(rec_long, on=["series_id", "h"], how="left")
    ratio = np.where(out["mean"] > 0, out["mean_rec"] / np.where(out["mean"] > 0, out["mean"], 1.0), 1.0)
    for c in ("p10", "p50", "p90"):
        out[c + "_rec"] = out[c] * ratio
    return out


def coherence_table(fc, meta):
    """Compares the total with the sum of its parts (stores and categories), before and after reconciliation."""
    rows = []
    stores = meta.loc[meta["level"] == 2, "series_id"]
    cats = meta.loc[meta["level"] == 3, "series_id"]
    total_id = meta.loc[meta["level"] == 0, "series_id"].iloc[0]
    for h, g in fc.groupby("h"):
        g = g.set_index("series_id")
        row = {"Week": int(h), "Total": g.loc[total_id, "mean"], "Sum of stores": g.loc[stores, "mean"].sum(),
               "Sum of categories": g.loc[cats, "mean"].sum()}
        if "mean_rec" in g:
            row.update({"Total (reconciled)": g.loc[total_id, "mean_rec"],
                        "Sum of stores (reconciled)": g.loc[stores, "mean_rec"].sum(),
                        "Sum of categories (reconciled)": g.loc[cats, "mean_rec"].sum()})
        rows.append(row)
    return pd.DataFrame(rows)


def history_frame(hist, last_week):
    """Weekly history with the week start date."""
    h = hist[hist["week_idx"] <= last_week].sort_values("week_idx").copy()
    h["week_start"] = pd.to_datetime(h["week_idx"].map(aps.week_start))
    return h
