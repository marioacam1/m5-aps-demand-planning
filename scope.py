"""Filter scope, weekly aggregation and period (week / month / quarter) summaries. No Streamlit."""
import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd

import aps_core as aps

HORIZON = aps.HORIZON
UNITS = ("Weeks", "Months", "Quarters")
MAX_BACK = {"Weeks": 52, "Months": 24, "Quarters": 12}


def resolve_scope(cats, stores, depts, skus, all_cats, all_stores, depts_by_cat, total_id):
    """Turns the checkbox selections into a list of (level, series_id) that together cover the scope exactly.

    cats / stores / depts / skus: selected values. depts_by_cat: {category: [departments]}.
    Hierarchy forecasts are reconciled, so summing series of different levels is consistent.
    """
    if skus:
        return [(7, s) for s in skus]
    if not cats or not stores:
        return []
    cats, stores, depts = list(cats), list(stores), set(depts)
    every_dept = all(set(depts_by_cat[c]) <= depts for c in cats)
    series = []
    if set(cats) == set(all_cats) and every_dept:
        series = [(0, total_id)] if set(stores) == set(all_stores) else [(2, s) for s in stores]
    else:
        whole = set(stores) == set(all_stores)
        for c in cats:
            sel = [d for d in depts_by_cat[c] if d in depts]
            if len(sel) == len(depts_by_cat[c]):
                series += [(3, c)] if whole else [(5, f"{s}|{c}") for s in stores]
            else:
                for d in sel:
                    series += [(4, d)] if whole else [(6, f"{s}|{d}") for s in stores]
    return series


def scope_label(cats, stores, depts, skus, all_cats, all_stores, depts_by_cat):
    if skus:
        return f"{len(skus)} SKU" + ("s" if len(skus) != 1 else "") + " · store CA_1"
    parts = ["All categories" if set(cats) == set(all_cats) else ", ".join(cats) or "No category"]
    parts.append("All stores" if set(stores) == set(all_stores) else (", ".join(stores) if len(stores) <= 3 else f"{len(stores)} stores") or "No store")
    every = all(set(depts_by_cat[c]) <= set(depts) for c in cats) if cats else True
    parts.append("All departments" if every else ", ".join(d for d in depts if d.split("_")[0] in cats))
    return " · ".join(parts)


def weekly_scope_frame(series, actual_getter, forecast_getter, last_week, first_week=1):
    """Weekly table of a scope: actual (past), forecast / low / high / p50 (future), ly (same week one year earlier).

    actual_getter(level, sid) -> Series of units indexed by week_idx.
    forecast_getter(level, sid) -> DataFrame with h, mean, p10, p50, p90 (13 rows) or None.
    Single series: exact quantiles. Several series: means add exactly; Low/High use the independence approximation
    mean ∓ sqrt(Σ deviation²), which can understate the real range when the series move together (flag `exact_band`).
    """
    weeks = np.arange(first_week, last_week + HORIZON + 1)
    act = pd.Series(0.0, index=np.arange(first_week, last_week + 1))
    fcs = []
    for level, sid in series:
        a = actual_getter(level, sid)
        act = act.add(a.reindex(act.index).fillna(0.0), fill_value=0.0)
        f = forecast_getter(level, sid)
        if f is None:
            raise ValueError(f"No forecast for {sid}.")
        fcs.append(f.sort_values("h").reset_index(drop=True))
    n = len(fcs[0])
    mean = np.sum([f["mean"].to_numpy() for f in fcs], axis=0)
    p50 = np.sum([f["p50"].to_numpy() for f in fcs], axis=0)
    if len(fcs) == 1:
        low, high = fcs[0]["p10"].to_numpy(), fcs[0]["p90"].to_numpy()
    else:
        dev_lo = np.sqrt(np.sum([(f["mean"].to_numpy() - f["p10"].to_numpy()).clip(min=0) ** 2 for f in fcs], axis=0))
        dev_hi = np.sqrt(np.sum([(f["p90"].to_numpy() - f["mean"].to_numpy()).clip(min=0) ** 2 for f in fcs], axis=0))
        low, high = np.maximum(mean - dev_lo, 0.0), mean + dev_hi
    df = pd.DataFrame({"week_idx": weeks})
    df["week_start"] = pd.to_datetime(df["week_idx"].map(aps.week_start))
    df["actual"] = df["week_idx"].map(act).where(df["week_idx"] <= last_week)
    fut = df["week_idx"] > last_week
    for name, vals in (("forecast", mean), ("low", low), ("high", high), ("p50", p50)):
        col = np.full(len(df), np.nan)
        col[fut.to_numpy()] = vals[: int(fut.sum())]
        df[name] = col
    df["ly"] = (df["week_idx"] - 52).map(act)
    df.attrs["exact_band"] = len(fcs) == 1
    return df


def period_key(week_idx, unit):
    if unit == "Weeks":
        return int(week_idx)
    d = aps.week_start(int(week_idx)) + timedelta(days=3)      # a week belongs to the period of its middle day
    return (d.year, d.month) if unit == "Months" else (d.year, (d.month - 1) // 3 + 1)


def _label(key, unit):
    if unit == "Weeks":
        d = aps.week_start(key)
        return f"W{key}", d.strftime("%b %d")
    if unit == "Months":
        return date(key[0], key[1], 1).strftime("%b %Y"), ""
    return f"Q{key[1]} {key[0]}", ""


def _period_end(key, unit):
    if unit == "Weeks":
        return aps.week_start(key) + timedelta(days=6)
    if unit == "Months":
        return date(key[0], key[1], calendar.monthrange(key[0], key[1])[1])
    q_end_month = key[1] * 3
    return date(key[0], q_end_month, calendar.monthrange(key[0], q_end_month)[1])


def build_periods(frame, unit, last_week, back, fwd):
    """Aggregates the weekly frame into periods around the current one.

    The current period is the one containing the first forecast week (the week being planned).
    Returns (periods, info): periods is a list of dicts; info has the available back / forward counts.
    """
    keys = [period_key(w, unit) for w in frame["week_idx"]]
    f = frame.assign(_k=keys)
    order = list(dict.fromkeys(keys))
    cur_key = period_key(last_week + 1, unit)
    ci = order.index(cur_key)
    horizon_end = aps.week_start(last_week + HORIZON) + timedelta(days=6)
    out = []
    for k in order:
        g = f[f["_k"] == k]
        act, fc = g["actual"].dropna(), g["forecast"].dropna()
        a_sum, f_sum = float(act.sum()) if len(act) else np.nan, float(fc.sum()) if len(fc) else np.nan
        ly = g["ly"].dropna()
        cons = g["consensus"] if "consensus" in g else pd.Series(np.nan, index=g.index)
        final_fut = cons.where(cons.notna(), g["forecast"]).dropna()
        has_cons = bool(cons.notna().any())
        c_sum = float(final_fut.sum()) if has_cons else np.nan
        plan = np.nansum([a_sum, float(final_fut.sum()) if len(final_fut) else np.nan])
        base_plan = np.nansum([a_sum, f_sum])
        pv = g["prev"] if "prev" in g else pd.Series(np.nan, index=g.index)
        prev = float(np.nansum([a_sum, float(pv.dropna().sum())])) if pv.notna().any() else np.nan
        ly_sum = float(ly.sum()) if len(ly) else np.nan
        label, sub = _label(k, unit)
        kind = "mixed" if len(act) and len(fc) else "forecast" if len(fc) else "actual"
        out.append({"key": k, "label": label, "sub": sub, "kind": kind, "is_current": k == cur_key, "actual": a_sum, "forecast": f_sum,
                    "consensus": c_sum, "plan": float(plan), "baseline_plan": float(base_plan), "prev": prev,
                    "vs_prev": (plan / prev - 1) if prev == prev and prev > 0 else np.nan,
                    "ly": ly_sum, "vs_ly": (plan / ly_sum - 1) if ly_sum and ly_sum > 0 else np.nan,
                    "low": float(g["low"].dropna().sum()) if g["low"].notna().any() else np.nan,
                    "high": float(g["high"].dropna().sum()) if g["high"].notna().any() else np.nan,
                    "partial": bool(kind != "actual" and unit != "Weeks" and _period_end(k, unit) > horizon_end)})
    info = {"max_back": min(ci, MAX_BACK[unit]), "max_fwd": len(order) - 1 - ci}
    back, fwd = min(back, info["max_back"]), min(fwd, info["max_fwd"])
    return out[ci - back: ci + fwd + 1], info


def yoy(plan, ly):
    return plan / ly - 1 if ly and ly > 0 else float("nan")
