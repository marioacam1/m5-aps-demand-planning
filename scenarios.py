"""Consensus and scenarios: manual overrides on top of the statistical forecast. No Streamlit."""
import numpy as np
import pandas as pd

PRESETS = ["No override", "Promotion (+20% in weeks 3–5)", "Demand surge (+10% every week)",
           "Demand drop (−10% every week)", "High case (use P90)", "Low case (use P10)"]
MAX_PCT = 500.0


def empty_overrides(n):
    return pd.DataFrame({"week": np.arange(1, n + 1), "override_pct": 0.0, "override_units": 0.0, "reason": ""})


def preset_overrides(name, base):
    """Initial overrides of a template. base: forecast with mean, p10, p90 (one row per week)."""
    adj = empty_overrides(len(base))
    if name.startswith("Promotion"):
        sel = adj["week"].between(3, 5)
        adj.loc[sel, "override_pct"], adj.loc[sel, "reason"] = 20.0, "Promotion"
    elif name.startswith("Demand surge"):
        adj["override_pct"], adj["reason"] = 10.0, "Demand surge"
    elif name.startswith("Demand drop"):
        adj["override_pct"], adj["reason"] = -10.0, "Demand drop"
    elif name.startswith("High case"):
        adj["override_pct"] = ((base["p90"].to_numpy() / np.maximum(base["mean"].to_numpy(), 1e-9)) - 1) * 100
        adj["reason"] = "High case band"
    elif name.startswith("Low case"):
        adj["override_pct"] = ((base["p10"].to_numpy() / np.maximum(base["mean"].to_numpy(), 1e-9)) - 1) * 100
        adj["reason"] = "Low case band"
    adj["override_pct"] = adj["override_pct"].round(1)
    return adj


def validate_overrides(adj):
    """Returns the list of problems (empty if everything is fine)."""
    problems = []
    for col in ("override_pct", "override_units"):
        if pd.to_numeric(adj[col], errors="coerce").isna().any():
            problems.append(f"'{col}' has empty or non-numeric values.")
    if not problems and (pd.to_numeric(adj["override_pct"]).abs() > MAX_PCT).any():
        problems.append(f"The % override must be between −{MAX_PCT:.0f}% and {MAX_PCT:.0f}%.")
    if not problems and (pd.to_numeric(adj["override_pct"]) <= -100).any():
        problems.append("An override of −100% or less removes all demand; use a value above −100%.")
    return problems


def apply_overrides(fc, adj):
    """Consensus forecast: value × (1 + %) + units, never below 0. Quantiles move the same way."""
    problems = validate_overrides(adj)
    if problems:
        raise ValueError(" ".join(problems))
    out = fc.sort_values("h").reset_index(drop=True).copy()
    pct = pd.to_numeric(adj["override_pct"]).to_numpy() / 100.0
    units = pd.to_numeric(adj["override_units"]).to_numpy()
    if len(pct) != len(out):
        raise ValueError("The overrides do not match the number of forecast weeks.")
    for c in ("mean", "p10", "p50", "p90"):
        out["consensus_" + c] = np.maximum(out[c].to_numpy() * (1 + pct) + units, 0.0)
    out["override_pct"], out["override_units"], out["reason"] = pct * 100, units, adj["reason"].to_numpy()
    return out


def compare(base, named):
    """Comparison table of totals. named: {name: result of apply_overrides}."""
    rows = [{"Scenario": "Statistical forecast", "Units": float(base["mean"].sum()), "Low case": float(base["p10"].sum()),
             "High case": float(base["p90"].sum())}]
    for name, res in named.items():
        rows.append({"Scenario": name, "Units": float(res["consensus_mean"].sum()), "Low case": float(res["consensus_p10"].sum()),
                     "High case": float(res["consensus_p90"].sum())})
    t = pd.DataFrame(rows)
    t["Δ units"] = t["Units"] - t.loc[0, "Units"]
    t["Δ %"] = t["Δ units"] / max(t.loc[0, "Units"], 1e-9) * 100
    return t


# ----------------------------------------------------------------------------- layered overrides (Sales + Marketing)
REASON_CODES = ["Promotion", "Tender", "Shortage", "Weather", "Other"]


def parse_cell(text, ref):
    """Parses an override cell in units. A plain number ('500') sets the week to that value; a signed number ('+200', '-150')
    adds or subtracts units from the layer above (`ref`). Blank = no override. Percentages are not accepted.

    Returns (value or None, error or ''). The value is never below 0.
    """
    t = str(text or "").strip().replace(",", "").replace(" ", "")
    if not t:
        return None, ""
    if "%" in t:
        return None, "type units (500, +200 or -150), not a percentage"
    try:
        v = float(t)
    except ValueError:
        return None, "type a number of units: 500 sets the week, +200 / -150 adds or subtracts"
    if t[0] in "+-":
        return max(ref + v, 0.0), ""
    return v, ""


def apply_layers(base, sales_txt, mkt_txt):
    """Cascade: Statistical baseline → Sales override → Marketing event → Final consensus.

    Each layer cell is a number of units: absolute (500) or a signed change (+200 / -150) of the previous layer. Quantiles are scaled by final / baseline.
    Returns (DataFrame, errors) — errors is {(layer, week_number): message}.
    """
    out = base.sort_values("h").reset_index(drop=True).copy()
    n = len(out)
    mean = out["mean"].to_numpy(float)
    sales, mkt, final = np.full(n, np.nan), np.full(n, np.nan), mean.copy()
    errors = {}
    for j in range(n):
        v, e = parse_cell(sales_txt[j], mean[j])
        if e:
            errors[("Sales override", j + 1)] = e
        if v is not None:
            sales[j] = v
            final[j] = v
        v, e = parse_cell(mkt_txt[j], final[j])
        if e:
            errors[("Marketing event", j + 1)] = e
        if v is not None:
            mkt[j] = v
            final[j] = v
    ratio = np.where(mean > 1e-9, final / np.maximum(mean, 1e-9), 1.0)
    shift = np.where(mean > 1e-9, 0.0, final - mean)
    for c in ("mean", "p10", "p50", "p90"):
        out["consensus_" + c] = np.maximum(out[c].to_numpy(float) * ratio + (shift if c != "mean" else 0.0), 0.0)
    out["consensus_mean"] = final
    out["sales"], out["marketing"] = sales, mkt
    out["changed"] = np.abs(final - mean) > 1e-9
    return out, errors
