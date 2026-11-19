"""Audit log, frozen versions and data-import validation. No Streamlit."""
import io

import numpy as np
import pandas as pd

AUDIT_COLUMNS = ["Timestamp", "User", "Scope", "Period", "Previous value", "New value", "Reason code", "Comment"]
REQUIRED = ["Date", "SKU", "Family", "Units", "Price", "Inventory"]
ALIASES = {"fecha": "Date", "date": "Date", "sku": "SKU", "item_id": "SKU", "familia": "Family", "family": "Family", "dept_id": "Family",
           "unidades": "Units", "units": "Units", "precio": "Price", "price": "Price", "sell_price": "Price",
           "inventario": "Inventory", "inventory": "Inventory", "on_hand": "Inventory"}


def audit_rows(scope, weeks, prev, new, reason, comment, user, ts):
    """One immutable audit row per modified week. weeks: list of (label)."""
    return [{"Timestamp": ts, "User": user, "Scope": scope, "Period": w, "Previous value": round(float(p), 2),
             "New value": round(float(n), 2), "Reason code": reason, "Comment": comment} for w, p, n in zip(weeks, prev, new)]


def commit_changes(committed, new_final, baseline, sales_txt, mkt_txt, week_ids, labels, reason, comment, user, ts, scope):
    """Compares the edited plan with what was committed before; returns (new committed dict, audit rows).

    committed: {week_idx: {"final", "sales", "mkt", "reason", "comment", "user", "ts"}} for this scope.
    """
    new_c, rows_w, rows_p, rows_n = {}, [], [], []
    for j, w in enumerate(week_ids):
        changed = abs(new_final[j] - baseline[j]) > 1e-9
        old = committed.get(w)
        old_val = old["final"] if old else baseline[j]
        if changed:
            new_c[w] = {"final": float(new_final[j]), "sales": sales_txt[j], "mkt": mkt_txt[j], "reason": reason, "comment": comment, "user": user, "ts": ts}
            if abs(new_final[j] - old_val) > 1e-9:
                rows_w.append(labels[j]); rows_p.append(old_val); rows_n.append(new_final[j])
        elif old:                                              # override removed
            rows_w.append(labels[j]); rows_p.append(old_val); rows_n.append(baseline[j])
    return new_c, audit_rows(scope, rows_w, rows_p, rows_n, reason, comment, user, ts)


def make_version(name, scope, measure, weekly_final, ts, user):
    """Frozen snapshot of the plan. weekly_final: {week_idx: units}."""
    return {"name": name, "scope": scope, "ts": ts, "user": user, "weeks": {int(k): float(v) for k, v in weekly_final.items()},
            "total": float(sum(weekly_final.values()))}


def compute_fva(actual, stat, cons):
    """Forecast Value Added of the overrides = WMAPE(statistical) − WMAPE(consensus). Positive = overrides helped (percentage points)."""
    actual, stat, cons = (np.asarray(x, float) for x in (actual, stat, cons))
    tot = actual.sum()
    if tot <= 0:
        return float("nan")
    return (np.abs(actual - stat).sum() - np.abs(actual - cons).sum()) / tot * 100


# ----------------------------------------------------------------------------- import validation
def read_upload(raw, name):
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw))


def validate_upload(df, known_skus):
    """Returns (clean DataFrame, report DataFrame with columns Level, Check, Detail, Rows). Errors block, warnings do not."""
    rep = []
    add = lambda lvl, chk, det, n=0: rep.append({"Level": lvl, "Check": chk, "Detail": det, "Rows": int(n)})
    df = df.rename(columns={c: ALIASES.get(str(c).strip().lower(), str(c).strip()) for c in df.columns})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        add("Error", "Required columns", "Missing: " + ", ".join(missing), 0)
        return None, pd.DataFrame(rep)
    df = df[REQUIRED].copy()
    if df.empty:
        add("Error", "Rows", "The file has no rows.", 0)
        return None, pd.DataFrame(rep)
    for c in REQUIRED:
        n = int(df[c].isna().sum())
        if n:
            add("Warning", "Null values", f"Column {c} has empty cells", n)
    d = pd.to_datetime(df["Date"], errors="coerce")
    bad = int(d.isna().sum() - df["Date"].isna().sum())
    if bad:
        add("Error", "Dates", "Some dates could not be read (use YYYY-MM-DD)", bad)
    for c in ("Units", "Price", "Inventory"):
        v = pd.to_numeric(df[c], errors="coerce")
        nn = int(v.isna().sum() - df[c].isna().sum())
        if nn:
            add("Error", "Numbers", f"Column {c} has non-numeric values", nn)
        neg = int((v < 0).sum())
        if neg:
            add("Error", "Negative values", f"Column {c} has negative values", neg)
        df[c] = v
    df["Date"] = d
    dup = int(df.duplicated(["Date", "SKU"]).sum())
    if dup:
        add("Warning", "Duplicates", "Repeated Date + SKU (the last one is kept)", dup)
    unk = int((~df["SKU"].isin(known_skus) & df["SKU"].notna()).sum())
    if unk:
        add("Warning", "Unknown SKUs", "SKUs that are not in the model's SKU sample are ignored for DOH and prices", unk)
    errors = [r for r in rep if r["Level"] == "Error"]
    if errors:
        return None, pd.DataFrame(rep)
    clean = df.dropna(subset=["Date", "SKU"]).drop_duplicates(["Date", "SKU"], keep="last").sort_values(["SKU", "Date"]).reset_index(drop=True)
    if not rep:
        add("OK", "All checks", "No problems found.", len(clean))
    return clean, pd.DataFrame(rep)


def days_on_hand(on_hand, weekly_forecast):
    """Days of coverage: stock ÷ average daily forecast demand. inf if no demand."""
    daily = np.asarray(weekly_forecast, float) / 7.0
    return np.where(daily > 1e-9, np.asarray(on_hand, float) / np.maximum(daily, 1e-9), np.inf)
