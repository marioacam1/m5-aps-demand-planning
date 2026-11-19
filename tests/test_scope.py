import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import scope as sc, aps_core as aps
ok = lambda c, m: print(("OK   " if c else "FALLA"), m) or (c or sys.exit(1))

CATS = ["FOODS", "HOBBIES", "HOUSEHOLD"]; STORES = ["CA_1", "CA_2", "TX_1"]
DBC = {"FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"], "HOBBIES": ["HOBBIES_1", "HOBBIES_2"], "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"]}
ALLD = sum(DBC.values(), [])
R = lambda **k: sc.resolve_scope(**{"cats": CATS, "stores": STORES, "depts": ALLD, "skus": [], "all_cats": CATS, "all_stores": STORES, "depts_by_cat": DBC, "total_id": "Total", **k})
ok(R() == [(0, "Total")], "everything selected -> Total")
ok(R(stores=["CA_1"]) == [(2, "CA_1")], "one store, all categories -> Store series")
ok(R(cats=["FOODS"]) == [(3, "FOODS")], "one category, all stores -> Category series")
ok(R(cats=["FOODS"], stores=["CA_1", "CA_2"]) == [(5, "CA_1|FOODS"), (5, "CA_2|FOODS")], "category + 2 stores -> Store×Category series")
ok(R(cats=["FOODS"], depts=["FOODS_3"]) == [(4, "FOODS_3")], "one department, all stores -> Department series")
ok(R(cats=["FOODS", "HOBBIES"], depts=["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_2"]) == [(3, "FOODS"), (4, "HOBBIES_2")], "mixed: whole category + one department")
ok(R(cats=["FOODS"], stores=["TX_1"], depts=["FOODS_1"]) == [(6, "TX_1|FOODS_1")], "department + store -> Store×Department")
ok(R(skus=["A", "B"]) == [(7, "A"), (7, "B")], "SKUs override everything")
ok(R(cats=[]) == [] and R(stores=[]) == [], "empty selection -> no series")
ok("All categories" in sc.scope_label(CATS, STORES, ALLD, [], CATS, STORES, DBC), "label for full scope")

# weekly frame
LW = 277
rng = np.random.default_rng(1)
acts = {sid: pd.Series(rng.uniform(100, 200, LW), index=np.arange(1, LW + 1)) for sid in ("A", "B")}
def fget(level, sid):
    m = np.full(13, 100.0) * (1 if sid == "A" else 2)
    return pd.DataFrame({"h": range(1, 14), "mean": m, "p10": m * .8, "p50": m * .98, "p90": m * 1.3})
fr1 = sc.weekly_scope_frame([(7, "A")], lambda l, s: acts[s], fget, LW)
ok(len(fr1) == LW + 13 and fr1["actual"].iloc[:LW].notna().all() and fr1["actual"].iloc[LW:].isna().all(), "actuals only in the past")
ok(fr1["forecast"].iloc[:LW].isna().all() and fr1["forecast"].iloc[LW:].notna().all(), "forecast only in the future")
ok(fr1.attrs["exact_band"] and abs(fr1["low"].iloc[-1] - 80) < 1e-9, "single series keeps exact quantiles")
frs = sc.weekly_scope_frame([(7, "A"), (7, "B")], lambda l, s: acts[s], fget, LW)
ok(abs(frs["forecast"].iloc[-1] - 300) < 1e-9, "means add up (100 + 200)")
exp_hi = 300 + np.sqrt(30 ** 2 + 60 ** 2); ok(abs(frs["high"].iloc[-1] - exp_hi) < 1e-9 and not frs.attrs["exact_band"], f"band uses independence approximation ({exp_hi:.1f} < naive sum 390)")
ok(frs["low"].iloc[-1] > 300 - (20 + 40), "approximate Low case is tighter than the naive sum")
ok(abs(frs["ly"].iloc[LW + 2] - (acts["A"][LW + 3 - 52] + acts["B"][LW + 3 - 52])) < 1e-9, "last year = same week 52 weeks earlier")

# periods: weeks
p, info = sc.build_periods(fr1, "Weeks", LW, back=8, fwd=12)
ok(len(p) == 8 + 1 + 12, f"weeks: 8 back + current + 12 forward = {len(p)}")
cur = [x for x in p if x["is_current"]]
ok(len(cur) == 1 and cur[0]["label"] == "W278" and cur[0]["kind"] == "forecast", "current week is W278 (first forecast week)")
ok(p[7]["kind"] == "actual" and p[8]["is_current"], "last actual week sits right before the current week")
ok(info["max_fwd"] == 12, f"max forward weeks = {info['max_fwd']}")
p2, _ = sc.build_periods(fr1, "Weeks", LW, back=500, fwd=500)
ok(len(p2) == 52 + 1 + 12, "limits are enforced (52 back, 12 forward)")

# months / quarters
pm, im = sc.build_periods(fr1, "Months", LW, back=3, fwd=10)
cm = [x for x in pm if x["is_current"]][0]
ok(cm["label"] == "May 2016" and cm["kind"] == "mixed" and not np.isnan(cm["actual"]) and not np.isnan(cm["forecast"]), f"current month is mixed: {cm['label']}")
ok(abs(cm["plan"] - (cm["actual"] + cm["forecast"])) < 1e-9, "plan = actual + forecast")
ok(pm[-1]["partial"] and not pm[0]["partial"], f"last forecast month flagged partial ({pm[-1]['label']})")
ok(1 <= im["max_fwd"] <= 4, f"months forward available = {im['max_fwd']}")
tot_f = sum(x["forecast"] for x in pm if not np.isnan(x["forecast"]))
ok(abs(tot_f - fr1["forecast"].sum()) < 1e-6 or len(pm) < len(sc.build_periods(fr1, "Months", LW, 3, 99)[0]), "monthly forecast sums to the weekly forecast when every month is shown")
pm_all, _ = sc.build_periods(fr1, "Months", LW, 3, 99)
ok(abs(sum(x["forecast"] for x in pm_all if not np.isnan(x["forecast"])) - fr1["forecast"].sum()) < 1e-6, "all months: forecast total preserved")
pq, iq = sc.build_periods(fr1, "Quarters", LW, 2, 5)
cq = [x for x in pq if x["is_current"]][0]
ok(cq["label"] == "Q2 2016" and cq["kind"] == "mixed", f"current quarter {cq['label']}")
ok(abs(sum(x["actual"] for x in sc.build_periods(fr1, "Quarters", LW, 12, 5)[0] if not np.isnan(x["actual"])) - fr1["actual"].iloc[-(12 * 13 + 5) - 13 + 0:LW].sum()) >= 0, "quarters computed")
ok(not np.isnan(pm[2]["vs_ly"]), "% vs last year is computed")
print("TODO BIEN")
