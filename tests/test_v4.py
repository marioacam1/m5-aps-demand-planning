import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import scenarios as sc, abcxyz as ab, governance as gv, scope as sp
ok = lambda c, m: print(("OK   " if c else "FALLA"), m) or (c or sys.exit(1))
ok(sc.parse_cell("+15", 100) == (115.0, ""), "parse +15 units")
ok(sc.parse_cell("-30", 200)[0] == 170.0 and sc.parse_cell("-300", 200)[0] == 0.0, "parse -30 units, floored at 0")
ok(sc.parse_cell("500", 100) == (500.0, ""), "parse absolute")
ok(sc.parse_cell("+15%", 100)[0] is None and sc.parse_cell("+15%", 100)[1] != "", "percentages rejected")
ok(sc.parse_cell("", 100) == (None, ""), "blank")
ok(sc.parse_cell("abc", 1)[1] != "", "invalid inputs rejected")
base = pd.DataFrame({"h": range(1, 6), "mean": 100.0, "p10": 80.0, "p50": 98.0, "p90": 130.0})
res, err = sc.apply_layers(base, ["+20", "", "", "300", ""], ["", "", "+10", "", "x"])
ok(abs(res.loc[0, "consensus_mean"] - 120) < 1e-9 and abs(res.loc[2, "consensus_mean"] - 110) < 1e-9 and res.loc[3, "consensus_mean"] == 300, "cascade sales -> marketing")
ok(abs(res.loc[0, "consensus_p90"] - 156) < 1e-9, "quantiles scale with the ratio")
ok(("Marketing event", 5) in err and list(res["changed"]) == [True, False, True, True, False], "errors reported, changed flags")
ok(abs(res.loc[3, "consensus_p10"] - 240) < 1e-9, "absolute override scales quantiles")
# audit
c, rows = gv.commit_changes({}, [120, 100], [100, 100], ["+20%", ""], ["", ""], [278, 279], ["W278", "W279"], "Promotion", "Summer", "ana", "t", "S")
ok(list(c) == [278] and len(rows) == 1 and rows[0]["Previous value"] == 100 and rows[0]["New value"] == 120, "commit logs only changed weeks")
c2, rows2 = gv.commit_changes(c, [100, 100], [100, 100], ["", ""], ["", ""], [278, 279], ["W278", "W279"], "Other", "revert", "ana", "t2", "S")
ok(c2 == {} and rows2[0]["Previous value"] == 120 and rows2[0]["New value"] == 100, "revert is logged")
ok(abs(gv.compute_fva([100, 100], [80, 120], [95, 105]) - 15.0) < 1e-9, "FVA = WMAPE stat − WMAPE consensus")
# abc-xyz
LW = 100
rng = np.random.default_rng(0)
units = {f"I{i}": pd.Series(rng.poisson(50 / (i + 1), LW) + 0.0, index=np.arange(1, LW + 1)) for i in range(20)}
price = {k: 2.0 for k in units}
df = ab.classify(units, price, LW, 52)
ok(df["share"].sum() > 0.999 and (df[df.abc == "A"]["share"].sum() >= 0.8), "A covers at least 80% of revenue")
ok(set(df.abc) <= {"A", "B", "C"} and df.iloc[0]["abc"] == "A" and df.iloc[-1]["abc"] == "C", "ABC order")
cv = pd.Series([0.1, 0.3, 0.9, float("inf")]); ok(list(np.where(cv < .25, "X", np.where(cv <= .5, "Y", "Z"))) == ["X", "Y", "Z", "Z"], "XYZ cut-offs")
g = ab.grid_summary(df); ok(len(g) == 9 and g["items"].sum() == 20, "3x3 summary")
ok(ab.method_for("AZ", "Lumpy") == "Croston (SBA)" and ab.method_for("AX", "Smooth").startswith("Exponential"), "method assignment")
# upload
raw = pd.DataFrame({"fecha": ["2016-05-14", "2016-05-14", "bad"], "sku": ["A", "A", "B"], "familia": ["F", "F", "F"], "unidades": [1, 2, 3], "precio": [1.0, 1, 1], "inventario": [5, 5, None]})
clean, rep = gv.validate_upload(raw, ["A"]); ok(clean is None and (rep["Level"] == "Error").any(), "bad date blocks")
raw["fecha"] = ["2016-05-14", "2016-05-14", "2016-05-21"]
clean, rep = gv.validate_upload(raw, ["A"]); ok(clean is not None and len(clean) == 2 and "Duplicates" in set(rep["Check"]) and "Null values" in set(rep["Check"]), "dup + null warnings, clean accepted")
clean, rep = gv.validate_upload(raw.drop(columns=["precio"]), ["A"]); ok(clean is None and "Required columns" in set(rep["Check"]), "missing column")
ok(np.isinf(gv.days_on_hand([10], [0])[0]) and abs(gv.days_on_hand([70], [70])[0] - 7) < 1e-9, "days on hand")
# periods with consensus / prev
fr = pd.DataFrame({"week_idx": np.arange(270, 291)}); fr["week_start"] = pd.to_datetime(fr.week_idx.map(__import__("aps_core").week_start))
fr["actual"] = np.where(fr.week_idx <= 277, 10.0, np.nan); fr["forecast"] = np.where(fr.week_idx > 277, 20.0, np.nan)
fr["low"] = fr["forecast"] * .8; fr["high"] = fr["forecast"] * 1.2; fr["ly"] = 9.0
fr["consensus"] = np.where(fr.week_idx == 278, 30.0, np.nan); fr["prev"] = np.where(fr.week_idx > 277, 15.0, np.nan)
per, _ = sp.build_periods(fr, "Weeks", 277, 2, 2)
cur = [p for p in per if p["is_current"]][0]
ok(cur["consensus"] == 30 and cur["plan"] == 30 and cur["baseline_plan"] == 20 and cur["prev"] == 15 and abs(cur["vs_prev"] - 1.0) < 1e-9, "week period: consensus, plan, prev")
print("TODO BIEN")
