import sys, io
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, math
import scenarios as sc, excel_export as xe
ok = lambda c, m: print(("OK   " if c else "FALLA"), m) or (c or sys.exit(1))

# 7) Escenarios
base = pd.DataFrame({"h": range(1, 14), "mean": 100.0, "p10": 80.0, "p50": 98.0, "p90": 130.0})
adj = sc.preset_overrides(sc.PRESETS[1], base)
ok((adj["override_pct"] == [0, 0, 20, 20, 20] + [0] * 8).all(), "plantilla de promoción: +20 % semanas 3 a 5")
res = sc.apply_overrides(base, adj)
ok(abs(res["consensus_mean"].sum() - (1300 + 60)) < 1e-9 and res.loc[2, "consensus_p90"] == 156, "consenso suma 1360 y P90 de la semana 3 es 156")
hi = sc.apply_overrides(base, sc.preset_overrides(sc.PRESETS[4], base))
ok(np.allclose(hi["consensus_mean"], base["p90"]), "escenario alto reproduce el P90")
lo = sc.apply_overrides(base, sc.preset_overrides(sc.PRESETS[5], base))
ok(np.allclose(lo["consensus_mean"], base["p10"]), "escenario bajo reproduce el P10")
a2 = sc.empty_overrides(13); a2["override_units"] = -500
ok((sc.apply_overrides(base, a2)["consensus_mean"] == 0).all(), "el consenso nunca baja de 0")
for bad in (-100, 900):
    a3 = sc.empty_overrides(13); a3.loc[0, "override_pct"] = bad
    try: sc.apply_overrides(base, a3); ok(False, f"debió rechazar {bad} %")
    except ValueError as e: ok(True, f"rechaza {bad} %: {e}")
a4 = sc.empty_overrides(13); a4["override_pct"] = a4["override_pct"].astype(object); a4.loc[0, "override_pct"] = None
try: sc.apply_overrides(base, a4); ok(False, "debió rechazar vacío")
except ValueError: ok(True, "rechaza celdas vacías")
cmp = sc.compare(base, {"Promo": res})
ok(abs(cmp.loc[1, "Δ %"] - 4.6153846) < 1e-4, f"comparativo: {cmp.loc[1, 'Δ %']:.2f} %")

# 8) Excel
data = xe.build_workbook({"Consenso": res, "Nombre muy largo " * 4: cmp, "Con/prohibido": adj})
x = pd.read_excel(io.BytesIO(data), sheet_name=None)
ok(len(x) == 3 and all(len(n) <= 31 for n in x) and len(x["Consenso"]) == 13, f"Excel con hojas {list(x)}")
print("TODO BIEN")
