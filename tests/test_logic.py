import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np, pandas as pd
import client, forecast_utils as fu, fake_endpoint as fe

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth/data")
hier = pd.read_csv(D / "history_hierarchy.csv.gz")
meta = hier.drop_duplicates("series_id")[["series_id", "level"]].reset_index(drop=True)
print("series:", len(meta), meta.level.value_counts().sort_index().to_dict())
by = {k: g for k, g in hier.groupby("series_id")}
LW = int(hier.week_idx.max())
srv, host = fe.start()
ok = lambda c, m: print(("OK   " if c else "FALLA"), m) or (c or sys.exit(1))

rows, skipped = fu.build_requests(meta, by, LW)
ok(len(rows) == 124 and not skipped, f"124 filas armadas, omitidas={skipped}")
fcs, info = client.query_endpoint(rows, host, "good-token", "m5-aps-forecast")
ok(len(fcs) == 124 and info["attempts"] == 1, f"respuesta válida de 124 series en {info['seconds']:.2f}s")
fc = fu.forecast_frame(fcs, dict(zip(meta.series_id, meta.level)))
ok(fc.shape[0] == 124 * 13, "tabla larga 124x13")
rec = fu.reconcile_bulk(fc, meta)
ct = fu.coherence_table(rec, meta)
gap0 = (ct["Total"] - ct["Sum of stores"]).abs().max(); gap1 = (ct["Total (reconciled)"] - ct["Sum of stores (reconciled)"]).abs().max()
ok(gap1 < 1e-6 < gap0, f"conciliación: brecha antes {gap0:.2f} y después {gap1:.2e}")
ok(((rec.p10_rec <= rec.p50_rec + 1e-6) & (rec.p50_rec <= rec.p90_rec + 1e-6)).all(), "cuantiles conciliados siguen ordenados")

# errores
def raises(exc, fn):
    try: fn()
    except exc as e: return e
    except Exception as e: print("tipo inesperado", type(e), e); return None
    return None
ok(raises(client.AuthError, lambda: client.query_endpoint(rows[:1], host, "malo", "m5-aps-forecast")), "401 -> AuthError sin reintento")
n = fe.STATE["calls"]; ok(raises(client.NotFound, lambda: client.query_endpoint(rows[:1], host, "good-token", "otro")), "404 -> NotFound")
bad = dict(rows[0], history=rows[0]["history"][:5]); e = raises(client.InvalidInput, lambda: client.query_endpoint([bad], host, "good-token", "m5-aps-forecast"))
ok(e and "26 to 300" in e.detail, "400 -> InvalidInput con detalle")
fe.STATE.update(fail_next=2, fail_status=503); sl = []
f2, i2 = client.query_endpoint(rows[:2], host, "good-token", "m5-aps-forecast", sleep=sl.append)
ok(i2["attempts"] == 3 and sl == [3, 6], f"503 x2 -> reintenta con esperas {sl} y tiene éxito")
fe.STATE.update(fail_next=99); sl = []
e = raises(client.Unavailable, lambda: client.query_endpoint(rows[:1], host, "good-token", "m5-aps-forecast", sleep=sl.append))
ok(e and sl == [3, 6, 12] and "4 attempts" in e.detail, f"503 persistente -> Unavailable tras 4 intentos {sl}"); fe.STATE["fail_next"] = 0
for mode, frag in [("bad_quantiles", "not ordered"), ("missing_field", "missing fields"), ("nan", "invalid value"), ("count", "Expected")]:
    fe.STATE["mode"] = mode
    e = raises(client.InvalidResponse, lambda: client.query_endpoint(rows[:3], host, "good-token", "m5-aps-forecast"))
    ok(e and frag in e.detail, f"respuesta inválida '{mode}' detectada: {e.detail[:60] if e else None}")
fe.STATE["mode"] = None
e = raises(client.Unavailable, lambda: client.query_endpoint(rows[:1], "http://127.0.0.1:9", "good-token", "m5", max_attempts=2, sleep=lambda s: None, timeout=(1, 1)))
ok(e, "servidor caído -> Unavailable")
ok(raises(client.ConfigError, lambda: client.query_endpoint(rows[:1], "", "t", "e")), "sin host -> ConfigError")
# items con huecos
items = pd.read_csv(D / "history_items.csv.gz"); iid = items.series_id.iloc[0]
r = fu.build_request(iid, 7, items[items.series_id == iid], LW); ok(r["history"] and len(r["history"]) >= 26, f"item {iid}: {len(r['history'])} semanas")
many = [dict(r, series_id=f"{iid}_{i}") for i in range(450)]
fm, im = client.query_many(many, host, "good-token", "m5-aps-forecast"); ok(len(fm) == 450, "query_many parte 450 series en 3 bloques")
print("TODO BIEN")
