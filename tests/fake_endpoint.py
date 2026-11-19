"""Servidor local que imita el contrato del endpoint (validación + pronóstico simple). Solo para pruebas."""
import json, math, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from datetime import date, timedelta

W1 = date(2011, 1, 29)
STATE = {"fail_next": 0, "fail_status": 503, "calls": 0, "delay": 0.0}

def forecast_row(r):
    h = np.array(r["history"], float); n = r["horizon"]; lw = r["last_week"]
    base = h[-13:].mean()
    weeks = [lw + i for i in range(1, n + 1)]
    mean = [base * (1 + .002 * i) for i in range(1, n + 1)]
    return {"series_id": r["series_id"], "status": "ok", "scale": float(base),
            "target_week": weeks, "week_start": [(W1 + timedelta(days=7 * (w - 1))).isoformat() for w in weeks],
            "mean": [round(m, 4) for m in mean], "p10": [round(m * .8, 4) for m in mean],
            "p50": [round(m * .98, 4) for m in mean], "p90": [round(m * 1.25, 4) for m in mean],
            "model_version": "1.3-semana277"}

def validate(rows):
    errs = []
    if not 1 <= len(rows) <= 200: errs.append("between 1 and 200 series")
    for i, r in enumerate(rows):
        h = r.get("history")
        if not isinstance(h, list) or not 26 <= len(h) <= 300: errs.append(f"fila {i}: history must have 26 to 300 weeks")
        elif any((not isinstance(v, (int, float))) or v < 0 or not math.isfinite(v) for v in h): errs.append(f"fila {i}: invalid history")
        if not isinstance(r.get("level"), int) or not 0 <= r["level"] <= 7: errs.append(f"fila {i}: invalid level")
    return errs

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        STATE["calls"] += 1
        if self.headers.get("Authorization") != "Bearer good-token": return self._send(401, {"message": "Invalid token"})
        if not self.path.endswith("/serving-endpoints/m5-aps-forecast/invocations"): return self._send(404, {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "does not exist"})
        if STATE["fail_next"] > 0:
            STATE["fail_next"] -= 1; return self._send(STATE["fail_status"], {"message": "temporarily unavailable"})
        body = self.rfile.read(int(self.headers["Content-Length"]))
        try: rows = json.loads(body)["dataframe_records"]
        except Exception: return self._send(400, {"message": "invalid body"})
        errs = validate(rows)
        if errs: return self._send(400, {"error_code": "BAD_REQUEST", "message": "Invalid input: " + " | ".join(errs[:8])})
        mode = STATE.get("mode")
        out = [forecast_row(r) for r in rows]
        if mode == "bad_quantiles": out[0]["p10"] = [v * 10 for v in out[0]["p10"]]
        if mode == "missing_field": out[0].pop("p90")
        if mode == "nan": out[0]["mean"][0] = -5
        if mode == "count": out = out[:-1]
        self._send(200, {"predictions": out})

def start():
    srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"
