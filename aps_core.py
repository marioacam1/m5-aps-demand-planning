"""Núcleo de planeación de demanda semanal (mini APS).

Un solo archivo que usan tres lugares, para que el entrenamiento y el servicio nunca se desalineen:
- el notebook de Databricks que entrena y evalúa;
- el modelo publicado en el endpoint;
- la página web (para armar las filas que se envían al endpoint).

Contenido: características normalizadas por escala, modelos de cuantiles (P10, P50, P90),
calibración conforme (CQR), conciliación jerárquica (MinT con pesos estructurales) y métricas.
"""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

VERSION = "1.3"

WEEK1_START = date(2011, 1, 29)   # sábado: inicio de la semana 1 de M5 (las semanas van de sábado a viernes)
HORIZON = 13                      # semanas hacia adelante
MIN_SCALE = 0.05                  # promedio mínimo de 13 semanas para poder pronosticar una serie
MIN_VALID_13 = 4                  # semanas con dato (de las últimas 13) para poder pronosticar
QUANTILES = (0.1, 0.5, 0.9)

LEVEL_NAMES = {
    0: "Total", 1: "Estado", 2: "Tienda", 3: "Categoría", 4: "Departamento",
    5: "Tienda × Categoría", 6: "Tienda × Departamento", 7: "Producto (CA_1)",
}

# Variables que recibe el modelo (todas conocidas en el momento del pronóstico)
FEATURES = ["level", "h", "l0", "l1", "l2", "l3", "m4", "m26", "m52", "cv26", "zero26",
            "hist", "ya", "ya3", "s1", "c1", "s2", "c2"]
CATEGORICAL = ["level"]
INPUT_COLUMNS = FEATURES + ["scale"]   # el servicio recibe también la escala para devolver unidades


# ----------------------------------------------------------------------------- calendario
def week_start(week):
    """Fecha de inicio (sábado) de la semana `week` (1 = 29 de enero de 2011)."""
    return WEEK1_START + timedelta(days=7 * (int(week) - 1))


def season_features(weeks):
    """Estacionalidad anual como dos armónicos (seno y coseno) del día del año a mitad de semana."""
    weeks = np.asarray(weeks, dtype=int)
    uniq, inverse = np.unique(weeks, return_inverse=True)
    doy = np.array([(week_start(w) + timedelta(days=3)).timetuple().tm_yday for w in uniq], dtype=float)
    ang = 2 * np.pi * doy / 365.25
    return (np.sin(ang)[inverse], np.cos(ang)[inverse], np.sin(2 * ang)[inverse], np.cos(2 * ang)[inverse])


# ----------------------------------------------------------------------------- series
def series_from_long(df, id_col, week_col, y_col, yr_col, n_weeks, level_col=None, level=None):
    """Pasa una tabla larga (una fila por serie y semana) a {id: {level, y, yr}} con arreglos de n_weeks.

    y  = demanda depurada (se usa para las variables); yr = demanda real (se usa como objetivo).
    Las semanas sin dato (producto aún no disponible) quedan en NaN.
    """
    out = {}
    for sid, g in df.groupby(id_col, sort=False):
        y = np.full(n_weeks, np.nan)
        yr = np.full(n_weeks, np.nan)
        idx = g[week_col].to_numpy(dtype=int) - 1
        y[idx] = g[y_col].to_numpy(dtype=float)
        yr[idx] = g[yr_col].to_numpy(dtype=float)
        lvl = int(g[level_col].iloc[0]) if level_col else int(level)
        out[sid] = {"level": lvl, "y": y, "yr": yr}
    return out


def _prefix(a):
    return np.concatenate([[0.0], np.cumsum(a)])


def _window(prefix, t, width):
    lo = np.maximum(0, t - width + 1)
    return prefix[t + 1] - prefix[lo]


def _take(arr, idx):
    """arr[idx] con NaN cuando el índice cae fuera de la serie."""
    ok = (idx >= 0) & (idx < len(arr))
    return np.where(ok, arr[np.clip(idx, 0, len(arr) - 1)], np.nan)


# ----------------------------------------------------------------------------- muestras
def build_samples(series, origins, horizons=range(1, HORIZON + 1), with_target=True):
    """Construye una fila por (serie, semana de origen, horizonte).

    series  : {id: {"level", "y", "yr"}}
    origins : lista de semanas de origen (1-based) o función f(id, info) -> lista.
              El origen es la última semana observada.
    Todas las variables se calculan solo con datos hasta el origen; el objetivo es la demanda
    real del semana origen + h, dividida entre la escala (promedio de las últimas 13 semanas).
    """
    horizons = np.asarray(list(horizons), dtype=int)
    if horizons.max() > 50:
        raise ValueError("El horizonte máximo es 50 semanas (la variable de año anterior usaría el futuro).")
    frames = []
    for sid, info in series.items():
        y, yr = info["y"], info["yr"]
        n = len(y)
        org = origins(sid, info) if callable(origins) else origins
        t = np.asarray(org, dtype=int) - 1
        t = t[(t >= 0) & (t < n)]
        if len(t) == 0:
            continue

        valid = ~np.isnan(y)
        zero = valid & (y == 0)
        cs, cn = _prefix(np.where(valid, y, 0.0)), _prefix(valid.astype(float))
        cq, cz = _prefix(np.where(valid, y * y, 0.0)), _prefix(zero.astype(float))

        n13, s13 = _window(cn, t, 13), _window(cs, t, 13)
        scale = np.where(n13 >= MIN_VALID_13, s13 / np.maximum(n13, 1), np.nan)
        scale = np.where(scale >= MIN_SCALE, scale, np.nan)

        n4 = _window(cn, t, 4)
        m4 = np.where(n4 >= 2, _window(cs, t, 4) / np.maximum(n4, 1), np.nan) / scale
        n26, s26, q26 = _window(cn, t, 26), _window(cs, t, 26), _window(cq, t, 26)
        mean26 = np.where(n26 >= 8, s26 / np.maximum(n26, 1), np.nan)
        m26 = mean26 / scale
        var26 = np.maximum(q26 / np.maximum(n26, 1) - mean26 ** 2, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cv26 = np.where(mean26 > 0, np.sqrt(var26) / mean26, np.nan)
        zero26 = np.where(n26 >= 8, _window(cz, t, 26) / np.maximum(n26, 1), np.nan)
        n52 = _window(cn, t, 52)
        m52 = np.where(n52 >= 26, _window(cs, t, 52) / np.maximum(n52, 1), np.nan) / scale
        hist = np.minimum(cn[t + 1], 104.0)
        lags = [_take(y, t - k) / scale for k in range(4)]

        idx = t[:, None] + horizons[None, :]                      # posición de la semana objetivo
        j = idx - 52                                              # misma semana del año anterior
        ya = _take(y, j) / scale[:, None]
        trio = np.stack([_take(y, j - 1), _take(y, j), _take(y, j + 1)], axis=-1)
        cnt = (~np.isnan(trio)).sum(axis=-1)
        ya3 = np.where(cnt > 0, np.nansum(trio, axis=-1) / np.maximum(cnt, 1), np.nan) / scale[:, None]
        s1, c1, s2, c2 = season_features((idx + 1).ravel())

        shape = idx.shape

        def col(v):
            return np.broadcast_to(np.asarray(v).reshape(-1, 1) if np.ndim(v) == 1 else v, shape).ravel()

        data = {
            "series_id": np.repeat(sid, idx.size),
            "level": np.repeat(info["level"], idx.size),
            "origin_week": col(t + 1),
            "h": np.broadcast_to(horizons[None, :], shape).ravel(),
            "target_week": (idx + 1).ravel(),
            "scale": col(scale),
            "l0": col(lags[0]), "l1": col(lags[1]), "l2": col(lags[2]), "l3": col(lags[3]),
            "m4": col(m4), "m26": col(m26), "m52": col(m52), "cv26": col(cv26), "zero26": col(zero26),
            "hist": col(hist), "ya": ya.ravel(), "ya3": ya3.ravel(),
            "s1": s1, "c1": c1, "s2": s2, "c2": c2,
        }
        y_true = _take(yr, idx).ravel()
        data["y_true"] = y_true
        data["y_ratio"] = y_true / data["scale"]
        fallback = data["scale"]
        data["b_naive"] = np.where(np.isnan(col(_take(y, t))), fallback, col(_take(y, t)))
        snaive = _take(y, j).ravel()
        data["b_snaive"] = np.where(np.isnan(snaive), fallback, snaive)
        data["b_ma13"] = fallback
        frame = pd.DataFrame(data)
        frame = frame[~frame["scale"].isna()]
        if with_target:
            frame = frame[~frame["y_true"].isna()]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- modelo
def make_model(quantile, max_iter=250, max_depth=8, learning_rate=0.08, min_samples_leaf=40, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, max_iter=max_iter, max_depth=max_depth,
        learning_rate=learning_rate, min_samples_leaf=min_samples_leaf, l2_regularization=1.0,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=seed)


# Dos familias de modelos: la jerarquía (series densas, razón cercana a 1) y los productos (muchos ceros).
# Mezclarlas en un solo modelo hacía que el cuantil P10 de la jerarquía se quedara pegado en 0.
GROUPS = {"hier": (0, 1, 2, 3, 4, 5, 6), "item": (7,)}


def make_mean_model(max_iter=250, max_depth=8, learning_rate=0.08, min_samples_leaf=40, seed=42):
    """Modelo de la media (pérdida de Poisson): el pronóstico puntual que sí es aditivo en la jerarquía."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf, l2_regularization=1.0,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=seed)


def fit_models(train, **kwargs):
    """Entrena, por familia de series, un modelo por cuantil (P10, P50, P90) y un modelo de la media."""
    models = {}
    for name, levels in GROUPS.items():
        part = train[train["level"].isin(levels)]
        if len(part) == 0:
            continue
        X = part[FEATURES].to_numpy(dtype=np.float32)
        y = part["y_ratio"].to_numpy(dtype=np.float32)
        models[name] = {q: make_model(q, **kwargs).fit(X, y) for q in QUANTILES}
        models[name]["mean"] = make_mean_model(**kwargs).fit(X, y)
    return models


def predict_ratio(models, df):
    """Predice (P10, P50, P90) en razón a la escala, sin cruces entre cuantiles ni valores negativos."""
    P = np.zeros((len(df), 3))
    for name, levels in GROUPS.items():
        if name not in models:
            continue
        mask = df["level"].isin(levels).to_numpy()
        if mask.any():
            X = df.loc[mask, FEATURES].to_numpy(dtype=np.float32)
            Pg = np.column_stack([models[name][q].predict(X) for q in QUANTILES])
            P[mask] = np.sort(np.maximum(Pg, 0.0), axis=1)
    return P


def predict_mean(models, df):
    """Predice la media en razón a la escala (nunca negativa)."""
    out = np.zeros(len(df))
    for name, levels in GROUPS.items():
        if name not in models:
            continue
        mask = df["level"].isin(levels).to_numpy()
        if mask.any():
            X = df.loc[mask, FEATURES].to_numpy(dtype=np.float32)
            out[mask] = np.maximum(models[name]["mean"].predict(X), 0.0)
    return out


def fit_cqr(P, y_ratio, groups, alpha=0.2, allow_shrink=False):
    """Calibración conforme de cuantiles (CQR) por grupo; devuelve el ajuste de ancho en razón a la escala.

    Por defecto solo ensancha el intervalo (ajuste >= 0): con pocas series por nivel, acortarlo con datos
    de un periodo corto hacía que la cobertura cayera en el periodo de prueba.
    """
    y_ratio, groups = np.asarray(y_ratio), np.asarray(groups)
    qhat = {}
    for g in np.unique(groups):
        m = groups == g
        scores = np.maximum(P[m, 0] - y_ratio[m], y_ratio[m] - P[m, 2])
        n = int(m.sum())
        k = min(math.ceil((n + 1) * (1 - alpha)), n)
        value = float(np.sort(scores)[k - 1])
        shrink = allow_shrink.get(int(g), False) if isinstance(allow_shrink, dict) else allow_shrink
        qhat[int(g)] = value if shrink else max(value, 0.0)
    return qhat


def apply_cqr(P, groups, qhat):
    adj = np.array([qhat.get(int(g), 0.0) for g in groups])
    out = P.copy()
    out[:, 0] = np.minimum(np.maximum(P[:, 0] - adj, 0.0), out[:, 1])
    out[:, 2] = np.maximum(P[:, 2] + adj, out[:, 1])
    return out


def apply_bias(P, groups, bias):
    """Multiplica los tres cuantiles por el factor de sesgo de cada nivel."""
    factor = np.array([bias.get(int(g), 1.0) for g in groups])
    return P * factor[:, None]


def calibrated_ratio(P, groups, calib):
    """Corrige el sesgo y calibra los intervalos. calib = {"bias": {...}, "qhat": {...}}."""
    return apply_cqr(apply_bias(P, groups, calib["bias"]), groups, calib["qhat"])


def calibrated_mean(m, groups, calib):
    """Corrige el sesgo de la media con el factor de cada nivel."""
    return apply_bias(np.asarray(m).reshape(-1, 1), groups, calib["bias"]).ravel()


def forecast_units(models, calib, df):
    """De filas con variables y escala a pronósticos P10/P50/P90 en unidades (calibrados)."""
    P = calibrated_ratio(predict_ratio(models, df), df["level"].to_numpy(), calib)
    return P * df["scale"].to_numpy()[:, None]


def forecast_mean_units(models, calib, df):
    """Pronóstico de la media en unidades (calibrado en sesgo)."""
    return calibrated_mean(predict_mean(models, df), df["level"].to_numpy(), calib) * df["scale"].to_numpy()


# ----------------------------------------------------------------------------- backtesting móvil y calibración
def rolling_backtest(series, cutoffs, eval_offsets=(0, 4, 8, 12), origins_fn=None, **fit_kwargs):
    """Reentrena con datos hasta cada corte y pronostica desde los orígenes corte + desplazamiento.

    Devuelve las predicciones crudas (razón a la escala: r10, r50, r90) de todas las ventanas.
    Cada ventana simula un pronóstico real: el modelo solo conoce el pasado de ese corte.
    """
    full = build_samples(series, origins_fn if origins_fn is not None else list(range(56, 270)))
    parts = []
    for cutoff in cutoffs:
        models = fit_models(full[full["target_week"] <= cutoff], **fit_kwargs)
        ev = build_samples(series, [cutoff + o for o in eval_offsets])
        P = predict_ratio(models, ev)
        ev = ev.copy()
        ev["window"] = cutoff
        ev["r10"], ev["r50"], ev["r90"] = P[:, 0], P[:, 1], P[:, 2]
        ev["rmean"] = predict_mean(models, ev)
        parts.append(ev)
    return pd.concat(parts, ignore_index=True)


def fit_bias(bt, lo=0.85, hi=1.15):
    """Factor por nivel = ventas reales / media pronosticada (ponderado por unidades), acotado a [lo, hi]."""
    out = {}
    for lvl, g in bt.groupby("level"):
        pred = float((g["rmean"] * g["scale"]).sum())
        out[int(lvl)] = float(np.clip(g["y_true"].sum() / pred, lo, hi)) if pred > 0 else 1.0
    return out


def fit_calibration(bt, use_bias=None, shrink=None, alpha=0.2):
    """Ajusta el factor de sesgo (solo en los niveles indicados) y el ancho de los intervalos."""
    use_bias = use_bias or {}
    factors = fit_bias(bt)
    bias = {lvl: (f if use_bias.get(lvl, False) else 1.0) for lvl, f in factors.items()}
    R = apply_bias(bt[["r10", "r50", "r90"]].to_numpy(), bt["level"].to_numpy(), bias)
    qhat = fit_cqr(R, bt["y_ratio"].to_numpy(), bt["level"].to_numpy(), alpha, shrink or False)
    return {"bias": bias, "qhat": qhat}


def out_of_fold(bt, use_bias=None, shrink=None, alpha=0.2):
    """Validación cruzada dejando una ventana fuera: calibra con las demás y pronostica la omitida."""
    parts = []
    for w in sorted(bt["window"].unique()):
        held = bt[bt["window"] == w]
        calib = fit_calibration(bt[bt["window"] != w], use_bias, shrink, alpha)
        R = held[["r10", "r50", "r90"]].to_numpy()
        P = calibrated_ratio(R, held["level"].to_numpy(), calib)
        m = calibrated_mean(held["rmean"].to_numpy(), held["level"].to_numpy(), calib)
        parts.append(attach_forecasts(held, R, P, held["rmean"].to_numpy(), m))
    return pd.concat(parts, ignore_index=True)


def choose_calibration(bt, alpha=0.2, bias_gain=0.02):
    """Elige por nivel, con validación cruzada, si conviene corregir el sesgo y si el intervalo puede acortarse.

    - Corrige el sesgo solo si baja el WAPE al menos `bias_gain` (relativo) fuera de muestra.
    - Permite acortar el intervalo solo si así la cobertura queda más cerca del objetivo (1 - alpha).
    """
    levels = sorted(bt["level"].unique())
    target = 1 - alpha

    def by_level(oof):
        rows = {}
        for lvl, g in oof.groupby("level"):
            y = g["y_true"].to_numpy()
            rows[int(lvl)] = {"wape": wape(y, g["mean"]), "bias": bias(y, g["mean"]),
                              "cov": float(np.mean((y >= g["p10"]) & (y <= g["p90"])))}
        return rows

    plain = by_level(out_of_fold(bt, {}, False, alpha))
    corrected = by_level(out_of_fold(bt, {l: True for l in levels}, False, alpha))
    use_bias = {l: bool(corrected[l]["wape"] < plain[l]["wape"] * (1 - bias_gain)) for l in levels}

    widen = by_level(out_of_fold(bt, use_bias, False, alpha))
    both = by_level(out_of_fold(bt, use_bias, True, alpha))
    shrink = {l: bool(abs(both[l]["cov"] - target) < abs(widen[l]["cov"] - target)) for l in levels}
    final = by_level(out_of_fold(bt, use_bias, shrink, alpha))

    table = pd.DataFrame([{
        "level": l, "level_name": LEVEL_NAMES[l],
        "wape_sin_corregir": plain[l]["wape"], "wape_con_correccion": corrected[l]["wape"],
        "sesgo_sin_corregir": plain[l]["bias"], "sesgo_con_correccion": corrected[l]["bias"],
        "corrige_sesgo": use_bias[l],
        "cobertura_solo_ensancha": widen[l]["cov"], "cobertura_bilateral": both[l]["cov"],
        "acorta_intervalo": shrink[l], "cobertura_final": final[l]["cov"],
    } for l in levels])
    return use_bias, shrink, table


def bias_by_window(bt):
    """Sesgo de la media sin corregir por nivel y ventana (para ver si es persistente o de un solo periodo)."""
    rows = []
    for (w, lvl), g in bt.groupby(["window", "level"]):
        rows.append({"window": int(w), "level_name": LEVEL_NAMES[int(lvl)],
                     "sesgo": bias(g["y_true"].to_numpy(), (g["rmean"] * g["scale"]).to_numpy())})
    return pd.DataFrame(rows).pivot(index="level_name", columns="window", values="sesgo")


# ----------------------------------------------------------------------------- conciliación
def summing_matrix(meta):
    """Matriz de suma S (series × series base) de la jerarquía. meta: series_id y level (0 a 6)."""
    meta = meta.reset_index(drop=True)
    bottom = meta.loc[meta["level"] == 6, "series_id"].tolist()
    parts = [b.split("|") for b in bottom]
    S = np.zeros((len(meta), len(bottom)))
    for i, (sid, lvl) in enumerate(zip(meta["series_id"], meta["level"])):
        for j, (store, dept) in enumerate(parts):
            cat, state = dept.split("_")[0], store.split("_")[0]
            if lvl == 0:
                ok = True
            elif lvl == 1:
                ok = state == sid
            elif lvl == 2:
                ok = store == sid
            elif lvl == 3:
                ok = cat == sid
            elif lvl == 4:
                ok = dept == sid
            elif lvl == 5:
                a, b = sid.split("|")
                ok = store == a and cat == b
            else:
                ok = sid == f"{store}|{dept}"
            S[i, j] = 1.0 if ok else 0.0
    return S


def reconcile_wls(Yhat, S):
    """Conciliación MinT con pesos estructurales: pronósticos coherentes (la suma de las partes es el total)."""
    lam = 1.0 / S.sum(axis=1)
    SW = S.T * lam
    G = np.linalg.solve(SW @ S, SW)
    bottom = np.maximum(G @ Yhat, 0.0)
    return S @ bottom


def reconcile_forecasts(res, meta):
    """Concilia la media de las series jerárquicas (nivel < 7) para que las partes sumen el total.

    Se concilia la media porque es aditiva; la mediana no lo es. Los cuantiles se reescalan con el mismo cambio
    relativo. Agrega mean_rec, p10_rec, p50_rec y p90_rec; el nivel de producto queda igual.
    """
    out = res.copy()
    out["mean_rec"] = out["mean"]
    for c in ("p10", "p50", "p90"):
        out[c + "_rec"] = out[c]
    hier = out[out["level"] < 7]
    if hier.empty:
        return out
    S = summing_matrix(meta)
    order = meta["series_id"].tolist()
    piv = hier.pivot_table(index="series_id", columns=["origin_week", "h"], values="mean", aggfunc="first")
    piv = piv.reindex(order).fillna(0.0)
    rec_arr = reconcile_wls(piv.to_numpy(), S)
    cols = piv.columns
    rec_long = pd.DataFrame({
        "series_id": np.repeat(piv.index.to_numpy(), len(cols)),
        "origin_week": np.tile(cols.get_level_values(0).to_numpy(), len(piv)),
        "h": np.tile(cols.get_level_values(1).to_numpy(), len(piv)),
        "mean_rec": rec_arr.ravel(),
    })
    key = hier[["series_id", "origin_week", "h"]].merge(rec_long, on=["series_id", "origin_week", "h"], how="left")
    mean = hier["mean"].to_numpy()
    ratio = np.where(mean > 0, key["mean_rec"].to_numpy() / np.where(mean > 0, mean, 1.0), 1.0)
    out.loc[hier.index, "mean_rec"] = key["mean_rec"].to_numpy()
    for c in ("p10", "p50", "p90"):
        out.loc[hier.index, c + "_rec"] = hier[c].to_numpy() * ratio
    return out


# ----------------------------------------------------------------------------- métricas
def wape(y, p):
    s = float(np.sum(y))
    return float(np.sum(np.abs(np.asarray(y) - np.asarray(p))) / s) if s > 0 else float("nan")


def bias(y, p):
    s = float(np.sum(y))
    return float((np.sum(p) - s) / s) if s > 0 else float("nan")


def pinball(y, p, q):
    d = np.asarray(y) - np.asarray(p)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def add_buckets(df):
    out = df.copy()
    out["h_bucket"] = pd.cut(out["h"], [0, 4, 8, 13], labels=["1-4", "5-8", "9-13"]).astype(str)
    out["level_name"] = out["level"].map(LEVEL_NAMES)
    return out


def accuracy_summary(res, by):
    """Tabla de precisión de la media (pronóstico puntual): WAPE con y sin conciliar, tres líneas base, valor
    agregado (FVA) frente a la mejor, sesgo; WAPE y sesgo del P50; cobertura del intervalo P10-P90 y fracción de
    casos por debajo de P10, P50 y P90 (confiabilidad de cada cuantil); pinball normalizado."""
    rows = []
    for key, g in res.groupby(by):
        y = g["y_true"].to_numpy()
        base = {"naive": wape(y, g["b_naive"]), "snaive": wape(y, g["b_snaive"]), "ma13": wape(y, g["b_ma13"])}
        valid = [v for v in base.values() if not math.isnan(v)]
        best = min(valid) if valid else float("nan")
        w_mean = wape(y, g["mean"])
        mean_y = float(np.mean(y))
        pin = np.mean([pinball(y, g["p10"], 0.1), pinball(y, g["p50"], 0.5), pinball(y, g["p90"], 0.9)])
        row = dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else (key,)))
        row.update({
            "n": len(g), "wape_media": w_mean,
            "wape_conciliada": wape(y, g["mean_rec"]) if "mean_rec" in g else float("nan"),
            "wape_naive": base["naive"], "wape_estacional": base["snaive"], "wape_prom13": base["ma13"],
            "fva_vs_mejor_base": best - w_mean if not math.isnan(best) else float("nan"),
            "sesgo_media": bias(y, g["mean"]),
            "wape_p50": wape(y, g["p50"]), "sesgo_p50": bias(y, g["p50"]),
            "cobertura_sin_calibrar": float(np.mean((y >= g["p10_raw"]) & (y <= g["p90_raw"]))),
            "cobertura_calibrada": float(np.mean((y >= g["p10"]) & (y <= g["p90"]))),
            "bajo_p10": float(np.mean(y < g["p10"])), "bajo_p50": float(np.mean(y <= g["p50"])),
            "bajo_p90": float(np.mean(y <= g["p90"])),
            "pinball_norm": float(pin / mean_y) if mean_y > 0 else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- ensamblado
def attach_forecasts(samples, P_raw, P_cal, m_raw=None, m_cal=None):
    """Une las predicciones (en unidades) a las filas de muestra para evaluar."""
    out = samples.copy()
    scale = out["scale"].to_numpy()
    for k, name in enumerate(("p10", "p50", "p90")):
        out[name + "_raw"] = P_raw[:, k] * scale
        out[name] = P_cal[:, k] * scale
    if m_raw is not None:
        out["mean_raw"] = np.asarray(m_raw) * scale
        out["mean"] = np.asarray(m_cal if m_cal is not None else m_raw) * scale
    return out
