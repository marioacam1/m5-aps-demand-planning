"""Charts (Altair). Actuals are always blue, forecasts orange, consensus green; numbers are printed on the points."""
import altair as alt
import pandas as pd

GRID = "#8A94A3"
INK = "#0B2545"
ACTUAL, FORECAST, CONSENSUS, SHADE = "#1F5FBF", "#F26B21", "#2E9E5B", "#FFF1E6"
AXIS_FMT = ",.2~f"      # 100000 -> 100,000 ; 0.5 -> 0.5 (never "500m")


def fmt_label(v):
    """Compact number for chart labels: 1.23M · 105.4k · 4,321 · 85 · 12.4 · 0.35."""
    v = float(v)
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.1f}k"
    if a >= 1e3:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _lab(df, col="value"):
    return df.assign(label=[fmt_label(v) for v in df[col]])
SCENARIO_COLORS = ["#F26B21", "#2E9E5B", "#7C4DFF", "#E91E63", "#00A3A3", "#8D6E63"]

# ABC-XYZ: green (AX) -> yellow -> red (CZ)
SEG_ORDER = ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]
SEG_COLORS = {"AX": "#1E9E5A", "AY": "#7CC04B", "BX": "#7CC04B", "AZ": "#F2D13C", "BY": "#F2D13C", "CX": "#F2D13C",
              "BZ": "#F4902F", "CY": "#F4902F", "CZ": "#D64545"}
SEG_TEXT = {"AX": "#FFFFFF", "AY": INK, "BX": INK, "AZ": INK, "BY": INK, "CX": INK, "BZ": "#FFFFFF", "CY": "#FFFFFF", "CZ": "#FFFFFF"}
CLASS_COLORS = {"Smooth": "#1E9E5A", "Erratic": "#F2B01E", "Intermittent": "#F4902F", "Lumpy": "#D64545", "No demand": "#9AA5B1"}
STATUS_COLORS = {"Critical": "#D64545", "Order now": "#F4902F", "Watch": "#F2D13C", "OK": "#1E9E5A"}
STATUS_TEXT = {"Critical": "#FFFFFF", "Order now": "#FFFFFF", "Watch": INK, "OK": "#FFFFFF"}


def _x_time(fmt="%b %d"):
    return alt.X("week_start:T", title=None, axis=alt.Axis(grid=True, gridColor=GRID, gridOpacity=0.25, gridDash=[2, 3],
                                                           domainColor=GRID, tickColor=GRID, format=fmt, labelAngle=-45))


def _y(title="Units per week"):
    return alt.Y("value:Q", title=title, scale=alt.Scale(zero=True),
                 axis=alt.Axis(grid=False, domainColor=GRID, tickColor=GRID, format=AXIS_FMT))


def _finish(chart, height=360):
    return chart.properties(height=height).configure_view(stroke=None).configure_axis(labelFontSize=12, titleFontSize=12)


def _tooltip(extra=()):
    return [alt.Tooltip("week_start:T", title="Week of", format="%Y-%m-%d"), alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("value:Q", title="Units", format=",.1f"), *extra]


def actuals_vs_forecast(hist, forecasts, band=None, label_every=1, x_format="%b %d", height=430, width=None, y_zero=True):
    """Actuals (blue) followed by one or more forecasts, with the forecast horizon shaded.

    hist: DataFrame week_start, units.
    forecasts: list of dicts {name, df (week_start, value), color, dy}; band: DataFrame week_start, low, high.
    Every forecast point carries its number; actuals are labelled every `label_every` weeks (the last one always).
    """
    a = hist[["week_start", "units"]].rename(columns={"units": "value"}).reset_index(drop=True).assign(series="Actuals")
    keep = ((len(a) - 1 - a.index) % max(label_every, 1) == 0)
    last = a.tail(1)
    names = ["Actuals"] + [f["name"] for f in forecasts]
    colors = [ACTUAL] + [f["color"] for f in forecasts]
    frames = [a]
    for f in forecasts:
        frames.append(pd.concat([last.assign(series=f["name"]), f["df"][["week_start", "value"]].assign(series=f["name"])]))
    long = pd.concat(frames, ignore_index=True)

    x, y = _x_time(x_format), _y()
    if not y_zero:
        y = alt.Y("value:Q", title="Units per week", scale=alt.Scale(zero=False, nice=True), axis=alt.Axis(grid=False, domainColor=GRID, tickColor=GRID, format=AXIS_FMT))
    color = alt.Color("series:N", scale=alt.Scale(domain=names, range=colors),
                      legend=alt.Legend(title=None, orient="top", symbolSize=220, labelFontSize=13, symbolStrokeWidth=4))
    x0 = last["week_start"].iloc[0]
    x1 = max(f["df"]["week_start"].max() for f in forecasts) + pd.Timedelta(days=3)
    shade = alt.Chart(pd.DataFrame({"x0": [x0], "x1": [x1]})).mark_rect(color=SHADE, opacity=1).encode(x="x0:T", x2="x1")
    tag = alt.Chart(pd.DataFrame({"x": [x0], "t": ["FORECAST HORIZON  →"]})).mark_text(
        align="left", baseline="top", dx=8, fontWeight="bold", fontSize=12, color=FORECAST).encode(x="x:T", y=alt.value(4), text="t:N")
    layers = [shade, tag]
    if band is not None:
        layers.append(alt.Chart(band).mark_area(opacity=0.22, color=FORECAST).encode(
            x=x, y=alt.Y("low:Q", title="Units per week"), y2="high:Q",
            tooltip=[alt.Tooltip("week_start:T", title="Week of", format="%Y-%m-%d"),
                     alt.Tooltip("low:Q", title="Low case", format=",.1f"), alt.Tooltip("high:Q", title="High case", format=",.1f")]))
    layers.append(alt.Chart(long).mark_line(strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=70)).encode(
        x=x, y=y, color=color, tooltip=_tooltip()))
    layers.append(alt.Chart(_lab(a[keep])).mark_text(dy=-13, fontSize=11, fontWeight="bold", color=ACTUAL).encode(
        x=x, y=y, text="label:N"))
    for f in forecasts:
        layers.append(alt.Chart(_lab(f["df"][["week_start", "value"]])).mark_text(dy=f.get("dy", -13), fontSize=11, fontWeight="bold", color=f["color"]).encode(
            x=x, y=y, text="label:N"))
    chart = _finish(alt.layer(*layers), height)
    return chart.properties(width=width) if width else chart


def scenario_chart(base, scenarios, height=360):
    """Weekly mean of the statistical forecast and of every saved scenario; numbers are printed for the first two lines."""
    parts = [base[["week_start", "mean"]].rename(columns={"mean": "value"}).assign(series="Statistical forecast")]
    for name, r in scenarios.items():
        parts.append(pd.DataFrame({"week_start": base["week_start"].to_numpy(), "value": r["consensus_mean"].to_numpy(), "series": name}))
    long = pd.concat(parts, ignore_index=True)
    names = list(dict.fromkeys(long["series"]))
    color = alt.Color("series:N", scale=alt.Scale(domain=names, range=SCENARIO_COLORS[:len(names)]),
                      legend=alt.Legend(title=None, orient="top", symbolSize=220, labelFontSize=13))
    x, y = _x_time(), _y()
    layers = [alt.Chart(long).mark_line(strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=70)).encode(x=x, y=y, color=color, tooltip=_tooltip())]
    for i, nm in enumerate(names[:2]):
        layers.append(alt.Chart(_lab(long[long["series"] == nm])).mark_text(dy=-13 if i == 0 else 16, fontSize=11, fontWeight="bold").encode(
            x=x, y=y, color=color, text="label:N"))
    return _finish(alt.layer(*layers), height)


def group_bars(df, category, series_col, value, colors, title_y="Units", height=330, pct=False):
    """Grouped bars with the number on top of each bar (percent labels when pct=True)."""
    names = list(colors)
    df = _lab(df, value) if not pct else df.assign(label=[f"{v * 100:.0f}%" for v in df[value]])
    base = alt.Chart(df).encode(x=alt.X(f"{category}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
                                xOffset=alt.XOffset(f"{series_col}:N", sort=names))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y(f"{value}:Q", title=title_y, axis=alt.Axis(grid=False, format=".0%" if pct else AXIS_FMT)),
        color=alt.Color(f"{series_col}:N", scale=alt.Scale(domain=names, range=[colors[n] for n in names]),
                        legend=alt.Legend(title=None, orient="top", symbolSize=200, labelFontSize=12)),
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{series_col}:N"), alt.Tooltip(f"{value}:Q", format=",.0f")])
    text = base.mark_text(dy=-8, fontSize=11, fontWeight="bold").encode(
        y=alt.Y(f"{value}:Q"), text="label:N",
        color=alt.Color(f"{series_col}:N", scale=alt.Scale(domain=names, range=[colors[n] for n in names]), legend=None))
    return _finish(alt.layer(bars, text), height)


def segment_heat(grid, value_col, title, fmt=",.0f", height=250):
    """ABC × XYZ matrix coloured by segment (green AX → red CZ) with the segment and its value written in each cell."""
    base = alt.Chart(grid).encode(
        x=alt.X("xyz:O", title="Demand variability (XYZ)  ·  X stable → Z erratic", axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=13)),
        y=alt.Y("abc:O", title="Revenue (ABC)", sort=["A", "B", "C"], axis=alt.Axis(labelAngle=0, labelFontSize=13)))
    rect = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=3).encode(
        color=alt.Color("segment:N", scale=alt.Scale(domain=SEG_ORDER, range=[SEG_COLORS[s] for s in SEG_ORDER]), legend=None),
        tooltip=[alt.Tooltip("segment:N", title="Segment"), alt.Tooltip(f"{value_col}:Q", title=title, format=fmt)])
    tcolor = alt.Color("text_color:N", scale=None, legend=None)
    code = base.mark_text(dy=-13, fontSize=13, fontWeight="bold").encode(text="segment:N", color=tcolor)
    val = base.mark_text(dy=9, fontSize=19, fontWeight="bold").encode(text=alt.Text(f"{value_col}:Q", format=fmt), color=tcolor)
    return _finish(alt.layer(rect, code, val).properties(title=alt.TitleParams(title, anchor="start", fontSize=14, color=INK)), height)


def class_bars(df, height=300):
    """SKU count by demand class, one colour per class, with the count on each bar."""
    names = [c for c in CLASS_COLORS if c in set(df["demand_class"])] + [c for c in df["demand_class"] if c not in CLASS_COLORS]
    base = alt.Chart(df).encode(x=alt.X("demand_class:N", sort=names, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("items:Q", title="SKUs", axis=alt.Axis(grid=False, format=AXIS_FMT)),
        color=alt.Color("demand_class:N", scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())), legend=None),
        tooltip=[alt.Tooltip("demand_class:N", title="Class"), alt.Tooltip("items:Q", title="SKUs", format=",.0f")])
    text = base.mark_text(dy=-8, fontSize=12, fontWeight="bold", color=INK).encode(y="items:Q", text=alt.Text("items:Q", format=",.0f"))
    return _finish(alt.layer(bars, text), height)


def adi_cv2_scatter(df, height=400):
    """ADI × CV² of the SKUs coloured by ABC-XYZ segment, with the Syntetos-Boylan cut-offs (1.32 and 0.49)."""
    ymax, xmax = float(max(df["cv2"].max(), 0.6)), float(max(df["adi"].max(), 1.5))
    pts = alt.Chart(df).mark_circle(size=80, opacity=0.85, stroke="white", strokeWidth=0.8).encode(
        x=alt.X("adi:Q", title="ADI  ·  average weeks between sales", axis=alt.Axis(grid=True, gridColor=GRID, gridOpacity=0.25, gridDash=[2, 3])),
        y=alt.Y("cv2:Q", title="CV²  ·  variability of sales size", axis=alt.Axis(grid=False)),
        color=alt.Color("abc_xyz:N", scale=alt.Scale(domain=SEG_ORDER, range=[SEG_COLORS[s] for s in SEG_ORDER]),
                        legend=alt.Legend(title="Segment", orient="right")),
        tooltip=["item_id", alt.Tooltip("abc_xyz:N", title="Segment"), alt.Tooltip("demand_class:N", title="Class"),
                 alt.Tooltip("adi:Q", format=".2f"), alt.Tooltip("cv2:Q", format=".2f")])
    vx = alt.Chart(pd.DataFrame({"x": [1.32]})).mark_rule(color=GRID, strokeDash=[5, 4]).encode(x="x:Q")
    hy = alt.Chart(pd.DataFrame({"y": [0.49]})).mark_rule(color=GRID, strokeDash=[5, 4]).encode(y="y:Q")
    quad = pd.DataFrame({"x": [1.0, 1.0, 1.4, 1.4], "y": [0.02, ymax * 0.97, 0.02, ymax * 0.97],
                         "t": ["SMOOTH", "ERRATIC", "INTERMITTENT", "LUMPY"]})
    labels = alt.Chart(quad).mark_text(align="left", baseline="middle", fontSize=14, fontWeight="bold", opacity=0.35, color=INK).encode(
        x="x:Q", y="y:Q", text="t:N")
    return _finish(alt.layer(labels, pts, vx, hy), height)


def inventory_chart(wk, safety_stock, height=380):
    """Projected inventory (forecast demand and High-case demand), planned orders as bars and the safety stock line."""
    long = pd.concat([
        wk[["week_start", "inventory_forecast"]].rename(columns={"inventory_forecast": "value"}).assign(series="Inventory · forecast demand"),
        wk[["week_start", "inventory_high"]].rename(columns={"inventory_high": "value"}).assign(series="Inventory · High-case demand")])
    names = ["Inventory · forecast demand", "Inventory · High-case demand"]
    color = alt.Color("series:N", scale=alt.Scale(domain=names, range=["#00897B", "#D64545"]),
                      legend=alt.Legend(title=None, orient="top", symbolSize=220, labelFontSize=12))
    x = _x_time()
    y = alt.Y("value:Q", title=measure, scale=alt.Scale(zero=True), axis=alt.Axis(grid=False, format=AXIS_FMT))
    orders = wk[wk["order"] > 0]
    bar = alt.Chart(orders).mark_bar(opacity=0.45, color="#7C4DFF", size=18).encode(
        x=x, y=alt.Y("order:Q", title="Units"), tooltip=[alt.Tooltip("week_start:T", title="Week of", format="%Y-%m-%d"),
                                                        alt.Tooltip("order:Q", title="Order", format=",.0f")])
    bar_t = alt.Chart(_lab(orders, "order")).mark_text(dy=-8, fontSize=11, fontWeight="bold", color="#7C4DFF").encode(
        x=x, y=alt.Y("order:Q"), text="label:N")
    lines = alt.Chart(long).mark_line(strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=70)).encode(
        x=x, y=y, color=color, tooltip=_tooltip())
    lab = alt.Chart(_lab(long[long["series"] == names[0]])).mark_text(dy=-13, fontSize=11, fontWeight="bold", color="#00897B").encode(
        x=x, y=y, text="label:N")
    ss = alt.Chart(pd.DataFrame({"y": [safety_stock], "t": [f"Safety stock {safety_stock:,.0f}"]}))
    rule = ss.mark_rule(color="#F2B01E", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
    rule_t = ss.mark_text(align="left", dx=4, dy=-6, fontSize=11, fontWeight="bold", color="#B8860B").encode(
        x=alt.value(0), y="y:Q", text="t:N")
    return _finish(alt.layer(bar, bar_t, lines, lab, rule, rule_t), height)


def period_chart(periods, height=420, px=90, prev_name="Compared plan"):
    """Forecast Summary line chart: Actuals (blue), Forecast (orange), Consensus (green) and an optional compared plan (purple, dotted).

    Wide spacing (`px` pixels per period) and a y-axis that does not start at zero keep the lines readable; the app shows
    the chart in a horizontally scrollable frame. Numbers sit above the lines, consensus numbers below.
    """
    df = pd.DataFrame(periods)
    df["xlabel"] = [p["label"] + (f" · {p['sub']}" if p["sub"] else "") + ("*" if p["partial"] else "") for p in periods]
    order = list(df["xlabel"])
    has_cons = bool(df["consensus"].notna().any())
    has_prev = bool(df["prev"].notna().any())
    act = df[df["kind"] == "actual"][["xlabel", "actual"]].rename(columns={"actual": "value"}).assign(series="Actuals")
    fc = df[df["kind"] != "actual"][["xlabel", "baseline_plan"]].rename(columns={"baseline_plan": "value"}).assign(series="Forecast")
    if len(act) and len(fc):                                   # join the lines so the forecast starts where the actuals end
        fc = pd.concat([act.tail(1).assign(series="Forecast"), fc], ignore_index=True)
    names, colors = ["Actuals", "Forecast"], [ACTUAL, FORECAST]
    series = [act, fc]
    if has_cons:
        series.append(df[df["kind"] != "actual"][["xlabel", "plan"]].rename(columns={"plan": "value"}).assign(series="Consensus"))
        names.append("Consensus"); colors.append(CONSENSUS)
    if has_prev:
        series.append(df[["xlabel", "prev"]].dropna().rename(columns={"prev": "value"}).assign(series=prev_name))
        names.append(prev_name); colors.append("#7B4FD0")
    long = pd.concat(series, ignore_index=True)
    x = alt.X("xlabel:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.5, paddingOuter=0.3),
              axis=alt.Axis(labelAngle=-40, labelFontSize=11, labelLimit=140, grid=True, gridColor=GRID, gridOpacity=0.25, gridDash=[2, 3]))
    y = alt.Y("value:Q", title="Units", scale=alt.Scale(zero=False, nice=True), axis=alt.Axis(grid=False, domainColor=GRID, tickColor=GRID, format=AXIS_FMT))
    color = alt.Color("series:N", scale=alt.Scale(domain=names, range=colors), legend=alt.Legend(title=None, orient="top", symbolSize=220, labelFontSize=13, symbolStrokeWidth=4))
    cur = df[df["is_current"]]
    layers = [alt.Chart(cur[["xlabel"]]).mark_rect(color=SHADE, opacity=1).encode(x=alt.X("xlabel:N", sort=order))]
    tip = [alt.Tooltip("xlabel:N", title="Period"), alt.Tooltip("series:N", title="Series"), alt.Tooltip("value:Q", title="Units", format=",.0f")]
    layers.append(alt.Chart(long[long["series"] != prev_name]).mark_line(strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=80)).encode(x=x, y=y, color=color, tooltip=tip))
    if has_prev:
        layers.append(alt.Chart(long[long["series"] == prev_name]).mark_line(strokeWidth=2.5, strokeDash=[3, 4], point=alt.OverlayMarkDef(filled=True, size=60)).encode(
            x=x, y=y, color=color, tooltip=tip))
    lab = long.assign(label=[fmt_label(v) for v in long["value"]])
    if len(fc) and len(act):
        lab = lab[~((lab["series"] == "Forecast") & (lab.index == len(act)))]       # the joining point is already labelled as an actual
    main = lab[lab["series"].isin(["Actuals", "Forecast"])]
    layers.append(alt.Chart(main).mark_text(dy=-14, fontSize=11, fontWeight="bold").encode(x=x, y=y, color=color, text="label:N"))
    if has_cons:
        layers.append(alt.Chart(lab[lab["series"] == "Consensus"]).mark_text(dy=18, fontSize=11, fontWeight="bold").encode(x=x, y=y, color=color, text="label:N"))
    if has_prev:
        layers.append(alt.Chart(lab[lab["series"] == prev_name]).mark_text(dy=18, fontSize=10, fontWeight="bold").encode(x=x, y=y, color=color, text="label:N"))
    layers.append(alt.Chart(cur.assign(t="▼ CURRENT", y=float(long["value"].max()))).mark_text(dy=-6, baseline="bottom", fontSize=11, fontWeight="bold", color=FORECAST).encode(
        x=x, y=alt.Y("y:Q", title=None, scale=alt.Scale(zero=False)), text="t:N"))
    return alt.layer(*layers).properties(height=height, width=max(len(df) * px, 640)).configure_view(stroke=None).configure_axis(labelFontSize=12, titleFontSize=12)


def acc_bars(df, x, value, color=None, color_map=None, title_y="", target=None, target_label="", signed=False, height=300):
    """Bars of a ratio (accuracy, coverage, bias) with the percentage on every bar and an optional target line."""
    df = df.assign(label=[(f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%") for v in df[value]])
    order = list(df[x])
    base = alt.Chart(df).encode(x=alt.X(f"{x}:N", sort=order, title=None, axis=alt.Axis(labelAngle=-25, labelFontSize=12, labelLimit=140)))
    enc = alt.Color(f"{color}:N", scale=alt.Scale(domain=list(color_map), range=list(color_map.values())), legend=None) if color else alt.value(ACTUAL)
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y(f"{value}:Q", title=title_y, axis=alt.Axis(grid=False, format=".0%")), color=enc,
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{value}:Q", format=".1%")])
    text = base.mark_text(dy=-8, fontSize=12, fontWeight="bold", color=INK).encode(y=alt.Y(f"{value}:Q"), text="label:N")
    layers = [bars, text]
    if target is not None:
        t = alt.Chart(pd.DataFrame({"y": [target], "t": [target_label]}))
        layers.append(t.mark_rule(color="#D64545", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q"))
        layers.append(t.mark_text(align="left", baseline="bottom", dx=4, dy=-3, fontSize=11, fontWeight="bold", color="#D64545").encode(x=alt.value(4), y="y:Q", text="t:N"))
    return _finish(alt.layer(*layers), height)


def backtest_chart(g, height=340):
    """Past forecast (orange) against what really happened (blue), with the Low-High band and numbers on the points."""
    g = g.sort_values("week_start").copy()
    long = pd.concat([g[["week_start", "y_true"]].rename(columns={"y_true": "value"}).assign(series="Actuals"),
                      g[["week_start", "mean"]].rename(columns={"mean": "value"}).assign(series="Forecast")])
    x, y = _x_time(), _y()
    color = alt.Color("series:N", scale=alt.Scale(domain=["Actuals", "Forecast"], range=[ACTUAL, FORECAST]),
                      legend=alt.Legend(title=None, orient="top", symbolSize=220, labelFontSize=13))
    band = alt.Chart(g).mark_area(opacity=0.2, color=FORECAST).encode(x=x, y=alt.Y("p10:Q", title="Units per week"), y2="p90:Q")
    line = alt.Chart(long).mark_line(strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=70)).encode(x=x, y=y, color=color, tooltip=_tooltip())
    lab = alt.Chart(_lab(long)).mark_text(dy=-13, fontSize=11, fontWeight="bold").encode(x=x, y=y, color=color, text="label:N")
    return _finish(alt.layer(band, line, lab), height)
