"""Demand Planning Workbench (mini APS) on a model served from Databricks.

The app contains no model: every forecast comes from an HTTP request to the Databricks Model Serving endpoint
(client.py). Historical data and the accuracy / segmentation tables are exports of the Gold tables.
"""
import json
import os
import time
from pathlib import Path

import altair as alt  # noqa: F401  (charts.py builds the Altair charts)
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import aps_core as aps
import charts
import client
import data_access
import excel_export
import forecast_utils as fu
import abcxyz as ab
import governance as gv
import scenarios as scn
import scope as sc
import ui

st.set_page_config(page_title="Demand Planning Workbench", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
COOLDOWN_SECONDS = 2
HORIZON = aps.HORIZON
LEVEL_EN = {0: "Total", 1: "State", 2: "Store", 3: "Category", 4: "Department", 5: "Store × Category", 6: "Store × Department", 7: "SKU"}
VIEWS = {"Category": 3, "Department": 4, "SKU": 7}
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ----------------------------------------------------------------------------- configuration and data
def get_config(key, default=""):
    try:
        value = st.secrets[key]
    except Exception:
        value = os.environ.get(key, default)
    return str(value).strip()


HOST, TOKEN = get_config("DATABRICKS_HOST"), get_config("DATABRICKS_TOKEN")
ENDPOINT = get_config("ENDPOINT_NAME", "m5-aps-forecast")
CONFIGURED = bool(HOST and TOKEN)


@st.cache_data(show_spinner="Loading data...")
def load():
    return data_access.load_all()


try:
    D = load()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

LAST_WEEK = D["last_week"]
CURRENT_WEEK = LAST_WEEK + 1                       # the week being planned: first forecast week
CURRENT_WEEK_START = aps.week_start(CURRENT_WEEK)
HIER_META = D["hier_meta"]
LEVEL_MAP = dict(zip(HIER_META["series_id"], HIER_META["level"]))
TOTAL_ID = HIER_META.loc[HIER_META["level"] == 0, "series_id"].iloc[0]
ITEMS = D["items_meta"]
CATS = sorted(HIER_META.loc[HIER_META["level"] == 3, "series_id"])
STORES = sorted(HIER_META.loc[HIER_META["level"] == 2, "series_id"])
ALL_DEPTS = sorted(HIER_META.loc[HIER_META["level"] == 4, "series_id"])
DEPTS_BY_CAT = {c: [d for d in ALL_DEPTS if d.startswith(c + "_")] for c in CATS}
fmt0 = lambda v: f"{v:,.0f}"
LAST_ACTUAL_DAY = D["manifest"].get("last_data_date", "")
AUDIT_FILE = Path(__file__).parent / "audit_log.csv"

# session state shared by every tab
ss = st.session_state
ss.setdefault("committed", {})      # {scope label: {week_idx: {final, sales, mkt, reason, comment, user, ts}}}
ss.setdefault("versions", [])       # frozen plan versions
ss.setdefault("upload", None)       # validated import (DataFrame)
ss.setdefault("upload_price", {})
if "audit" not in ss:
    try:
        ss["audit"] = pd.read_csv(AUDIT_FILE).to_dict("records") if AUDIT_FILE.exists() else []
    except Exception:
        ss["audit"] = []

# average prices (the dataset has no price per series): SKU price = 52-week revenue / units; categories and departments use the SKU sample
_rev = ITEMS.groupby("cat_id")[["revenue_52w", "units_52w"]].sum()
CAT_PRICE = (_rev["revenue_52w"] / _rev["units_52w"].clip(lower=1e-9)).to_dict()
_rev = ITEMS.groupby("dept_id")[["revenue_52w", "units_52w"]].sum()
DEPT_PRICE = (_rev["revenue_52w"] / _rev["units_52w"].clip(lower=1e-9)).to_dict()
ITEM_PRICE = (ITEMS.set_index("item_id")["revenue_52w"] / ITEMS.set_index("item_id")["units_52w"].clip(lower=1e-9)).to_dict()
ALL_PRICE = float(ITEMS["revenue_52w"].sum() / max(ITEMS["units_52w"].sum(), 1e-9))


def item_price(sid):
    return ss["upload_price"].get(sid, ITEM_PRICE.get(sid, ALL_PRICE))


def price_of(level, sid):
    if level == 7:
        return item_price(sid)
    if level in (3, 5):
        return CAT_PRICE.get(sid.split("|")[-1], ALL_PRICE)
    if level in (4, 6):
        return DEPT_PRICE.get(sid.split("|")[-1], ALL_PRICE)
    return ALL_PRICE


def log_audit(rows):
    if not rows:
        return
    ss["audit"] = ss["audit"] + rows
    try:
        pd.DataFrame(rows).to_csv(AUDIT_FILE, mode="a", header=not AUDIT_FILE.exists(), index=False)
    except Exception:
        pass            # the log still lives in the session


# ----------------------------------------------------------------------------- endpoint access
def call_endpoint(rows):
    """Query with debounce and clear messages. Returns (forecasts, info) or None on error."""
    now = time.time()
    if now - st.session_state.get("last_call", 0) < COOLDOWN_SECONDS:
        st.warning("Please wait a couple of seconds before requesting again.")
        return None
    st.session_state["last_call"] = now
    try:
        with st.spinner("Running the statistical forecast on Databricks. If the model was idle, the first run can take 1–2 minutes..."):
            got = client.query_many(rows, HOST, TOKEN, ENDPOINT)
        st.session_state.pop("run_error", None)
        return got
    except client.ApiError as err:
        detail = err.detail + (f" (HTTP {err.status})" if err.status else "")
        st.session_state["run_error"] = (isinstance(err, client.Unavailable), f"{err.user_message}\n\nTechnical detail: {detail}")
    except ValueError as err:
        st.session_state["run_error"] = (False, f"The request could not be built: {err}")
    return None


def run_forecast():
    """One planning run: hierarchy (124 series) + the 302 SKUs, 13 weeks ahead. Hierarchy is reconciled."""
    h_rows, skipped = fu.build_requests(HIER_META, D["hier_by_series"], LAST_WEEK, HORIZON)
    i_rows = []
    for sid in ITEMS["item_id"]:
        try:
            i_rows.append(fu.build_request(sid, 7, D["items_by_series"][sid], LAST_WEEK, HORIZON))
        except (ValueError, KeyError) as exc:
            skipped.append((sid, str(exc)))
    got = call_endpoint(h_rows + i_rows)
    if not got:
        return False
    forecasts, info = got
    allfc = fu.forecast_frame(forecasts, {**LEVEL_MAP, **{r["series_id"]: 7 for r in i_rows}})
    hier_fc = allfc[allfc["level"] < 7].reset_index(drop=True)
    note = ""
    try:
        hier_fc = fu.reconcile_bulk(hier_fc, HIER_META)
    except ValueError as err:
        note = f"Reconciliation skipped: {err}"
    st.session_state["run"] = {"hier": hier_fc, "items": allfc[allfc["level"] == 7].reset_index(drop=True), "info": info, "week": LAST_WEEK,
                               "when": time.time(), "note": note, "skipped": skipped, "model_version": str(allfc["model_version"].iloc[0])}
    return True


def stat_forecast(level, sid):
    """Statistical forecast of a series (columns mean, p10, p50, p90; reconciled for hierarchy levels) or None."""
    run = st.session_state.get("run")
    if not run or run["week"] != LAST_WEEK:
        return None
    src = run["items"] if level == 7 else run["hier"]
    g = src[src["series_id"] == sid].sort_values("h").copy()
    if g.empty:
        return None
    if level < 7:
        for c in ("mean", "p10", "p50", "p90"):
            if c + "_rec" in g:
                g[c] = g[c + "_rec"]
    return g.reset_index(drop=True)


def actual_series(level, sid):
    h = D["items_by_series"][sid] if level == 7 else D["hier_by_series"][sid]
    return h.set_index("week_idx")["units"]


def hist_for(level, sid):
    return fu.history_frame(D["items_by_series"][sid] if level == 7 else D["hier_by_series"][sid], LAST_WEEK)


def series_options(level):
    if level == 7:
        meta = ITEMS.sort_values("revenue_52w", ascending=False)
        info = meta.set_index("item_id")
        return meta["item_id"].tolist(), (lambda s: f"{s}   ·   {info.loc[s, 'abc_xyz']}   ·   {info.loc[s, 'demand_class']}")
    return HIER_META.loc[HIER_META["level"] == level, "series_id"].tolist(), str


def seg_style(v):
    c = charts.SEG_COLORS.get(v)
    return f"background-color:{c};color:{charts.SEG_TEXT[v]};font-weight:800;text-align:center" if c else ""


CLASS_TINT = {"Smooth": "#DDF3E7", "Erratic": "#FFF3D1", "Intermittent": "#FFE5CC", "Lumpy": "#FADBD8", "No demand": "#E8ECF0"}
class_style = lambda v: f"background-color:{CLASS_TINT[v]};font-weight:600" if v in CLASS_TINT else ""


def show_scrollable(chart, height):
    """Shows a wide chart in a frame with a horizontal scroll bar, so long timelines are never squeezed."""
    components.html(chart.to_html(embed_options={"actions": False}), height=height + 60, scrolling=True)


# ----------------------------------------------------------------------------- o9-style filter toolbar
def _set_all(keys, value):
    for k in keys:
        st.session_state[k] = value


def filter_toolbar(prefix):
    """Category / Store / Department / SKU filters as drop-downs with check boxes. Returns the resolved scope or None."""
    ss = st.session_state
    ck = lambda kind, v: f"{prefix}_{kind}_{v}"
    for c in CATS:
        ss.setdefault(ck("cat", c), True)
    for s in STORES:
        ss.setdefault(ck("st", s), True)
    for d in ALL_DEPTS:
        ss.setdefault(ck("dp", d), True)
    cats = [c for c in CATS if ss[ck("cat", c)]]
    stores = [s for s in STORES if ss[ck("st", s)]]
    visible = [d for c in cats for d in DEPTS_BY_CAT[c]]
    depts = [d for d in visible if ss[ck("dp", d)]]
    sku_pool = ITEMS[ITEMS["cat_id"].isin(cats) & ITEMS["dept_id"].isin(depts)].sort_values("revenue_52w", ascending=False)["item_id"].tolist()
    sku_key = ck("sku", "sel")
    ss[sku_key] = [s for s in ss.get(sku_key, []) if s in sku_pool]

    def pop(col, label):
        return col.popover(label, use_container_width=True) if hasattr(st, "popover") else col.expander(label)

    def all_none(kind, keys):
        b1, b2 = st.columns(2)
        b1.button("Select all", key=ck(kind, "_all"), on_click=_set_all, args=(keys, True), use_container_width=True)
        b2.button("Clear", key=ck(kind, "_none"), on_click=_set_all, args=(keys, False), use_container_width=True)

    cols = st.columns([1, 1, 1, 1, 1.2])
    with pop(cols[0], f"Category · {len(cats)}/{len(CATS)}  ▾"):
        all_none("cat", [ck("cat", c) for c in CATS])
        for c in CATS:
            st.checkbox(c, key=ck("cat", c))
    with pop(cols[1], f"Store · {len(stores)}/{len(STORES)}  ▾"):
        all_none("st", [ck("st", s) for s in STORES])
        for state in sorted({s.split("_")[0] for s in STORES}):
            st.caption(f"State {state}")
            for s in [x for x in STORES if x.startswith(state + "_")]:
                st.checkbox(s, key=ck("st", s))
    with pop(cols[2], f"Department · {len(depts)}/{len(visible)}  ▾"):
        all_none("dp", [ck("dp", d) for d in visible])
        for c in cats:
            st.caption(c)
            for d in DEPTS_BY_CAT[c]:
                st.checkbox(d, key=ck("dp", d))
    with pop(cols[3], f"SKU · {len(ss[sku_key]) or 'all'}  ▾"):
        st.caption("SKU-level data covers store CA_1 only. Picking SKUs overrides the Store filter.")
        st.multiselect("SKUs (type to search)", sku_pool, key=sku_key)
    skus = ss[sku_key]
    series = sc.resolve_scope(cats, stores, depts, skus, CATS, STORES, DEPTS_BY_CAT, TOTAL_ID)
    label = sc.scope_label(cats, stores, depts, skus, CATS, STORES, DEPTS_BY_CAT)
    ui.chips([("Category", "all" if len(cats) == len(CATS) else ", ".join(cats) or "none", ui.BLUE),
              ("Store", "CA_1 (SKU data)" if skus else ("all" if len(stores) == len(STORES) else f"{len(stores)} selected"), ui.PURPLE),
              ("Department", "all" if len(depts) == len(visible) else f"{len(depts)} selected", ui.TEAL),
              ("SKU", f"{len(skus)} selected" if skus else "all", ui.ORANGE)])
    if not series:
        st.warning("Select at least one category, one store and one department (or pick SKUs).")
        return None
    return {"series": series, "label": label, "cats": cats, "stores": stores, "depts": depts, "skus": skus, "pool": sku_pool}


def scope_frame_for(series, money=False):
    if money:
        act = lambda l, sid: actual_series(l, sid) * price_of(l, sid)

        def fc(l, sid):
            f = stat_forecast(l, sid)
            if f is None:
                return None
            f = f.copy()
            for c in ("mean", "p10", "p50", "p90"):
                f[c] = f[c] * price_of(l, sid)
            return f
        return sc.weekly_scope_frame(series, act, fc, LAST_WEEK)
    return sc.weekly_scope_frame(series, actual_series, stat_forecast, LAST_WEEK)


def scope_frame(sel, money=False):
    try:
        return scope_frame_for(sel["series"], money)
    except ValueError as err:
        st.error(str(err))
        return None


# ----------------------------------------------------------------------------- header and sidebar
ui.hero("📦 Demand Planning Workbench", "Statistical forecasting · KPIs & ABC-XYZ · Consensus, versions & audit",
        ["Walmart M5 retail data", f"Actuals through {LAST_ACTUAL_DAY}", f"Current week: W{CURRENT_WEEK} · {CURRENT_WEEK_START:%Y-%m-%d}", "13-week horizon"])

with st.sidebar:
    st.markdown("### Planning run")
    if CONFIGURED and not st.session_state.get("auto_ran"):
        st.session_state["auto_ran"] = True
        run_forecast()
    if CONFIGURED and st.button("↻  Re-run statistical forecast", type="primary", use_container_width=True):
        run_forecast()
    run = st.session_state.get("run")
    if run:
        ui.notice(f"<b>Forecast ready</b><br>{run['info']['series']} series in {run['info']['seconds']:.1f} s<br>Model {run['model_version']}", ui.GREEN)
        if run["note"]:
            st.warning(run["note"])
    elif CONFIGURED:
        ui.notice("<b>Forecast not available yet.</b><br>Press the button to run it.", ui.AMBER)
    else:
        ui.notice("<b>Model connection not configured.</b><br>Add DATABRICKS_HOST and DATABRICKS_TOKEN to the app secrets.", ui.RED)
    err = st.session_state.get("run_error")
    if err:
        (st.warning if err[0] else st.error)(err[1])
    ss["user"] = st.text_input("Planner name (for the audit log)", value=ss.get("user", "planner"), max_chars=30, key="user_name").strip() or "planner"
    st.caption(f"Endpoint: `{ENDPOINT}`  ·  aps_core v{aps.VERSION}")
    st.markdown("---")
    st.markdown("**Legend**")
    st.markdown(f"{ui.badge_html('Actuals', charts.ACTUAL)} what really sold<br>{ui.badge_html('Forecast', charts.FORECAST)} what the model expects<br>"
                f"{ui.badge_html('Consensus', charts.CONSENSUS)} forecast after your overrides", unsafe_allow_html=True)
    st.caption("W = week number of the dataset (W1 = 2011-01-29). Weeks run Saturday to Friday.")

tab_ov, tab_fc, tab_kpi, tab_con, tab_data, tab_about = st.tabs(["📊 Overview", "🎯 ABC-XYZ", "📐 KPIs", "🤝 Consensus & Scenarios", "🗂️ Data & Audit", "ℹ️ Acerca de"])


# ----------------------------------------------------------------------------- Overview = Forecast Summary
def scope_cols(sel):
    """Category / Store / Department / SKU values of the current filters, as (short text, full text) pairs."""
    if sel["skus"]:
        meta = ITEMS[ITEMS["item_id"].isin(sel["skus"])]
        cats, stores, depts, skus = sorted(meta["cat_id"].unique()), ["CA_1"], sorted(meta["dept_id"].unique()), list(sel["skus"])
    else:
        cats, stores, depts, skus = sel["cats"], sel["stores"], sel["depts"], []
    all_depts = [d for c_ in sel["cats"] for d in DEPTS_BY_CAT[c_]]

    def one(vals, is_all, noun, limit=3):
        if is_all:
            return ("All", "All")
        full = ", ".join(vals)
        return (full if len(vals) <= limit else f"{len(vals)} {noun}", full)
    return [one(cats, not sel["skus"] and set(cats) == set(CATS), "categories"),
            one(stores, not sel["skus"] and set(stores) == set(STORES), "stores"),
            one(depts, not sel["skus"] and set(depts) == set(all_depts) and set(sel["cats"]) == set(CATS), "departments"),
            one(skus, not skus, "SKUs")]


def version_names(label):
    return [v["name"] for v in ss["versions"] if v["scope"] == label]


def version_weeks(label, name):
    return next(v["weeks"] for v in ss["versions"] if v["scope"] == label and v["name"] == name)


def overlay(frame, frame_u, label, plan_choice, cmp_choice, money):
    """Adds `consensus` (committed overrides or a frozen version) and `prev` (compared version) columns, converted to the shown measure."""
    ratio = (frame["forecast"] / frame_u["forecast"]).where(frame_u["forecast"] > 0, 1.0).fillna(1.0) if money else 1.0
    out = frame.copy()
    out["consensus"], out["prev"] = np.nan, np.nan
    if plan_choice == "Current plan":
        com = ss["committed"].get(label, {})
        if com:
            out["consensus"] = frame["week_idx"].map({w: e["final"] for w, e in com.items()}) * ratio
    else:
        out["consensus"] = frame["week_idx"].map(version_weeks(label, plan_choice)) * ratio
    if cmp_choice != "None":
        out["prev"] = frame["week_idx"].map(version_weeks(label, cmp_choice)) * ratio
    return out


def breakdown(sel, dim, money, unit, back, fwd):
    """Roll-up / drill-down: baseline plan (actuals + statistical forecast) of each member of the chosen dimension."""
    if dim == "Category":
        members = [(c, dict(cats=[c], stores=sel["stores"], depts=[d for d in sel["depts"] if d in DEPTS_BY_CAT[c]], skus=[])) for c in sel["cats"]]
    elif dim == "Department":
        members = [(d, dict(cats=[d.rsplit("_", 1)[0]], stores=sel["stores"], depts=[d], skus=[])) for d in sel["depts"]]
    elif dim == "Store":
        members = [(s_, dict(cats=sel["cats"], stores=[s_], depts=sel["depts"], skus=[])) for s_ in sel["stores"]]
    else:
        members = [(k, dict(cats=sel["cats"], stores=sel["stores"], depts=sel["depts"], skus=[k])) for k in (sel["skus"] or sel["pool"][:15])]
    rows, periods_ref = [], None
    for name, kw in members:
        series = sc.resolve_scope(kw["cats"], kw["stores"], kw["depts"], kw["skus"], CATS, STORES, DEPTS_BY_CAT, TOTAL_ID)
        if not series:
            continue
        try:
            fr = scope_frame_for(series, money)
        except ValueError:
            continue
        per, _ = sc.build_periods(fr, unit, LAST_WEEK, back, fwd)
        periods_ref = periods_ref or per
        tot, ly = sum(p["baseline_plan"] for p in per), sum(p["ly"] for p in per if p["ly"] == p["ly"])
        rows.append((name, [p["baseline_plan"] for p in per], (tot / ly - 1) if ly > 0 else float("nan"), False))
    return rows, periods_ref


def render_overview():
    ui.section("Forecast Summary", "Filter the scope with the drop-downs, then choose how to group time")
    sel = filter_toolbar("ov")
    if sel is None:
        return
    if not ss.get("run"):
        st.info("The forecast is not available yet. Press **Re-run statistical forecast** in the sidebar.")
        return
    label = sel["label"]
    frame_u = scope_frame(sel)
    if frame_u is None:
        return

    ui.section("View options", "Time buckets, plan version and comparison")
    c = st.columns([1.4, 1, 1])
    money = False
    unit = c[0].radio("Group by", list(sc.UNITS), horizontal=True, key="ov_unit")
    frame = frame_u
    _, info = sc.build_periods(frame, unit, LAST_WEEK, 999, 999)
    dflt = {"Weeks": (8, 12), "Months": (6, 3), "Quarters": (4, 1)}[unit]
    back = c[1].number_input(f"{unit} back", min_value=0, max_value=max(info["max_back"], 0), value=min(dflt[0], info["max_back"]), step=1, key=f"ov_back_{unit}")
    fwd = c[2].number_input(f"{unit} forward", min_value=0, max_value=max(info["max_fwd"], 0), value=min(dflt[1], info["max_fwd"]), step=1, key=f"ov_fwd_{unit}")
    names = version_names(label)
    c = st.columns([1.3, 1.3, 2.4])
    plan_choice = c[0].selectbox("Plan shown", ["Current plan"] + names, key="ov_plan")
    cmp_choice = c[1].selectbox("Compare against", ["None"] + [n for n in names if n != plan_choice], key="ov_cmp",
                                help="Compare the plan shown with a frozen version (e.g. last cycle's official plan).")
    c[2].caption(f"Up to {info['max_back']} {unit.lower()} back and {info['max_fwd']} forward available. Current {unit[:-1].lower()} = the one containing "
                 f"W{CURRENT_WEEK} (week of {CURRENT_WEEK_START:%Y-%m-%d}). **Current plan** = statistical forecast plus the overrides you committed for this exact scope.")
    frame = overlay(frame, frame_u, label, plan_choice, cmp_choice, money)
    pre = ""
    fm = lambda v: f"{v:,.0f}"
    measure = "Units"

    fin = frame["consensus"].where(frame["consensus"].notna(), frame["forecast"])
    cur_v = float(fin[frame["week_idx"] == CURRENT_WEEK].iloc[0])
    fut = frame["week_idx"] > LAST_WEEK
    f13, ly13 = float(fin[fut].sum()), float(frame.loc[fut, "ly"].sum())
    past13 = (frame["week_idx"] > LAST_WEEK - 13) & (frame["week_idx"] <= LAST_WEEK)
    a13, aly13 = float(frame.loc[past13, "actual"].sum()), float(frame.loc[past13, "ly"].sum())
    k = st.columns(4)
    ui.kpi(k[0], f"Current week · W{CURRENT_WEEK}", fm(cur_v), f"units · week of {CURRENT_WEEK_START:%b %d}", ui.ORANGE, "#5B6B7F")
    d1 = (f13 / ly13 - 1) if ly13 > 0 else None
    ui.kpi(k[1], "Plan · next 13 weeks", fm(f13), f"{d1:+.1%} vs same weeks last year" if d1 is not None else "", ui.ORANGE, ui.delta_color(d1) if d1 is not None else "#5B6B7F")
    d2 = (a13 / aly13 - 1) if aly13 > 0 else None
    ui.kpi(k[2], "Actuals · last 13 weeks", fm(a13), f"{d2:+.1%} vs same weeks last year" if d2 is not None else "", ui.BLUE, ui.delta_color(d2) if d2 is not None else "#5B6B7F")
    ui.kpi(k[3], "Scope", label.split(" · ")[0] if len(label) > 28 else label, f"{len(sel['series'])} series combined", ui.PURPLE, "#5B6B7F")

    ui.section("Forecast Summary matrix", "Closed periods on the left of the dark line, open periods on the right")
    periods, _ = sc.build_periods(frame, unit, LAST_WEEK, int(back), int(fwd))
    st.markdown('<div class="lgd"><b style="background:#1F5FBF">Blue</b> = actuals &nbsp; <b style="background:#F26B21">Orange</b> = statistical forecast &nbsp; '
                '<b style="background:#2E9E5B">Green</b> = consensus &nbsp; <b style="background:#7B4FD0">Purple</b> = compared plan &nbsp; '
                '<b style="background:#0B2545">CURRENT</b> = period being planned &nbsp; * = forecast covers only part of the period</div>', unsafe_allow_html=True)
    dec = 1 if (not money and len(sel["series"]) == 1 and sel["series"][0][0] == 7) else 0
    st.markdown(ui.period_grid(periods, dec=dec, band=(unit == "Weeks"), band_exact=frame_u.attrs.get("exact_band", True), prefix=pre,
                               prev_name=f"Plan: {cmp_choice}", cons_name="Consensus forecast" if plan_choice == "Current plan" else f"Version: {plan_choice}",
                               scope_cols=scope_cols(sel)),
                unsafe_allow_html=True)
    if unit == "Weeks" and not frame_u.attrs.get("exact_band", True):
        st.caption("Low / High case for several series combined are approximate (they assume the series move independently).")
    if plan_choice == "Current plan" and not ss["committed"].get(label):
        st.caption("No consensus overrides are committed for this exact scope yet. Create them in the Consensus & Scenarios tab with the same filters.")

    ui.section("Demand plan chart", "Scroll sideways to see every period; hover for the numbers")
    show_scrollable(charts.period_chart(periods, prev_name=f"Plan: {cmp_choice}"), 420)
    st.caption("Blue line = actuals · orange line = statistical forecast · green line = consensus plan (numbers below) · purple dotted = compared plan. The axis does not start at zero so the movement is easier to see.")

    ui.section("Roll-up / drill-down", "The same periods broken down by Category, Department, Store or SKU (statistical baseline)")
    dims = ["SKU"] if sel["skus"] else ["Category", "Department", "Store", "SKU"]
    dim = st.radio("Break down by", dims, horizontal=True, key="ov_dim")
    if dim == "SKU" and not sel["skus"]:
        st.caption("Showing the 15 SKUs with most revenue within the selected categories and departments (store CA_1). Pick SKUs in the filter to choose others.")
    rows, per_ref = breakdown(sel, dim, money, unit, int(back), int(fwd))
    if rows and per_ref:
        tot_row = ("Scope total", [p["baseline_plan"] for p in sc.build_periods(scope_frame(sel, money), unit, LAST_WEEK, int(back), int(fwd))[0]],
                   float("nan"), True)
        t_ly = sum(p["ly"] for p in per_ref if p["ly"] == p["ly"])
        st.markdown(ui.breakdown_table(rows + [tot_row], per_ref, prefix=pre), unsafe_allow_html=True)
        st.caption("Each row is its own forecast, reconciled with the hierarchy, so members add up to the scope total (SKU rows are a sample and do not).")
    else:
        st.info("Nothing to break down with the current filters.")

    ui.section("Freeze a version", "Take a snapshot of the current plan to compare it later")
    f = st.columns([3, 1])
    vname = f[0].text_input("Version name", value="Official plan", max_chars=40, key="ov_vname")
    f[1].markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if f[1].button("🧊 Freeze version", use_container_width=True, key="btn_freeze"):
        com = ss["committed"].get(label, {})
        fu_ = frame_u[frame_u["week_idx"] > LAST_WEEK]
        weekly = {int(w): float(com[w]["final"]) if w in com else float(v) for w, v in zip(fu_["week_idx"], fu_["forecast"])}
        nm = vname.strip() or "Version"
        taken = version_names(label)
        if nm in taken:
            nm = f"{nm} ({len(taken) + 1})"
        ss["versions"] = ss["versions"] + [gv.make_version(nm, label, "Units", weekly, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), ss.get("user", "planner"))]
        st.success(f"Version “{nm}” frozen for this scope. Pick it in **Plan shown** or **Compare against**.")
        st.rerun()
    if names:
        st.caption("Frozen versions for this scope: " + ", ".join(names))



with tab_ov:
    render_overview()


# ----------------------------------------------------------------------------- KPIs
SEG_MEANING = {
    "AX": ("Top revenue · stable", "Automate replenishment; highest service level; review weekly."),
    "AY": ("Top revenue · moderate variability", "Frequent review; keep safety stock; use scenarios for promotions."),
    "AZ": ("Top revenue · erratic", "Plan with sales input; protect with safety stock; review every cycle."),
    "BX": ("Mid revenue · stable", "Automate with the standard service level."),
    "BY": ("Mid revenue · moderate variability", "Standard policy; monthly review."),
    "BZ": ("Mid revenue · erratic", "Moderate buffer; consider ordering against actual demand."),
    "CX": ("Low revenue · stable", "Automate with lean stock; low touch."),
    "CY": ("Low revenue · moderate variability", "Lean stock; periodic review."),
    "CZ": ("Low revenue · erratic", "Minimize stock; make-to-order or delisting candidate."),
}
BIAS_COLORS = {"Within ±5%": "#1E9E5A", "Between ±5% and ±10%": "#F2B01E", "Beyond ±10%": "#D64545"}


def _model_wape(row):
    w = row.get("wape_conciliada") if int(row["level"]) < 7 else None
    return row["wape_media"] if w is None or pd.isna(w) else w


def render_accuracy():
    al = D["acc_level"]
    if al is None or "wape_media" not in al:
        st.info("The accuracy tables were not found in the data folder (accuracy_level.csv).")
        return
    al = al.sort_values("level").reset_index(drop=True)
    al["Level"] = al["level"].map(LEVEL_EN)
    al["model_wape"] = [_model_wape(r) for _, r in al.iterrows()]
    al["accuracy"] = 1 - al["model_wape"]
    ui.section("Forecast accuracy", "How close the forecast was to what really happened in past tests (back-tests)")
    pick = st.selectbox("Hierarchy level", al["Level"].tolist(), index=min(3, len(al) - 1), key="kpi_level")
    r = al[al["Level"] == pick].iloc[0]
    acc = float(r["accuracy"])
    acol = ui.GREEN if acc > 0.85 else ui.AMBER if acc >= 0.70 else ui.RED
    trend = ""
    aw = D["acc_window"]
    if aw is not None and len(aw) >= 2:
        aw = aw.sort_values("window")
        w_acc = 1 - aw.apply(lambda x: x["wape_conciliada"] if pd.notna(x.get("wape_conciliada")) else x["wape_media"], axis=1).to_numpy()
        dlt = (w_acc[-1] - w_acc[-2]) * 100
        trend = f"{'▲' if dlt >= 0 else '▼'} {abs(dlt):.1f} pp vs previous cutoff"
    k = st.columns(4)
    ui.kpi(k[0], "Forecast accuracy", f"{acc:.1%}", ("1 − WMAPE · " + trend) if trend else "1 − WMAPE · higher is better", acol, acol)
    bias = float(r["sesgo_media"])
    bcol = ui.GREEN if abs(bias) <= 0.05 else ui.AMBER if abs(bias) <= 0.10 else ui.RED
    ui.kpi(k[1], "Forecast bias", f"{bias:+.1%}", "over-forecasting (+)" if bias > 0 else "under-forecasting (−)", bcol, bcol)
    fva = float(r["fva_vs_mejor_base"]) * 100 if pd.notna(r.get("fva_vs_mejor_base")) else float("nan")
    ui.kpi(k[2], "FVA · model vs naive", f"{fva:+.1f} pp", "WMAPE saved vs best simple method", ui.GREEN if fva > 0 else ui.RED, ui.delta_color(fva))
    cov = float(r["cobertura_calibrada"])
    ccol = ui.GREEN if abs(cov - 0.8) <= 0.05 else ui.AMBER
    ui.kpi(k[3], "Range coverage", f"{cov:.1%}", "weeks inside Low–High · target 80%", ccol, ccol)
    st.caption("Accuracy colour: green above 85%, amber 70–85%, red below 70%. WMAPE = Σ|actual − forecast| ÷ Σ actual. "
               "FVA of the overrides (WMAPE statistical − WMAPE consensus) appears once real sales exist for a committed override.")

    left, right = st.columns(2)
    rows = []
    for _, x in al.iterrows():
        rows += [{"Level": x["Level"], "Method": "Model", "Accuracy": x["accuracy"]}, {"Level": x["Level"], "Method": "Seasonal naive", "Accuracy": 1 - x["wape_estacional"]}]
    left.markdown("**Accuracy by hierarchy level** · model vs the same week last year")
    left.altair_chart(charts.group_bars(pd.DataFrame(rows), "Level", "Method", "Accuracy", {"Model": "#1E9E5A", "Seasonal naive": "#C5CCD6"}, title_y="Accuracy", pct=True), use_container_width=True)
    if D["acc_horizon"] is not None:
        h = D["acc_horizon"].copy()
        h["Horizon"] = h["h_bucket"].map({"1-4": "Weeks 1–4", "5-8": "Weeks 5–8", "9-13": "Weeks 9–13"}).fillna(h["h_bucket"])
        h["accuracy"] = 1 - h.apply(lambda x: x["wape_conciliada"] if pd.notna(x.get("wape_conciliada")) else x["wape_media"], axis=1)
        right.markdown("**Accuracy by forecast horizon**")
        right.altair_chart(charts.acc_bars(h, "Horizon", "accuracy", title_y="Accuracy"), use_container_width=True)

    with st.expander("More detail: bias and range coverage by level"):
        a, b = st.columns(2)
        bias_df = al[["Level", "sesgo_media"]].rename(columns={"sesgo_media": "bias"})
        bias_df["band"] = bias_df["bias"].map(lambda v: "Within ±5%" if abs(v) <= 0.05 else "Between ±5% and ±10%" if abs(v) <= 0.10 else "Beyond ±10%")
        a.markdown("**Bias by level** (positive = forecast above actuals)")
        a.altair_chart(charts.acc_bars(bias_df, "Level", "bias", color="band", color_map=BIAS_COLORS, title_y="Bias", signed=True), use_container_width=True)
        cov_df = al[["Level", "cobertura_calibrada"]].rename(columns={"cobertura_calibrada": "coverage"})
        b.markdown("**Range coverage by level**")
        b.altair_chart(charts.acc_bars(cov_df, "Level", "coverage", title_y="Coverage", target=0.8, target_label="Target 80%"), use_container_width=True)
    with st.expander("What do these KPIs mean?"):
        st.markdown("- **Forecast accuracy** = 1 − WMAPE. 90% means the forecast missed 10% of the units sold, in total.\n"
                    "- **Bias** = (total forecast − total actuals) ÷ total actuals. Positive: usually too high; negative: too low. Within ±5% is healthy.\n"
                    "- **FVA** = percentage points of WMAPE the model saves compared with the best simple method (naive, seasonal naive or 13-week average).\n"
                    "- **Range coverage** = share of weeks in which actual demand fell between the Low and High case. The target is 80%.")


def _seg_card(seg, items, share):
    bg, fg = charts.SEG_COLORS[seg], charts.SEG_TEXT[seg]
    return (f'<div style="background:{bg};color:{fg};border-radius:12px;padding:10px 6px;text-align:center;line-height:1.25">'
            f'<div style="font-size:1.5rem;font-weight:800">{seg}</div><div style="font-size:.95rem;font-weight:700">{items} SKUs</div>'
            f'<div style="font-size:.85rem">{share:.1%} of sales</div></div>')


def _pick_segment(seg):
    ss["abc_seg"] = None if ss.get("abc_seg") == seg else seg


def render_abc_tab():
    ui.section("ABC-XYZ Analysis", "Filter the SKUs, read the summary, pick a segment in the 9-box matrix and drill down")
    ui.notice("<b>ABC</b> = Pareto of revenue: <b>A</b> = the SKUs that make the first 80% of sales, <b>B</b> the next 15%, <b>C</b> the last 5%. "
              "<b>XYZ</b> = variability of weekly demand (CV): <b>X</b> &lt; 0.25, <b>Y</b> 0.25–0.50, <b>Z</b> &gt; 0.50. SKU data covers store CA_1, so the Store filter does not change this tab.", ui.BLUE)
    sel = filter_toolbar("abc")
    if sel is None:
        return
    win = st.selectbox("History window used for the classification", list(ab.WINDOWS), index=2, key="abc_win")
    units_by = {k_: actual_series(7, k_) for k_ in ITEMS["item_id"]}
    full = ab.classify(units_by, {k_: item_price(k_) for k_ in ITEMS["item_id"]}, LAST_WEEK, ab.WINDOWS[win], ITEMS)
    run = ss.get("run")
    fc13 = {}
    if run:
        for k_ in full["item_id"]:
            f = stat_forecast(7, k_)
            fc13[k_] = float(f["mean"].sum()) if f is not None else float("nan")
    full["fc13"] = full["item_id"].map(fc13) if fc13 else float("nan")
    keep = set(sel["skus"]) if sel["skus"] else set(sel["pool"])
    sub = full[full["item_id"].isin(keep)].copy()
    if sub.empty:
        st.info("No SKUs match the current filters.")
        return
    sub["share"] = sub["revenue"] / max(sub["revenue"].sum(), 1e-9)
    n = len(sub)
    st.caption(f"Segments are assigned across all {len(full)} SKUs of the sample and then filtered, so a SKU keeps its segment whatever you filter.")

    ui.section("ABC-XYZ summary", "The nine segments for the SKUs in the filters")
    k = st.columns(5)
    ui.kpi(k[0], "SKUs in scope", fmt0(n), "store CA_1 sample", ui.BLUE, "#5B6B7F")
    ui.kpi(k[1], "Class A SKUs", f"{int((sub['abc'] == 'A').sum())}", f"{sub.loc[sub['abc'] == 'A', 'share'].sum():.0%} of revenue in scope", ui.GREEN, "#5B6B7F")
    ax = int((sub["segment"] == "AX").sum())
    ui.kpi(k[2], "AX · stable & valuable", f"{ax / n:.1%}", f"{ax} SKUs · safe to automate", "#1E9E5A", "#5B6B7F")
    cz = sub[sub["segment"] == "CZ"]
    ui.kpi(k[3], "CZ · erratic & low value", f"{len(cz) / n:.1%}", f"{len(cz)} SKUs · {cz['share'].sum():.1%} of sales", ui.RED, "#5B6B7F")
    hard = int(sub["demand_class"].isin(["Intermittent", "Lumpy"]).sum())
    ui.kpi(k[4], "Hard to forecast", f"{hard / n:.1%}", "intermittent or lumpy demand", ui.AMBER, "#5B6B7F")
    seg_rows = {}
    for g_ in charts.SEG_ORDER:
        x = sub[sub["segment"] == g_]
        finite = x["cv"].replace([np.inf], np.nan)
        seg_rows[g_] = {"items": len(x), "share_items": len(x) / n, "share_rev": float(x["share"].sum()), "units": float(x["units"].sum()),
                        "cv": float(finite.mean()) if finite.notna().any() else float("nan"), "fc13": float(x["fc13"].sum()) if len(x) and x["fc13"].notna().any() else float("nan")}
    fin = sub["cv"].replace([np.inf], np.nan)
    totals = {"items": n, "share_items": 1.0, "share_rev": 1.0, "units": float(sub["units"].sum()), "cv": float(fin.mean()) if fin.notna().any() else float("nan"),
              "fc13": float(sub["fc13"].sum()) if sub["fc13"].notna().any() else float("nan")}
    st.markdown(ui.segment_summary_grid(seg_rows, scope_cols(sel), totals), unsafe_allow_html=True)

    ui.section("9-box matrix", "Click a segment to filter the drill-down below (click it again to clear)")
    g = ab.grid_summary(sub).set_index("segment")
    hc = st.columns([0.35, 1, 1, 1])
    for j, lbl in enumerate(["", "X · stable", "Y · variable", "Z · erratic"]):
        hc[j].markdown(f"<div style='text-align:center;font-weight:700;color:#3D4F66'>{lbl}</div>", unsafe_allow_html=True)
    for a, alabel in zip("ABC", ["A · top", "B · mid", "C · low"]):
        rc = st.columns([0.35, 1, 1, 1])
        rc[0].markdown(f"<div style='font-weight:700;color:#3D4F66;padding-top:1.4rem'>{alabel}</div>", unsafe_allow_html=True)
        for j, x in enumerate("XYZ"):
            seg = a + x
            rc[1 + j].markdown(_seg_card(seg, int(g.loc[seg, "items"]), float(sub.loc[sub["segment"] == seg, "share"].sum())), unsafe_allow_html=True)
            on = ss.get("abc_seg") == seg
            rc[1 + j].button(("✔ Filtering " if on else "Filter ") + seg, key=f"abc_btn_{seg}", on_click=_pick_segment, args=(seg,), use_container_width=True)

    ui.section("Roll-up / drill-down", "Break the SKUs down by Category, Department or SKU" + (f" · only segment {ss['abc_seg']}" if ss.get("abc_seg") else ""))
    dim = st.radio("Break down by", ["Category", "Department", "SKU"], horizontal=True, key="abc_dim")
    view = sub[sub["segment"] == ss["abc_seg"]] if ss.get("abc_seg") else sub
    if view.empty:
        st.info("No SKUs in that segment for the current filters.")
        return
    if dim in ("Category", "Department"):
        col = "cat_id" if dim == "Category" else "dept_id"
        rows = []
        for name, x in view.groupby(col):
            r = {dim: name}
            r.update({g_: int((x["segment"] == g_).sum()) for g_ in charts.SEG_ORDER})
            r["Total SKUs"], r["Revenue share"] = len(x), float(x["share"].sum())
            r["Forecast next 13 wks"] = float(x["fc13"].sum()) if x["fc13"].notna().any() else float("nan")
            rows.append(r)
        t = pd.DataFrame(rows)
        tot = {dim: "Total", **{g_: int(t[g_].sum()) for g_ in charts.SEG_ORDER}, "Total SKUs": int(t["Total SKUs"].sum()), "Revenue share": float(t["Revenue share"].sum()),
               "Forecast next 13 wks": float(t["Forecast next 13 wks"].sum())}
        t = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
        sty = t.style.format({"Revenue share": "{:.1%}", "Forecast next 13 wks": "{:,.0f}"})
        for g_ in charts.SEG_ORDER:
            sty = ui.style_map(sty, (lambda v, c_=charts.SEG_COLORS[g_]: f"background-color:{c_}66;font-weight:700" if v else "color:#B7C0CC"), [g_])
        st.dataframe(sty, hide_index=True, use_container_width=True)
        st.caption("Each cell = number of SKUs of that segment; colours follow the 9-box matrix (green = AX … red = CZ).")
    else:
        f = st.columns([2, 2])
        q = f[0].text_input("Search SKU", "", key="abc_search")
        sel_cls = f[1].multiselect("Demand class", sorted(view["demand_class"].dropna().unique()), key="abc_cls")
        if q.strip():
            view = view[view["item_id"].str.contains(q.strip(), case=False, regex=False)]
        if sel_cls:
            view = view[view["demand_class"].isin(sel_cls)]
        show = view[["item_id", "dept_id", "segment", "demand_class", "units", "revenue", "share", "cv", "fc13", "method"]].rename(columns={
            "item_id": "SKU", "dept_id": "Department", "segment": "Segment", "demand_class": "Demand class", "units": "Units (window)", "revenue": "Revenue (window, $)",
            "share": "Revenue share", "cv": "CV", "fc13": "Forecast 13 wks", "method": "Recommended method"})
        sty = show.style.format({"Units (window)": "{:,.0f}", "Revenue (window, $)": "${:,.0f}", "Revenue share": "{:.2%}", "CV": "{:.2f}", "Forecast 13 wks": "{:,.0f}"})
        sty = ui.style_map(ui.style_map(sty, seg_style, ["Segment"]), class_style, ["Demand class"])
        st.dataframe(sty, hide_index=True, use_container_width=True, height=420)
        st.download_button("⬇ Download SKU list (CSV)", show.round(2).to_csv(index=False).encode("utf-8"), file_name="abc_xyz_skus.csv", mime="text/csv")
        st.caption(f"{len(show)} SKUs shown.")
    with st.expander("Segment playbook: suggested approach and statistical method"):
        pb = pd.DataFrame([{"Segment": k_, "Profile": v[0], "Suggested approach": v[1], "Recommended statistical method": ab.METHODS[k_]} for k_, v in SEG_MEANING.items()])
        st.dataframe(ui.style_map(pb.style, seg_style, ["Segment"]), hide_index=True, use_container_width=True)
        st.caption("The method is a recommendation by segment (Z segments use Croston when demand is intermittent or lumpy). The deployed endpoint uses one global model for every series.")


def _demo_inventory():
    rng = np.random.default_rng(11)
    rows = []
    for k_ in ITEMS["item_id"]:
        f = stat_forecast(7, k_)
        if f is not None:
            rows.append({"SKU": k_, "on_hand": float(round(f["mean"].head(4).mean() * rng.uniform(0.3, 9.0)))})
    return pd.DataFrame(rows)


def _set_demo(v):
    ss["demo_inv"] = v


def render_inventory():
    ui.section("Inventory health", "Days on hand (DOH) and stock-out alerts: current stock against the projected demand")
    inv = None
    up = ss.get("upload")
    if up is not None and up["Inventory"].notna().any():
        last = up.dropna(subset=["Inventory"]).sort_values("Date").groupby("SKU").tail(1)
        inv = last[last["SKU"].isin(ITEMS["item_id"])][["SKU", "Inventory"]].rename(columns={"Inventory": "on_hand"})
        src = f"Inventory from your imported file (latest row per SKU, {len(inv)} SKUs)."
    elif ss.get("demo_inv"):
        inv = _demo_inventory()
        src = "ILLUSTRATIVE DEMO INVENTORY — random stock levels, not real data."
    if inv is None or inv.empty:
        ui.notice("<b>No inventory data yet.</b><br>Import a file with an <b>Inventory</b> column in the <b>Data & Audit</b> tab, or switch on a demo to see how the KPIs work.", ui.AMBER)
        st.button("Use illustrative demo inventory", key="btn_demo", on_click=_set_demo, args=(True,))
        return
    if ss.get("demo_inv") and not (up is not None and up["Inventory"].notna().any()):
        ui.notice("<b>Illustrative demo.</b> The stock levels below are random so you can see the screen working. Import your real inventory in <b>Data & Audit</b>.", ui.RED)
        st.button("Turn demo off", key="btn_demo_off", on_click=_set_demo, args=(False,))
    st.caption(src)
    c = st.columns(3)
    lead = c[0].number_input("Replenishment lead time (days)", min_value=1, max_value=120, value=14, key="inv_lead")
    over = c[1].number_input("Overstock above (days of cover)", min_value=15, max_value=365, value=90, key="inv_over")
    fam = c[2].multiselect("Department filter", sorted(ITEMS["dept_id"].unique()), key="inv_dept")
    rows = []
    meta = ITEMS.set_index("item_id")
    for _, r in inv.iterrows():
        f = stat_forecast(7, r["SKU"])
        if f is None:
            continue
        wk = f["mean"].to_numpy()
        cum = np.cumsum(wk)
        short = np.nonzero(cum > r["on_hand"])[0]
        doh = float(gv.days_on_hand([r["on_hand"]], [wk[:4].mean()])[0])
        rows.append({"SKU": r["SKU"], "Department": meta.loc[r["SKU"], "dept_id"], "Segment": meta.loc[r["SKU"], "abc_xyz"], "On hand": r["on_hand"],
                     "Forecast next 4 wks": wk[:4].sum(), "DOH": doh, "Stock-out week": f"W{int(f['target_week'].iloc[short[0]])}" if len(short) else "none in 13 wks",
                     "Short in 13 wks": max(cum[-1] - r["on_hand"], 0.0), "Revenue 52w": meta.loc[r["SKU"], "revenue_52w"]})
    t = pd.DataFrame(rows)
    if fam:
        t = t[t["Department"].isin(fam)]
    if t.empty:
        st.info("No SKUs to show.")
        return
    t["Status"] = np.where(t["DOH"] < lead, "At risk", np.where(t["DOH"] > over, "Overstock", "Healthy"))
    daily = max(t["Forecast next 4 wks"].sum() / 28.0, 1e-9)
    risk = t[t["Status"] == "At risk"]
    k = st.columns(5)
    cover = t["On hand"].sum() / daily
    ui.kpi(k[0], "Days on hand (overall)", f"{cover:,.0f}", "total stock ÷ daily forecast demand", ui.GREEN if cover >= lead else ui.RED, "#5B6B7F")
    ui.kpi(k[1], "SKUs at stock-out risk", f"{len(risk)}", f"DOH below the {lead}-day lead time", ui.RED if len(risk) else ui.GREEN, "#5B6B7F")
    ui.kpi(k[2], "Share of SKUs at risk", f"{len(risk) / len(t):.0%}", f"of {len(t)} SKUs", ui.AMBER, "#5B6B7F")
    ui.kpi(k[3], "Overstocked SKUs", f"{(t['Status'] == 'Overstock').sum()}", f"more than {over} days of cover", ui.PURPLE, "#5B6B7F")
    ui.kpi(k[4], "Units short in 13 weeks", fmt0(t["Short in 13 wks"].sum()), "demand above current stock", ui.ORANGE, "#5B6B7F")
    t = t.sort_values(["Status", "DOH"], key=lambda s_: s_.map({"At risk": 0, "Overstock": 1, "Healthy": 2}) if s_.name == "Status" else s_)
    status_c = {"At risk": "background-color:#FADBD8;color:#9B2D2D;font-weight:800", "Overstock": "background-color:#E9DDF9;color:#5B3FA0;font-weight:800", "Healthy": "background-color:#DDF3E7;color:#14663A;font-weight:800"}
    show = t.drop(columns=["Revenue 52w"])
    sty = show.style.format({"On hand": "{:,.0f}", "Forecast next 4 wks": "{:,.0f}", "DOH": "{:,.0f}", "Short in 13 wks": "{:,.0f}"})
    sty = ui.style_map(ui.style_map(sty, lambda v: status_c.get(v, ""), ["Status"]), seg_style, ["Segment"])
    st.dataframe(sty, hide_index=True, use_container_width=True, height=420)
    st.download_button("⬇ Download inventory alerts (CSV)", show.round(1).to_csv(index=False).encode("utf-8"), file_name="inventory_alerts.csv", mime="text/csv")
    st.caption("DOH = on-hand stock ÷ (average forecast demand of the next 4 weeks ÷ 7). At risk = DOH below the lead time: the stock would run out before a new order could arrive.")


def render_kpis():
    t1, t3 = st.tabs(["📐 Forecast accuracy", "📦 Inventory health"])
    with t1:
        render_accuracy()
    with t3:
        render_inventory()


with tab_kpi:
    render_kpis()


with tab_fc:
    render_abc_tab()


# ----------------------------------------------------------------------------- Consensus & Scenarios
N_PAST = 3


def _units_text(pct, mean):
    """Template values are shown as units: the baseline moved by the template percentage, rounded."""
    return f"{max(mean * (1 + pct / 100.0), 0.0):.0f}" if abs(pct) > 1e-9 else ""


def render_consensus():
    ui.section("Consensus & scenarios", "Layer Sales and Marketing overrides (in units) on top of the statistical baseline, then commit them with a reason")
    sel = filter_toolbar("con")
    if sel is None:
        return
    if not ss.get("run"):
        st.info("The forecast is not available yet. Press **Re-run statistical forecast** in the sidebar.")
        return
    frame = scope_frame(sel)
    if frame is None:
        return
    label = sel["label"]
    fut = frame[frame["week_idx"] > LAST_WEEK].reset_index(drop=True)
    past = frame[frame["week_idx"] <= LAST_WEEK].tail(N_PAST).reset_index(drop=True)
    n = len(fut)
    base = pd.DataFrame({"h": range(1, n + 1), "target_week": fut["week_idx"], "week_start": fut["week_start"], "mean": fut["forecast"],
                         "p10": fut["low"], "p50": fut["p50"], "p90": fut["high"]})
    hist = frame[frame["week_idx"] <= LAST_WEEK].tail(13)[["week_start", "actual"]].rename(columns={"actual": "units"})
    com = ss["committed"].get(label, {})
    gen = ss.setdefault(f"gen|{label}", 0)
    chart_slot = st.container()

    ui.section("Override grid", "Orange rows are yours to edit. Past weeks are locked; you can change the current week and beyond")
    ui.notice("<b>How to type (always in units):</b> a <b>plain number</b> such as <b>500</b> sets that week to exactly 500 units; a number with a sign, such as <b>+200</b> or <b>-150</b>, "
              "adds or subtracts units from the layer above. Leave a cell empty for no change. Order: Statistical baseline → Sales override → Marketing event → Final consensus.", ui.ORANGE)
    preset = st.selectbox("Start from a template", scn.PRESETS, key="con_preset")
    start = scn.preset_overrides(preset, base)
    seed_sales = [(com[w]["sales"] if (preset == scn.PRESETS[0] and w in com) else _units_text(start.loc[j, "override_pct"], base.loc[j, "mean"])) for j, w in enumerate(fut["week_idx"])]
    seed_mkt = [(com[w]["mkt"] if (preset == scn.PRESETS[0] and w in com) else "") for w in fut["week_idx"]]
    widths = [2.2] + [1] * (N_PAST + n)
    key_base = f"{label}|{preset}|{gen}"
    stat_total = float(base["mean"].sum())

    cs = st.columns(widths)
    ui.grid_label(cs[0], "Week")
    for i, r in past.iterrows():
        ui.grid_cell(cs[1 + i], f"🔒 W{int(r['week_idx'])}<br><small>{r['week_start']:%b %d}</small>", "gh-past")
    for j, r in fut.iterrows():
        cur = j == 0
        ui.grid_cell(cs[1 + N_PAST + j], f"{'▶ CURRENT<br>' if cur else ''}W{int(r['week_idx'])}<br><small>{r['week_start']:%b %d}</small>", "gh-cur" if cur else "gh-fut")
    dec = 1 if (len(sel["series"]) == 1 and sel["series"][0][0] == 7) else 0
    cs = st.columns(widths)
    ui.grid_label(cs[0], "Actuals")
    for i, r in past.iterrows():
        ui.grid_cell(cs[1 + i], f"{r['actual']:,.{dec}f}", "g-act")
    for j in range(n):
        ui.grid_cell(cs[1 + N_PAST + j], "–", "g-empty")
    cs = st.columns(widths)
    ui.grid_label(cs[0], "Statistical baseline <small>(read-only)</small>")
    for i in range(N_PAST):
        ui.grid_cell(cs[1 + i], "–", "g-empty")
    for j, r in fut.iterrows():
        ui.grid_cell(cs[1 + N_PAST + j], f"{r['forecast']:,.{dec}f}", "g-fc")
    sales_txt, mkt_txt = [], []
    for name, store_list, seeds, tag in (("✏️ Sales override", sales_txt, seed_sales, "sal"), ("✏️ Marketing event", mkt_txt, seed_mkt, "mkt")):
        cs = st.columns(widths)
        ui.grid_label(cs[0], name, "row-override")
        for i in range(N_PAST):
            ui.grid_cell(cs[1 + i], "🔒", "g-lock")
        for j in range(n):
            store_list.append(cs[1 + N_PAST + j].text_input(f"{name} week {j + 1}", value=seeds[j], key=f"{tag}|{key_base}|{j}", label_visibility="collapsed", placeholder="units"))
    res, errors = scn.apply_layers(base, sales_txt, mkt_txt)
    cs = st.columns(widths)
    ui.grid_label(cs[0], "Final consensus <small>(calculated)</small>", "row-consensus")
    for i in range(N_PAST):
        ui.grid_cell(cs[1 + i], "–", "g-empty")
    for j in range(n):
        ui.grid_cell(cs[1 + N_PAST + j], f"{res.loc[j, 'consensus_mean']:,.{dec}f}{' ✎' if res.loc[j, 'changed'] else ''}", "g-cons")
    cs = st.columns(widths)
    ui.grid_label(cs[0], "Δ vs baseline (units)")
    for i in range(N_PAST):
        ui.grid_cell(cs[1 + i], "–", "g-empty")
    for j in range(n):
        d = res.loc[j, "consensus_mean"] - base.loc[j, "mean"]
        ui.grid_cell(cs[1 + N_PAST + j], f"{d:+,.{dec}f}" if abs(d) > 1e-9 else "·", "g-empty")
    for (layer, wk), msg in errors.items():
        st.error(f"{layer}, week {wk}: {msg}")
    changed = int(res["changed"].sum())
    st.caption(f"{changed} week(s) with an override ✎. Locked weeks are history and cannot be changed. Final consensus is never below 0; Low / High case move in proportion.")

    ui.section("Commit or revert", "A reason code and a short comment are required to save a change")
    c = st.columns([1.2, 2.5, 1.2, 1.2])
    reason = c[0].selectbox("Reason code", scn.REASON_CODES, key="con_reason_code")
    comment = c[1].text_input("Comment", value="", max_chars=120, key=f"con_comment|{label}", placeholder="e.g. Summer promotion agreed with Sales")
    c[2].markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    c[3].markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    do_commit = c[2].button("✅ Commit", type="primary", use_container_width=True, key="btn_commit")
    do_revert = c[3].button("↩ Revert to baseline", use_container_width=True, key="btn_revert")
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    user = ss.get("user", "planner")
    labels = [f"W{int(w)} ({d:%b %d})" for w, d in zip(fut["week_idx"], fut["week_start"])]
    wk_ids = [int(w) for w in fut["week_idx"]]
    if do_commit:
        if errors:
            st.error("Fix the highlighted cells before committing.")
        elif changed and not comment.strip():
            st.error("Add a short comment: every override needs a justification.")
        elif not changed and not com:
            st.info("There are no overrides to commit.")
        else:
            new_c, rows = gv.commit_changes(com, res["consensus_mean"].to_numpy(), base["mean"].to_numpy(), sales_txt, mkt_txt, wk_ids, labels,
                                            reason, comment.strip() or "Overrides removed", user, now, label)
            ss["committed"][label] = new_c
            log_audit(rows)
            st.success(f"Committed {len(new_c)} overridden week(s); {len(rows)} change(s) written to the audit log. The Overview now shows this consensus for this scope.")
            com = new_c
    if do_revert:
        if com:
            _, rows = gv.commit_changes(com, base["mean"].to_numpy(), base["mean"].to_numpy(), [""] * n, [""] * n, wk_ids, labels, "Other", "Reverted to statistical baseline", user, now, label)
            log_audit(rows)
        ss["committed"][label] = {}
        ss[f"gen|{label}"] = gen + 1
        st.rerun()

    if com:
        ui.section("Committed overrides", "Orange-bordered cells are overridden; hover a cell to see the reason and who changed it")
        items = []
        for j, w in enumerate(wk_ids):
            e = com.get(w)
            items.append({"label": f"W{w}<br><small>{fut.loc[j, 'week_start']:%b %d}</small>", "cur": j == 0, "base": base.loc[j, "mean"], "final": e["final"] if e else base.loc[j, "mean"],
                          "changed": bool(e), "dec": dec, "tip": f"{e['reason']}: {e['comment']} ({e['user']}, {e['ts']})" if e else ""})
        st.markdown(ui.committed_grid(items), unsafe_allow_html=True)

    with chart_slot:
        ui.section("Forecast vs consensus", "Blue = actuals · orange = statistical forecast · green = consensus after your overrides")
        show_scrollable(charts.actuals_vs_forecast(
            hist, [{"name": "Statistical forecast", "df": base.assign(value=base["mean"]), "color": charts.FORECAST, "dy": -14},
                   {"name": "Consensus forecast", "df": res.assign(value=res["consensus_mean"]), "color": charts.CONSENSUS, "dy": 20}],
            label_every=2, height=400, width=(len(hist) + n) * 72, y_zero=False), 400)
        st.caption("Scroll sideways to see every week. Forecast numbers sit above the orange line, consensus numbers below the green line.")

    with st.expander("Top-down allocation: distribute the consensus change to the children"):
        delta = res["consensus_mean"].to_numpy() - base["mean"].to_numpy()
        if sel["skus"]:
            kids = list(sel["skus"])
            w52 = np.array([float(actual_series(7, k_).reindex(range(LAST_WEEK - 51, LAST_WEEK + 1)).fillna(0).sum()) for k_ in kids])
            lvl, kind = 7, "SKU"
        else:
            kids = list(sel["depts"])
            w52 = np.array([float(actual_series(4, d).reindex(range(LAST_WEEK - 51, LAST_WEEK + 1)).fillna(0).sum()) for d in kids])
            lvl, kind = 4, "Department"
        if not changed:
            st.info("Type an override above to see how it would be distributed.")
        elif w52.sum() <= 0:
            st.info("No history to compute the weights.")
        else:
            w = w52 / w52.sum()
            alloc = pd.DataFrame({kind: kids, "Historical weight (52w)": w, "Δ units (13 weeks)": delta.sum() * w})
            whole = bool(sel["skus"]) or set(sel["stores"]) == set(STORES)
            stat13 = [float(stat_forecast(lvl, k_)["mean"].sum()) if (whole and stat_forecast(lvl, k_) is not None) else float("nan") for k_ in kids]
            alloc["Statistical (13 weeks)"] = stat13
            alloc["New plan (13 weeks)"] = alloc["Statistical (13 weeks)"] + alloc["Δ units (13 weeks)"]
            st.dataframe(alloc.style.format({"Historical weight (52w)": "{:.1%}", "Δ units (13 weeks)": "{:+,.0f}", "Statistical (13 weeks)": "{:,.0f}", "New plan (13 weeks)": "{:,.0f}"}),
                         hide_index=True, use_container_width=True)
            st.caption("The change of the scope is split among its children in proportion to their share of the last 52 weeks of sales. "
                       + ("" if whole else "Children statistical totals are shown only when all stores are selected (department forecasts are chain-wide)."))
            st.download_button("⬇ Download allocation (CSV)", alloc.round(2).to_csv(index=False).encode("utf-8"), file_name="top_down_allocation.csv", mime="text/csv")

    store = ss.setdefault("scenarios", {}).setdefault(label, {})
    total_cons = float(res["consensus_mean"].sum())
    k = st.columns(3)
    ui.kpi(k[0], "Statistical forecast · 13 weeks", fmt0(stat_total), "units", ui.ORANGE, "#5B6B7F")
    dd = total_cons / max(stat_total, 1e-9) - 1
    ui.kpi(k[1], "Consensus forecast · 13 weeks", fmt0(total_cons), f"{dd:+.1%} vs statistical", ui.GREEN, ui.delta_color(dd))
    ui.kpi(k[2], "Saved scenarios", len(store), "for this scope", ui.PURPLE, "#5B6B7F")

    ui.section("Save and compare scenarios", "Save the current overrides under a name to compare them (frozen plan versions live in the Overview)")
    cc = st.columns([3, 1])
    name = cc[0].text_input("Scenario name", value="Consensus", max_chars=40, key="con_name")
    cc[1].markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if cc[1].button("💾 Save scenario", use_container_width=True, key="btn_save_scn"):
        if not name.strip():
            st.error("Enter a name for the scenario.")
        else:
            store[name.strip()] = res
            st.success(f"Scenario “{name.strip()}” saved.")
    shown = dict(store)
    if name.strip() and name.strip() not in store:
        shown[f"{name.strip()} (unsaved)"] = res
    cmp_t = scn.compare(base, shown)
    st.altair_chart(charts.scenario_chart(base, shown), use_container_width=True)
    sty = cmp_t.style.format({"Units": "{:,.0f}", "Low case": "{:,.0f}", "High case": "{:,.0f}", "Δ units": "{:+,.0f}", "Δ %": "{:+.1f}%"})
    sty = ui.style_map(sty, lambda v: f"color:{'#1E9E5A' if v > 0 else '#D64545' if v < 0 else '#5B6B7F'};font-weight:800", ["Δ units", "Δ %"])
    st.dataframe(sty, hide_index=True, use_container_width=True)

    stat_sheet = base[["week_start", "mean", "p10", "p50", "p90"]].copy()
    stat_sheet[["mean", "p10", "p50", "p90"]] = stat_sheet[["mean", "p10", "p50", "p90"]].round(2)
    stat_sheet["week_start"] = stat_sheet["week_start"].dt.strftime("%Y-%m-%d")
    stat_sheet.columns = ["Week starting", "Forecast", "Low case (P10)", "Median (P50)", "High case (P90)"]
    ov_sheet = pd.DataFrame({"Week": labels, "Sales override": sales_txt, "Marketing event": mkt_txt, "Final consensus": res["consensus_mean"].round(2)})
    sheets = {"Statistical forecast": stat_sheet, "Overrides": ov_sheet}
    for nm, r in shown.items():
        sheets[f"Scenario {nm}"[:31]] = r[["h", "target_week", "consensus_mean", "consensus_p10", "consensus_p50", "consensus_p90"]].round(2).rename(columns={
            "h": "Week", "target_week": "Model week", "consensus_mean": "Consensus", "consensus_p10": "Low case (P10)", "consensus_p50": "Median (P50)", "consensus_p90": "High case (P90)"})
    sheets["Comparison"] = cmp_t.round(2)
    if ss["audit"]:
        sheets["Audit log"] = pd.DataFrame(ss["audit"])
    sheets["Info"] = pd.DataFrame({"Item": ["Scope", "Series combined", "Last actual week", "Model version", "Generated"],
                                   "Value": [label, len(sel["series"]), LAST_WEEK, ss["run"]["model_version"], pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")]})
    st.download_button("⬇ Download scenarios (Excel)", excel_export.build_workbook(sheets), file_name="scenarios.xlsx", mime=XLSX)


with tab_con:
    render_consensus()


# ----------------------------------------------------------------------------- Data & Audit
def render_data():
    ui.section("Import data", "Upload a CSV or Excel file; it is validated before use")
    ui.notice("<b>Required columns:</b> Date, SKU, Family, Units, Price, Inventory. <b>Price</b> feeds the revenue used by ABC-XYZ and <b>Inventory</b> feeds Days on hand. "
              "Units and Date are validated; the forecast model keeps using the Databricks history.", ui.BLUE)
    tmpl = pd.DataFrame({"Date": f"{LAST_ACTUAL_DAY}", "SKU": ITEMS["item_id"].head(5), "Family": ITEMS["dept_id"].head(5), "Units": 0,
                         "Price": [round(item_price(k_), 2) for k_ in ITEMS["item_id"].head(5)], "Inventory": ""})
    st.download_button("⬇ Download CSV template", tmpl.to_csv(index=False).encode("utf-8"), file_name="import_template.csv", mime="text/csv")
    f = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="upl")
    if f is not None:
        try:
            raw = gv.read_upload(f.getvalue(), f.name)
        except Exception as err:
            st.error(f"The file could not be read: {err}")
            raw = None
        if raw is not None:
            clean, rep = gv.validate_upload(raw, set(ITEMS["item_id"]))
            lv = {"Error": "background-color:#FADBD8;color:#9B2D2D;font-weight:800", "Warning": "background-color:#FFF3D1;color:#8A5A00;font-weight:800", "OK": "background-color:#DDF3E7;color:#14663A;font-weight:800"}
            st.markdown("**Validation report**")
            st.dataframe(ui.style_map(rep.style, lambda v: lv.get(v, ""), ["Level"]), hide_index=True, use_container_width=True)
            if clean is None:
                st.error("The file was not loaded. Fix the errors above and upload it again.")
            else:
                prices = clean.dropna(subset=["Price"]).sort_values("Date").groupby("SKU").tail(1)
                ss["upload"] = clean
                ss["upload_price"] = {r["SKU"]: float(r["Price"]) for _, r in prices.iterrows() if r["SKU"] in set(ITEMS["item_id"])}
                st.success(f"Loaded {len(clean):,} valid rows · {clean['SKU'].nunique()} SKUs · prices applied to {len(ss['upload_price'])} model SKUs.")
    if ss.get("upload") is not None:
        with st.expander("Preview of the loaded data"):
            st.dataframe(ss["upload"].head(200), hide_index=True, use_container_width=True)
        if st.button("Remove imported data", key="btn_rm_upload"):
            ss["upload"], ss["upload_price"] = None, {}
            st.rerun()

    ui.section("Audit log", "Immutable record of every committed override or revert")
    if ss["audit"]:
        aud = pd.DataFrame(ss["audit"]).reindex(columns=gv.AUDIT_COLUMNS)
        st.dataframe(aud.iloc[::-1], hide_index=True, use_container_width=True, height=min(60 + 36 * len(aud), 420))
        st.download_button("⬇ Download audit log (CSV)", aud.to_csv(index=False).encode("utf-8"), file_name="audit_log.csv", mime="text/csv")
        st.caption("Entries can only be added, never edited or deleted from the app. They are also appended to audit_log.csv next to app.py when the app has write access.")
    else:
        st.info("No overrides have been committed yet. Commit one in the Consensus & Scenarios tab and it will appear here.")

    ui.section("Frozen versions", "Snapshots of the plan, created from the Overview")
    if ss["versions"]:
        vt = pd.DataFrame([{"Version": v["name"], "Scope": v["scope"], "Frozen at": v["ts"], "By": v["user"], "13-week plan (units)": v["total"]} for v in ss["versions"]])
        st.dataframe(vt.style.format({"13-week plan (units)": "{:,.0f}"}), hide_index=True, use_container_width=True)
    else:
        st.info("No versions yet. Use **Freeze version** in the Overview.")


with tab_data:
    render_data()


# ----------------------------------------------------------------------------- Acerca de (Spanish)
def render_about():
    ui.section("Arquitectura")
    st.code("Kaggle M5 → Volume (landing) → Pipeline Lakeflow (Bronze → Silver → Gold)\n"
            "            → Job programado → Modelo en MLflow / Unity Catalog → Endpoint (Model Serving)\n"
            "            → Esta aplicación (Streamlit) ── HTTP POST ──▶ endpoint", language="text")
    ui.section("Metodología")
    st.markdown("- **Pronóstico:** demanda semanal con horizonte de 13 semanas; modelos de cuantiles con gradient boosting (Low / High case) más un modelo de media, calibrados con predicción conforme.\n"
                "- **Jerarquía:** Category, Department y los niveles superiores se concilian (MinT) para que las partes sumen el total.\n"
                "- **Semana actual:** la primera semana pronosticada (W278). La fecha simulada de «hoy» es el final de los datos; las semanas anteriores son históricas y no se pueden modificar.\n"
                "- **Consenso por capas:** Statistical baseline → Sales override → Marketing event → Final consensus. Cada celda se captura en unidades: un número simple (500) fija la semana en ese valor y un número con signo (+200 / −150) suma o resta unidades. Cada cambio exige un código de motivo y un comentario, y queda en el log de auditoría.\n"
                "- **Versiones:** una versión congelada es una foto del plan; se puede mostrar o comparar contra el plan actual (variación % contra el plan anterior).\n"
                "- **Ingreso para ABC:** unidades × precio del SKU (ingreso de 52 semanas entre sus unidades, o el Price del archivo importado).\n"
                "- **ABC-XYZ:** ABC por Pareto de ingreso (A hasta 80 %, B hasta 95 %, C el resto); XYZ por coeficiente de variación (X < 0.25, Y 0.25–0.50, Z > 0.50), recalculable con 3, 6 o 12 meses. Cada segmento tiene un método estadístico recomendado.\n"
                "- **KPIs:** WMAPE, accuracy (1 − WMAPE), sesgo, valor agregado del pronóstico y cobertura del rango, calculados con pruebas retrospectivas (back-tests) sobre 5 fechas de corte. DOH = inventario ÷ demanda diaria pronosticada.")
    ui.section("Contrato de la API", "POST /serving-endpoints/{endpoint}/invocations")
    st.markdown("Una fila por serie: `series_id`, `level` (0–7), `last_week`, `horizon` (1–13) e `history` (26–300 valores semanales ≥ 0). "
                "La respuesta incluye `mean`, `p10`, `p50` y `p90` por semana, en unidades. Hasta 200 series por consulta; una entrada inválida se rechaza con HTTP 400 y un mensaje.")
    st.markdown("**Manejo de errores en la aplicación:** tiempo de conexión de 20 s y de lectura de 90 s; hasta 4 intentos con esperas de 3, 6 y 12 s ante tiempo agotado, "
                "errores de conexión, 408/425/429 y 5xx; sin reintento en 400/401/403/404; cada respuesta se valida (campos, longitudes, números finitos y no negativos, "
                "cuantiles ordenados, fechas).")
    ui.section("Probar la conexión")
    if st.button("Consultar el endpoint (Total, 4 semanas)", key="btn_ping"):
        row = fu.build_request(TOTAL_ID, 0, D["hier_by_series"][TOTAL_ID], LAST_WEEK, 4)
        got = call_endpoint([row])
        if got:
            f, i = got
            st.success(f"Respuesta válida en {i['seconds']:.2f} s · intentos: {i['attempts']} · modelo {f[0]['model_version']}")
            st.json({"series_id": f[0]["series_id"], "mean": f[0]["mean"], "p10": f[0]["p10"], "p90": f[0]["p90"]})
        elif ss.get("run_error"):
            e = ss["run_error"]
            (st.warning if e[0] else st.error)(e[1])
    ui.section("Limitaciones conocidas")
    st.markdown("- El Low case de SKUs individuales suele ser 0 porque muchos SKUs venden solo algunas semanas; el pronóstico y el High case no se ven afectados.\n"
                "- Al combinar varias series, la suma de las medias es exacta, pero el rango Low–High es una aproximación que supone independencia entre series.\n"
                "- Las clases ABC-XYZ se calculan con la muestra de SKUs de la tienda CA_1 y con el historial de la ventana elegida; se traslapan en parte con el periodo de las pruebas retrospectivas.\n"
                "- **FVA de los overrides** (error estadístico − error del consenso) solo se puede medir cuando existan ventas reales posteriores a un override confirmado; en esta simulación todavía no las hay. La métrica FVA mostrada compara el modelo contra el mejor método simple.\n"
                "- El ingreso de cada SKU usa un precio promedio (el dataset no trae precio por semana); el inventario proviene del archivo importado o de una demo ilustrativa claramente marcada.\n"
                "- La importación valida y usa Price e Inventory; no reemplaza el historial que alimenta al modelo en Databricks.\n"
                "- Los overrides, versiones y el log viven en la sesión del navegador; el log también se anexa a `audit_log.csv` si la app puede escribir en disco (en Streamlit Community Cloud ese archivo no es permanente).\n"
                "- El endpoint escala a cero: tras un periodo inactivo, la primera ejecución puede tardar de 1 a 2 minutos.\n"
                "- La aplicación cubre una muestra estratificada de 302 SKUs de la tienda CA_1; el modelo atiende toda la jerarquía.")


with tab_about:
    render_about()
