"""Visual helpers: CSS, header, KPI cards, notices and badges."""
import html

import streamlit as st

NAVY, BLUE, ORANGE, GREEN, RED, AMBER, PURPLE, TEAL = "#0B2545", "#1F5FBF", "#F26B21", "#2E9E5B", "#D64545", "#F2B01E", "#7C4DFF", "#00897B"

CSS = f"""
<style>
.block-container {{padding-top: 1.2rem; max-width: 1500px;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #14407A 55%, {BLUE} 100%); color: #fff; border-radius: 14px;
        padding: 20px 26px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;}}
.hero-t {{font-size: 1.75rem; font-weight: 800; letter-spacing: .01em;}}
.hero-s {{font-size: .95rem; opacity: .85; margin-top: 2px;}}
.chip {{display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
        padding: 4px 12px; font-size: .8rem; margin-left: 6px; white-space: nowrap;}}
.kpi {{background: #fff; border: 1px solid #DCE4F0; border-left: 7px solid var(--c); border-radius: 12px; padding: 12px 16px;
       box-shadow: 0 1px 3px rgba(11,37,69,.08); height: 100%;}}
.kpi .l {{font-size: .74rem; font-weight: 700; color: #5B6B7F; text-transform: uppercase; letter-spacing: .06em;}}
.kpi .v {{font-size: 1.9rem; font-weight: 800; color: {NAVY}; line-height: 1.25;}}
.kpi .s {{font-size: .85rem; font-weight: 600; color: var(--sc);}}
.sec {{display: flex; align-items: baseline; gap: 10px; border-bottom: 3px solid {BLUE}; padding-bottom: 4px; margin: 18px 0 10px 0;}}
.sec .t {{font-size: 1.25rem; font-weight: 800; color: {NAVY};}}
.sec .h {{font-size: .88rem; color: #5B6B7F;}}
.note {{border-radius: 10px; padding: 12px 16px; margin: 8px 0; border: 1px solid var(--c); background: var(--bg); color: {NAVY};}}
.note b {{color: var(--c);}}
.badge {{display: inline-block; border-radius: 6px; padding: 2px 10px; font-weight: 800; font-size: .85rem;}}
.step {{background: #fff; border: 1px solid #DCE4F0; border-top: 5px solid var(--c); border-radius: 12px; padding: 12px 14px; height: 100%;}}
.step .n {{font-size: .75rem; font-weight: 800; color: var(--c); letter-spacing: .08em;}}
.step .t {{font-weight: 800; color: {NAVY}; margin: 2px 0 4px 0;}}
.step .d {{font-size: .86rem; color: #3D4F66;}}
.stTabs [data-baseweb="tab"] {{font-weight: 700; font-size: 1rem; padding: 10px 16px;}}
.stTabs [aria-selected="true"] {{color: {BLUE};}}
[data-testid="stSidebar"] {{border-right: 1px solid #CBD7EA;}}
</style>
"""

TINTS = {BLUE: "#EAF1FD", ORANGE: "#FFF1E6", GREEN: "#E7F6EE", RED: "#FDECEC", AMBER: "#FFF7E0", PURPLE: "#F0EBFF", TEAL: "#E3F4F2"}


CSS2 = """
<style>
.pgrid-wrap {overflow-x: auto; border: 1px solid #DCE4F0; border-radius: 12px; background: #fff; margin: 6px 0 10px 0;}
table.pgrid {border-collapse: separate; border-spacing: 0; width: 100%; font-size: .86rem;}
table.pgrid th, table.pgrid td {padding: 7px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid #EEF2F8; min-width: 78px;}
table.pgrid th.rowlbl, table.pgrid td.rowlbl {position: sticky; left: 0; z-index: 2; text-align: left; font-weight: 700; background: #F7F9FC; color: #0B2545; min-width: 150px; border-right: 2px solid #DCE4F0;}
table.pgrid th {color: #fff; font-weight: 700; text-align: center; vertical-align: bottom;}
table.pgrid th .lbl {font-size: .9rem;} table.pgrid th .sub {font-size: .72rem; opacity: .9; font-weight: 500;}
table.pgrid th .curtag {font-size: .62rem; letter-spacing: .08em; background: #0B2545; border-radius: 4px; padding: 1px 6px; margin-bottom: 3px; display: inline-block; visibility: hidden;}
table.pgrid th.cur .curtag {visibility: visible;}
.h-act {background: #1F5FBF;} .h-fc {background: #F26B21;} .h-mix {background: linear-gradient(90deg, #1F5FBF 0%, #1F5FBF 45%, #F26B21 55%, #F26B21 100%);}
table.pgrid td.cur {background: #FFF4E8 !important; border-left: 2px solid #F26B21; border-right: 2px solid #F26B21;}
table.pgrid td.c-act {background: #EAF1FD; color: #14407A; font-weight: 700;} table.pgrid td.c-fc {background: #FFF1E6; color: #B84A0A; font-weight: 700;}
table.pgrid td.c-plan {font-weight: 800; color: #0B2545;} table.pgrid td.c-ly {color: #6B7A8D;} table.pgrid td.empty {color: #B7C0CC;}
table.pgrid th.scp {background: #0B2545; text-align: left; min-width: 84px;}
table.pgrid td.scp {text-align: left; font-weight: 700; color: #0B2545; background: #F7F9FC; vertical-align: middle; white-space: normal; max-width: 150px; border-right: 1px solid #DCE4F0;}
table.pgrid td.c-cons {background: #D4EEDF; color: #145A32; font-weight: 800;} table.pgrid td.c-prev {background: #EFE9FB; color: #5B3FA0; font-weight: 700;}
table.pgrid th.fs, table.pgrid td.fs {border-left: 3px solid #0B2545;}
table.pgrid td.modcell {outline: 3px solid #F26B21; outline-offset: -3px; cursor: help;}
tr.brk-tot td {background: #F0F4FA !important; font-weight: 800;}
table.pgrid td.c-lo {background: #FDECEC; color: #9B2D2D;} table.pgrid td.c-hi {background: #E7F6EE; color: #1B7A47;}
.up {color: #1E9E5A; font-weight: 800;} .down {color: #D64545; font-weight: 800;}
.lgd {font-size: .82rem; color: #5B6B7F; margin-bottom: 8px;} .lgd b {padding: 1px 8px; border-radius: 4px; color: #fff;}
table.ftab {border-collapse: collapse; width: 100%; background: #fff; border: 1px solid #DCE4F0; border-radius: 12px; overflow: hidden; font-size: .92rem;}
table.ftab th {background: #0B2545; color: #fff; padding: 9px 12px; text-align: left; font-size: .8rem; letter-spacing: .04em;}
table.ftab td {padding: 8px 12px; border-bottom: 1px solid #EEF2F8; vertical-align: middle;}
table.ftab tr.cur td {background: #FFF4E8; border-top: 2px solid #F26B21; border-bottom: 2px solid #F26B21;}
table.ftab tr.tot td {background: #F0F4FA; font-weight: 800;}
table.ftab .exp {font-weight: 800; color: #B84A0A; font-size: 1.02rem;} table.ftab .rng {color: #3D4F66;}
.rbar {position: relative; height: 12px; background: #EEF2F8; border-radius: 6px; min-width: 150px;}
.rfill {position: absolute; top: 0; height: 12px; background: #F9C79B; border-radius: 6px;} .rmark {position: absolute; top: -3px; width: 4px; height: 18px; background: #F26B21; border-radius: 2px;}
.tag {font-size: .65rem; font-weight: 800; letter-spacing: .06em; background: #F26B21; color: #fff; border-radius: 4px; padding: 1px 6px; margin-left: 6px;}
.fchip {display: inline-block; border-radius: 999px; padding: 3px 12px; margin: 2px 6px 2px 0; font-size: .82rem; font-weight: 700; color: #fff;}
.gcell {text-align: center; padding: 6px 2px; font-size: .86rem; border-radius: 6px; line-height: 1.2;}
.gh-past {background: #D9DEE6; color: #5B6B7F;} .gh-cur {background: #0B2545; color: #fff;} .gh-fut {background: #F26B21; color: #fff;}
.g-act {background: #EAF1FD; color: #14407A; font-weight: 700;} .g-fc {background: #FFF1E6; color: #B84A0A; font-weight: 700;}
.g-cons {background: #D4EEDF; color: #145A32; font-weight: 800;} .g-lock {background: #F0F2F5; color: #98A2B0;} .g-empty {color: #B7C0CC;}
.rl {font-weight: 800; font-size: .88rem; color: #0B2545; padding-top: 6px;}
div[data-testid="stHorizontalBlock"]:has(.row-override) {background: #FFD9B0; border: 2px solid #F26B21; border-radius: 10px; padding: 6px 6px 2px 6px; margin: 4px 0;}
div[data-testid="stHorizontalBlock"]:has(.row-consensus) {background: #E3F4EA; border: 2px solid #2E9E5B; border-radius: 10px; padding: 6px; margin: 4px 0;}
div[data-testid="stHorizontalBlock"]:has(.row-override) [data-testid="stTextInput"] input, div[data-testid="stHorizontalBlock"]:has(.row-override) [data-testid="stNumberInput"] input {font-size: .82rem; padding: 4px 2px; text-align: center; font-weight: 700;}
div[data-testid="stHorizontalBlock"]:has(.row-override) [data-testid="stNumberInputStepUp"], div[data-testid="stHorizontalBlock"]:has(.row-override) [data-testid="stNumberInputStepDown"] {display: none;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(CSS2, unsafe_allow_html=True)


def chips(items):
    """items: [(label, text, color)] -> row of coloured chips."""
    st.markdown("".join(f'<span class="fchip" style="background:{c}">{html.escape(l)}: {html.escape(t)}</span>' for l, t, c in items), unsafe_allow_html=True)


def _num(v, dec=0):
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:,.{dec}f}"


def _pct(v, cur=""):
    if v is None or v != v:
        return f'<td class="empty{cur}">–</td>'
    return f'<td class="{"up" if v >= 0 else "down"}{cur}">{v:+.1%}</td>'


def period_grid(periods, dec=0, band=False, band_exact=True, prefix="", prev_name="", cons_name="Consensus forecast", scope_cols=None):
    """Forecast Summary grid: periods as columns, measures as rows. Blue = actuals, orange = forecast, CURRENT marked."""
    head = ('<th class="scp">Category</th><th class="scp">Store</th><th class="scp">Department</th><th class="scp">SKU</th><th class="rowlbl">Measure</th>' if scope_cols
            else '<th class="rowlbl"></th>')
    first_open = next((i for i, p in enumerate(periods) if p["kind"] != "actual"), -1)
    for i, p in enumerate(periods):
        cls = {"actual": "h-act", "forecast": "h-fc", "mixed": "h-mix"}[p["kind"]] + (" cur" if p["is_current"] else "") + (" fs" if i == first_open else "")
        sub = f'<div class="sub">{html.escape(p["sub"])}</div>' if p["sub"] else ""
        head += (f'<th class="{cls}"><div class="curtag">CURRENT</div><div class="lbl">{html.escape(p["label"])}{"*" if p["partial"] else ""}</div>{sub}</th>')

    state = {"n": 0}

    def scope_cells(n_rows):
        if not scope_cols or state["n"]:
            return ""
        return "".join(f'<td class="scp" rowspan="{n_rows}" title="{html.escape(full, quote=True)}">{html.escape(short)}</td>' for short, full in scope_cols)

    def row(name, fn, css=""):
        cells = ""
        for i, p in enumerate(periods):
            cur = (" cur" if p["is_current"] else "") + (" fs" if i == first_open else "")
            cells += fn(p, cur)
        lead = scope_cells(state["total"])
        state["n"] += 1
        return f'<tr>{lead}<td class="rowlbl {css}">{html.escape(name)}</td>{cells}</tr>'

    def val(key, cls):
        def f(p, cur):
            v = p[key]
            if v != v:
                return f'<td class="empty{cur}">–</td>'
            return f'<td class="{cls}{cur}">{prefix}{_num(v, dec)}</td>'
        return f
    has_cons = any(p["consensus"] == p["consensus"] for p in periods)
    has_prev = any(p["prev"] == p["prev"] for p in periods)
    state["total"] = 5 + int(has_cons) + 2 * int(has_prev) + (2 if band else 0)
    rows = [row("Actuals", val("actual", "c-act")), row("Statistical forecast", val("forecast", "c-fc"))]
    if has_cons:
        rows.append(row(cons_name, val("consensus", "c-cons")))
    rows.append(row("Final plan", val("plan", "c-plan")))
    rows.append(row("Last year", val("ly", "c-ly")))
    rows.append(row("YoY % (vs last year)", lambda p, cur: _pct(p["vs_ly"], cur)))
    if has_prev:
        rows.append(row(prev_name or "Compared plan", val("prev", "c-prev")))
        rows.append(row("% vs compared plan", lambda p, cur: _pct(p["vs_prev"], cur)))
    if band:
        approx = "" if band_exact else " (approx.)"
        rows.append(row("Low case" + approx, val("low", "c-lo")))
        rows.append(row("High case" + approx, val("high", "c-hi")))
    return f'<div class="pgrid-wrap"><table class="pgrid"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def forecast_table(rows, total):
    """Plain-language weekly table. rows: dicts label, expected, low, high, ly_pct, current; total: dict expected, low, high, ly_pct."""
    top = max([r["high"] for r in rows] + [1e-9]) * 1.05

    def bar(r):
        l, h, m = r["low"] / top * 100, (r["high"] - r["low"]) / top * 100, r["expected"] / top * 100
        return f'<div class="rbar"><div class="rfill" style="left:{l:.1f}%;width:{max(h, 1):.1f}%"></div><div class="rmark" style="left:{m:.1f}%"></div></div>'

    body = ""
    for r in rows:
        tag = '<span class="tag">CURRENT WEEK</span>' if r["current"] else ""
        ly = _pct(r["ly_pct"])
        body += (f'<tr class="{"cur" if r["current"] else ""}"><td><b>{html.escape(r["label"])}</b>{tag}</td><td class="exp">{_num(r["expected"], r["dec"])}</td>'
                 f'<td class="rng">{_num(r["low"], r["dec"])} – {_num(r["high"], r["dec"])}</td><td>{bar(r)}</td>{ly}</tr>')
    d = rows[0]["dec"]
    body += (f'<tr class="tot"><td>13-week total</td><td class="exp">{_num(total["expected"], d)}</td><td class="rng">{_num(total["low"], d)} – {_num(total["high"], d)}</td>'
             f'<td>sum of the weekly figures</td>{_pct(total["ly_pct"])}</tr>')
    head = ("<tr><th>WEEK</th><th>EXPECTED DEMAND<br>(Forecast)</th><th>REALISTIC RANGE<br>(Low – High case)</th>"
            "<th>WHERE THE FORECAST SITS IN THE RANGE</th><th>VS SAME WEEK<br>LAST YEAR</th></tr>")
    return f'<table class="ftab"><thead>{head}</thead><tbody>{body}</tbody></table>'


def grid_cell(col, text, css):
    col.markdown(f'<div class="gcell {css}">{text}</div>', unsafe_allow_html=True)


def grid_label(col, text, marker=""):
    mk = f'<span class="{marker}"></span>' if marker else ""
    col.markdown(f'<div class="rl">{text}</div>{mk}', unsafe_allow_html=True)


def hero(title, subtitle, chips):
    chip_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(f'<div class="hero"><div><div class="hero-t">{html.escape(title)}</div>'
                f'<div class="hero-s">{html.escape(subtitle)}</div></div><div>{chip_html}</div></div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", color=BLUE, sub_color=None):
    col.markdown(f'<div class="kpi" style="--c:{color};--sc:{sub_color or color}"><div class="l">{html.escape(label)}</div>'
                 f'<div class="v">{html.escape(str(value))}</div><div class="s">{html.escape(sub)}&nbsp;</div></div>', unsafe_allow_html=True)


def delta_color(x):
    return GREEN if x >= 0 else RED


def section(title, hint=""):
    st.markdown(f'<div class="sec"><span class="t">{html.escape(title)}</span><span class="h">{html.escape(hint)}</span></div>', unsafe_allow_html=True)


def notice(text_html, color=BLUE):
    """text_html may contain <b> and <br>; callers pass only trusted literals."""
    st.markdown(f'<div class="note" style="--c:{color};--bg:{TINTS.get(color, "#F4F7FB")}">{text_html}</div>', unsafe_allow_html=True)


def badge_html(text, bg, fg="#fff"):
    return f'<span class="badge" style="background:{bg};color:{fg}">{html.escape(text)}</span>'


def step_card(col, number, title, desc, color):
    col.markdown(f'<div class="step" style="--c:{color}"><div class="n">STEP {number}</div><div class="t">{html.escape(title)}</div>'
                 f'<div class="d">{html.escape(desc)}</div></div>', unsafe_allow_html=True)


def style_map(styler, func, subset):
    """Styler.map (pandas ≥ 2.1) with fallback to applymap."""
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(func, subset=subset)


def breakdown_table(rows, periods, prefix="", dec=0):
    """Roll-up / drill-down matrix: one row per member (category, department, store or SKU), plan per period. rows: [(name, [values], yoy)]."""
    first_open = next((i for i, p in enumerate(periods) if p["kind"] != "actual"), -1)
    head = '<th class="rowlbl">Member</th>' + "".join(
        f'<th class="{ {"actual": "h-act", "forecast": "h-fc", "mixed": "h-mix"}[p["kind"]]}{" cur" if p["is_current"] else ""}{" fs" if i == first_open else ""}"><div class="curtag">CURRENT</div><div class="lbl">{html.escape(p["label"])}</div></th>'
        for i, p in enumerate(periods)) + '<th class="h-mix">Total</th><th class="h-mix">YoY %</th>'
    body = ""
    for k, (name, vals, yoy_v, is_total) in enumerate(rows):
        cells = "".join(f'<td class="{"cur" if periods[i]["is_current"] else ""}{" fs" if i == first_open else ""}">{prefix}{_num(v, dec)}</td>' for i, v in enumerate(vals))
        tot = sum(v for v in vals if v == v)
        body += (f'<tr class="{"brk-tot" if is_total else ""}"><td class="rowlbl">{html.escape(name)}</td>{cells}<td class="c-plan">{prefix}{_num(tot, dec)}</td>{_pct(yoy_v)}</tr>')
    return f'<div class="pgrid-wrap"><table class="pgrid"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def committed_grid(items, prefix=""):
    """Read-only view of committed overrides. items: [{week, label, base, final, changed, tip}]. Modified cells get an orange border and a hover tooltip."""
    head = '<th class="rowlbl"></th>' + "".join(f'<th class="{"cur" if it.get("cur") else ""}">{html.escape(it["label"])}</th>' for it in items)
    r1 = "".join(f'<td class="c-fc">{_num(it["base"], it["dec"])}</td>' for it in items)
    r2 = "".join(f'<td class="c-cons{" modcell" if it["changed"] else ""}"{(" title=" + chr(34) + html.escape(it["tip"], quote=True) + chr(34)) if it["changed"] else ""}>{_num(it["final"], it["dec"])}{" ✎" if it["changed"] else ""}</td>' for it in items)
    return (f'<div class="pgrid-wrap"><table class="pgrid"><thead><tr>{head}</tr></thead><tbody><tr><td class="rowlbl">Statistical baseline</td>{r1}</tr>'
            f'<tr><td class="rowlbl">Committed consensus</td>{r2}</tr></tbody></table></div>')


def segment_summary_grid(seg_rows, scope_cols, totals):
    """ABC-XYZ summary: the 9 segments as columns, measures as rows. seg_rows: {segment: dict(items, share_items, share_rev, units, cv, fc13)}."""
    from_chart = {"AX": "#1E9E5A", "AY": "#7CC04B", "BX": "#7CC04B", "AZ": "#F2D13C", "BY": "#F2D13C", "CX": "#F2D13C", "BZ": "#F4902F", "CY": "#F4902F", "CZ": "#D64545"}
    dark = {"AX", "BZ", "CY", "CZ"}
    segs = list(from_chart)
    head = ('<th class="scp">Category</th><th class="scp">Store</th><th class="scp">Department</th><th class="scp">SKU</th><th class="rowlbl">Measure</th>'
            + "".join(f'<th style="background:{from_chart[g]};color:{"#fff" if g in dark else "#0B2545"}"><div class="lbl">{g}</div></th>' for g in segs) + '<th class="h-mix">Total</th>')
    measures = [("SKUs", "items", lambda v: f"{v:,.0f}"), ("% of SKUs", "share_items", lambda v: f"{v:.1%}"), ("% of revenue", "share_rev", lambda v: f"{v:.1%}"),
                ("Units (window)", "units", lambda v: f"{v:,.0f}"), ("Avg CV", "cv", lambda v: f"{v:.2f}"), ("Forecast next 13 wks", "fc13", lambda v: f"{v:,.0f}")]
    body = ""
    for n, (name, key, fm) in enumerate(measures):
        lead = "".join(f'<td class="scp" rowspan="{len(measures)}" title="{html.escape(full, quote=True)}">{html.escape(short)}</td>' for short, full in scope_cols) if n == 0 else ""
        cells = "".join(f'<td>{fm(seg_rows[g][key]) if seg_rows[g][key] == seg_rows[g][key] else "–"}</td>' for g in segs)
        tv = totals[key]
        body += f'<tr>{lead}<td class="rowlbl">{html.escape(name)}</td>{cells}<td class="c-plan">{fm(tv) if tv == tv else "–"}</td></tr>'
    return f'<div class="pgrid-wrap"><table class="pgrid"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
