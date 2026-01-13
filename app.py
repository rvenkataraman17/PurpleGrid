import os
from urllib.parse import urlparse

import pandas as pd
import streamlit as st


# ------------------------
# Page setup
# ------------------------
st.set_page_config(
    page_title="Siemens Energy – Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Siemens Energy – Strategy Dashboard")


# ------------------------
# Config / Governance
# ------------------------
# Governance: unit discipline (you said EURm everywhere, plus % for margin-type KPIs)
ALLOWED_UNITS = {"EURm", "%"}

# Governance: Siemens Energy numbers must come from Siemens Energy IR pages (if source_ref exists)
def is_se_ir_url(url: str) -> bool:
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    try:
        u = urlparse(url)
        domain_ok = u.netloc.endswith("siemens-energy.com")
        path_ok = ("investor-relations" in u.path) or ("investor" in u.path)
        return domain_ok and path_ok
    except Exception:
        return False


def safe_dt(series):
    return pd.to_datetime(series, errors="coerce")


def fmt_value(value, unit):
    """Consistent formatting for KPI values."""
    try:
        v = float(value)
    except Exception:
        return f"{value} {unit}".strip()

    if str(unit).strip() == "%":
        return f"{v:.1f}%"
    # Default EURm formatting
    return f"{v:,.0f} {unit}".strip()


def fmt_delta(delta, unit):
    if delta is None or pd.isna(delta):
        return None
    try:
        d = float(delta)
    except Exception:
        return None

    if str(unit).strip() == "%":
        return f"{d:+.1f}%"
    return f"{d:+,.0f} {unit}".strip()


def require_columns(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"{df_name} is missing required columns: {missing}")
        st.stop()


# ------------------------
# Load data
# ------------------------
@st.cache_data
def load_data():
    kpis = pd.read_csv("data/company_kpis_2025.csv")
    guidance = pd.read_csv("data/guidance_2025.csv")
    peers = pd.read_csv("data/peer_kpis_2025.csv")

    intel_path = "data/intelligence_live.csv" if os.path.exists("data/intelligence_live.csv") else "data/intelligence_2025.csv"
    intelligence = pd.read_csv(intel_path)

    strategy = pd.read_csv("data/strategy_2025.csv")
    return kpis, guidance, peers, intelligence, strategy, intel_path


kpis, guidance, peers, intelligence, strategy, intel_path = load_data()


# ------------------------
# Normalize / Expected columns (soft expectations)
# ------------------------
# KPI file should ideally include:
# company, period, period_end, basis, metric, value, unit, commentary, source_ref, confidence
# We'll degrade gracefully if some columns are absent, but period dropdown + trends need period/period_end.

# Ensure essential columns exist where needed
require_columns(kpis, ["metric", "value", "unit"], "company_kpis_2025.csv")
require_columns(guidance, ["metric", "unit"], "guidance_2025.csv")
require_columns(peers, ["company", "metric", "value", "unit"], "peer_kpis_2025.csv")

# Parse dates if present
if "period_end" in kpis.columns:
    kpis["period_end"] = safe_dt(kpis["period_end"])
else:
    kpis["period_end"] = pd.NaT

if "date" in strategy.columns:
    strategy["date"] = safe_dt(strategy["date"])

if "date_utc" in intelligence.columns:
    intelligence["date_utc"] = safe_dt(intelligence["date_utc"])
elif "date" in intelligence.columns:
    intelligence["date"] = safe_dt(intelligence["date"])


# ------------------------
# Sidebar controls + Governance panel
# ------------------------
st.sidebar.header("Controls")

# Period dropdown (best practice: add a period column)
selected_period = None
kpis_view = kpis.copy()

if "period" in kpis.columns and kpis["period"].notna().any():
    periods = sorted(kpis["period"].dropna().unique().tolist())
    selected_period = st.sidebar.selectbox("Reporting period", periods, index=len(periods) - 1)
    kpis_view = kpis[kpis["period"] == selected_period].copy()
else:
    st.sidebar.info("Tip: add a 'period' column in company_kpis_2025.csv to enable period dropdown.")

# Optional metric selector for charts
available_metrics = sorted(kpis["metric"].dropna().unique().tolist())
metric_selected = st.sidebar.selectbox("Trend metric", available_metrics, index=0 if available_metrics else 0)

# Intelligence window
days = st.sidebar.slider("Intel lookback (days)", min_value=7, max_value=120, value=30, step=1)

# Governance checks
with st.sidebar.expander("Governance checks", expanded=True):
    msgs = []

    # Units (allow EURm and % only)
    units_in_view = sorted(kpis_view["unit"].dropna().unique().tolist())
    if any(u not in ALLOWED_UNITS for u in units_in_view):
        msgs.append(("❌ Unit governance failed", f"Unexpected units: {units_in_view}. Use EURm and % only."))
    else:
        msgs.append(("✅ Units OK", f"Units: {units_in_view}"))

    # Single period snapshot
    if "period" in kpis_view.columns:
        if kpis_view["period"].nunique() != 1:
            msgs.append(("❌ Mixed periods in snapshot", "Snapshot should contain exactly one period."))
        else:
            msgs.append(("✅ Single period snapshot", f"{kpis_view['period'].iloc[0]}"))

    # IR-only source gate (only if source_ref exists)
    if "source_ref" in kpis_view.columns:
        bad = kpis_view[~kpis_view["source_ref"].apply(is_se_ir_url)]
        if len(bad) > 0:
            msgs.append(("❌ Source governance failed", f"{len(bad)} KPI rows have non-IR sources."))
        else:
            msgs.append(("✅ KPI sources are IR", "OK"))

    # Confidence visibility (if present)
    if "confidence" in kpis_view.columns:
        conf = kpis_view["confidence"].fillna("Unknown").value_counts().to_dict()
        msgs.append(("ℹ Confidence distribution", str(conf)))

    for t, d in msgs:
        st.write(f"**{t}**")
        st.caption(d)

# Hard stop if IR governance fails (recommended once you start using source_ref)
if "source_ref" in kpis_view.columns:
    bad = kpis_view[~kpis_view["source_ref"].apply(is_se_ir_url)]
    if len(bad) > 0:
        st.error("Governance stop: KPI rows include non-Investor-Relations sources.")
        st.dataframe(bad[["metric", "value", "unit", "source_ref"]], use_container_width=True)
        st.stop()


# ------------------------
# Tabs
# ------------------------
tab_perf, tab_gp, tab_intel, tab_strategy = st.tabs(
    ["📊 Performance", "🏁 Guidance & Peers", "🧠 Strategic Intelligence", "🧭 Strategy"]
)


# ========================
# TAB 1: Performance
# ========================
with tab_perf:
    st.subheader("Performance Snapshot")

    # Snapshot caption
    caption = []
    if selected_period:
        caption.append(f"Reporting basis: **{selected_period}**")
    if "as_of" in kpis_view.columns and kpis_view["as_of"].notna().any():
        caption.append(f"As of: **{str(kpis_view['as_of'].dropna().iloc[0])}**")
    if "basis" in kpis_view.columns and kpis_view["basis"].notna().any():
        caption.append(f"Status: **{kpis_view['basis'].dropna().iloc[0]}**")
    if caption:
        st.caption(" | ".join(caption))

    # Delta vs previous period (requires period_end and multi-period history)
    prev_map = {}
    if kpis["period_end"].notna().any() and kpis_view["period_end"].notna().any():
        snap_end = kpis_view["period_end"].dropna().iloc[0]
        hist_before = kpis[kpis["period_end"] < snap_end].copy()
        if len(hist_before) > 0:
            prev_last = hist_before.sort_values("period_end").groupby("metric").tail(1)
            prev_map = dict(zip(prev_last["metric"], prev_last["value"]))

    # KPI tiles
    cards = kpis_view.to_dict(orient="records")
    cols = st.columns(3)

    for i, row in enumerate(cards):
        metric = row.get("metric", "—")
        value = row.get("value", "—")
        unit = row.get("unit", "")
        commentary = row.get("commentary", "")
        delta = None

        if metric in prev_map:
            try:
                delta = float(value) - float(prev_map[metric])
            except Exception:
                delta = None

        with cols[i % 3]:
            st.metric(
                label=metric,
                value=fmt_value(value, unit),
                delta=fmt_delta(delta, unit),
                help=commentary
            )

    st.divider()

    st.subheader("Trend (time series)")

    trend = kpis[kpis["metric"] == metric_selected].copy() if metric_selected else pd.DataFrame()
    trend = trend.sort_values("period_end")

    if len(trend) == 0:
        st.info("No trend data available.")
    elif trend["period_end"].isna().all():
        st.warning("Add a 'period_end' column (YYYY-MM-DD) to enable true trend charts.")
        st.dataframe(trend, use_container_width=True)
    else:
        chart = trend[["period_end", "value"]].dropna().set_index("period_end")
        st.line_chart(chart)
        cols_to_show = [c for c in ["period", "period_end", "basis", "value", "unit", "confidence", "source_ref", "commentary"] if c in trend.columns]
        st.dataframe(trend[cols_to_show], use_container_width=True)


# ========================
# TAB 2: Guidance & Peers
# ========================
with tab_gp:
    st.subheader("Guidance vs Actual")

    # Expect guidance columns: low/mid/high optional; allow mid-only too
    # We'll try to match by metric + unit (+ period if present in both)
    guidance_view = guidance.copy()

    # Filter guidance to the selected period if possible
    if selected_period and "period" in guidance_view.columns:
        guidance_view = guidance_view[guidance_view["period"] == selected_period].copy()

    # Build actuals for join
    actuals = kpis_view[["metric", "value", "unit"]].copy()

    # Determine guidance column set
    has_mid = "mid" in guidance_view.columns
    has_low_high = ("low" in guidance_view.columns) and ("high" in guidance_view.columns)

    # Merge
    merged = actuals.merge(guidance_view, on=["metric", "unit"], how="left", suffixes=("", "_g"))

    # Compute deltas vs midpoint if available
    if has_mid:
        merged["delta_vs_mid"] = pd.to_numeric(merged["value"], errors="coerce") - pd.to_numeric(merged["mid"], errors="coerce")

    # Status vs band
    if has_low_high:
        v = pd.to_numeric(merged["value"], errors="coerce")
        lo = pd.to_numeric(merged["low"], errors="coerce")
        hi = pd.to_numeric(merged["high"], errors="coerce")
        merged["within_band"] = (v >= lo) & (v <= hi)

    # Display as KPI tiles with "within band" color cues
    cols = st.columns(3)
    for i, row in merged.iterrows():
        metric = row["metric"]
        unit = row["unit"]
        value = row["value"]

        delta_txt = None
        if has_mid and pd.notna(row.get("delta_vs_mid")):
            delta_txt = fmt_delta(row["delta_vs_mid"], unit)

        with cols[i % 3]:
            # Show a simple status line under the metric
            st.metric(metric, fmt_value(value, unit), delta=delta_txt)
            if has_low_high and pd.notna(row.get("within_band")):
                if bool(row["within_band"]):
                    st.success("Within guidance band")
                else:
                    st.error("Outside guidance band")

    st.divider()
    st.subheader("Peers (snapshot comparison)")

    # Filter peers by selected period if possible
    peers_view = peers.copy()
    if selected_period and "period" in peers_view.columns:
        peers_view = peers_view[peers_view["period"] == selected_period].copy()

    peer_metric = st.selectbox("Peer metric", sorted(peers_view["metric"].dropna().unique().tolist()))
    peer_unit = st.selectbox("Unit", sorted(peers_view[peers_view["metric"] == peer_metric]["unit"].dropna().unique().tolist()))

    p = peers_view[(peers_view["metric"] == peer_metric) & (peers_view["unit"] == peer_unit)].copy()

    # Add Siemens Energy from actuals if matches
    se_row = kpis_view[(kpis_view["metric"] == peer_metric) & (kpis_view["unit"] == peer_unit)][["metric", "value", "unit"]].copy()
    if len(se_row) > 0:
        se_val = se_row.iloc[0]["value"]
        p = pd.concat([
            pd.DataFrame([{"company": "Siemens Energy", "metric": peer_metric, "value": se_val, "unit": peer_unit}]),
            p
        ], ignore_index=True)

    # Bar chart + table
    p["value_num"] = pd.to_numeric(p["value"], errors="coerce")
    p = p.sort_values("value_num", ascending=False)

    st.bar_chart(p.set_index("company")["value_num"])
    cols_to_show = [c for c in ["company", "period", "metric", "value", "unit", "basis", "confidence", "source_ref"] if c in p.columns]
    st.dataframe(p[cols_to_show], use_container_width=True)


# ========================
# TAB 3: Strategic Intelligence
# ========================
with tab_intel:
    st.subheader("Strategic Intelligence")
    st.caption(f"Source: **{intel_path}**")

    # Normalize expected columns from your collector
    # Preferred: date_utc, category, entity, headline, source_name, source_url, region, signal_type, impact_area, expected_impact, confidence, action_required
    # Degrade gracefully if your file is older schema
    if "date_utc" in intelligence.columns:
        intelligence["date_dt"] = safe_dt(intelligence["date_utc"])
    elif "date" in intelligence.columns:
        intelligence["date_dt"] = safe_dt(intelligence["date"])
    else:
        intelligence["date_dt"] = pd.NaT

    # Lookback filter
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    intel = intelligence.copy()
    if intel["date_dt"].notna().any():
        intel = intel[intel["date_dt"] >= cutoff].copy()

    # Filters
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    categories = sorted(intel["category"].dropna().unique().tolist()) if "category" in intel.columns else []
    entities = sorted(intel["entity"].dropna().unique().tolist()) if "entity" in intel.columns else []
    regions = sorted(intel["region"].dropna().unique().tolist()) if "region" in intel.columns else []

    with c1:
        cat_sel = st.multiselect("Category", categories, default=categories)
    with c2:
        ent_sel = st.multiselect("Entity", entities, default=entities)
    with c3:
        reg_sel = st.multiselect("Region", regions, default=regions)
    with c4:
        q = st.text_input("Search", placeholder="e.g., HVDC, GE Vernova, tender, regulation...")

    if categories:
        intel = intel[intel["category"].isin(cat_sel)]
    if entities:
        intel = intel[intel["entity"].isin(ent_sel)]
    if regions:
        intel = intel[intel["region"].isin(reg_sel)]

    if q.strip():
        s = q.strip().lower()
        text_cols = [c for c in ["headline", "description", "strategic_implication", "signal_type", "impact_area"] if c in intel.columns]
        mask = False
        for col in text_cols:
            mask = mask | intel[col].astype(str).str.lower().str.contains(s, na=False)
        intel = intel[mask]

    # Critical signals (Head of Strategy view)
    st.markdown("### 🚨 Critical signals (needs attention)")
    if "expected_impact" in intel.columns:
        crit = intel[(intel["expected_impact"] == "Negative") | (intel.get("action_required", "").astype(str) != "")]
    else:
        # fallback if old schema: show latest
        crit = intel.copy()

    crit = crit.sort_values("date_dt", ascending=False)

    show_cols = [c for c in [
        "date_utc", "category", "entity", "headline", "signal_type", "impact_area",
        "expected_impact", "confidence", "action_required", "source_url"
    ] if c in crit.columns]

    st.dataframe(crit[show_cols].head(30), use_container_width=True)

    st.divider()

    st.markdown("### 📈 Signal volume by category")
    if "category" in intel.columns and len(intel) > 0:
        st.bar_chart(intel["category"].value_counts())
    else:
        st.info("No category data available to chart.")

    st.markdown("### All signals")
    all_cols = [c for c in [
        "date_utc", "category", "entity", "headline", "region",
        "signal_type", "impact_area", "expected_impact", "confidence", "source_url"
    ] if c in intel.columns]
    st.dataframe(intel.sort_values("date_dt", ascending=False)[all_cols], use_container_width=True)


# ========================
# TAB 4: Strategy
# ========================
with tab_strategy:
    st.subheader("Strategic Direction (Management Signals)")

    if "theme" in strategy.columns:
        themes = ["All"] + sorted(strategy["theme"].dropna().unique().tolist())
        tsel = st.selectbox("Theme", themes)
    else:
        tsel = "All"

    search = st.text_input("Search strategy statements", placeholder="e.g., Grid Tech, HVDC, backlog, margin...")

    sview = strategy.copy()
    if tsel != "All" and "theme" in sview.columns:
        sview = sview[sview["theme"] == tsel]

    if search.strip():
        s = search.strip().lower()
        text_cols = [c for c in ["statement", "implication"] if c in sview.columns]
        mask = False
        for col in text_cols:
            mask = mask | sview[col].astype(str).str.lower().str.contains(s, na=False)
        sview = sview[mask]

    cols_to_show = [c for c in ["date", "theme", "statement", "implication"] if c in sview.columns]
    st.dataframe(sview[cols_to_show].sort_values("date", ascending=False), use_container_width=True)
