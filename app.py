import os
from urllib.parse import urlparse

import pandas as pd
import streamlit as st


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Siemens Energy – Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Siemens Energy – Strategy Dashboard")


# ---------------------------
# Governance configuration
# ---------------------------
ALLOWED_UNITS = {"EURm", "%", "ratio"}

# Siemens Energy KPIs must be backed by Siemens Energy Investor Relations sources.
# (Adjust allowed_path if your IR URLs use different path segments.)
ALLOWED_DOMAIN = "siemens-energy.com"
REQUIRED_PATH_TOKENS = ["investor-relations", "investor"]  # allow either token

# Strict governance: block Low confidence from being shown in KPI tiles
BLOCK_CONFIDENCE = {"Low"}

# Visual best practices we implement:
# - KPI tiles + deltas
# - RAG vs guidance band
# - sparklines-like trends via line charts per metric
# - "critical signals" intelligence section
# (RAG and executive dashboard principles are widely used in performance reporting) :contentReference[oaicite:1]{index=1}


def safe_dt(series):
    return pd.to_datetime(series, errors="coerce")


def fmt_value(value, unit):
    try:
        v = float(value)
    except Exception:
        return f"{value} {unit}".strip()

    u = str(unit).strip().lower()
    if u == "%":
        return f"{v:.1f}%"
    if u == "ratio":
        return f"{v:.2f}x"
    # EURm
    return f"{v:,.0f} {unit}".strip()


def fmt_delta(delta, unit):
    if delta is None or pd.isna(delta):
        return None
    try:
        d = float(delta)
    except Exception:
        return None

    u = str(unit).strip().lower()
    if u == "%":
        return f"{d:+.1f}%"
    if u == "ratio":
        return f"{d:+.2f}x"
    return f"{d:+,.0f} {unit}".strip()


def is_se_ir_url(url: str) -> bool:
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    try:
        u = urlparse(url)
        domain_ok = u.netloc.endswith(ALLOWED_DOMAIN)
        path_ok = any(tok in (u.path or "").lower() for tok in REQUIRED_PATH_TOKENS)
        return domain_ok and path_ok
    except Exception:
        return False


def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ {name} missing required columns: {missing}")
        st.stop()


def governance_validate_kpis(kpis: pd.DataFrame):
    # Required schema
    require_cols(
        kpis,
        ["company", "period", "period_end", "basis", "metric", "value", "unit", "source_ref", "confidence", "commentary"],
        "company_kpis.csv"
    )

    # Units
    bad_units = sorted(set(kpis["unit"].dropna()) - ALLOWED_UNITS)
    if bad_units:
        st.error(f"❌ Unit governance failed. Unexpected units: {bad_units}. Allowed: {sorted(ALLOWED_UNITS)}")
        st.stop()

    # Period_end parse
    kpis["period_end"] = safe_dt(kpis["period_end"])
    if kpis["period_end"].isna().any():
        st.error("❌ period_end contains invalid dates. Use YYYY-MM-DD (e.g., 2024-09-30).")
        st.stop()

    # IR-only source gate
    bad_src = kpis[~kpis["source_ref"].apply(is_se_ir_url)]
    if len(bad_src) > 0:
        st.error("❌ Source governance failed: KPI rows include non-Investor-Relations sources for Siemens Energy.")
        st.dataframe(bad_src[["period", "metric", "source_ref", "confidence"]], use_container_width=True)
        st.stop()

    # Confidence must be present (High/Medium/Low)
    # We don't block Medium, but we DO block Low for KPI tiles (below).
    return kpis


def governance_validate_guidance(guidance: pd.DataFrame):
    require_cols(guidance, ["company", "period", "metric", "unit", "low", "mid", "high", "source_ref", "confidence"], "guidance.csv")
    bad_units = sorted(set(guidance["unit"].dropna()) - ALLOWED_UNITS)
    if bad_units:
        st.error(f"❌ Guidance unit governance failed: {bad_units}")
        st.stop()
    return guidance


def pick_latest_intel_file():
    return "data/intelligence_live.csv" if os.path.exists("data/intelligence_live.csv") else "data/intelligence_2025.csv"


# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    kpis = pd.read_csv("data/company_kpis_2025.csv")
    guidance = pd.read_csv("data/guidance_2025.csv") if os.path.exists("data/guidance_2025.csv") else pd.DataFrame()
    peers = pd.read_csv("data/peer_kpis_2025.csv") if os.path.exists("data/peer_kpis_2025.csv") else pd.DataFrame()

    intel_path = pick_latest_intel_file()
    intelligence = pd.read_csv(intel_path) if os.path.exists(intel_path) else pd.DataFrame()

    strategy = pd.read_csv("data/strategy_2025.csv") if os.path.exists("data/strategy_2025.csv") else pd.DataFrame()
    return kpis, guidance, peers, intelligence, intel_path, strategy


kpis, guidance, peers, intelligence, intel_path, strategy = load_data()


# ---------------------------
# Governance checks (hard)
# ---------------------------
kpis = governance_validate_kpis(kpis)

if not guidance.empty:
    guidance = governance_validate_guidance(guidance)

# Normalize intelligence if present
if not intelligence.empty:
    if "date_utc" in intelligence.columns:
        intelligence["date_dt"] = safe_dt(intelligence["date_utc"])
    elif "date" in intelligence.columns:
        intelligence["date_dt"] = safe_dt(intelligence["date"])
    else:
        intelligence["date_dt"] = pd.NaT


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")

# Yearly basis: periods like 2024_FY, 2025_FY
periods = sorted(kpis["period"].dropna().unique().tolist())
selected_period = st.sidebar.selectbox("Reporting period (yearly)", periods, index=len(periods) - 1)

# View filters
lookback_days = st.sidebar.slider("Intel lookback (days)", 7, 120, 30, 1)

# Show/hide medium confidence in tiles (Low is always blocked)
show_medium = st.sidebar.checkbox("Include Medium confidence in KPI tiles", value=True)

allowed_conf_for_tiles = {"High"} | ({"Medium"} if show_medium else set())

# Company fixed to Siemens Energy for KPI file; if you expand later, keep it filterable.
company = "Siemens Energy"


# ---------------------------
# Prepare views
# ---------------------------
kpis_view = kpis[(kpis["company"] == company) & (kpis["period"] == selected_period)].copy()
kpis_view = kpis_view.sort_values("metric")

# Block Low confidence from KPI tiles (strict)
blocked = kpis_view[kpis_view["confidence"].isin(BLOCK_CONFIDENCE)].copy()
kpis_tiles = kpis_view[kpis_view["confidence"].isin(allowed_conf_for_tiles)].copy()

# Snapshot date
snapshot_end = kpis_view["period_end"].max()

# Previous period for deltas (yearly -> previous FY)
prev_candidates = kpis[(kpis["company"] == company) & (kpis["period_end"] < snapshot_end)].copy()
prev_last = prev_candidates.sort_values("period_end").groupby("metric").tail(1)
prev_map = dict(zip(prev_last["metric"], prev_last["value"]))


# ---------------------------
# Tabs
# ---------------------------
tab_perf, tab_guidance, tab_intel, tab_strategy = st.tabs(
    ["📊 Performance", "🏁 Guidance", "🧠 Strategic Intelligence", "🧭 Strategy"]
)

# ===========================
# TAB: PERFORMANCE
# ===========================
with tab_perf:
    st.subheader("Executive Snapshot (Decision-grade)")

    # Basis banner
    st.caption(
        f"Company: **{company}**  |  Reporting basis: **{selected_period}**  |  Period end: **{snapshot_end.date()}**"
    )

    # If any KPIs are blocked, make it loud
    if len(blocked) > 0:
        st.error("Blocked KPIs: **Low confidence** items are not shown in the KPI tiles. Fix source_ref / confidence to use them.")
        st.dataframe(blocked[["metric", "value", "unit", "confidence", "source_ref"]], use_container_width=True)

    # KPI tiles (3 columns)
    cards = kpis_tiles.to_dict(orient="records")
    cols = st.columns(3)

    for i, row in enumerate(cards):
        metric = row["metric"]
        unit = row["unit"]
        value = row["value"]
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
                help=row.get("commentary", "")
            )

    st.divider()

    # Trend section (multi-year yearly = 2 points now; grows as you add more years)
    st.subheader("Trends (FY time series)")

    metric_selected = st.selectbox(
        "Select KPI for trend",
        sorted(kpis[kpis["company"] == company]["metric"].unique().tolist()),
    )

    ts = kpis[(kpis["company"] == company) & (kpis["metric"] == metric_selected)].copy()
    ts = ts.sort_values("period_end")

    # Show small table + chart
    st.line_chart(ts.set_index("period_end")["value"])
    st.dataframe(
        ts[["period", "period_end", "basis", "value", "unit", "confidence", "source_ref", "commentary"]],
        use_container_width=True
    )


# ===========================
# TAB: GUIDANCE (RAG)
# ===========================
with tab_guidance:
    st.subheader("Guidance vs Actual (RAG)")

    if guidance.empty:
        st.info("No guidance file found. Add data/guidance_2025.csv to enable this view.")
    else:
        g = guidance[(guidance["company"] == company) & (guidance["period"] == selected_period)].copy()
        if len(g) == 0:
            st.warning("No guidance rows match the selected period.")
        else:
            # Join
            a = kpis_view[["metric", "value", "unit", "confidence"]].copy()
            merged = a.merge(g, on=["metric", "unit"], how="left", suffixes=("_act", "_g"))

            # Block Low confidence actuals from guidance evaluation too
            merged_eval = merged[~merged["confidence"].isin(BLOCK_CONFIDENCE)].copy()

            # Compute status vs band
            merged_eval["value_num"] = pd.to_numeric(merged_eval["value"], errors="coerce")
            merged_eval["low_num"] = pd.to_numeric(merged_eval["low"], errors="coerce")
            merged_eval["mid_num"] = pd.to_numeric(merged_eval["mid"], errors="coerce")
            merged_eval["high_num"] = pd.to_numeric(merged_eval["high"], errors="coerce")

            def rag(row):
                v, lo, hi = row["value_num"], row["low_num"], row["high_num"]
                if pd.isna(v) or pd.isna(lo) or pd.isna(hi):
                    return "Grey"
                if lo <= v <= hi:
                    return "Green"
                # If close to band (within 5% of band), amber; else red
                band = max(abs(hi - lo), 1e-9)
                dist = min(abs(v - lo), abs(v - hi))
                return "Amber" if dist <= 0.05 * band else "Red"

            merged_eval["RAG"] = merged_eval.apply(rag, axis=1)
            merged_eval["delta_vs_mid"] = merged_eval["value_num"] - merged_eval["mid_num"]

            # Tiles with RAG labels
            cols = st.columns(3)
            for i, row in merged_eval.iterrows():
                with cols[i % 3]:
                    st.metric(
                        row["metric"],
                        fmt_value(row["value_num"], row["unit"]),
                        delta=fmt_delta(row["delta_vs_mid"], row["unit"]),
                    )
                    rag_val = row["RAG"]
                    if rag_val == "Green":
                        st.success("On track (within guidance band)")
                    elif rag_val == "Amber":
                        st.warning("Watch (near guidance band)")
                    elif rag_val == "Red":
                        st.error("Off track (outside guidance band)")
                    else:
                        st.info("No band / insufficient data")

            st.divider()
            st.subheader("Guidance table (audit)")
            st.dataframe(
                merged_eval[[
                    "metric", "unit", "value_num", "low_num", "mid_num", "high_num",
                    "delta_vs_mid", "RAG", "confidence", "source_ref_g", "confidence_g"
                ]].rename(columns={
                    "value_num": "actual",
                    "low_num": "guid_low",
                    "mid_num": "guid_mid",
                    "high_num": "guid_high",
                    "source_ref_g": "guidance_source",
                    "confidence_g": "guidance_confidence"
                }),
                use_container_width=True
            )


# ===========================
# TAB: STRATEGIC INTELLIGENCE
# ===========================
with tab_intel:
    st.subheader("Strategic Intelligence (Signals, not noise)")
    if intelligence.empty:
        st.info("No intelligence file found. Use the daily collector to create data/intelligence_live.csv.")
    else:
        st.caption(f"Feed source: **{intel_path}**")

        # Governance expectation: categories
        required_cols = ["category", "entity"]
        for c in required_cols:
            if c not in intelligence.columns:
                st.error(f"Intelligence file missing required column: {c}")
                st.stop()

        # Lookback filter
        if "date_dt" in intelligence.columns and intelligence["date_dt"].notna().any():
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
            intel = intelligence[intelligence["date_dt"] >= cutoff].copy()
        else:
            intel = intelligence.copy()

        # Filters
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        cats = sorted(intel["category"].dropna().unique().tolist()) if "category" in intel.columns else []
        ents = sorted(intel["entity"].dropna().unique().tolist()) if "entity" in intel.columns else []
        regs = sorted(intel["region"].dropna().unique().tolist()) if "region" in intel.columns else []

        with c1:
            cat_sel = st.multiselect("Category", cats, default=cats)
        with c2:
            ent_sel = st.multiselect("Entity", ents, default=ents)
        with c3:
            reg_sel = st.multiselect("Region", regs, default=regs)
        with c4:
            q = st.text_input("Search", placeholder="HVDC, GE Vernova, tender, regulation...")

        if cats:
            intel = intel[intel["category"].isin(cat_sel)]
        if ents:
            intel = intel[intel["entity"].isin(ent_sel)]
        if regs and "region" in intel.columns:
            intel = intel[intel["region"].isin(reg_sel)]

        if q.strip():
            s = q.strip().lower()
            text_cols = [c for c in ["headline", "description", "signal_type", "impact_area", "strategic_implication"] if c in intel.columns]
            mask = False
            for col in text_cols:
                mask = mask | intel[col].astype(str).str.lower().str.contains(s, na=False)
            intel = intel[mask]

        # Top line charts
        left, right = st.columns([1, 1])
        with left:
            st.caption("Signal volume by category")
            st.bar_chart(intel["category"].value_counts())
        with right:
            if "expected_impact" in intel.columns:
                st.caption("Impact mix")
                st.bar_chart(intel["expected_impact"].value_counts())

        st.divider()

        # Critical signals (Head of Strategy default)
        st.markdown("### 🚨 Critical signals (requires attention)")
        if "expected_impact" in intel.columns:
            crit = intel[(intel["expected_impact"] == "Negative") | (intel.get("action_required", "").astype(str).str.strip() != "")]
        else:
            crit = intel.copy()

        sort_col = "date_dt" if "date_dt" in crit.columns else None
        if sort_col:
            crit = crit.sort_values(sort_col, ascending=False)

        show_cols = [c for c in [
            "date_utc", "category", "entity", "headline", "signal_type", "impact_area",
            "expected_impact", "confidence", "action_required", "source_url"
        ] if c in crit.columns]
        st.dataframe(crit[show_cols].head(40), use_container_width=True)

        st.divider()

        st.markdown("### All signals (filtered)")
        all_cols = [c for c in [
            "date_utc", "category", "entity", "headline", "region",
            "signal_type", "impact_area", "expected_impact", "confidence", "source_url"
        ] if c in intel.columns]
        if sort_col:
            intel = intel.sort_values(sort_col, ascending=False)
        st.dataframe(intel[all_cols], use_container_width=True)


# ===========================
# TAB: STRATEGY
# ===========================
with tab_strategy:
    st.subheader("Management Signals & Implications")

    if strategy.empty:
        st.info("No strategy file found. Add data/strategy_2025.csv.")
    else:
        # Basic filters
        if "theme" in strategy.columns:
            themes = ["All"] + sorted(strategy["theme"].dropna().unique().tolist())
            theme_sel = st.selectbox("Theme", themes)
        else:
            theme_sel = "All"

        search = st.text_input("Search", placeholder="Grid Technologies, HVDC, margin, backlog...")

        sview = strategy.copy()
        if theme_sel != "All" and "theme" in sview.columns:
            sview = sview[sview["theme"] == theme_sel]

        if search.strip():
            s = search.strip().lower()
            cols = [c for c in ["statement", "implication"] if c in sview.columns]
            mask = False
            for c in cols:
                mask = mask | sview[c].astype(str).str.lower().str.contains(s, na=False)
            sview = sview[mask]

        # Show most recent first
        if "date" in sview.columns:
            sview["date"] = safe_dt(sview["date"])
            sview = sview.sort_values("date", ascending=False)

        show_cols = [c for c in ["date", "theme", "statement", "implication"] if c in sview.columns]
        st.dataframe(sview[show_cols], use_container_width=True)
