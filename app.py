import os
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

# ---------------------------
# Page setup
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
BLOCK_CONFIDENCE = {"Low"}  # strict: never show Low in tiles/charts

SIEMENS_IR_ROOT = "https://www.siemens-energy.com/global/en/home/investor-relations.html"
GEV_IR_ROOT = "https://www.gevernova.com/investors"
SCHNEIDER_IR_ROOT = "https://www.se.com/ww/en/about-us/investor-relations/"

# Strict domain allowlist per company
ALLOWED_DOMAINS_BY_COMPANY = {
    "Siemens Energy": ["siemens-energy.com"],
    "GE Vernova": ["gevernova.com"],
    "Schneider Electric": ["se.com"],
}

# Optional: require path token for Siemens specifically (keeps it tight)
REQUIRED_PATH_TOKENS_BY_COMPANY = {
    "Siemens Energy": ["investor-relations", "investor"],
    "GE Vernova": ["investor"],  # keep looser; site varies
    "Schneider Electric": ["investor-relations", "investor"],
}


# ---------------------------
# Helpers
# ---------------------------
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


def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ {name} missing required columns: {missing}")
        st.stop()


def url_allowed_for_company(url: str, company: str) -> bool:
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    try:
        u = urlparse(url)
        domains = ALLOWED_DOMAINS_BY_COMPANY.get(company, [])
        domain_ok = any(u.netloc.endswith(d) for d in domains)

        tokens = REQUIRED_PATH_TOKENS_BY_COMPANY.get(company, [])
        path = (u.path or "").lower()
        path_ok = True if not tokens else any(t in path for t in tokens)

        return domain_ok and path_ok
    except Exception:
        return False


def apply_metric_mapping(df: pd.DataFrame, mapping: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    Optional: map raw/source metric names to canonical metric names.
    mapping columns: source_metric, canonical_metric
    """
    if mapping is None or mapping.empty:
        return df
    require_cols(mapping, ["source_metric", "canonical_metric"], "metric_map.csv")
    if "metric" not in df.columns:
        st.error(f"{df_name} missing column 'metric' required for mapping.")
        st.stop()

    m = dict(zip(mapping["source_metric"].astype(str), mapping["canonical_metric"].astype(str)))
    df["metric"] = df["metric"].astype(str).map(lambda x: m.get(x, x))
    return df


def governance_validate_kpis(df: pd.DataFrame, df_name: str):
    require_cols(
        df,
        ["company", "period", "period_end", "basis", "metric", "value", "unit", "source_ref", "confidence", "commentary"],
        df_name
    )

    # unit governance
    bad_units = sorted(set(df["unit"].dropna()) - ALLOWED_UNITS)
    if bad_units:
        st.error(f"❌ Unit governance failed in {df_name}. Unexpected units: {bad_units}. Allowed: {sorted(ALLOWED_UNITS)}")
        st.stop()

    # period_end must parse
    df["period_end"] = safe_dt(df["period_end"])
    if df["period_end"].isna().any():
        st.error(f"❌ {df_name}: period_end has invalid dates. Use YYYY-MM-DD (e.g., 2024-09-30).")
        st.stop()

    # source_ref per company allowlist
    bad_rows = []
    for comp in df["company"].dropna().unique():
        sub = df[df["company"] == comp]
        bad = sub[~sub["source_ref"].apply(lambda u: url_allowed_for_company(u, comp))]
        if len(bad) > 0:
            bad_rows.append(bad[["company", "period", "metric", "source_ref", "confidence"]])

    if bad_rows:
        st.error(f"❌ Source governance failed in {df_name}: some rows are not from approved Investor Relations domains/paths.")
        st.dataframe(pd.concat(bad_rows, ignore_index=True), use_container_width=True)
        st.stop()

    return df


def governance_validate_guidance(df: pd.DataFrame, df_name: str):
    require_cols(
        df,
        ["company", "period", "metric", "unit", "low", "mid", "high", "source_ref", "confidence"],
        df_name
    )
    bad_units = sorted(set(df["unit"].dropna()) - ALLOWED_UNITS)
    if bad_units:
        st.error(f"❌ Unit governance failed in {df_name}. Unexpected units: {bad_units}.")
        st.stop()

    # source_ref per company allowlist
    bad_rows = []
    for comp in df["company"].dropna().unique():
        sub = df[df["company"] == comp]
        bad = sub[~sub["source_ref"].apply(lambda u: url_allowed_for_company(u, comp))]
        if len(bad) > 0:
            bad_rows.append(bad[["company", "period", "metric", "source_ref", "confidence"]])

    if bad_rows:
        st.error(f"❌ Source governance failed in {df_name}: some rows are not from approved IR domains/paths.")
        st.dataframe(pd.concat(bad_rows, ignore_index=True), use_container_width=True)
        st.stop()

    return df


def load_intelligence():
    # prefer live feed if you set up daily job, else fallback
    p = "data/intelligence_live.csv"
    if os.path.exists(p):
        return pd.read_csv(p), p
    p2 = "data/intelligence_2025.csv"
    if os.path.exists(p2):
        return pd.read_csv(p2), p2
    return pd.DataFrame(), ""


# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    kpis = pd.read_csv("data/company_kpis_2025.csv")
    guidance = pd.read_csv("data/guidance_2025.csv") if os.path.exists("data/guidance_2025.csv") else pd.DataFrame()
    peers = pd.read_csv("data/peer_kpis_2025.csv") if os.path.exists("data/peer_kpis_2025.csv") else pd.DataFrame()
    strategy = pd.read_csv("data/strategy_2025.csv") if os.path.exists("data/strategy_2025.csv") else pd.DataFrame()

    metric_map = pd.read_csv("data/metric_map.csv") if os.path.exists("data/metric_map.csv") else pd.DataFrame()
    intelligence, intel_path = load_intelligence()

    return kpis, guidance, peers, intelligence, intel_path, strategy, metric_map


kpis, guidance, peers, intelligence, intel_path, strategy, metric_map = load_data()

# Apply mapping (optional)
kpis = apply_metric_mapping(kpis, metric_map, "company_kpis_2025.csv")
if not peers.empty:
    peers = apply_metric_mapping(peers, metric_map, "peer_kpis_2025.csv")
if not guidance.empty:
    guidance = apply_metric_mapping(guidance, metric_map, "guidance_2025.csv")

# Governance validation (hard stops)
kpis = governance_validate_kpis(kpis, "company_kpis_2025.csv")
if not peers.empty:
    peers = governance_validate_kpis(peers, "peer_kpis_2025.csv")
if not guidance.empty:
    guidance = governance_validate_guidance(guidance, "guidance_2025.csv")

# Normalize intelligence dates if present
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

company = "Siemens Energy"

# Yearly periods only
periods = sorted(kpis[kpis["company"] == company]["period"].dropna().unique().tolist())
if not periods:
    st.error("No periods found for Siemens Energy in company_kpis_2025.csv.")
    st.stop()

selected_period = st.sidebar.selectbox("Reporting period (FY)", periods, index=len(periods) - 1)

# Confidence toggle: allow Medium to show (Low always blocked)
show_medium = st.sidebar.checkbox("Include Medium confidence", value=True)
allowed_conf = {"High"} | ({"Medium"} if show_medium else set())

# Intel lookback
lookback_days = st.sidebar.slider("Intel lookback (days)", 7, 120, 30, 1)

with st.sidebar.expander("Source roots (reference)", expanded=False):
    st.write(f"Siemens Energy IR: {SIEMENS_IR_ROOT}")
    st.write(f"GE Vernova IR: {GEV_IR_ROOT}")
    st.write(f"Schneider IR: {SCHNEIDER_IR_ROOT}")

# ---------------------------
# Views: Siemens KPIs for selected period
# ---------------------------
kpis_view_all = kpis[(kpis["company"] == company) & (kpis["period"] == selected_period)].copy()
kpis_view_all = kpis_view_all.sort_values("metric")

blocked = kpis_view_all[kpis_view_all["confidence"].isin(BLOCK_CONFIDENCE)].copy()
kpis_tiles = kpis_view_all[kpis_view_all["confidence"].isin(allowed_conf)].copy()

snapshot_end = kpis_view_all["period_end"].max()

# Previous FY values for delta (YoY)
prev_candidates = kpis[(kpis["company"] == company) & (kpis["period_end"] < snapshot_end)].copy()
prev_last = prev_candidates.sort_values("period_end").groupby("metric").tail(1)
prev_map = dict(zip(prev_last["metric"], prev_last["value"]))

# ---------------------------
# Tabs
# ---------------------------
tab_perf, tab_peers, tab_intel, tab_strategy = st.tabs(
    ["📊 Performance", "🏁 Peers & Guidance", "🧠 Strategic Intelligence", "🧭 Strategy"]
)

# ===========================
# TAB: PERFORMANCE
# ===========================
with tab_perf:
    st.subheader("Executive Snapshot (Decision-driven)")

    st.caption(
        f"Company: **{company}**  |  Period: **{selected_period}**  |  Period end: **{snapshot_end.date()}**  |  Display confidence: **{sorted(allowed_conf)}**"
    )

    if len(blocked) > 0:
        st.error("Blocked KPIs: **Low confidence** items are excluded from tiles and charts. Fix them before using in decisions.")
        st.dataframe(blocked[["metric", "value", "unit", "confidence", "source_ref"]], use_container_width=True)

    # KPI tiles
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

    st.subheader("Trends (FY series)")

    metric_selected = st.selectbox(
        "Select KPI",
        sorted(kpis[kpis["company"] == company]["metric"].unique().tolist()),
    )

    ts = kpis[(kpis["company"] == company) & (kpis["metric"] == metric_selected)].copy()
    ts = ts[ts["confidence"].isin(allowed_conf)].sort_values("period_end")

    if len(ts) == 0:
        st.warning("No High/Medium confidence time-series points for this metric.")
    else:
        st.line_chart(ts.set_index("period_end")["value"])

    st.dataframe(
        ts[["period", "period_end", "basis", "value", "unit", "confidence", "source_ref", "commentary"]],
        use_container_width=True
    )

# ===========================
# TAB: PEERS & GUIDANCE
# ===========================
with tab_peers:
    st.subheader("Peers (GE Vernova & Schneider Electric) + Guidance")

    # --- Guidance (Siemens only) ---
    st.markdown("### Siemens Energy: Guidance vs Actual (RAG)")
    if guidance.empty:
        st.info("No guidance_2025.csv found. Add it to enable RAG guidance view.")
    else:
        g = guidance[(guidance["company"] == company) & (guidance["period"] == selected_period)].copy()
        if len(g) == 0:
            st.warning("No Siemens guidance rows match the selected period.")
        else:
            a = kpis_view_all[["metric", "value", "unit", "confidence"]].copy()
            a = a[a["confidence"].isin(allowed_conf)]

            merged = a.merge(g, on=["metric", "unit"], how="left", suffixes=("_act", "_g"))
            for c in ["value", "low", "mid", "high"]:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")

            def rag(row):
                v, lo, hi = row["value"], row["low"], row["high"]
                if pd.isna(v) or pd.isna(lo) or pd.isna(hi):
                    return "Grey"
                if lo <= v <= hi:
                    return "Green"
                band = max(abs(hi - lo), 1e-9)
                dist = min(abs(v - lo), abs(v - hi))
                return "Amber" if dist <= 0.05 * band else "Red"

            merged["RAG"] = merged.apply(rag, axis=1)
            merged["delta_vs_mid"] = merged["value"] - merged["mid"]

            cols = st.columns(3)
            for i, row in merged.iterrows():
                with cols[i % 3]:
                    st.metric(row["metric"], fmt_value(row["value"], row["unit"]), fmt_delta(row["delta_vs_mid"], row["unit"]))
                    if row["RAG"] == "Green":
                        st.success("On track")
                    elif row["RAG"] == "Amber":
                        st.warning("Watch")
                    elif row["RAG"] == "Red":
                        st.error("Off track")
                    else:
                        st.info("No guidance band")

            st.dataframe(
                merged[["metric", "unit", "value", "low", "mid", "high", "delta_vs_mid", "RAG", "source_ref_g", "confidence_g"]]
                .rename(columns={"value": "actual", "source_ref_g": "guidance_source", "confidence_g": "guidance_confidence"}),
                use_container_width=True
            )

    st.divider()

    # --- Peers compare ---
    st.markdown("### Peer comparison (same KPI, same unit, same FY)")

    # Keep only two competitors
    peer_universe = ["GE Vernova", "Schneider Electric"]

    # Filter peer data for selected period and confidence (exclude Low)
    p = peers[(peers["company"].isin(peer_universe)) & (peers["period"] == selected_period)].copy()
    p = p[p["confidence"].isin(allowed_conf)]

    # Siemens point for same period
    se = kpis_view_all.copy()
    se = se[se["confidence"].isin(allowed_conf)]

    # KPI selection (intersection)
    common_metrics = sorted(set(se["metric"].unique()).intersection(set(p["metric"].unique()))) if len(p) > 0 else sorted(se["metric"].unique())
    peer_metric = st.selectbox("KPI", common_metrics)

    # Unit must match
    se_unit = se[se["metric"] == peer_metric]["unit"].iloc[0] if len(se[se["metric"] == peer_metric]) else None
    if se_unit is None:
        st.warning("Selected KPI not available for Siemens Energy in this period.")
        st.stop()

    # Build comparison table
    rows = []
    # Siemens
    se_val = se[se["metric"] == peer_metric]["value"].iloc[0] if len(se[se["metric"] == peer_metric]) else None
    rows.append({"company": "Siemens Energy", "value": se_val, "unit": se_unit})

    # Peers
    p2 = p[(p["metric"] == peer_metric) & (p["unit"] == se_unit)].copy()
    for _, r in p2.iterrows():
        rows.append({"company": r["company"], "value": r["value"], "unit": r["unit"]})

    comp = pd.DataFrame(rows)
    comp["value_num"] = pd.to_numeric(comp["value"], errors="coerce")
    comp = comp.dropna(subset=["value_num"]).sort_values("value_num", ascending=False)

    if len(comp) == 0:
        st.warning("No High/Medium confidence peer values for this KPI + unit.")
    else:
        st.bar_chart(comp.set_index("company")["value_num"])
        st.dataframe(comp[["company", "value_num", "unit"]].rename(columns={"value_num": "value"}), use_container_width=True)

# ===========================
# TAB: STRATEGIC INTELLIGENCE
# ===========================
with tab_intel:
    st.subheader("Strategic Intelligence (Signals)")

    if intelligence.empty:
        st.info("No intelligence file found. Add intelligence_2025.csv or set up intelligence_live.csv (daily).")
    else:
        if intel_path:
            st.caption(f"Feed: **{intel_path}**")

        # Lookback
        if "date_dt" in intelligence.columns and intelligence["date_dt"].notna().any():
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
            intel = intelligence[intelligence["date_dt"] >= cutoff].copy()
        else:
            intel = intelligence.copy()

        # Required columns for your intelligence governance
        # Ideal columns: date_utc, category, entity, headline, region, signal_type, impact_area, expected_impact, confidence, action_required, source_url
        # We'll degrade gracefully if some are missing.
        for needed in ["category", "entity"]:
            if needed not in intel.columns:
                st.error(f"Intelligence file is missing required column: {needed}")
                st.stop()

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
            cols = [c for c in ["headline", "description", "signal_type", "impact_area", "strategic_implication"] if c in intel.columns]
            mask = False
            for c in cols:
                mask = mask | intel[c].astype(str).str.lower().str.contains(s, na=False)
            intel = intel[mask]

        # Category volume chart
        left, right = st.columns([1, 1])
        with left:
            st.caption("Signal volume by category")
            st.bar_chart(intel["category"].value_counts())

        with right:
            if "expected_impact" in intel.columns:
                st.caption("Impact mix")
                st.bar_chart(intel["expected_impact"].value_counts())

        st.divider()

        # Critical signals
        st.markdown("### 🚨 Critical signals (attention)")
        if "expected_impact" in intel.columns:
            crit = intel[(intel["expected_impact"] == "Negative") | (intel.get("action_required", "").astype(str).str.strip() != "")]
        else:
            crit = intel.copy()

        sort_col = "date_dt" if "date_dt" in crit.columns else None
        if sort_col:
            crit = crit.sort_values(sort_col, ascending=False)

        show_cols = [c for c in [
            "date_utc", "category", "entity", "headline", "region",
            "signal_type", "impact_area", "expected_impact", "confidence", "action_required", "source_url"
        ] if c in crit.columns]
        st.dataframe(crit[show_cols].head(50), use_container_width=True)

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
    st.subheader("Strategy (Management signals & implications)")

    if strategy.empty:
        st.info("No strategy_2025.csv found.")
    else:
        if "theme" in strategy.columns:
            themes = ["All"] + sorted(strategy["theme"].dropna().unique().tolist())
            theme_sel = st.selectbox("Theme", themes)
        else:
            theme_sel = "All"

        search = st.text_input("Search", placeholder="Grid, HVDC, margin, backlog, restructuring...")

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

        if "date" in sview.columns:
            sview["date"] = safe_dt(sview["date"])
            sview = sview.sort_values("date", ascending=False)

        show_cols = [c for c in ["date", "theme", "statement", "implication"] if c in sview.columns]
        st.dataframe(sview[show_cols], use_container_width=True)
