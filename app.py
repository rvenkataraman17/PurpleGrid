import os
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Siemens Energy – Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Siemens Energy – Strategy Dashboard")

# ---------------------------
# Governance config
# ---------------------------
ALLOWED_UNITS = {"EURm", "%", "ratio"}
BLOCK_CONFIDENCE = {"Low"}

IR_ROOTS = {
    "Siemens Energy": "https://www.siemens-energy.com/global/en/home/investor-relations.html",
    "GE Vernova": "https://www.gevernova.com/investors",
    "Schneider Electric": "https://www.se.com/ww/en/about-us/investor-relations/",
}

ALLOWED_DOMAINS = {
    "Siemens Energy": ["siemens-energy.com"],
    "GE Vernova": ["gevernova.com"],
    "Schneider Electric": ["se.com"],
}

REQUIRED_PATH_TOKENS = {
    "Siemens Energy": ["investor", "investor-relations", "press-releases", "earnings-release"],
    "GE Vernova": ["invest", "investor", "news", "press"],
    "Schneider Electric": ["investor", "investor-relations", "assets", "document", "release"],
}


def safe_dt(series):
    """
    Parse dates robustly and normalize to timezone-naive UTC timestamps.
    This avoids pandas errors when comparing tz-aware and tz-naive datetimes.
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # Convert tz-aware UTC -> tz-naive (still UTC clock time)
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        pass
    return s



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


def url_allowed(company: str, url: str) -> bool:
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    try:
        u = urlparse(url)
        domain_ok = any(u.netloc.endswith(d) for d in ALLOWED_DOMAINS.get(company, []))
        path = (u.path or "").lower()
        token_ok = any(t in path for t in REQUIRED_PATH_TOKENS.get(company, []))
        return domain_ok and token_ok
    except Exception:
        return False


def apply_metric_mapping(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping is None or mapping.empty:
        return df
    require_cols(mapping, ["source_metric", "canonical_metric"], "metric_map.csv")
    m = dict(zip(mapping["source_metric"].astype(str), mapping["canonical_metric"].astype(str)))
    df["metric"] = df["metric"].astype(str).map(lambda x: m.get(x, x))
    return df


def governance_validate_kpis(df: pd.DataFrame, name: str):
    require_cols(df, ["company","period","period_end","basis","metric","value","unit","source_ref","confidence","commentary"], name)

    bad_units = sorted(set(df["unit"].dropna()) - ALLOWED_UNITS)
    if bad_units:
        st.error(f"❌ Unit governance failed in {name}. Unexpected units: {bad_units}")
        st.stop()

    df["period_end"] = safe_dt(df["period_end"])
    if df["period_end"].isna().any():
        st.error(f"❌ {name}: period_end has invalid dates. Use YYYY-MM-DD.")
        st.stop()

    bad = df[~df.apply(lambda r: url_allowed(r["company"], r["source_ref"]), axis=1)]
    if len(bad) > 0:
        st.error(f"❌ Source governance failed in {name}: some rows not from approved IR sources.")
        st.dataframe(bad[["company","period","metric","source_ref","confidence"]], use_container_width=True)
        st.stop()

    return df


def governance_validate_market(df: pd.DataFrame, name: str):
    require_cols(df, ["company","date","close","currency","source_ref","confidence"], name)
    df["date"] = safe_dt(df["date"])
    return df


@st.cache_data
def load_data():
    kpis = pd.read_csv("data/company_kpis_2025.csv")
    peers = pd.read_csv("data/peer_kpis_2025.csv") if os.path.exists("data/peer_kpis_2025.csv") else pd.DataFrame()
    guidance = pd.read_csv("data/guidance_2025.csv") if os.path.exists("data/guidance_2025.csv") else pd.DataFrame()
    market = pd.read_csv("data/market_data.csv") if os.path.exists("data/market_data.csv") else pd.DataFrame()
    metric_map = pd.read_csv("data/metric_map.csv") if os.path.exists("data/metric_map.csv") else pd.DataFrame()

    intel_path = "data/intelligence_live.csv" if os.path.exists("data/intelligence_live.csv") else "data/intelligence_2025.csv"
    intelligence = pd.read_csv(intel_path) if os.path.exists(intel_path) else pd.DataFrame()

    strategy = pd.read_csv("data/strategy_2025.csv") if os.path.exists("data/strategy_2025.csv") else pd.DataFrame()
    return kpis, peers, guidance, market, intelligence, intel_path, strategy, metric_map


kpis, peers, guidance, market, intelligence, intel_path, strategy, metric_map = load_data()

kpis = apply_metric_mapping(kpis, metric_map)
if not peers.empty:
    peers = apply_metric_mapping(peers, metric_map)
if not guidance.empty:
    guidance = apply_metric_mapping(guidance, metric_map)

kpis = governance_validate_kpis(kpis, "company_kpis_2025.csv")
if not peers.empty:
    peers = governance_validate_kpis(peers, "peer_kpis_2025.csv")
if not market.empty:
    market = governance_validate_market(market, "market_data.csv")

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
periods = sorted(kpis[(kpis["company"] == company)]["period"].dropna().unique().tolist())

if not periods:
    st.error("No reporting periods found for 'Siemens Energy' in data/company_kpis_2025.csv. Add rows with company='Siemens Energy' and a period like '2025_FY'.")
    st.stop()

selected_period = st.sidebar.selectbox("Reporting period (FY)", periods, index=len(periods)-1)


show_medium = st.sidebar.checkbox("Include Medium confidence", value=True)
allowed_conf = {"High"} | ({"Medium"} if show_medium else set())
intel_days = st.sidebar.slider("Intel lookback (days)", 7, 120, 30, 1)

with st.sidebar.expander("Approved sources", expanded=False):
    for k, v in IR_ROOTS.items():
        st.write(f"{k}: {v}")

tab_perf, tab_peers, tab_intel, tab_strategy = st.tabs(
    ["📊 Performance", "🏁 Peers & Market", "🧠 Strategic Intelligence", "🧭 Strategy"]
)

# ===========================
# PERFORMANCE
# ===========================
with tab_perf:
    sv_all = kpis[(kpis["company"] == company) & (kpis["period"] == selected_period)].copy().sort_values("metric")
    blocked = sv_all[sv_all["confidence"].isin(BLOCK_CONFIDENCE)]
    sv = sv_all[sv_all["confidence"].isin(allowed_conf)]

    snapshot_end = sv_all["period_end"].max()
    prev = kpis[(kpis["company"] == company) & (kpis["period_end"] < snapshot_end)]
    prev_last = prev.sort_values("period_end").groupby("metric").tail(1)
    prev_map = dict(zip(prev_last["metric"], prev_last["value"]))

    st.subheader("Executive Snapshot")
    st.caption(f"Company: **{company}** | Period: **{selected_period}** | Period end: **{snapshot_end.date()}**")

    if len(blocked) > 0:
        st.error("Blocked (Low confidence) KPIs are excluded. Fix extraction/source before using.")
        st.dataframe(blocked[["metric","value","unit","confidence","source_ref"]], use_container_width=True)

    cols = st.columns(3)
    for i, row in enumerate(sv.to_dict(orient="records")):
        metric, value, unit = row["metric"], row["value"], row["unit"]
        delta = None
        if metric in prev_map:
            try:
                delta = float(value) - float(prev_map[metric])
            except Exception:
                delta = None
        with cols[i % 3]:
            st.metric(metric, fmt_value(value, unit), fmt_delta(delta, unit), help=row.get("commentary", ""))

    st.divider()
    st.subheader("Trend")
    metric_selected = st.selectbox("Select KPI", sorted(kpis[kpis["company"] == company]["metric"].unique().tolist()))
    ts = kpis[(kpis["company"] == company) & (kpis["metric"] == metric_selected) & (kpis["confidence"].isin(allowed_conf))].sort_values("period_end")
    if len(ts) > 0:
        st.line_chart(ts.set_index("period_end")["value"])
    st.dataframe(ts[["period","period_end","basis","value","unit","confidence","source_ref","commentary"]], use_container_width=True)

# ===========================
# PEERS & MARKET
# ===========================
with tab_peers:
    st.subheader("Peers & Market")

    peer_universe = ["GE Vernova", "Schneider Electric"]

    # Market section
    st.markdown("### Market snapshot")
    if market.empty:
        st.info("market_data.csv not found yet. Add it + run scripts/fetch_market_data.py via your existing fetch_intel.yml.")
    else:
        market["date"] = pd.to_datetime(market["date"], errors="coerce")
        latest = market.sort_values("date").groupby("company").tail(1)
        latest = latest[latest["confidence"].isin(allowed_conf)]
        st.dataframe(latest[["company","date","close","currency","confidence","source_ref"]], use_container_width=True)

        st.markdown("### Share price trend")
        company_sel = st.selectbox("Company for share price trend", sorted(market["company"].dropna().unique().tolist()))
        m = market[market["company"] == company_sel].sort_values("date")
        if len(m) > 1:
            st.line_chart(m.set_index("date")[["close"]])
        else:
            st.info("Not enough history yet. After a few days, you'll see a trend.")

    st.divider()

    # Peer KPI comparison
    st.markdown("### Peer KPI comparison (same FY, same unit)")
    if peers.empty:
        st.info("peer_kpis_2025.csv is empty/missing. Populate via your daily job.")
    else:
        se = kpis[(kpis["company"] == company) & (kpis["period"] == selected_period) & (kpis["confidence"].isin(allowed_conf))].copy()
        p = peers[(peers["company"].isin(peer_universe)) & (peers["period"] == selected_period) & (peers["confidence"].isin(allowed_conf))].copy()

        common_metrics = sorted(set(se["metric"]).intersection(set(p["metric"]))) if len(p) else sorted(se["metric"].unique())
        peer_metric = st.selectbox("KPI to compare", common_metrics)

        se_row = se[se["metric"] == peer_metric]
        if len(se_row) == 0:
            st.warning("Siemens value missing for selected KPI (or blocked).")
        else:
            unit = se_row["unit"].iloc[0]
            rows = [{"company": "Siemens Energy", "value": float(se_row["value"].iloc[0]), "unit": unit}]
            p2 = p[(p["metric"] == peer_metric) & (p["unit"] == unit)]
            for _, r in p2.iterrows():
                rows.append({"company": r["company"], "value": float(r["value"]), "unit": r["unit"]})

            comp = pd.DataFrame(rows).dropna()
            if len(comp) > 0:
                st.bar_chart(comp.set_index("company")["value"])
                comp_show = comp.copy()
                comp_show["value"] = comp_show["value"].map(lambda v: fmt_value(v, unit))
                st.dataframe(comp_show[["company","value","unit"]], use_container_width=True)

# ===========================
# STRATEGIC INTELLIGENCE
# ===========================
with tab_intel:
    st.subheader("Strategic Intelligence (signals)")
    if intelligence.empty:
        st.info("No intelligence feed found (intelligence_live.csv or intelligence_2025.csv).")
    else:
        st.caption(f"Feed: **{intel_path}**")

        intel = intelligence.copy()
        if "date_dt" in intel.columns and intel["date_dt"].notna().any():
            cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=intel_days)).tz_convert(None)

if "date_dt" in intel.columns and intel["date_dt"].notna().any():
    intel = intel[intel["date_dt"] >= cutoff]


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
            q = st.text_input("Search", placeholder="HVDC, grid, tender, regulation...")

        if cats:
            intel = intel[intel["category"].isin(cat_sel)]
        if ents:
            intel = intel[intel["entity"].isin(ent_sel)]
        if regs and "region" in intel.columns:
            intel = intel[intel["region"].isin(reg_sel)]

        if q.strip():
            s = q.strip().lower()
            cols = [c for c in ["headline","description","signal_type","impact_area","strategic_implication"] if c in intel.columns]
            mask = False
            for c in cols:
                mask = mask | intel[c].astype(str).str.lower().str.contains(s, na=False)
            intel = intel[mask]

        left, right = st.columns([1, 1])
        with left:
            if "category" in intel.columns:
                st.caption("Volume by category")
                st.bar_chart(intel["category"].value_counts())
        with right:
            if "expected_impact" in intel.columns:
                st.caption("Impact mix")
                st.bar_chart(intel["expected_impact"].value_counts())

        st.divider()
        st.markdown("### 🚨 Critical signals")
        if "expected_impact" in intel.columns:
            crit = intel[(intel["expected_impact"] == "Negative") | (intel.get("action_required","").astype(str).str.strip() != "")]
        else:
            crit = intel.copy()

        if "date_dt" in crit.columns:
            crit = crit.sort_values("date_dt", ascending=False)

        cols_show = [c for c in ["date_utc","category","entity","headline","region","signal_type","impact_area","expected_impact","confidence","action_required","source_url"] if c in crit.columns]
        st.dataframe(crit[cols_show].head(50), use_container_width=True)

        st.divider()
        st.markdown("### All signals (filtered)")
        cols_all = [c for c in ["date_utc","category","entity","headline","region","signal_type","impact_area","expected_impact","confidence","source_url"] if c in intel.columns]
        if "date_dt" in intel.columns:
            intel = intel.sort_values("date_dt", ascending=False)
        st.dataframe(intel[cols_all], use_container_width=True)

# ===========================
# STRATEGY
# ===========================
with tab_strategy:
    st.subheader("Strategy (management signals)")
    if strategy.empty:
        st.info("No strategy_2025.csv found.")
    else:
        if "theme" in strategy.columns:
            themes = ["All"] + sorted(strategy["theme"].dropna().unique().tolist())
            tsel = st.selectbox("Theme", themes)
        else:
            tsel = "All"
        search = st.text_input("Search", placeholder="Grid, HVDC, margin, backlog...")

        sview = strategy.copy()
        if tsel != "All" and "theme" in sview.columns:
            sview = sview[sview["theme"] == tsel]
        if search.strip():
            s = search.strip().lower()
            cols = [c for c in ["statement","implication"] if c in sview.columns]
            mask = False
            for c in cols:
                mask = mask | sview[c].astype(str).str.lower().str.contains(s, na=False)
            sview = sview[mask]
        if "date" in sview.columns:
            sview["date"] = safe_dt(sview["date"])
            sview = sview.sort_values("date", ascending=False)
        cols_show = [c for c in ["date","theme","statement","implication"] if c in sview.columns]
        st.dataframe(sview[cols_show], use_container_width=True)
