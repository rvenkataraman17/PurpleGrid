import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Siemens Energy – Strategy Dashboard (2025)",
    layout="wide"
)

st.title("Siemens Energy – Strategy Dashboard (2025)")

# ------------------------
# Load data
# ------------------------
@st.cache_data
def load_data():
    kpis = pd.read_csv("data/company_kpis_2025.csv")
    strategy = pd.read_csv("data/strategy_2025.csv")
    intelligence = pd.read_csv("data/intelligence_2025.csv")
    return kpis, strategy, intelligence

kpis, strategy, intelligence = load_data()

# ------------------------
# SECTION 1: 2025 Performance Snapshot
# ------------------------
st.header("2025 Performance Snapshot")

cols = st.columns(len(kpis))

for idx, row in kpis.iterrows():
    with cols[idx]:
        st.metric(
            label=row["metric"],
            value=f'{row["value"]} {row["unit"]}',
            help=row["commentary"]
        )

st.divider()

# ------------------------
# SECTION 2: Strategic Direction
# ------------------------
st.header("Strategic Direction (Management Signals)")

themes = ["All"] + sorted(strategy["theme"].unique().tolist())
selected_theme = st.selectbox("Filter by theme", themes)

if selected_theme != "All":
    strategy_view = strategy[strategy["theme"] == selected_theme]
else:
    strategy_view = strategy

st.dataframe(
    strategy_view[["date", "theme", "statement", "implication"]],
    use_container_width=True
)

st.divider()

# ------------------------
# SECTION 3: Strategic Intelligence Feed
# ------------------------
st.header("Strategic Intelligence Feed")

col1, col2, col3 = st.columns(3)

with col1:
    category_filter = st.multiselect(
        "Category",
        options=intelligence["category"].unique(),
        default=intelligence["category"].unique()
    )

with col2:
    region_filter = st.multiselect(
        "Region",
        options=intelligence["region"].unique(),
        default=intelligence["region"].unique()
    )

with col3:
    entity_filter = st.multiselect(
        "Entity (Trend / Competitor / Policy)",
        options=intelligence["entity"].unique(),
        default=intelligence["entity"].unique()
    )

filtered_intel = intelligence[
    (intelligence["category"].isin(category_filter)) &
    (intelligence["region"].isin(region_filter)) &
    (intelligence["entity"].isin(entity_filter))
]

st.dataframe(
    filtered_intel[[
        "date",
        "category",
        "entity",
        "region",
        "description",
        "strategic_implication"
    ]],
    use_container_width=True
)
