import csv
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
import feedparser


OUTPUT_PATH = "data/intelligence_live.csv"
SOURCES_PATH = "data/intel_sources.csv"

# Governance: only allow these categories
ALLOWED_CATEGORIES = {"Industry", "Competition", "Customers", "Regulation"}

# Simple keyword-based signal typing (you can refine later)
SIGNAL_RULES = [
    ("Regulation", ["regulation", "directive", "policy", "sanction", "tariff", "consultation"], "Policy/Regulatory"),
    ("Competition", ["wins", "award", "contract", "order", "backlog", "tender"], "Competitive move"),
    ("Customers", ["tso", "utility", "grid operator", "tender", "framework"], "Customer signal"),
    ("Industry", ["hvdc", "grid", "transformer", "switchgear", "interconnector"], "Market/Industry"),
]

IMPACT_RULES = [
    (["delay", "shortage", "sanction", "tariff", "investigation", "penalty"], "Negative"),
    (["award", "wins", "record", "accelerate", "growth", "expands", "increases"], "Positive"),
]

HEADERS = [
    "date_utc",
    "category",
    "entity",
    "headline",
    "source_name",
    "source_url",
    "region",
    "signal_type",
    "impact_area",
    "expected_impact",
    "confidence",
    "action_required",
    "id",
]


def stable_id(*parts: str) -> str:
    raw = "||".join([p or "" for p in parts]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def classify_signal(category: str, text: str) -> str:
    t = text.lower()
    for cat, kws, label in SIGNAL_RULES:
        if category == cat and any(k in t for k in kws):
            return label
    # fallback by category
    return {
        "Industry": "Market/Industry",
        "Competition": "Competitive move",
        "Customers": "Customer signal",
        "Regulation": "Policy/Regulatory",
    }.get(category, "Other")


def classify_impact(text: str) -> str:
    t = text.lower()
    for kws, label in IMPACT_RULES:
        if any(k in t for k in kws):
            return label
    return "Neutral"


def default_impact_area(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["hvdc", "grid", "switchgear", "transformer", "substation"]):
        return "Grid"
    if any(k in t for k in ["wind", "turbine", "offshore"]):
        return "Wind"
    if any(k in t for k in ["service", "maintenance"]):
        return "Service"
    return "Enterprise"


def fetch_rss(source: Dict) -> List[Dict]:
    url = source["url"]
    feed = feedparser.parse(url)
    out = []

    for e in feed.entries[:50]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        # Keep UTC date as now if parsing fails
        dt_utc = datetime.now(timezone.utc)

        # feedparser often includes parsed time
        if getattr(e, "published_parsed", None):
            dt_utc = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)

        headline = title if title else link
        cat = source["category"]
        entity = source["entity"]

        signal_type = classify_signal(cat, headline)
        expected_impact = classify_impact(headline)
        impact_area = default_impact_area(headline)

        # Governance defaults
        confidence = "Medium" if source.get("priority", "Medium") != "High" else "High"
        action_required = "Yes" if expected_impact == "Negative" else ""

        rid = stable_id(source["source_name"], headline, link)

        out.append({
            "date_utc": dt_utc.strftime("%Y-%m-%d"),
            "category": cat,
            "entity": entity,
            "headline": headline,
            "source_name": source["source_name"],
            "source_url": link or url,
            "region": source.get("region", "Global"),
            "signal_type": signal_type,
            "impact_area": impact_area,
            "expected_impact": expected_impact,
            "confidence": confidence,
            "action_required": action_required,
            "id": rid,
        })

    return out


def main():
    if not os.path.exists(SOURCES_PATH):
        raise FileNotFoundError(f"Missing {SOURCES_PATH}. Create it first.")

    sources = pd.read_csv(SOURCES_PATH).to_dict(orient="records")

    # Governance: category allowlist
    for s in sources:
        if s.get("category") not in ALLOWED_CATEGORIES:
            raise ValueError(f"Source category not allowed: {s.get('category')} in {s}")

    all_rows: List[Dict] = []

    for s in sources:
        stype = str(s.get("type", "")).lower().strip()
        if stype == "rss":
            all_rows.extend(fetch_rss(s))
        else:
            # Keep it strict: start with RSS only.
            # Add HTML scraping later for very stable, owned pages (regulators/company PR).
            continue

    df_new = pd.DataFrame(all_rows, columns=HEADERS)

    # Load existing and dedupe by id
    if os.path.exists(OUTPUT_PATH):
        df_old = pd.read_csv(OUTPUT_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # Keep last 120 days to prevent file bloat
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")

    # Use timezone-naive UTC cutoff (prevents pandas comparison crash)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=120)

    df = df[df["date_utc"].notna()]
    df = df[df["date_utc"] >= cutoff]

    df["date_utc"] = df["date_utc"].dt.strftime("%Y-%m-%d")

    df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    if __name__ == "__main__":
    main()

