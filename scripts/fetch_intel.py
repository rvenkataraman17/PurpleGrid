import csv
import hashlib
import os
from datetime import datetime, timedelta

import feedparser
import pandas as pd

OUTPUT_PATH = "data/intelligence_live.csv"
SOURCES_PATH = "data/intel_sources.csv"

# Only allow these categories (governance)
ALLOWED_CATEGORIES = {"Industry", "Competition", "Customers", "Regulation"}

HEADERS = [
    "date_utc",
    "category",
    "entity",
    "region",
    "headline",
    "source_name",
    "source_url",
    "signal_type",
    "expected_impact",
    "confidence",
    "action_required",
    "id",
]

SIGNAL_KEYWORDS = {
    "Policy/Regulatory": ["regulation", "directive", "policy", "sanction", "tariff", "consultation", "law", "rule"],
    "Competitive move": ["wins", "award", "contract", "order", "backlog", "tender", "partnership", "acquisition"],
    "Customer signal": ["tso", "utility", "grid operator", "framework", "tender", "procurement"],
    "Market/Industry": ["hvdc", "grid", "transformer", "switchgear", "substation", "interconnector", "data center"],
}

NEGATIVE_WORDS = ["delay", "shortage", "sanction", "tariff", "investigation", "penalty", "lawsuit", "recall"]
POSITIVE_WORDS = ["award", "wins", "record", "accelerate", "growth", "expands", "increase", "ramp", "upgrade"]


def stable_id(*parts: str) -> str:
    raw = "||".join([p or "" for p in parts]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def classify_signal(text: str) -> str:
    t = text.lower()
    for label, kws in SIGNAL_KEYWORDS.items():
        if any(k in t for k in kws):
            return label
    return "General"


def classify_impact(text: str) -> str:
    t = text.lower()
    if any(k in t for k in NEGATIVE_WORDS):
        return "Negative"
    if any(k in t for k in POSITIVE_WORDS):
        return "Positive"
    return "Neutral"


def parse_entry_date(entry) -> str:
    # Always write date_utc as YYYY-MM-DD (string). No timezone dtype issues.
    if getattr(entry, "published_parsed", None):
        dt = datetime(*entry.published_parsed[:6])
        return dt.strftime("%Y-%m-%d")
    if getattr(entry, "updated_parsed", None):
        dt = datetime(*entry.updated_parsed[:6])
        return dt.strftime("%Y-%m-%d")
    return datetime.utcnow().strftime("%Y-%m-%d")


def fetch_rss(source: dict) -> list[dict]:
    feed = feedparser.parse(source["url"])
    rows = []

    for e in getattr(feed, "entries", [])[:50]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title:
            continue

        date_utc = parse_entry_date(e)
        signal_type = classify_signal(title)
        impact = classify_impact(title)

        confidence = "High" if str(source.get("priority", "")).strip().lower() == "high" else "Medium"
        action_required = "Yes" if impact == "Negative" else ""

        rid = stable_id(source.get("source_name", ""), title, link)

        rows.append({
            "date_utc": date_utc,
            "category": source["category"],
            "entity": source.get("entity", ""),
            "region": source.get("region", "Global"),
            "headline": title,
            "source_name": source.get("source_name", ""),
            "source_url": link if link else source["url"],
            "signal_type": signal_type,
            "expected_impact": impact,
            "confidence": confidence,
            "action_required": action_required,
            "id": rid,
        })

    return rows


def main():
    if not os.path.exists(SOURCES_PATH):
        raise FileNotFoundError(f"Missing {SOURCES_PATH}. Create it first.")

    sources = pd.read_csv(SOURCES_PATH).to_dict(orient="records")

    # Governance: only allowed categories
    for s in sources:
        if s.get("category") not in ALLOWED_CATEGORIES:
            raise ValueError(f"Source category not allowed: {s.get('category')}")

    all_rows = []
    for s in sources:
        stype = str(s.get("type", "")).strip().lower()
        if stype == "rss":
            all_rows.extend(fetch_rss(s))

    df_new = pd.DataFrame(all_rows, columns=HEADERS)

    # Load old
    if os.path.exists(OUTPUT_PATH):
        df_old = pd.read_csv(OUTPUT_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # Dedupe
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])

    # Keep last 120 days (string-based date filter to avoid tz/dtype issues)
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    cutoff = datetime.utcnow() - timedelta(days=120)
    df = df[df["date_utc"].notna()]
    df = df[df["date_utc"] >= cutoff]
    df = df.sort_values(["date_utc", "category", "entity"], ascending=[False, True, True])

    # Write back as YYYY-MM-DD strings
    df["date_utc"] = df["date_utc"].dt.strftime("%Y-%m-%d")
    df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
