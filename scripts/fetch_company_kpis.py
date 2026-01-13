import csv
import re
import pandas as pd
import requests

OUT = "data/company_kpis_2025.csv"
HEADERS = ["company","period","period_end","basis","metric","value","unit","source_ref","confidence","commentary"]

UA = {"User-Agent": "Mozilla/5.0"}

SIEMENS_URLS = [
    # FY2025 results press release (contains margin, FCF pre tax, net income in text)
    "https://www.siemens-energy.com/global/en/home/press-releases/siemens-energy-fulfills-all-commitments---and-increases-mid-term.html",
    # IR landing (fallback)
    "https://www.siemens-energy.com/global/en/home/investor-relations.html",
]

def get_text(url: str) -> str:
    r = requests.get(url, timeout=30, headers=UA)
    r.raise_for_status()
    return r.text

def to_float(num_str: str) -> float:
    return float(num_str.replace(",", "").strip())

def extract_siemens_fy2025(text: str, src: str):
    rows = []

    # Margin at 6%
    m = re.search(r"margin\s+at\s+(\d+(\.\d+)?)\s*%", text, re.IGNORECASE)
    if m:
        rows.append(("Profit margin before special items", float(m.group(1)), "%", "Extracted margin statement"))

    # Free cash flow pre tax ... €4.663 billion
    m = re.search(r"Free cash flow pre tax.*?€\s*([\d\.,]+)\s*billion", text, re.IGNORECASE)
    if m:
        bn = to_float(m.group(1))
        rows.append(("Free cash flow pre tax", bn * 1000.0, "EURm", "Converted from EUR bn to EURm"))

    # Net income at €1.685 billion
    m = re.search(r"Net income.*?€\s*([\d\.,]+)\s*billion", text, re.IGNORECASE)
    if m:
        bn = to_float(m.group(1))
        rows.append(("Net income", bn * 1000.0, "EURm", "Converted from EUR bn to EURm"))

    out = []
    for metric, value, unit, comm in rows:
        out.append({
            "company": "Siemens Energy",
            "period": "2025_FY",
            "period_end": "2025-09-30",
            "basis": "Actual",
            "metric": metric,
            "value": value,
            "unit": unit,
            "source_ref": src,
            "confidence": "High",
            "commentary": comm,
        })
    return out

def main():
    all_rows = []

    for url in SIEMENS_URLS:
        try:
            text = get_text(url)
        except Exception:
            continue
        all_rows.extend(extract_siemens_fy2025(text, url))

    df = pd.DataFrame(all_rows, columns=HEADERS)

    # Write even if empty (so file exists). Dashboard will populate once extraction succeeds.
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(df)} Siemens KPI rows -> {OUT}")

if __name__ == "__main__":
    main()
