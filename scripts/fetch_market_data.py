import csv
from datetime import datetime
import pandas as pd
import requests

# Simple, no-key daily close data via Stooq CSV
# Siemens Energy (Xetra): ENR.DE -> enr.de
# Schneider (Paris): SU.PA -> su.pa
# GE Vernova (NYSE): GEV -> gev.us
STOOQ_URL = "https://stooq.com/q/l/?s={symbol}&i=d"

TICKERS = {
    "Siemens Energy": {"symbol": "enr.de", "currency": "EUR"},
    "Schneider Electric": {"symbol": "su.pa", "currency": "EUR"},
    "GE Vernova": {"symbol": "gev.us", "currency": "USD"},
}

OUT = "data/market_data.csv"
HEADERS = ["company", "date", "close", "currency", "source_ref", "confidence"]

def fetch_stooq_close(symbol: str):
    url = STOOQ_URL.format(symbol=symbol)
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        return None
    # Symbol,Date,Time,Open,High,Low,Close,Volume
    parts = lines[1].split(",")
    if len(parts) < 7:
        return None
    date = parts[1]
    close = float(parts[6])
    return date, close, url

def main():
    rows = []
    for company, meta in TICKERS.items():
        res = fetch_stooq_close(meta["symbol"])
        if not res:
            continue
        date, close, src = res
        rows.append({
            "company": company,
            "date": date,
            "close": close,
            "currency": meta["currency"],
            "source_ref": f"https://stooq.com/q/?s={meta['symbol']}",
            "confidence": "Medium"  # market quotes are not IR; keep Medium
        })

    df = pd.DataFrame(rows, columns=HEADERS)

    # append if file exists (keep history), else create
    try:
        old = pd.read_csv(OUT)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["company", "date"], keep="last")
    except Exception:
        pass

    df = df.sort_values(["company", "date"])
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(df)} rows to {OUT}")

if __name__ == "__main__":
    main()
