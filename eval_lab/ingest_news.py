"""Financial news headlines for the universe, from two free sources with no key:

    python -m eval_lab.ingest_news --provider gdelt   [--days 84] [--window-days 14]
    python -m eval_lab.ingest_news --provider google  [--days 84] [--window-days 7]

    gdelt   GDELT 2.0 DOC API: exact first-seen timestamps, up to 250 articles per
            query, but it enforces one request per ~5 seconds and blocks bursts
            for a long time.
    google  Google News RSS search: up to 100 articles per query and tolerant of
            pacing, but older items carry a placeholder time (07:00 GMT) instead of
            the real publish time, so those are stored with time_precision = "day".
            (Google News feeds are offered for personal, non-commercial use.)

For each company and window the company's names are searched together with
market vocabulary (stock, shares, earnings, ...). Search is only the first
filter: "Apple" still finds dessert recipes, which is why
`eval_lab.news_sentiment` checks relevance with entity recognition afterwards.

Each finished window is logged, so re-running resumes where it stopped (the
latest window is always refreshed).
"""
import argparse
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from eval_lab import companies
from eval_lab.db import news_db

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_RSS = "https://news.google.com/rss/search"
MARKET_TERMS = "(stock OR shares OR earnings OR investors OR analyst OR revenue OR outlook)"
PAUSE_SECONDS = {"gdelt": 10, "google": 3}
PLACEHOLDER_TIMES = {(7, 0, 0), (0, 0, 0)}   # Google News times that mean "date only"
BACKOFF_SECONDS = [120, 300, 600, 900, 900]

SCHEMA = """
CREATE TABLE IF NOT EXISTS news_articles (
    provider       VARCHAR,     -- gdelt | google
    article_id     VARCHAR,     -- hash of the URL
    query_ticker   VARCHAR,     -- the company whose search returned it
    url            VARCHAR,
    domain         VARCHAR,
    title          VARCHAR,
    seen_at        TIMESTAMP,   -- first seen (gdelt) or published (google), UTC
    time_precision VARCHAR,     -- exact | day
    language       VARCHAR,
    source_country VARCHAR
);
CREATE TABLE IF NOT EXISTS news_fetch_log (
    provider     VARCHAR,
    ticker       VARCHAR,
    window_start TIMESTAMP,
    window_end   TIMESTAMP,
    n_articles   INTEGER,
    fetched_at   TIMESTAMP
);
"""


def query_for(ticker: str, provider: str = "gdelt") -> str:
    """GDELT rejects quoted single words and very short phrases, and only allows
    parentheses around OR'd terms, so: quote multi-word or punctuated names, drop
    names under four characters unless they're a plain word (IBM), and only
    parenthesize when there's more than one term."""
    terms = []
    for name in companies.universe()[ticker]["names"]:
        punctuated = " " in name or "-" in name or "&" in name
        if punctuated and len(name) >= 4:
            terms.append(f'"{name}"')
        elif not punctuated and len(name) >= 3:
            terms.append(name)
    if len(ticker) >= 4 and ticker not in terms:
        terms.append(ticker)
    names = terms[0] if len(terms) == 1 else f"({' OR '.join(terms)})"
    return f"{names} {MARKET_TERMS}" + (" sourcelang:english" if provider == "gdelt" else "")


def fetch_gdelt(session: requests.Session, query: str, start: datetime, end: datetime) -> list[dict]:
    params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": 250, "sort": "datedesc",
              "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")}
    for wait in [0, *BACKOFF_SECONDS]:
        if wait:
            print(f"    rate limited; waiting {wait}s")
            time.sleep(wait)
        response = session.get(GDELT_API, params=params, timeout=60)
        if response.status_code == 429:
            continue
        response.raise_for_status()
        if not response.text.strip():
            return []
        try:
            return response.json().get("articles", [])
        except ValueError:
            raise RuntimeError(f"GDELT rejected the query: {response.text.strip()[:200]}")
    raise RuntimeError("still rate limited after backing off; try again later")


def fetch_google(session: requests.Session, query: str, start: datetime, end: datetime) -> list[dict]:
    """Google News RSS; `before:` is exclusive, so the window end is pushed a day."""
    q = f"{query} after:{start:%Y-%m-%d} before:{end + timedelta(days=1):%Y-%m-%d}"
    response = session.get(GOOGLE_RSS, params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}, timeout=60)
    response.raise_for_status()
    articles = []
    for item in ET.fromstring(response.content).iter("item"):
        source = item.find("source")
        title = item.findtext("title") or ""
        publisher = source.text if source is not None else ""
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        published = pd.to_datetime(item.findtext("pubDate"), utc=True).tz_localize(None)
        articles.append({
            "url": item.findtext("link"), "title": title,
            "domain": re.sub(r"^https?://(www\.)?", "", source.get("url", "")) if source is not None else None,
            "published": published,
            "time_precision": "day" if (published.hour, published.minute, published.second) in PLACEHOLDER_TIMES else "exact",
        })
    return articles


FETCHERS = {"gdelt": fetch_gdelt, "google": fetch_google}


def to_frame(articles: list[dict], ticker: str, provider: str) -> pd.DataFrame:
    rows = []
    for a in articles:
        if not a.get("title") or not a.get("url"):
            continue
        rows.append({
            "provider": provider,
            "article_id": hashlib.sha1(a["url"].encode()).hexdigest()[:16],
            "query_ticker": ticker,
            "url": a["url"],
            "domain": a.get("domain"),
            "title": a["title"].strip(),
            "seen_at": pd.to_datetime(a["seendate"], format="%Y%m%dT%H%M%SZ") if "seendate" in a else a["published"],
            "time_precision": a.get("time_precision", "exact"),
            "language": a.get("language", "English"),
            "source_country": a.get("sourcecountry"),
        })
    return pd.DataFrame(rows).drop_duplicates("article_id") if rows else pd.DataFrame()


def windows(days: int, window_days: int) -> list[tuple[datetime, datetime]]:
    end = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    out = []
    while start < end:
        out.append((start, min(start + timedelta(days=window_days), end)))
        start += timedelta(days=window_days)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=list(FETCHERS), default="gdelt")
    parser.add_argument("--days", type=int, default=84, help="how far back to go (GDELT's DOC API covers ~3 months)")
    parser.add_argument("--window-days", type=int, help="default: 14 for gdelt, 7 for google (smaller cap per query)")
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()

    con = news_db()
    con.execute(SCHEMA)
    tickers = args.tickers or list(companies.universe())
    provider = args.provider
    spans = windows(args.days, args.window_days or (14 if provider == "gdelt" else 7))
    done = {(t, pd.Timestamp(s)) for t, s in con.execute(
        "SELECT ticker, window_start FROM news_fetch_log WHERE provider = ?", [provider]).fetchall()}
    session = requests.Session()
    total = 0
    for ticker in tickers:
        query = query_for(ticker, provider)
        for i, (start, end) in enumerate(spans):
            latest = i == len(spans) - 1
            if (ticker, pd.Timestamp(start)) in done and not latest:
                continue
            try:
                df = to_frame(FETCHERS[provider](session, query, start, end), ticker, provider)
            except RuntimeError as exc:
                if "rejected" not in str(exc):
                    raise
                print(f"{ticker:6} skipped: {exc}")
                break
            if len(df):
                con.execute("DELETE FROM news_articles WHERE provider = ? AND query_ticker = ? AND article_id IN (SELECT UNNEST(?))",
                            [provider, ticker, list(df["article_id"])])
                con.register("_news", df)
                con.execute("INSERT INTO news_articles BY NAME SELECT * FROM _news")
                con.unregister("_news")
            con.execute("DELETE FROM news_fetch_log WHERE provider = ? AND ticker = ? AND window_start = ?", [provider, ticker, start])
            con.execute("INSERT INTO news_fetch_log VALUES (?, ?, ?, ?, ?, now())", [provider, ticker, start, end, len(df)])
            total += len(df)
            capped = "  (hit the per-query cap)" if len(df) >= (250 if provider == "gdelt" else 100) else ""
            print(f"{ticker:6} {start:%Y-%m-%d} to {end:%Y-%m-%d}: {len(df):3} articles{capped}")
            time.sleep(PAUSE_SECONDS[provider])
    n = con.execute("SELECT count(*), count(DISTINCT article_id) FROM news_articles").fetchone()
    print(f"\nfetched {total} this run; {n[0]} rows ({n[1]} distinct articles) stored")


if __name__ == "__main__":
    main()
