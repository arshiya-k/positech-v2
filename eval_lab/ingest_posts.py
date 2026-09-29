"""Normalize PosiTech's scraped news / Reddit / Twitter CSVs into source.duckdb.

Each scraper stored dates differently (ISO 8601, Unix epoch seconds, Twitter's
"Mon Dec 12 07:23:35 +0000 2022"), so they're parsed per source into UTC.
VADER is re-run on the stored text: for news, PosiTech scored the article body
but saved only the title, so the stored score can't be reproduced from the data.

    python -m eval_lab.ingest_posts
"""
import hashlib

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from eval_lab.db import POSITECH_DATA, replace_rows, source_db

SOURCES = {
    "news": ("news_data.csv", "title"),
    "reddit": ("reddit_data.csv", "post"),
    "twitter": ("twitter_data.csv", "post"),
}


def parse_posted_at(source: str, raw: pd.Series) -> pd.Series:
    if source == "reddit":
        ts = pd.to_datetime(raw.astype(float), unit="s", utc=True)
    elif source == "twitter":
        ts = pd.to_datetime(raw, format="%a %b %d %H:%M:%S %z %Y", utc=True)
    else:
        ts = pd.to_datetime(raw, format="ISO8601", utc=True)
    return ts.dt.tz_localize(None)


def post_id(source: str, link: str, text: str) -> str:
    return hashlib.sha1(f"{source}|{link}|{text}".encode()).hexdigest()[:12]


def load_source(source: str, vader: SentimentIntensityAnalyzer) -> pd.DataFrame:
    filename, text_col = SOURCES[source]
    raw = pd.read_csv(POSITECH_DATA / filename)
    text = raw[text_col].fillna("").astype(str)
    df = pd.DataFrame({
        "source": source,
        "ticker": raw["keyword"],
        "posted_at": parse_posted_at(source, raw["date"]),
        "text": text,
        "link": raw["link"],
        "vader_stored": raw["sentiment_score"],
        "vader_recomputed": text.map(lambda t: vader.polarity_scores(t)["compound"]),
        "is_retweet": text.str.startswith("RT @"),
        "n_chars": text.str.len(),
    })
    df["post_id"] = [post_id(source, l, t) for l, t in zip(df["link"].astype(str), df["text"])]
    normalized = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df["is_duplicate"] = normalized.duplicated()
    return df


def main() -> None:
    vader = SentimentIntensityAnalyzer()
    df = pd.concat([load_source(s, vader) for s in SOURCES], ignore_index=True)
    df = df.drop_duplicates("post_id")
    con = source_db()
    replace_rows(con, "posts", df, "source")
    print(df.groupby("source").agg(
        posts=("post_id", "size"),
        first=("posted_at", "min"),
        last=("posted_at", "max"),
        duplicates=("is_duplicate", "sum"),
        retweets=("is_retweet", "sum"),
    ).to_string())


if __name__ == "__main__":
    main()
