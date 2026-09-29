"""Daily prices, corporate actions and earnings announcements from Yahoo Finance
into source.duckdb.

yfinance's `Close` is already split-adjusted and `Adj Close` is split- and
dividend-adjusted. Neither is the price that actually printed that day, which
is what a point-in-time ratio like trailing P/E needs, so `close_unadjusted` is
rebuilt from the split history. Yahoo folds spin-offs into that history as
fractional "splits"; they're kept as `spinoff_adjustment` so they aren't mistaken
for share splits that change per-share figures.

Each earnings announcement gets a `reaction_date`: the first session that could
trade on the news. A 4:00pm release reacts the next day; a 7:00am release reacts
the same day. Getting this wrong is a classic event-study error.

    python -m eval_lab.ingest_market [--tickers AAPL NVDA]
"""
import argparse
from fractions import Fraction

import pandas as pd
import yfinance as yf

from eval_lab.db import load_config, replace_rows, source_db


def unadjust_for_splits(close: pd.Series, splits: pd.Series) -> pd.Series:
    """Undo split adjustment: multiply each price by every split ratio that came after it."""
    factor = pd.Series(1.0, index=close.index)
    for split_date, ratio in splits[splits > 0].items():
        factor[close.index < split_date] *= ratio
    return close * factor


def is_share_split(ratio: float) -> bool:
    """True for real splits (2-for-1, 3-for-2, 1-for-10...). Yahoo also records
    spin-offs as "splits" with ratios like 1.061 (the price adjustment for the
    distributed shares); those don't change the share count or per-share figures."""
    simple = Fraction(ratio).limit_denominator(10)
    return abs(float(simple) - ratio) < 1e-3 and not 0.8 < ratio < 1.25


def fetch(ticker: str, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = yf.Ticker(ticker).history(start=start, auto_adjust=False, actions=True)
    if hist.empty:
        raise RuntimeError(f"no price history returned for {ticker}")
    hist.index = hist.index.tz_localize(None).normalize()
    splits, dividends = hist["Stock Splits"], hist["Dividends"]

    prices = pd.DataFrame({
        "ticker": ticker,
        "date": hist.index.date,
        "open": hist["Open"].to_numpy(),
        "high": hist["High"].to_numpy(),
        "low": hist["Low"].to_numpy(),
        "close": hist["Close"].to_numpy(),
        "adj_close": hist["Adj Close"].to_numpy(),
        "close_unadjusted": unadjust_for_splits(hist["Close"], splits).to_numpy(),
        "volume": hist["Volume"].fillna(0).astype("int64").to_numpy(),
    })
    actions = pd.concat([
        pd.DataFrame({"date": splits[splits > 0].index.date,
                      "action_type": ["split" if is_share_split(r) else "spinoff_adjustment" for r in splits[splits > 0]],
                      "value": splits[splits > 0].to_numpy()}),
        pd.DataFrame({"date": dividends[dividends > 0].index.date, "action_type": "dividend", "value": dividends[dividends > 0].to_numpy()}),
    ])
    actions.insert(0, "ticker", ticker)
    return prices, actions


MARKET_OPEN, MARKET_CLOSE = pd.Timedelta(hours=9, minutes=30), pd.Timedelta(hours=16)


def announcement_timing(announced_at_et: pd.Timestamp) -> str:
    time_of_day = announced_at_et - announced_at_et.normalize()
    if time_of_day == pd.Timedelta(0):
        return "unknown"  # Yahoo gives midnight when it has only a date
    if time_of_day < MARKET_OPEN:
        return "before_open"
    return "after_close" if time_of_day >= MARKET_CLOSE else "during_market"


def reaction_date(announced_at_et: pd.Timestamp, timing: str, trading_days: pd.DatetimeIndex):
    """First trading session whose prices could reflect the announcement."""
    day = announced_at_et.normalize()
    i = trading_days.searchsorted(day, side="right" if timing == "after_close" else "left")
    return trading_days[i].date() if i < len(trading_days) else None


def fetch_earnings(ticker: str, start: str, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    raw = yf.Ticker(ticker).get_earnings_dates(limit=100)
    if raw is None or raw.empty:
        return pd.DataFrame()
    announced = raw.index.tz_convert("America/New_York").tz_localize(None)
    df = pd.DataFrame({
        "ticker": ticker,
        "announced_at_et": announced,
        "eps_estimate": raw["EPS Estimate"].to_numpy(),
        "eps_reported": raw["Reported EPS"].to_numpy(),
        "surprise_pct": raw["Surprise(%)"].to_numpy(),
    })
    df = df[(df["announced_at_et"] >= pd.Timestamp(start)) & df["eps_reported"].notna()]
    df["timing"] = df["announced_at_et"].map(announcement_timing)
    df["reaction_date"] = [reaction_date(a, t, trading_days) for a, t in zip(df["announced_at_et"], df["timing"])]
    return df.drop_duplicates(["ticker", "announced_at_et"])


def sync_companies(con, universe: dict) -> None:
    """Insert new tickers from the universe and refresh their config-driven columns,
    leaving the EDGAR-derived columns (cik, name, fiscal_year_end_month) alone."""
    rows = pd.DataFrame([
        {
            "ticker": c["ticker"],
            "sector": c["sector"],
            "fy_convention": c.get("fy_convention", "end_year"),
            "in_positech": bool(c.get("positech", False)),
        }
        for c in universe["companies"]
    ])
    con.register("_universe", rows)
    con.execute("INSERT INTO companies BY NAME SELECT * FROM _universe WHERE ticker NOT IN (SELECT ticker FROM companies)")
    con.execute("""
        UPDATE companies SET sector = u.sector, fy_convention = u.fy_convention, in_positech = u.in_positech
        FROM _universe u WHERE companies.ticker = u.ticker
    """)
    con.unregister("_universe")


def main() -> None:
    universe = load_config("universe.yaml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", help="subset of tickers (default: whole universe + benchmark)")
    args = parser.parse_args()
    tickers = args.tickers or [c["ticker"] for c in universe["companies"]] + [universe["benchmark"]]

    con = source_db()
    sync_companies(con, universe)
    failed = []
    for ticker in tickers:
        try:
            prices, actions = fetch(ticker, universe["start_date"])
        except Exception as exc:  # yfinance raises a grab-bag of exception types
            failed.append(ticker)
            print(f"{ticker:6} FAILED: {exc}")
            continue
        replace_rows(con, "prices", prices, "ticker")
        replace_rows(con, "corporate_actions", actions, "ticker", keys=[ticker])
        n_splits = int(actions["action_type"].isin(["split", "spinoff_adjustment"]).sum())
        n_earnings = "-"
        if ticker != universe["benchmark"]:
            try:
                earnings = fetch_earnings(ticker, universe["start_date"], pd.DatetimeIndex(pd.to_datetime(prices["date"])))
            except Exception as exc:  # the earnings calendar is scraped and fails independently of prices
                earnings = pd.DataFrame()
                print(f"{ticker:6} earnings calendar unavailable: {exc}")
            replace_rows(con, "earnings", earnings, "ticker", keys=[ticker])
            n_earnings = len(earnings)
        print(f"{ticker:6} {len(prices):5} days  {n_splits} splits  {len(actions) - n_splits} dividends  {n_earnings} earnings")
    if failed:
        raise SystemExit(f"failed tickers: {' '.join(failed)} (re-run with --tickers to retry)")


if __name__ == "__main__":
    main()
