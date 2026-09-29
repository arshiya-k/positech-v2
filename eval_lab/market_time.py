"""US equity session times and event-window returns."""
import pandas as pd


def session_close_utc(dates) -> pd.DatetimeIndex:
    """4:00pm New York on each date, in naive UTC. Handles daylight saving time:
    the close is 21:00 UTC in winter and 20:00 UTC in summer."""
    local = pd.DatetimeIndex(pd.to_datetime(dates)).normalize() + pd.Timedelta(hours=16)
    return local.tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)


def event_returns(events: pd.DataFrame, prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """For each event (ticker, at = naive UTC timestamp), abnormal returns vs the benchmark over:

        reaction      last close at or before the event -> first close after it
        next_session  the session after that (does the news predict anything further?)

    Returns one row per event with reaction_date, reaction_abnormal, next_session_abnormal.
    """
    closes = {t: g.set_index("date")["adj_close"].sort_index() for t, g in prices.groupby("ticker")}
    bench = closes[benchmark]
    close_times = {t: session_close_utc(s.index) for t, s in closes.items()}
    rows = []
    for e in events.itertuples():
        series = closes.get(e.ticker)
        if series is None:
            rows.append({})
            continue
        i = close_times[e.ticker].searchsorted(pd.Timestamp(e.at), side="right") - 1
        row = {}
        for window, (a, b) in {"reaction": (i, i + 1), "next_session": (i + 1, i + 2)}.items():
            if a < 0 or b >= len(series):
                continue
            d0, d1 = series.index[a], series.index[b]
            if d0 in bench.index and d1 in bench.index:
                row[f"{window}_abnormal"] = (series.iloc[b] / series.iloc[a] - 1) - (bench[d1] / bench[d0] - 1)
                if window == "reaction":
                    row["reaction_date"] = d1
        rows.append(row)
    return pd.DataFrame(rows, index=events.index)
