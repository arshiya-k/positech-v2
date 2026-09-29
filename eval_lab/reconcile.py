"""Reconcile a data vendor (Yahoo Finance) against SEC filings.

Two sources for the same number rarely agree perfectly, and the reasons are
what a data team has to understand:

    match                      within rounding
    matches_first_reported     vendor shows the original figure; SEC data has since restated it
    split_basis                per-share or share-count figure on a different basis, explained by a recorded split
    unrecorded_basis_change    per-share figure off by a clean ratio (2x, 10x...) that no recorded split explains
    matches_adjacent_period    vendor value belongs to the SEC's prior or next period (period mapping)
    scale                      off by a power of 1,000
    street_vs_gaap             earnings-calendar EPS (analysts' adjusted figure) vs GAAP diluted EPS
    unexplained                a real difference to investigate (often a definition difference)

Rows only one source has are split by reason:

    vendor_only_metric         the company never reports this line (e.g. no operating income subtotal);
                               the vendor computes its own
    q4_not_filed               a fourth quarter, which companies never file on its own
    missing_in_sec / missing_in_vendor   other coverage gaps

It also reports how far apart the two sources' period-end dates are.

    python -m eval_lab.reconcile [--tickers AAPL JPM] [--skip-fetch]

Writes reports/reconciliation.csv (one row per comparison) and reports/reconciliation_summary.csv.
"""
import argparse

import numpy as np
import pandas as pd
import yfinance as yf

from eval_lab.db import load_config, replace_rows, source_db, write_report

VENDOR = "yahoo"
LINE_ITEMS = {
    "Total Revenue": "revenue",
    "Operating Income": "operating_income",
    "Net Income": "net_income",
    "Diluted EPS": "eps_diluted",
    "Diluted Average Shares": "diluted_shares",
}
PERIOD_MATCH_DAYS = 10     # Yahoo labels periods by month end; 52/53-week years end a few days earlier
REL_TOLERANCE = 0.005
EPS_TOLERANCE = 0.011      # EPS is reported to the cent
SCALES = [1e3, 1e6, 1e-3, 1e-6]
CLEAN_RATIOS = [2, 3, 4, 5, 10, 20]
PER_SHARE = ("eps_diluted", "diluted_shares", "eps_street_vs_gaap")


def fetch_vendor(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    frames = []
    for frequency, statement in [("annual", stock.income_stmt), ("quarterly", stock.quarterly_income_stmt)]:
        if statement is None or statement.empty:
            continue
        rows = statement.loc[statement.index.intersection(list(LINE_ITEMS))]
        stacked = rows.rename(index=LINE_ITEMS).stack().dropna()
        long = pd.DataFrame({"metric": stacked.index.get_level_values(0),
                             "period_end": stacked.index.get_level_values(1), "value": stacked.to_numpy()})
        frames.append(long.assign(ticker=ticker, vendor=VENDOR, frequency=frequency))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["period_end"] = pd.to_datetime(out["period_end"]).dt.date
    out["value"] = out["value"].astype(float)
    return out


def close_enough(a: float, b: float, metric: str) -> bool:
    if metric == "eps_diluted":
        return abs(a - b) <= EPS_TOLERANCE
    return abs(a - b) <= REL_TOLERANCE * max(abs(b), 1.0)


def split_factors_after(splits: pd.DataFrame, ticker: str, date) -> list[float]:
    """Every cumulative factor the later splits could have applied: a figure may have
    been restated for some of the splits since the period but not the rest."""
    later = splits[(splits["ticker"] == ticker) & (splits["date"] > pd.Timestamp(date))].sort_values("date")["value"]
    return [float(later.iloc[i:].prod()) for i in range(len(later))]


def basis_status(vendor_value: float, sec_value: float, split_factors: list[float], metric: str) -> str | None:
    """Is the vendor figure the SEC figure restated by a split factor, or by a clean
    ratio no recorded split explains? Share counts scale up with a split, EPS down."""
    if not sec_value or metric not in PER_SHARE:
        return None
    def fits(factor: float) -> bool:
        target = sec_value * factor if metric == "diluted_shares" else sec_value / factor
        # vendor EPS is rounded to the cent, so allow for that rounding
        return abs(vendor_value - target) <= max(0.02 * abs(target), EPS_TOLERANCE if metric != "diluted_shares" else 0)
    if any(fits(f) or fits(1 / f) for f in split_factors):
        return "split_basis"
    if any(fits(r) or fits(1 / r) for r in CLEAN_RATIOS):
        return "unrecorded_basis_change"
    return None


def classify(vendor_value: float, sec: pd.Series, neighbors: list[float], split_factors: list[float], metric: str) -> str:
    if close_enough(vendor_value, sec["value"], metric):
        return "match"
    if pd.notna(sec["first_reported_value"]) and close_enough(vendor_value, sec["first_reported_value"], metric):
        return "matches_first_reported"
    basis = basis_status(vendor_value, sec["value"], split_factors, metric)
    if basis:
        return basis
    if any(close_enough(vendor_value, n, metric) for n in neighbors):
        return "matches_adjacent_period"
    if sec["value"] and any(abs(vendor_value / sec["value"] / f - 1) < 0.02 for f in SCALES):
        return "scale"
    return "unexplained"


def reconcile_statements(vendor: pd.DataFrame, sec: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Compare each vendor figure with the SEC figure whose period ends nearest to it."""
    rows = []
    year_ends = sec[sec["frequency"] == "annual"].groupby("ticker")["period_end"].apply(set).to_dict()
    for (ticker, metric, frequency), v in vendor.groupby(["ticker", "metric", "frequency"]):
        s = sec[(sec["ticker"] == ticker) & (sec["metric"] == metric) & (sec["frequency"] == frequency)].sort_values("period_end")
        reported_at_all = ((sec["ticker"] == ticker) & (sec["metric"] == metric)).any()
        matched_sec = set()
        for vr in v.itertuples():
            gap = (s["period_end"] - pd.Timestamp(vr.period_end)).dt.days.abs() if len(s) else pd.Series(dtype=float)
            if not len(gap) or gap.min() > PERIOD_MATCH_DAYS:
                at_year_end = any(abs((pd.Timestamp(vr.period_end) - e).days) <= PERIOD_MATCH_DAYS for e in year_ends.get(ticker, ()))
                status = ("vendor_only_metric" if not reported_at_all
                          else "q4_not_filed" if frequency == "quarterly" and at_year_end else "missing_in_sec")
                rows.append({"ticker": ticker, "metric": metric, "frequency": frequency, "vendor_period_end": vr.period_end,
                             "vendor_value": vr.value, "status": status})
                continue
            i = gap.idxmin()
            sr = s.loc[i]
            matched_sec.add(i)
            position = s.index.get_loc(i)
            neighbors = [s.iloc[j]["value"] for j in (position - 1, position + 1) if 0 <= j < len(s)]
            factors = split_factors_after(splits, ticker, sr["period_end"])
            rows.append({
                "ticker": ticker, "metric": metric, "frequency": frequency,
                "vendor_period_end": vr.period_end, "sec_period_end": sr["period_end"].date(),
                "period_end_offset_days": int((pd.Timestamp(vr.period_end) - sr["period_end"]).days),
                "fiscal_year": sr["fiscal_year"], "fiscal_quarter": sr.get("fiscal_quarter"),
                "vendor_value": vr.value, "sec_value": sr["value"], "sec_first_reported": sr["first_reported_value"],
                "sec_is_derived": bool(sr.get("is_derived", False)),
                "rel_diff": (vr.value - sr["value"]) / abs(sr["value"]) if sr["value"] else np.nan,
                "status": classify(vr.value, sr, neighbors, factors, metric),
            })
        # SEC periods inside the vendor's date range that the vendor doesn't carry.
        if len(v) and len(s):
            lo, hi = pd.Timestamp(min(v["period_end"])), pd.Timestamp(max(v["period_end"]))
            window = s[(s["period_end"] >= lo - pd.Timedelta(days=PERIOD_MATCH_DAYS)) & (s["period_end"] <= hi + pd.Timedelta(days=PERIOD_MATCH_DAYS))]
            for i, sr in window.drop(index=list(matched_sec), errors="ignore").iterrows():
                rows.append({"ticker": ticker, "metric": metric, "frequency": frequency, "sec_period_end": sr["period_end"].date(),
                             "fiscal_year": sr["fiscal_year"], "sec_value": sr["value"], "status": "missing_in_vendor"})
    return pd.DataFrame(rows)


def reconcile_street_eps(earnings: pd.DataFrame, quarterly: pd.DataFrame, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """Earnings-calendar EPS vs GAAP diluted EPS for the quarter the announcement reports.

    The announcement comes 2-10 weeks after the quarter ends. Q4 has no GAAP
    quarterly EPS in the filings at all (see ingest_fundamentals), so those rows
    show up as `no_gaap_quarter`.
    """
    eps = quarterly[quarterly["metric"] == "eps_diluted"].sort_values("period_end")
    rows = []
    for e in earnings.itertuples():
        candidates = eps[(eps["ticker"] == e.ticker) & (eps["period_end"] < e.announced_at_et)
                         & (eps["period_end"] >= e.announced_at_et - pd.Timedelta(days=75))]
        base = {"ticker": e.ticker, "metric": "eps_street_vs_gaap", "frequency": "quarterly",
                "vendor_period_end": e.announced_at_et.date(), "vendor_value": e.eps_reported}
        if candidates.empty:
            rows.append({**base, "status": "no_gaap_quarter"})
            continue
        q = candidates.iloc[-1]
        if abs(e.eps_reported - q["value"]) <= EPS_TOLERANCE:
            status = "match"
        else:
            factors = split_factors_after(splits, e.ticker, q["period_end"]) if splits is not None else []
            status = basis_status(e.eps_reported, q["value"], factors, "eps_street_vs_gaap") or "street_vs_gaap"
        rows.append({
            **base, "sec_period_end": q["period_end"].date(), "fiscal_year": q["fiscal_year"],
            "fiscal_quarter": q["fiscal_quarter"], "sec_value": q["value"],
            "rel_diff": (e.eps_reported - q["value"]) / abs(q["value"]) if q["value"] else np.nan,
            "status": status,
        })
    return pd.DataFrame(rows)


def main() -> None:
    universe = load_config("universe.yaml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--skip-fetch", action="store_true", help="reuse vendor data already in the database")
    args = parser.parse_args()
    tickers = args.tickers or [c["ticker"] for c in universe["companies"]]

    con = source_db()
    if not args.skip_fetch:
        for ticker in tickers:
            try:
                vendor = fetch_vendor(ticker)
            except Exception as exc:  # yfinance scraping fails in many ways
                print(f"{ticker:6} vendor statements unavailable: {exc}")
                continue
            replace_rows(con, "vendor_fundamentals", vendor, "ticker", keys=[ticker])
            print(f"{ticker:6} {len(vendor):3} vendor figures")

    vendor = con.sql("SELECT * FROM vendor_fundamentals").df()
    annual = con.sql("SELECT *, 'annual' AS frequency FROM fundamentals").df()
    quarterly = con.sql("SELECT *, 'quarterly' AS frequency FROM fundamentals_quarterly").df()
    if annual.empty:
        raise SystemExit("No SEC fundamentals loaded: run `python -m eval_lab.ingest_fundamentals` first.")
    splits = con.sql("SELECT ticker, date, value FROM corporate_actions WHERE action_type = 'split'").df()
    splits["date"] = pd.to_datetime(splits["date"])
    sec = pd.concat([annual, quarterly], ignore_index=True)
    sec["period_end"] = pd.to_datetime(sec["period_end"])
    earnings = con.sql("SELECT * FROM earnings").df()

    detail = pd.concat([
        reconcile_statements(vendor[vendor["ticker"].isin(tickers)], sec, splits),
        reconcile_street_eps(earnings[earnings["ticker"].isin(tickers)], sec[sec["frequency"] == "quarterly"], splits),
    ], ignore_index=True)
    summary = detail.groupby(["metric", "frequency", "status"]).size().rename("n").reset_index()
    summary["share_of_metric"] = summary["n"] / summary.groupby(["metric", "frequency"])["n"].transform("sum")
    write_report(detail, "reconciliation.csv")
    write_report(summary, "reconciliation_summary.csv")

    print("\n" + summary.pivot_table(index=["metric", "frequency"], columns="status", values="n", fill_value=0).to_string())
    offsets = detail["period_end_offset_days"].dropna()
    if len(offsets):
        print(f"\nvendor period-end label differs from the filing's in {(offsets != 0).mean():.0%} of matches "
              f"(median {offsets[offsets != 0].abs().median():.0f} days)")


if __name__ == "__main__":
    main()
