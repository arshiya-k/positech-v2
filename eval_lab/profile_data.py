"""Data-quality profile of the source database, plus a price anomaly detector
whose precision and recall are measured by injecting known errors.

    python -m eval_lab.profile_data [--z 6]

Writes:
    reports/data_quality.csv        one row per issue found (check, severity, ticker, detail)
    reports/price_anomalies.csv     extreme moves in the real data, labeled with what explains them
    reports/price_detector_pr.csv   precision / recall by threshold and error type
"""
import argparse

import numpy as np
import pandas as pd

from eval_lab import companies
from eval_lab.db import load_config, source_db, write_report

SPLIT_MATCH = 0.02         # a multi-version EPS is "split-adjusted" if it moved by the split ratio within 2%
Z_THRESHOLDS = [3, 4, 5, 6, 8, 10]
INJECTION_KINDS = ["spike", "decimal_slip", "missed_split"]


def issue(check: str, severity: str, ticker: str | None = None, detail: str = "", **extra) -> dict:
    return {"check": check, "severity": severity, "ticker": ticker, "detail": detail, **extra}


# ---------------------------------------------------------------- fundamentals

def check_fundamentals(con, universe: dict) -> list[dict]:
    f = con.sql("SELECT * FROM fundamentals").df()
    if f.empty:
        return [issue("fundamentals_loaded", "warn", detail="no fundamentals yet: run ingest_fundamentals")]
    out = []
    years = universe["fiscal_years"]
    have = set(zip(f["ticker"], f["metric"], f["fiscal_year"]))
    for metric in sorted(f["metric"].unique()):
        for ticker in [c["ticker"] for c in universe["companies"]]:
            missing = [y for y in years if (ticker, metric, y) not in have]
            if missing:
                out.append(issue("missing_value", "warn", ticker, f"{metric} missing for FY{', FY'.join(map(str, missing))}", metric=metric))

    dup = f.groupby(["ticker", "metric", "fiscal_year"]).size()
    for (ticker, metric, fy), n in dup[dup > 1].items():
        out.append(issue("duplicate_fiscal_year", "error", ticker, f"{metric} FY{fy} maps to {n} period ends", metric=metric))

    splits = con.sql("SELECT ticker, date, value FROM corporate_actions WHERE action_type = 'split'").df()
    for r in f[f["n_versions"] > 1].itertuples():
        later = splits[(splits["ticker"] == r.ticker) & (splits["date"] > r.period_end)]
        cumulative = float(later["value"].prod()) if len(later) else 1.0
        moved = r.first_reported_value / r.value if r.value else np.nan
        if r.metric == "diluted_shares":
            moved = 1 / moved if moved else np.nan
        explained = cumulative > 1 and abs(moved / cumulative - 1) < SPLIT_MATCH
        change = r.value / r.first_reported_value - 1 if r.first_reported_value else np.nan
        out.append(issue(
            "split_adjusted" if explained else "restated", "info", r.ticker,
            f"{r.metric} FY{r.fiscal_year}: first reported {r.first_reported_value:,.4g}, latest {r.value:,.4g} ({change:+.1%})",
            metric=r.metric,
        ))

    for r in f[(f["metric"] == "revenue") & (f["value"].abs() < 1e8)].itertuples():
        out.append(issue("scale_suspect", "warn", r.ticker, f"revenue FY{r.fiscal_year} = {r.value:,.0f} USD looks like thousands or millions", metric="revenue"))
    for r in f[(f["metric"] == "eps_diluted") & (f["value"].abs() > 1000)].itertuples():
        out.append(issue("scale_suspect", "warn", r.ticker, f"EPS FY{r.fiscal_year} = {r.value:,.2f}", metric="eps_diluted"))

    wide = f.pivot_table(index=["ticker", "fiscal_year"], columns="metric", values="value")
    if {"operating_income", "revenue"} <= set(wide.columns):
        for (ticker, fy), row in wide[wide["operating_income"] > wide["revenue"]].iterrows():
            out.append(issue("identity_violation", "error", ticker, f"FY{fy} operating income exceeds revenue"))

    companies = con.sql("SELECT ticker, fiscal_year_end_month, fy_convention FROM companies").df()
    for r in companies[companies["fiscal_year_end_month"].notna() & (companies["fiscal_year_end_month"] != 12)].itertuples():
        out.append(issue("non_calendar_fiscal_year", "info", r.ticker, f"fiscal year ends in month {int(r.fiscal_year_end_month)}, labelled by {r.fy_convention}"))
    return out


def check_quarterly(con, universe: dict) -> list[dict]:
    q = con.sql("SELECT * FROM fundamentals_quarterly").df()
    if q.empty:
        return [issue("quarterly_loaded", "warn", detail="no quarterly fundamentals yet: run ingest_fundamentals")]
    # Quarters and full years are restated at different times (a spin-off recasts
    # later filings but never the original 10-Qs), so compare within a vintage:
    # latest vs latest, and as-first-reported vs as-first-reported. Only flag a year
    # when neither vintage adds up.
    annual = con.sql("SELECT ticker, metric, fiscal_year, value, first_reported_value FROM fundamentals").df()
    out = []
    counts = q[q["metric"] == "revenue"].groupby(["ticker", "fiscal_year"])["fiscal_quarter"].nunique()
    for (ticker, fy), n in counts[(counts < 4) & counts.index.get_level_values(1).isin(universe["fiscal_years"])].items():
        out.append(issue("missing_quarters", "warn", ticker, f"revenue FY{fy}: only {n} of 4 quarters", metric="revenue"))

    additive = q[q["metric"].isin(["revenue", "operating_income", "net_income"])]
    sums = additive.groupby(["ticker", "metric", "fiscal_year"]).agg(
        latest=("value", "sum"), first=("first_reported_value", "sum"), n=("fiscal_quarter", "nunique"))
    sums = sums[sums["n"] == 4].join(annual.set_index(["ticker", "metric", "fiscal_year"]), how="inner")
    def off(q, y):
        return (q - y).abs() > 0.005 * y.abs()
    bad = sums[off(sums["latest"], sums["value"]) & off(sums["first"], sums["first_reported_value"])]
    first_year = min(universe["fiscal_years"])
    for (ticker, metric, fy), r in bad.iterrows():
        out.append(issue(
            "quarter_sum_mismatch", "warn" if fy >= first_year else "info", ticker,
            f"{metric} FY{fy}: quarters sum to {r['latest']:,.0f} vs full year {r['value']:,.0f} ({r['latest'] / r['value'] - 1:+.2%}) "
            f"on latest figures, {r['first'] / r['first_reported_value'] - 1:+.2%} as first reported",
            metric=metric))
    for r in q[q["is_derived"] & (q["metric"] == "revenue") & (q["value"] <= 0)].itertuples():
        out.append(issue("negative_derived_q4", "error", r.ticker, f"derived Q4 revenue FY{r.fiscal_year} = {r.value:,.0f}", metric="revenue"))
    derived = q[q["is_derived"]]
    if len(derived):
        out.append(issue("derived_q4", "info", None, f"{len(derived)} Q4 values derived from 10-K minus 9-month YTD (Q4 is never filed on its own)"))
    return out


def check_earnings(con) -> list[dict]:
    e = con.sql("SELECT * FROM earnings ORDER BY ticker, announced_at_et").df()
    if e.empty:
        return [issue("earnings_loaded", "warn", detail="no earnings calendar yet: run ingest_market")]
    out = []
    for ticker, g in e.groupby("ticker"):
        gaps = g["announced_at_et"].diff().dt.days
        for when, days in zip(g["announced_at_et"][gaps > 120], gaps[gaps > 120]):
            out.append(issue("earnings_gap", "warn", ticker, f"{days:.0f} days without an announcement before {when:%Y-%m-%d}"))
        unknown = (g["timing"] == "unknown").sum()
        if unknown:
            out.append(issue("earnings_time_unknown", "warn", ticker, f"{unknown} announcements have no time of day, so the reaction session is a guess"))
        if g["eps_estimate"].isna().any():
            out.append(issue("missing_estimate", "info", ticker, f"{int(g['eps_estimate'].isna().sum())} announcements with no consensus estimate"))
    return out


def explain_anomalies(flagged: pd.DataFrame, con) -> pd.DataFrame:
    """Label each flagged move with a known cause: an earnings reaction session
    (or the day after, when news keeps digesting) or a corporate action."""
    earnings = con.sql("SELECT ticker, reaction_date FROM earnings WHERE reaction_date IS NOT NULL").df()
    actions = con.sql("SELECT ticker, date FROM corporate_actions WHERE action_type = 'split'").df()
    def dates(frame, col, offsets):
        return {(t, d + pd.offsets.BDay(k)) for t, d in zip(frame["ticker"], pd.to_datetime(frame[col])) for k in offsets}
    earnings_days, split_days = dates(earnings, "reaction_date", [0, 1]), dates(actions, "date", [-1, 0, 1])
    keys = list(zip(flagged["ticker"], pd.to_datetime(flagged["date"])))
    flagged = flagged.copy()
    flagged["explained_by"] = ["earnings" if k in earnings_days else "split" if k in split_days else "unexplained" for k in keys]
    return flagged


# ---------------------------------------------------------------- prices

def check_prices(con, benchmark: str) -> list[dict]:
    p = con.sql("SELECT * FROM prices ORDER BY ticker, date").df()
    if p.empty:
        return [issue("prices_loaded", "warn", detail="no prices yet: run ingest_market")]
    out = []
    for r in p[(p["close"] <= 0) | (p["low"] > p["high"])].itertuples():
        out.append(issue("invalid_price", "error", r.ticker, f"{r.date:%Y-%m-%d}: close {r.close}, low {r.low}, high {r.high}"))

    bench_days = set(p.loc[p["ticker"] == benchmark, "date"])
    for ticker, g in p.groupby("ticker"):
        missing = bench_days - set(g["date"]) if ticker != benchmark else set()
        if missing:
            out.append(issue("missing_trading_days", "warn", ticker, f"{len(missing)} benchmark trading days missing"))
        runs = (g["close"].diff() != 0).cumsum()
        longest = int(g.groupby(runs).size().max())
        if longest >= 5:
            out.append(issue("stale_price", "warn", ticker, f"close unchanged for {longest} consecutive days"))

    spinoffs = con.sql("SELECT ticker, date, value FROM corporate_actions WHERE action_type = 'spinoff_adjustment'").df()
    for s in spinoffs.itertuples():
        out.append(issue("spinoff_recorded_as_split", "warn", s.ticker,
                         f"{s.date:%Y-%m-%d}: vendor split history has ratio {s.value:g}, a spin-off price adjustment, not a share split"))
    splits = con.sql("SELECT ticker, date, value FROM corporate_actions WHERE action_type = 'split'").df()
    for s in splits.itertuples():
        g = p[p["ticker"] == s.ticker].set_index("date")
        if s.date not in g.index or g.index.get_loc(s.date) == 0:
            continue
        i = g.index.get_loc(s.date)
        adjusted_move = g["close"].iloc[i] / g["close"].iloc[i - 1] - 1
        traded_ratio = g["close_unadjusted"].iloc[i - 1] / g["close_unadjusted"].iloc[i]
        ok = abs(adjusted_move) < 0.25
        out.append(issue(
            "split_adjustment" if ok else "split_not_adjusted", "info" if ok else "error", s.ticker,
            f"{s.value:g}-for-1 on {s.date:%Y-%m-%d}: adjusted close moved {adjusted_move:+.1%}, traded price ratio {traded_ratio:.2f}",
        ))
    return out


# ---------------------------------------------------------------- posts

def check_posts(con, universe: dict) -> list[dict]:
    posts = con.sql("SELECT * FROM posts").df()
    if posts.empty:
        return [issue("posts_loaded", "warn", detail="no posts yet: run ingest_posts")]
    out = []
    for source, g in posts.groupby("source"):
        out.append(issue("date_range", "info", None, f"{source}: {len(g)} posts, {g['posted_at'].min():%Y-%m-%d} to {g['posted_at'].max():%Y-%m-%d}", source=source))
        if g["is_duplicate"].any():
            out.append(issue("duplicate_text", "warn", None, f"{source}: {int(g['is_duplicate'].sum())} posts repeat an earlier post's text", source=source))
        if g["is_retweet"].any():
            out.append(issue("retweets", "info", None, f"{source}: {int(g['is_retweet'].sum())} retweets", source=source))
        drift = (g["vader_stored"] - g["vader_recomputed"]).abs() > 0.2
        if drift.any():
            out.append(issue("vader_not_reproducible", "warn", None, f"{source}: {drift.mean():.0%} of stored VADER scores differ from a re-run on the stored text by >0.2", source=source))
        mentions = pd.Series([companies.mentions(t, k) for t, k in zip(g["text"], g["ticker"])], index=g.index)
        if (~mentions).any():
            out.append(issue("possible_off_topic", "warn", None, f"{source}: {(~mentions).mean():.0%} of posts never mention their ticker or company name", source=source))
    return out


# ---------------------------------------------------------------- anomaly detector

def robust_z(returns: pd.Series, window: int = 60) -> pd.Series:
    """How unusual each return is relative to the trailing window, using median and MAD
    so the anomalies themselves don't inflate the yardstick."""
    center = returns.rolling(window, min_periods=20).median().shift(1)
    spread = (returns - center).abs().rolling(window, min_periods=20).median().shift(1)
    return (returns - center) / (1.4826 * spread)


def detect(prices: pd.DataFrame, benchmark: str, z_threshold: float, market_z: float = 4.0) -> pd.DataFrame:
    """Flag daily moves far outside a stock's own recent volatility, ignoring days
    when the benchmark itself moved abnormally (market-wide shocks)."""
    scored = []
    for ticker, g in prices.sort_values("date").groupby("ticker"):
        g = g[["ticker", "date", "close"]].copy()
        g["ret"] = np.log(g["close"]).diff()
        g["z"] = robust_z(g["ret"])
        scored.append(g)
    scored = pd.concat(scored)
    market = scored.loc[scored["ticker"] == benchmark].set_index("date")["z"].abs()
    scored = scored[scored["ticker"] != benchmark]
    market_shock = scored["date"].map(market).fillna(0) > market_z
    return scored[(scored["z"].abs() > z_threshold) & ~market_shock]


def inject(prices: pd.DataFrame, benchmark: str, n_per_kind: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corrupt a copy of `prices` with known errors. Returns (corrupted, truth)."""
    rng = np.random.default_rng(seed)
    corrupted = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    tickers = [t for t in corrupted["ticker"].unique() if t != benchmark]
    truth = []
    for kind in INJECTION_KINDS:
        for _ in range(n_per_kind):
            ticker = rng.choice(tickers)
            rows = corrupted.index[corrupted["ticker"] == ticker]
            i = rows[int(rng.integers(80, len(rows) - 2))]
            if kind == "spike":           # bad print that reverts the next day
                corrupted.loc[i, "close"] *= 1 + rng.choice([-1, 1]) * rng.uniform(0.15, 0.40)
            elif kind == "decimal_slip":  # price keyed in the wrong unit for a day
                corrupted.loc[i, "close"] *= rng.choice([10, 0.1])
            else:                         # history before a split never adjusted
                corrupted.loc[rows[0]:i - 1, "close"] *= rng.choice([2, 3, 4, 10])
            truth.append({"ticker": ticker, "date": corrupted.loc[i, "date"], "kind": kind,
                          "echo_date": corrupted.loc[i + 1, "date"]})
    return corrupted, pd.DataFrame(truth)


def evaluate_detector(prices: pd.DataFrame, benchmark: str, n_per_kind: int = 25, seed: int = 7) -> pd.DataFrame:
    corrupted, truth = inject(prices, benchmark, n_per_kind, seed)
    injected = set(zip(truth["ticker"], truth["date"]))
    # A one-day spike also makes the next day's reversal look extreme; that's an echo, not a false alarm.
    echoes = set(zip(truth.loc[truth["kind"] != "missed_split", "ticker"], truth.loc[truth["kind"] != "missed_split", "echo_date"]))
    rows = []
    for z in Z_THRESHOLDS:
        before = set(zip(*detect(prices, benchmark, z)[["ticker", "date"]].to_numpy().T)) if len(prices) else set()
        after = set(zip(*detect(corrupted, benchmark, z)[["ticker", "date"]].to_numpy().T))
        new_flags = after - before
        false_alarms = new_flags - injected - echoes
        caught = truth.apply(lambda r: (r["ticker"], r["date"]) in after, axis=1)
        tp = int(caught.sum())
        precision = tp / (tp + len(false_alarms)) if tp + len(false_alarms) else np.nan
        for kind in INJECTION_KINDS + ["all"]:
            mask = truth["kind"] == kind if kind != "all" else slice(None)
            rows.append({
                "z_threshold": z, "error_kind": kind, "injected": int(caught[mask].size),
                "detected": int(caught[mask].sum()), "recall": caught[mask].mean(),
                "precision_all_kinds": precision, "false_alarms": len(false_alarms),
                "baseline_flags_on_clean_data": len(before),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--z", type=float, default=6.0, help="z threshold for the review list (default 6)")
    args = parser.parse_args()

    universe = load_config("universe.yaml")
    benchmark = universe["benchmark"]
    con = source_db()
    issues = pd.DataFrame(check_fundamentals(con, universe) + check_quarterly(con, universe) + check_prices(con, benchmark)
                          + check_earnings(con) + check_posts(con, universe))
    print(f"data quality -> {write_report(issues, 'data_quality.csv')}")
    print(issues.groupby(["severity", "check"]).size().to_string(), "\n")

    prices = con.sql("SELECT ticker, date, close FROM prices").df()
    if prices["ticker"].nunique() > 1:
        flagged = explain_anomalies(detect(prices, benchmark, args.z).sort_values("z", key=abs, ascending=False), con)
        print(f"{len(flagged)} real moves above |z|={args.z:g} -> {write_report(flagged, 'price_anomalies.csv')}")
        print(flagged["explained_by"].value_counts().to_string(), "\n")
        pr = evaluate_detector(prices, benchmark)
        print(f"detector precision/recall -> {write_report(pr, 'price_detector_pr.csv')}")
        print(pr[pr["error_kind"] == "all"][["z_threshold", "recall", "precision_all_kinds", "false_alarms"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
