"""Annual (10-K) and quarterly (10-Q) fundamentals from SEC EDGAR XBRL
"companyfacts" into source.duckdb.

SEC's fair-access policy requires a User-Agent that identifies you:

    export SEC_USER_AGENT="Your Name your.email@example.com"
    python -m eval_lab.ingest_fundamentals [--tickers AAPL NVDA]

Every 10-K repeats the prior years' figures, so each period appears in several
filings. We keep the latest value (what a data vendor shows today) and the
first reported value (what investors saw at the time), and count distinct
versions: more than one means a restatement or, for per-share figures, a split.

Companies don't file a Q4 report: the fourth quarter only exists inside the
10-K's full-year figure. For additive metrics (revenue, operating income, net
income) Q4 is derived as full year minus the 9-month year-to-date figure from
the Q3 10-Q, and marked `is_derived`. Both figures are taken as originally
reported: later 10-Ks recast the full year for discontinued operations, but old
10-Qs are never recast, so mixing vintages would dump the restatement into Q4. Per-share EPS is not additive (the share
count changes every quarter), so Q4 EPS is left missing rather than guessed.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

from eval_lab.db import load_config, replace_rows, source_db
from eval_lab.ingest_market import sync_companies

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
PERIODS = {
    # 52/53-week fiscal years run 364 or 371 days; quarters 91 or 98.
    "annual": {"days": (350, 380), "forms": {"10-K", "10-K/A"}},
    "quarter": {"days": (80, 100), "forms": {"10-Q", "10-Q/A", "10-K", "10-K/A"}},
    "nine_months": {"days": (260, 285), "forms": {"10-Q", "10-Q/A"}},
}
QUARTERLY_METRICS = ["revenue", "operating_income", "net_income", "eps_diluted", "diluted_shares"]
ADDITIVE = {"revenue", "operating_income", "net_income"}


def sec_get(session: requests.Session, url: str) -> dict:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    time.sleep(0.15)  # SEC allows at most 10 requests per second
    return response.json()


def fiscal_year_label(period_end: pd.Timestamp, convention: str) -> int:
    """52/53-week years can end a few days into January: J&J's fiscal 2020 ended
    Jan 3, 2021. Stepping back 10 days labels those by the year they mostly cover,
    while late-January year ends (NVDA, WMT) keep their end-year label."""
    if convention == "start_year":
        return period_end.year - 1
    return (period_end - pd.Timedelta(days=10)).year


def extract_metric(facts: dict, ticker: str, metric: str, spec: dict, convention: str,
                   period: str = "annual") -> pd.DataFrame:
    low, high = PERIODS[period]["days"]
    forms = PERIODS[period]["forms"]
    rows = []
    for rank, concept in enumerate(spec["concepts"]):
        taxonomy, _, name = concept.partition(":")
        units = facts.get("facts", {}).get(taxonomy, {}).get(name, {}).get("units", {})
        for obs in units.get(spec["unit"], []):
            if obs.get("form") not in forms:
                continue
            end = pd.Timestamp(obs["end"])
            start = pd.Timestamp(obs["start"]) if "start" in obs else pd.NaT
            if spec["kind"] == "duration" and (pd.isna(start) or not low <= (end - start).days <= high):
                continue
            rows.append({
                "ticker": ticker, "metric": metric, "source_concept": concept, "rank": rank,
                "unit": spec["unit"], "period_start": start, "period_end": end,
                "value": float(obs["val"]), "accn": obs["accn"], "filed": pd.Timestamp(obs["filed"]),
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("period_end")
    # Filers sometimes date the same period differently across filings (Deere's
    # FY2015 ends Oct 31 in one 10-K and Nov 1 in the next); treat end dates within
    # two weeks of each other as one period and keep the latest filing's dates.
    df["period"] = (df["period_end"].diff().dt.days.fillna(np.inf) > 14).cumsum()
    if spec.get("select") == "largest":
        # Some filers tag a subtotal with a generic concept (P&G's 2014 10-K tags a
        # ~$28B figure as us-gaap:Revenues next to $84B of SalesRevenueNet). A total
        # is never smaller than its parts, so take the concept with the largest
        # latest value for each period; priority order breaks ties.
        latest_by_concept = df.sort_values("filed").groupby(["period", "rank"])["value"].last().reset_index()
        latest_by_concept["size"] = latest_by_concept["value"].abs()
        chosen = latest_by_concept.sort_values(["period", "size", "rank"], ascending=[True, False, True]).drop_duplicates("period")
        df = df.merge(chosen[["period", "rank"]], on=["period", "rank"])
    else:
        df = df[df["rank"] == df.groupby("period")["rank"].transform("min")]
    df = df.sort_values("filed")
    versions = df.groupby("period")["value"]
    latest = df.groupby("period").tail(1).set_index("period")
    latest["first_reported_value"] = versions.first()
    latest["n_versions"] = versions.nunique()
    latest = latest.reset_index(drop=True).drop(columns="rank")
    latest["fiscal_year"] = latest["period_end"].map(lambda d: fiscal_year_label(d, convention))
    return latest


def merge_facts(fact_sets: list[dict]) -> dict:
    """Combine companyfacts from several filer entities (e.g. a company and the
    holding company that later replaced it) into one set of observations."""
    merged = {"entityName": " / ".join(f.get("entityName", "") for f in fact_sets), "facts": {}}
    for facts in fact_sets:
        for taxonomy, concepts in facts.get("facts", {}).items():
            for concept, body in concepts.items():
                units = merged["facts"].setdefault(taxonomy, {}).setdefault(concept, {"units": {}})["units"]
                for unit, observations in body.get("units", {}).items():
                    units.setdefault(unit, []).extend(observations)
    return merged


def company_fundamentals(facts: dict, ticker: str, metrics: dict, convention: str) -> pd.DataFrame:
    frames = [f for f in (extract_metric(facts, ticker, m, spec, convention) for m, spec in metrics.items()) if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Balance sheets in a 10-K can include dates that aren't fiscal year ends; keep
    # instant values only on the period ends that annual income statements use.
    duration = [m for m, spec in metrics.items() if spec["kind"] == "duration"]
    year_ends = set(df.loc[df["metric"].isin(duration), "period_end"])
    return df[df["metric"].isin(duration) | df["period_end"].isin(year_ends)]


def assign_fiscal_quarters(df: pd.DataFrame, fiscal_years: pd.DataFrame) -> pd.DataFrame:
    """Place each sub-annual period inside the fiscal year that contains it and
    number it by how far its end date is from the start of that year."""
    out = []
    for fy in fiscal_years.itertuples():
        inside = df[(df["period_start"] >= fy.period_start - pd.Timedelta(days=7))
                    & (df["period_end"] <= fy.period_end + pd.Timedelta(days=7))].copy()
        inside["fiscal_year"] = fy.fiscal_year
        inside["fiscal_quarter"] = ((inside["period_end"] - fy.period_start).dt.days / 91.3).round().clip(1, 4).astype(int)
        out.append(inside)
    return pd.concat(out, ignore_index=True) if out else df.head(0).assign(fiscal_quarter=pd.Series(dtype=int))


def quarterly_fundamentals(facts: dict, ticker: str, metrics: dict, annual: pd.DataFrame) -> pd.DataFrame:
    fiscal_years = (annual[annual["period_start"].notna()].sort_values("period_end")
                    .drop_duplicates("fiscal_year", keep="last")[["fiscal_year", "period_start", "period_end"]])
    if len(fiscal_years):
        # The fiscal year in progress has 10-Qs but no 10-K yet; give its quarters a home.
        last = fiscal_years.iloc[-1]
        start = last["period_end"] + pd.Timedelta(days=1)
        fiscal_years = pd.concat([fiscal_years, pd.DataFrame([{
            "fiscal_year": last["fiscal_year"] + 1, "period_start": start,
            "period_end": start + pd.DateOffset(years=1) - pd.Timedelta(days=1)}])], ignore_index=True)
    frames = []
    for metric in [m for m in QUARTERLY_METRICS if m in metrics]:
        spec = metrics[metric]
        quarters = extract_metric(facts, ticker, metric, spec, "end_year", "quarter")
        if quarters.empty:
            quarters = pd.DataFrame(columns=["fiscal_year", "fiscal_quarter", "period_end", "value",
                                             "first_reported_value", "source_concept"])
        else:
            quarters = assign_fiscal_quarters(quarters.drop(columns="fiscal_year"), fiscal_years)
            quarters = quarters.drop_duplicates(["fiscal_year", "fiscal_quarter"], keep="last").assign(is_derived=False, derivation=None)
            frames.append(quarters)
        if metric not in ADDITIVE:
            continue

        nine = extract_metric(facts, ticker, metric, spec, "end_year", "nine_months")
        nine = assign_fiscal_quarters(nine.drop(columns="fiscal_year"), fiscal_years) if not nine.empty else nine
        full_year = annual[annual["metric"] == metric].set_index("fiscal_year")
        reported = set(zip(quarters["fiscal_year"], quarters["fiscal_quarter"]))
        derived = []
        for fy, year in full_year.iterrows():
            if (fy, 4) in reported:
                continue
            ytd = nine[nine["fiscal_year"] == fy] if not nine.empty else nine
            q123 = quarters[(quarters["fiscal_year"] == fy) & quarters["fiscal_quarter"].isin([1, 2, 3])]
            # Both figures must use the same XBRL concept, or the difference mixes definitions.
            ytd = ytd[ytd["source_concept"] == year["source_concept"]] if len(ytd) else ytd
            q123 = q123[q123["source_concept"] == year["source_concept"]]
            if len(ytd):
                ytd = ytd.iloc[-1]
                first = year["first_reported_value"] - ytd["first_reported_value"]
                start, how = ytd["period_end"] + pd.Timedelta(days=1), "as reported: full year minus 9-month YTD"
            elif len(q123) == 3:
                first = year["first_reported_value"] - q123["first_reported_value"].sum()
                start, how = q123["period_end"].max() + pd.Timedelta(days=1), "as reported: full year minus Q1+Q2+Q3"
            else:
                continue
            value, versions = first, 1
            derived.append({
                "ticker": ticker, "metric": metric, "fiscal_year": fy, "fiscal_quarter": 4,
                "period_start": start, "period_end": year["period_end"], "value": value,
                "first_reported_value": first, "n_versions": versions, "unit": spec["unit"],
                "source_concept": year["source_concept"], "accn": year["accn"], "filed": year["filed"],
                "is_derived": True, "derivation": how,
            })
        frames.append(pd.DataFrame(derived))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    universe = load_config("universe.yaml")
    metrics = load_config("concepts.yaml")["metrics"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()

    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        raise SystemExit('Set SEC_USER_AGENT="Your Name your.email@example.com" (required by SEC).')
    session = requests.Session()
    session.headers["User-Agent"] = user_agent

    companies = {c["ticker"]: c for c in universe["companies"]}
    tickers = args.tickers or list(companies)
    ciks = {row["ticker"]: str(row["cik_str"]).zfill(10) for row in sec_get(session, TICKER_MAP_URL).values()}

    con = source_db()
    sync_companies(con, universe)
    skipped = []
    for ticker in tickers:
        convention = companies[ticker].get("fy_convention", "end_year")
        # SEC's ticker file maps a ticker to its *current* filer. When a company
        # reorganizes under a new parent, the history stays with the old CIK, so
        # universe.yaml can list every CIK whose filings belong to the ticker.
        entity_ciks = [str(c).zfill(10) for c in companies[ticker].get("ciks", [])] or [ciks[ticker]]
        facts = merge_facts([sec_get(session, FACTS_URL.format(cik=cik)) for cik in entity_ciks])
        df = company_fundamentals(facts, ticker, metrics, convention)
        if df.empty:
            skipped.append(ticker)
            print(f"{ticker:6} no annual 10-K data under CIK {', '.join(entity_ciks)} ({facts.get('entityName')}); "
                  "if the ticker moved to a new filer, add its old CIK under `ciks` in universe.yaml")
            continue
        replace_rows(con, "fundamentals", df, "ticker")
        quarterly = quarterly_fundamentals(facts, ticker, metrics, df)
        replace_rows(con, "fundamentals_quarterly", quarterly, "ticker", keys=[ticker])

        revenue_ends = df.loc[df["metric"] == "revenue", "period_end"]
        fye_month = int(revenue_ends.dt.month.mode().iloc[0]) if len(revenue_ends) else None
        con.execute(
            "UPDATE companies SET cik = ?, name = ?, fiscal_year_end_month = ? WHERE ticker = ?",
            [",".join(entity_ciks), facts.get("entityName"), fye_month, ticker],
        )
        restated = int((df["n_versions"] > 1).sum())
        print(f"{ticker:6} {len(df):4} rows  {df['fiscal_year'].min()}-{df['fiscal_year'].max()}  "
              f"FYE month {fye_month}  {restated} multi-version values  "
              f"{len(quarterly)} quarterly rows ({int(quarterly.get('is_derived', pd.Series(dtype=bool)).sum())} derived Q4)")
    if skipped:
        raise SystemExit(f"\nskipped (no data): {' '.join(skipped)}")


if __name__ == "__main__":
    main()
