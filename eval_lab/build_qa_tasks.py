"""Build the financial Q&A benchmark: templated questions whose gold answers are
computed from the source data, then a stratified random sample of them.

Every template also records the plausible *wrong* answers (prior year's figure,
split-adjusted EPS, price return instead of total return, ...), so whoever
answers the questions (a person, a model, a vendor feed) can be scored on *why*
an answer was wrong, not just whether it was.

    python -m eval_lab.build_qa_tasks

Writes the sample to eval.duckdb (qa_tasks) and reports/qa_benchmark.csv.
"""
import json

import numpy as np
import pandas as pd

from eval_lab.db import eval_db, load_config, source_db, write_report
from eval_lab.sampling import representativeness, stratified_sample

TASK_TYPES = {
    "lookup_revenue": "factual_lookup",
    "lookup_eps": "factual_lookup",
    "lookup_q4_revenue": "factual_lookup",
    "calc_revenue_growth": "calculation",
    "calc_operating_margin": "calculation",
    "calc_pe": "market_calc",
    "market_total_return": "market_calc",
    "earnings_reaction": "market_calc",
    "reasoning_margin_drivers": "reasoning",
}


class Facts:
    """Lookups over the source data that the templates share."""

    def __init__(self, con):
        f = con.sql("SELECT * FROM fundamentals").df()
        if f.empty:
            raise SystemExit("No fundamentals loaded: run `python -m eval_lab.ingest_fundamentals` first.")
        index = ["ticker", "fiscal_year"]
        self.latest = f.pivot_table(index=index, columns="metric", values="value", aggfunc="last")
        self.first = f.pivot_table(index=index, columns="metric", values="first_reported_value", aggfunc="last")
        self.concept = f.pivot_table(index=index, columns="metric", values="source_concept", aggfunc="last")
        self.period_end = f.groupby(index)["period_end"].max()
        self.companies = con.sql("SELECT * FROM companies").df().set_index("ticker")
        prices = con.sql("SELECT ticker, date, close, adj_close, close_unadjusted FROM prices ORDER BY date").df()
        self.prices = {t: g.set_index("date") for t, g in prices.groupby("ticker")}
        self.splits = con.sql("SELECT ticker, date, value FROM corporate_actions WHERE action_type = 'split'").df()
        self.quarterly = con.sql("SELECT * FROM fundamentals_quarterly").df()
        self.earnings = con.sql("SELECT * FROM earnings WHERE timing != 'unknown' AND reaction_date IS NOT NULL").df()

    def get(self, table: pd.DataFrame, ticker: str, fy: int, metric: str) -> float | None:
        try:
            value = table.at[(ticker, fy), metric]
        except KeyError:
            return None
        return None if pd.isna(value) else float(value)

    def concept_for(self, ticker: str, fy: int, metric: str) -> str | None:
        try:
            return self.concept.at[(ticker, fy), metric]
        except KeyError:
            return None

    def name(self, ticker: str) -> str:
        name = self.companies.at[ticker, "name"] if ticker in self.companies.index else None
        return name if isinstance(name, str) and name else ticker

    def price_on_or_before(self, ticker: str, date) -> pd.Series | None:
        g = self.prices.get(ticker)
        if g is None:
            return None
        window = g.loc[:pd.Timestamp(date)]
        return window.iloc[-1] if len(window) else None

    def quarter(self, ticker: str, fy: int, q: int, metric: str) -> pd.Series | None:
        rows = self.quarterly[(self.quarterly["ticker"] == ticker) & (self.quarterly["fiscal_year"] == fy)
                              & (self.quarterly["fiscal_quarter"] == q) & (self.quarterly["metric"] == metric)]
        return rows.iloc[-1] if len(rows) else None

    def session_return(self, ticker: str, date, column: str = "adj_close") -> float | None:
        """Close-to-close return of the session on `date` (from the prior session's close)."""
        g = self.prices.get(ticker)
        if g is None or pd.Timestamp(date) not in g.index:
            return None
        i = g.index.get_loc(pd.Timestamp(date))
        return float(g[column].iloc[i] / g[column].iloc[i - 1] - 1) if i > 0 else None

    def neighbor_session(self, ticker: str, date, offset: int):
        g = self.prices.get(ticker)
        i = g.index.get_loc(pd.Timestamp(date)) + offset
        return g.index[i] if 0 <= i < len(g) else None

    def split_factor_after(self, ticker: str, date) -> float:
        later = self.splits[(self.splits["ticker"] == ticker) & (self.splits["date"] > pd.Timestamp(date))]
        return float(later["value"].prod()) if len(later) else 1.0

    def traps(self, ticker: str) -> list[str]:
        month = self.companies.at[ticker, "fiscal_year_end_month"] if ticker in self.companies.index else None
        return ["fiscal_calendar"] if pd.notna(month) and int(month) != 12 else []


def task(template: str, ticker: str, fy: int, prompt: str, gold, unit: str, tol_type: str, tolerance: float,
         alts: dict, plausible: tuple, traps: list, reference: dict) -> dict:
    return {
        "task_id": f"{template}-{ticker}-{fy}", "template_id": template, "task_type": TASK_TYPES[template],
        "ticker": ticker, "fiscal_year": fy, "prompt": prompt, "gold_value": gold, "unit": unit,
        "tol_type": tol_type, "tolerance": tolerance,
        "alt_values": json.dumps({k: v for k, v in alts.items() if v is not None and gold is not None and not np.isclose(v, gold, rtol=1e-3)}),
        "plausible_low": plausible[0], "plausible_high": plausible[1],
        "trap_tags": ",".join(sorted(set(traps))), "reference": json.dumps(reference, default=str),
    }


def pct(a, b):
    return None if a is None or b in (None, 0) else (a / b - 1) * 100


def neighbor_range(values: list, low=0.5, high=2.0) -> tuple:
    values = [v for v in values if v is not None]
    return (low * min(values), high * max(values)) if values else (None, None)


def templates_for(x: Facts, ticker: str, fy: int) -> list[dict]:
    out = []
    name, end = x.name(ticker), x.period_end.get((ticker, fy))
    if end is None:
        return out
    L, F = x.latest, x.first
    rev, rev_prev, rev_next = (x.get(L, ticker, y, "revenue") for y in (fy, fy - 1, fy + 1))
    opinc, opinc_prev = x.get(L, ticker, fy, "operating_income"), x.get(L, ticker, fy - 1, "operating_income")
    eps_first, eps_latest = x.get(F, ticker, fy, "eps_diluted"), x.get(L, ticker, fy, "eps_diluted")
    split_after = x.split_factor_after(ticker, end)
    # What the EPS looks like on today's share basis. When no later filing restates
    # the year (Netflix's 2020 EPS after its 2025 10-for-1 split), compute it.
    eps_today = eps_latest if eps_latest is not None and eps_first is not None and not np.isclose(eps_latest, eps_first) \
        else (eps_first / split_after if eps_first is not None and split_after > 1 else None)
    base_traps = x.traps(ticker)
    period = {"fiscal_year": fy, "period_end": f"{end:%Y-%m-%d}"}

    if rev is not None:
        first_rev = x.get(F, ticker, fy, "revenue")
        out.append(task(
            "lookup_revenue", ticker, fy,
            f"What was {name}'s ({ticker}) total revenue for fiscal year {fy}, as reported in its 10-K? Answer in USD millions.",
            rev / 1e6, "USD millions", "rel", 0.005,
            {"prior_year": rev_prev and rev_prev / 1e6, "next_year": rev_next and rev_next / 1e6,
             "first_reported": first_rev and first_rev / 1e6},
            tuple(v and v / 1e6 for v in neighbor_range([rev_prev, rev_next])),
            base_traps + (["restated"] if first_rev and not np.isclose(first_rev, rev) else []),
            {**period, "revenue_usd": rev, "first_reported_revenue_usd": first_rev,
             "xbrl_concept": x.concept_for(ticker, fy, "revenue")},
        ))

    if eps_first is not None:
        out.append(task(
            "lookup_eps", ticker, fy,
            f"What diluted EPS did {name} ({ticker}) report for fiscal year {fy} in that year's 10-K "
            "(as originally reported, not adjusted for later stock splits)? Answer in USD per share.",
            eps_first, "USD/share", "rel", 0.01,
            {"split_adjusted": eps_today, "prior_year": x.get(F, ticker, fy - 1, "eps_diluted"),
             "next_year": x.get(F, ticker, fy + 1, "eps_diluted")},
            (None, None),
            base_traps + (["split_adjustment"] if split_after > 1 else []),
            {**period, "eps_diluted_as_reported": eps_first, "eps_diluted_split_adjusted_today": eps_latest,
             "stock_splits_since": split_after},
        ))

    if rev is not None and rev_prev:
        prior_growth = pct(rev_prev, x.get(L, ticker, fy - 2, "revenue"))
        out.append(task(
            "calc_revenue_growth", ticker, fy,
            f"By what percentage did {name}'s ({ticker}) revenue grow in fiscal year {fy} versus fiscal year {fy - 1}? "
            "Answer in percent, to one decimal place.",
            pct(rev, rev_prev), "percent", "abs", 0.1,
            {"prior_year_growth": prior_growth}, (-60.0, 150.0), base_traps,
            {**period, f"revenue_fy{fy}_usd": rev, f"revenue_fy{fy - 1}_usd": rev_prev},
        ))

    if rev and opinc is not None:
        net = x.get(L, ticker, fy, "net_income")
        prev_margin = opinc_prev / rev_prev * 100 if opinc_prev is not None and rev_prev else None
        out.append(task(
            "calc_operating_margin", ticker, fy,
            f"What was {name}'s ({ticker}) operating margin in fiscal year {fy}? Answer in percent, to one decimal place.",
            opinc / rev * 100, "percent", "abs", 0.1,
            {"net_margin_instead": net and net / rev * 100, "prior_year": prev_margin}, (-50.0, 70.0), base_traps,
            {**period, "revenue_usd": rev, "operating_income_usd": opinc, "net_income_usd": net},
        ))

    q4 = x.quarter(ticker, fy, 4, "revenue")
    if q4 is not None and q4["is_derived"] and rev:
        q3 = x.quarter(ticker, fy, 3, "revenue")
        out.append(task(
            "lookup_q4_revenue", ticker, fy,
            f"What was {name}'s ({ticker}) revenue for the fourth quarter of fiscal year {fy} (the three months ended "
            f"{q4['period_end']:%B %d, %Y})? Answer in USD millions.",
            q4["value"] / 1e6, "USD millions", "rel", 0.005,
            {"full_year": rev / 1e6, "nine_months_ytd": (rev - q4["value"]) / 1e6,
             "third_quarter": None if q3 is None else q3["value"] / 1e6},
            (0.1 * rev / 4 / 1e6, 0.6 * rev / 1e6), base_traps + ["derived_q4"],
            {**period, "q4_revenue_usd": q4["value"], "full_year_revenue_usd": rev,
             "derivation": q4["derivation"], "q4_period_start": f"{q4['period_start']:%Y-%m-%d}"},
        ))

    close = x.price_on_or_before(ticker, end)
    if eps_first and eps_first > 0 and close is not None:
        traded, adjusted = float(close["close_unadjusted"]), float(close["close"])
        out.append(task(
            "calc_pe", ticker, fy,
            f"Compute {name}'s ({ticker}) trailing P/E ratio at the end of fiscal year {fy}: the closing share price "
            f"on the last trading day on or before the fiscal year-end date ({end:%B %d, %Y}), divided by diluted EPS "
            f"for fiscal year {fy} as reported in that year's 10-K. Round to one decimal place.",
            traded / eps_first, "ratio", "rel", 0.02,
            {"split_adjusted_price": adjusted / eps_first,
             "split_adjusted_eps": eps_today and traded / eps_today},
            (0.0, 150.0), base_traps + ["point_in_time"] + (["split_adjustment"] if split_after > 1 else []),
            {**period, "price_date": f"{close.name:%Y-%m-%d}", "close_as_traded": traded,
             "close_split_adjusted_today": adjusted, "eps_diluted_as_reported": eps_first},
        ))

    start, stop = x.price_on_or_before(ticker, f"{fy - 1}-12-31"), x.price_on_or_before(ticker, f"{fy}-12-31")
    if start is not None and stop is not None and start.name.year == fy - 1:
        total = pct(float(stop["adj_close"]), float(start["adj_close"]))
        price_only = pct(float(stop["close"]), float(start["close"]))
        out.append(task(
            "market_total_return", ticker, fy,
            f"What was {name}'s ({ticker}) total shareholder return in calendar year {fy}, including reinvested dividends? "
            f"Use closing prices on the last trading day of {fy - 1} and of {fy}. Answer in percent, to one decimal place.",
            total, "percent", "abs", 0.5,
            {"price_return_only": price_only}, (-80.0, 300.0),
            ["total_vs_price_return"] if abs(total - price_only) > 0.5 else [],
            {"calendar_year": fy, "start_date": f"{start.name:%Y-%m-%d}", "end_date": f"{stop.name:%Y-%m-%d}",
             "total_return_pct": total, "price_return_pct": price_only},
        ))

    if rev and rev_prev and opinc is not None and opinc_prev is not None:
        out.append(task(
            "reasoning_margin_drivers", ticker, fy,
            f"Explain what drove the change in {name}'s ({ticker}) operating margin from fiscal year {fy - 1} to fiscal "
            f"year {fy}. Quantify revenue growth, operating income growth and the margin change in percentage points, "
            "then give the most likely explanation.",
            None, "text", None, None, {}, (None, None), base_traps,
            {**period, f"revenue_fy{fy}_usd": rev, f"revenue_fy{fy - 1}_usd": rev_prev,
             f"operating_income_fy{fy}_usd": opinc, f"operating_income_fy{fy - 1}_usd": opinc_prev,
             "revenue_growth_pct": pct(rev, rev_prev), "operating_income_growth_pct": pct(opinc, opinc_prev),
             "margin_change_pp": (opinc / rev - opinc_prev / rev_prev) * 100},
        ))
    return out


TIMING_TEXT = {"before_open": "before the market opened", "after_close": "after the market closed",
               "during_market": "during market hours"}


def earnings_templates(x: Facts, ticker: str, benchmark: str, years: list[int]) -> list[dict]:
    """One task per earnings announcement: the abnormal return of the first session
    that could trade on the news. Using the announcement-day session for an
    after-close release is the trap."""
    out = []
    events = x.earnings[(x.earnings["ticker"] == ticker) & x.earnings["announced_at_et"].dt.year.isin(years)]
    for e in events.itertuples():
        react = pd.Timestamp(e.reaction_date)
        stock, market = x.session_return(ticker, react), x.session_return(benchmark, react)
        if stock is None or market is None:
            continue
        wrong = x.neighbor_session(ticker, react, -1 if e.timing == "after_close" else 1)
        wrong_abnormal = None
        if wrong is not None:
            ws, wm = x.session_return(ticker, wrong), x.session_return(benchmark, wrong)
            wrong_abnormal = None if ws is None or wm is None else (ws - wm) * 100
        t = task(
            "earnings_reaction", ticker, int(e.announced_at_et.year),
            f"{x.name(ticker)} ({ticker}) announced quarterly results on {e.announced_at_et:%B %d, %Y} at "
            f"{e.announced_at_et.strftime('%I:%M %p').lstrip('0')} New York time, {TIMING_TEXT[e.timing]}. What was the stock's abnormal return "
            "versus SPY (stock return minus SPY return, close to close, dividends included) in the first full trading "
            "session that could react to the news? Answer in percent, to two decimal places.",
            (stock - market) * 100, "percent", "abs", 0.05,
            {"raw_return_not_abnormal": stock * 100, "wrong_session": wrong_abnormal},
            (-40.0, 40.0), ["announcement_timing", "abnormal_vs_raw"],
            {"announced_at_et": f"{e.announced_at_et:%Y-%m-%d %H:%M}", "timing": e.timing,
             "reaction_session": f"{react:%Y-%m-%d}", "stock_return_pct": stock * 100, "spy_return_pct": market * 100,
             "eps_estimate": e.eps_estimate, "eps_reported": e.eps_reported, "surprise_pct": e.surprise_pct},
        )
        t["task_id"] = f"earnings_reaction-{ticker}-{e.announced_at_et:%Y%m%d}"
        out.append(t)
    return out


def cap_terciles(x: Facts) -> pd.Series:
    """Market cap = latest traded price x latest diluted share count, bucketed into
    terciles *within this universe* (so "small" still means a large company)."""
    caps = {}
    shares_by_ticker = x.latest["diluted_shares"].dropna() if "diluted_shares" in x.latest else pd.Series(dtype=float)
    for ticker in x.companies.index:
        close = x.prices.get(ticker)
        shares = shares_by_ticker[shares_by_ticker.index.get_level_values("ticker") == ticker]
        if close is not None and len(shares):
            caps[ticker] = float(close["close_unadjusted"].iloc[-1]) * float(shares.iloc[-1])
    caps = pd.Series(caps)
    return pd.qcut(caps, 3, labels=["small", "mid", "large"]).astype(str) if len(caps) >= 3 else caps.map(lambda _: "unknown")


def main() -> None:
    universe, sampling = load_config("universe.yaml"), load_config("sampling.yaml")
    cfg = sampling["qa"]
    x = Facts(source_db())
    tickers = [c["ticker"] for c in universe["companies"]]
    pool = pd.DataFrame(
        [t for ticker in tickers for fy in universe["fiscal_years"] for t in templates_for(x, ticker, fy)]
        + [t for ticker in tickers for t in earnings_templates(x, ticker, universe["benchmark"], universe["fiscal_years"])]
    )
    pool["sector"] = pool["ticker"].map(x.companies["sector"])
    pool["cap_tercile"] = pool["ticker"].map(cap_terciles(x)).fillna("unknown")

    missing = [t for t in cfg["quotas"] if t not in set(pool["template_id"])]
    if missing:
        print(f"no candidates for {', '.join(missing)} (is the data for them loaded?)")
    sample = pd.concat([
        stratified_sample(pool[pool["template_id"] == template], n, cfg["strata"], sampling["seed"] + i)
        for i, (template, n) in enumerate(cfg["quotas"].items())
    ])
    con = eval_db()
    con.execute("DELETE FROM qa_tasks")
    con.register("_tasks", sample)
    con.execute("INSERT INTO qa_tasks BY NAME SELECT * FROM _tasks")

    summary = pd.concat([
        representativeness(pool[pool["template_id"] == t], sample[sample["template_id"] == t], cfg["strata"]).assign(template_id=t)
        for t in cfg["quotas"] if t in set(pool["template_id"])
    ])
    write_report(summary, "qa_sampling_summary.csv")
    write_report(sample.drop(columns=["reference"]), "qa_benchmark.csv")
    traps = sample["trap_tags"].str.split(",").explode().replace("", np.nan).dropna().value_counts()
    print(f"{len(pool)} candidate tasks -> {len(sample)} sampled")
    print(sample.groupby(["task_type", "template_id"]).size().to_string())
    print("\ntrap coverage in sample:\n" + traps.to_string())


if __name__ == "__main__":
    main()
