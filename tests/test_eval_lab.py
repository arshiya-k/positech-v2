import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from eval_lab.build_qa_tasks import Facts, templates_for
from eval_lab.db import PKG
from eval_lab.ingest_fundamentals import extract_metric
from eval_lab.ingest_market import unadjust_for_splits
from eval_lab.ingest_posts import parse_posted_at
from eval_lab.profile_data import evaluate_detector
from eval_lab.rubric import Rubric
from eval_lab.sampling import allocate, stratified_sample


# ---------------------------------------------------------------- sampling

def test_allocate_is_proportional_with_a_floor_of_one():
    sizes = pd.Series({"a": 100, "b": 10, "c": 1})
    alloc = allocate(sizes, 20)
    assert alloc.sum() == 20
    assert (alloc >= 1).all()
    assert alloc["a"] > alloc["b"] >= alloc["c"]


def test_allocate_never_exceeds_stratum_size():
    alloc = allocate(pd.Series({"a": 2, "b": 3}), 100)
    assert alloc.to_dict() == {"a": 2, "b": 3}


def test_allocate_fewer_draws_than_strata_prefers_large_strata():
    alloc = allocate(pd.Series({"a": 50, "b": 40, "c": 5, "d": 5}), 2)
    assert alloc.sum() == 2 and alloc["a"] == 1 and alloc["b"] == 1


def test_stratified_sample_is_reproducible():
    df = pd.DataFrame({"s": list("aaaabbbccd") * 10, "x": range(100)})
    one, two = (stratified_sample(df, 20, ["s"], seed=1) for _ in range(2))
    assert one["x"].tolist() == two["x"].tolist()
    assert set(one["s"]) == set("abcd")


# ---------------------------------------------------------------- ingestion

def test_unadjust_for_splits_restores_traded_price():
    idx = pd.to_datetime(["2020-08-27", "2020-08-28", "2020-08-31"])
    close = pd.Series([125.0, 124.8, 129.0], index=idx)
    splits = pd.Series([0.0, 0.0, 4.0], index=idx)
    assert unadjust_for_splits(close, splits).round(1).tolist() == [500.0, 499.2, 129.0]


def test_parse_posted_at_handles_each_scraper_format():
    assert parse_posted_at("news", pd.Series(["2022-12-09T14:29:00Z"]))[0] == pd.Timestamp("2022-12-09 14:29")
    assert parse_posted_at("reddit", pd.Series([1670594286.0]))[0] == pd.Timestamp("2022-12-09 13:58:06")
    assert parse_posted_at("twitter", pd.Series(["Mon Dec 12 07:23:35 +0000 2022"]))[0] == pd.Timestamp("2022-12-12 07:23:35")


def _obs(val, start, end, filed, form="10-K"):
    return {"val": val, "start": start, "end": end, "filed": filed, "form": form, "accn": filed}


def test_extract_metric_tracks_versions_and_skips_quarters():
    facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [
        _obs(12.00, "2019-09-29", "2020-09-26", "2020-10-30"),   # as reported
        _obs(3.00, "2019-09-29", "2020-09-26", "2021-10-29"),    # same year, split-adjusted in a later 10-K
        _obs(2.50, "2020-06-28", "2020-09-26", "2020-10-30"),    # a quarter inside the 10-K
        _obs(9.99, "2019-09-29", "2020-09-26", "2020-08-01", form="10-Q"),
    ]}}}}}
    spec = {"kind": "duration", "unit": "USD/shares", "concepts": ["us-gaap:EarningsPerShareDiluted"]}
    df = extract_metric(facts, "AAPL", "eps_diluted", spec, "end_year")
    assert len(df) == 1
    row = df.iloc[0]
    assert (row["value"], row["first_reported_value"], row["n_versions"], row["fiscal_year"]) == (3.0, 12.0, 2, 2020)


def test_extract_metric_start_year_convention():
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [_obs(1e11, "2023-01-30", "2024-01-28", "2024-03-13")]}}}}}
    spec = {"kind": "duration", "unit": "USD", "concepts": ["us-gaap:Revenues"]}
    assert extract_metric(facts, "HD", "revenue", spec, "start_year").iloc[0]["fiscal_year"] == 2023


# ---------------------------------------------------------------- rubric

def test_rubric_values_and_ui():
    rubric = Rubric.load("sentiment")
    assert rubric.allowed_values("sentiment") == [-2, -1, 0, 1, 2]
    assert rubric.is_ordinal("relevance") and not rubric.is_ordinal("content_type")
    assert [d["key"] for d in rubric.to_ui()["dimensions"]] == ["relevance", "sentiment", "forward_looking", "content_type"]


# ---------------------------------------------------------------- task templates

def _synthetic_source():
    con = duckdb.connect()
    con.execute((PKG / "sql" / "source_schema.sql").read_text())
    con.execute("INSERT INTO companies VALUES ('ACME', '1', 'Acme Corp', 'Industrials', 9, 'end_year', false)")
    rows = []
    for fy, rev, opinc, eps in [(2019, 100e9, 20e9, 8.0), (2020, 110e9, 25e9, 10.0), (2021, 121e9, 30e9, 12.0)]:
        end = pd.Timestamp(f"{fy}-09-30")
        for metric, value, first in [("revenue", rev, rev), ("operating_income", opinc, opinc),
                                     ("net_income", opinc * 0.8, opinc * 0.8), ("eps_diluted", eps / 4, eps),
                                     ("diluted_shares", 4e9, 1e9)]:
            rows.append(("ACME", metric, fy, end - pd.Timedelta(days=365), end, value, first, 2, "USD", "x", "a", end))
    con.executemany("INSERT INTO fundamentals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    dates = pd.bdate_range("2018-06-01", "2022-06-30")
    split_day = pd.Timestamp("2022-01-03")
    traded = np.where(dates < split_day, 200.0, 50.0)
    prices = pd.DataFrame({"ticker": "ACME", "date": dates, "open": 50.0, "high": 50.0, "low": 50.0,
                           "close": 50.0, "adj_close": 50.0, "close_unadjusted": traded, "volume": 1})
    con.register("p", prices)
    con.execute("INSERT INTO prices BY NAME SELECT * FROM p")
    con.execute("INSERT INTO corporate_actions VALUES ('ACME', '2022-01-03', 'split', 4.0)")
    return con


def test_templates_compute_gold_answers_and_traps():
    facts = Facts(_synthetic_source())
    tasks = {t["template_id"]: t for t in templates_for(facts, "ACME", 2020)}
    assert tasks["lookup_revenue"]["gold_value"] == pytest.approx(110_000)
    assert tasks["calc_revenue_growth"]["gold_value"] == pytest.approx(10.0)
    assert tasks["calc_operating_margin"]["gold_value"] == pytest.approx(25 / 110 * 100)
    assert "net_margin_instead" in json.loads(tasks["calc_operating_margin"]["alt_values"])

    eps = tasks["lookup_eps"]
    assert eps["gold_value"] == 10.0 and "split_adjustment" in eps["trap_tags"]
    assert json.loads(eps["alt_values"])["split_adjusted"] == 2.5

    pe = tasks["calc_pe"]  # traded $200 / reported EPS $10, not the split-adjusted $50
    assert pe["gold_value"] == pytest.approx(20.0)
    assert {"point_in_time", "split_adjustment", "fiscal_calendar"} <= set(pe["trap_tags"].split(","))
    assert json.loads(pe["alt_values"])["split_adjusted_price"] == pytest.approx(5.0)


# ---------------------------------------------------------------- anomaly detector

def test_detector_catches_injected_errors():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", periods=600)
    frames = [pd.DataFrame({"ticker": t, "date": dates, "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))})
              for t in ["A", "B", "C", "D", "SPY"]]
    result = evaluate_detector(pd.concat(frames, ignore_index=True), "SPY", n_per_kind=10, seed=1)
    overall = result[(result["error_kind"] == "all") & (result["z_threshold"] == 6)].iloc[0]
    assert overall["recall"] >= 0.9 and overall["precision_all_kinds"] >= 0.9


# ---------------------------------------------------------------- quarterly fundamentals

def test_q4_is_derived_from_full_year_minus_nine_months():
    from eval_lab.ingest_fundamentals import company_fundamentals, quarterly_fundamentals

    revenue = [
        _obs(400.0, "2022-10-01", "2023-09-30", "2023-11-03"),                 # FY2023 in the 10-K
        _obs(90.0, "2022-10-01", "2022-12-31", "2023-02-03", form="10-Q"),     # Q1
        _obs(95.0, "2023-01-01", "2023-04-01", "2023-05-05", form="10-Q"),     # Q2
        _obs(100.0, "2023-04-02", "2023-07-01", "2023-08-04", form="10-Q"),    # Q3
        _obs(285.0, "2022-10-01", "2023-07-01", "2023-08-04", form="10-Q"),    # 9-month YTD
    ]
    eps = [_obs(4.0, "2022-10-01", "2023-09-30", "2023-11-03"), _obs(0.9, "2022-10-01", "2022-12-31", "2023-02-03", form="10-Q")]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": revenue}},
                                   "EarningsPerShareDiluted": {"units": {"USD/shares": eps}}}}}
    metrics = {
        "revenue": {"kind": "duration", "unit": "USD", "concepts": ["us-gaap:Revenues"]},
        "operating_income": {"kind": "duration", "unit": "USD", "concepts": ["us-gaap:OperatingIncomeLoss"]},
        "net_income": {"kind": "duration", "unit": "USD", "concepts": ["us-gaap:NetIncomeLoss"]},
        "eps_diluted": {"kind": "duration", "unit": "USD/shares", "concepts": ["us-gaap:EarningsPerShareDiluted"]},
    }
    annual = company_fundamentals(facts, "ACME", metrics, "end_year")
    q = quarterly_fundamentals(facts, "ACME", metrics, annual)
    rev = q[q["metric"] == "revenue"].set_index("fiscal_quarter")
    assert rev["value"].to_dict() == {1: 90.0, 2: 95.0, 3: 100.0, 4: 115.0}
    assert rev.loc[4, "is_derived"] and "9-month" in rev.loc[4, "derivation"]
    assert not rev.loc[1:3, "is_derived"].any()
    assert set(q.loc[q["metric"] == "eps_diluted", "fiscal_quarter"]) == {1}  # Q4 EPS is never derived


# ---------------------------------------------------------------- earnings

@pytest.mark.parametrize("stamp, timing, expected", [
    ("2024-05-02 16:30", "after_close", "2024-05-03"),
    ("2024-05-03 16:05", "after_close", "2024-05-06"),   # Friday evening reacts on Monday
    ("2024-05-02 07:00", "before_open", "2024-05-02"),
    ("2024-05-02 12:00", "during_market", "2024-05-02"),
    ("2024-05-02 00:00", "unknown", "2024-05-02"),
])
def test_earnings_timing_and_reaction_session(stamp, timing, expected):
    from eval_lab.ingest_market import announcement_timing, reaction_date

    at = pd.Timestamp(stamp)
    assert announcement_timing(at) == timing
    assert str(reaction_date(at, timing, pd.bdate_range("2024-04-29", "2024-05-10"))) == expected


def test_earnings_reaction_task_uses_next_session_for_after_close_release():
    from eval_lab.build_qa_tasks import earnings_templates

    con = _synthetic_source()
    dates = pd.bdate_range("2021-01-04", "2021-01-08")
    for ticker, closes in [("ACME", [100, 100, 110, 121, 121]), ("SPY", [100, 100, 101, 101, 101])]:
        con.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
        con.register("p", pd.DataFrame({"ticker": ticker, "date": dates, "close": closes, "adj_close": closes,
                                        "close_unadjusted": closes, "open": 0.0, "high": 0.0, "low": 0.0, "volume": 1}))
        con.execute("INSERT INTO prices BY NAME SELECT * FROM p")
    con.execute("INSERT INTO earnings VALUES ('ACME', '2021-01-06 16:05', 'after_close', '2021-01-07', 1.0, 1.2, 20.0)")
    [t] = earnings_templates(Facts(con), "ACME", "SPY", [2021])
    assert t["gold_value"] == pytest.approx(10.0)                    # Jan 7: +10% vs SPY flat
    alts = json.loads(t["alt_values"])
    assert alts["wrong_session"] == pytest.approx(9.0)               # Jan 6: +10% vs SPY +1%
    assert "4:05 PM" in t["prompt"] and t["task_id"] == "earnings_reaction-ACME-20210106"


def test_explain_anomalies_labels_earnings_and_splits():
    from eval_lab.profile_data import explain_anomalies

    con = duckdb.connect()
    con.execute((PKG / "sql" / "source_schema.sql").read_text())
    con.execute("INSERT INTO earnings VALUES ('A', '2024-05-02 16:30', 'after_close', '2024-05-03', 1, 1, 0)")
    con.execute("INSERT INTO corporate_actions VALUES ('B', '2024-06-10', 'split', 10)")
    flagged = pd.DataFrame({"ticker": ["A", "A", "B", "A"],
                            "date": pd.to_datetime(["2024-05-03", "2024-05-06", "2024-06-10", "2024-07-01"])})
    assert explain_anomalies(flagged, con)["explained_by"].tolist() == ["earnings", "earnings", "split", "unexplained"]


# ---------------------------------------------------------------- reconciliation

@pytest.mark.parametrize("vendor, metric, split, expected", [
    (100.2, "revenue", 1, "match"),
    (95.0, "revenue", 1, "matches_first_reported"),
    (100_000.0, "revenue", 1, "scale"),
    (80.0, "revenue", 1, "matches_adjacent_period"),
    (140.0, "revenue", 1, "unexplained"),
    (2.50, "eps_diluted", 4, "split_basis"),
    (5.00, "eps_diluted", 1, "unrecorded_basis_change"),
    (10.01, "eps_diluted", 1, "match"),
    (13.3, "eps_diluted", 1, "unexplained"),
])
def test_reconcile_classify(vendor, metric, split, expected):
    from eval_lab.reconcile import classify

    sec = pd.Series({"value": 10.0 if metric == "eps_diluted" else 100.0,
                     "first_reported_value": 10.0 if metric == "eps_diluted" else 95.0})
    assert classify(vendor, sec, [80.0], [split] if split != 1 else [], metric) == expected


def test_reconcile_labels_missing_rows_by_reason():
    from eval_lab.reconcile import reconcile_statements

    sec = pd.DataFrame({"ticker": "ACME", "metric": "revenue", "frequency": ["annual", "quarterly"],
                        "period_end": pd.to_datetime(["2023-12-31", "2023-09-30"]), "value": [400.0, 100.0],
                        "first_reported_value": [400.0, 100.0], "fiscal_year": 2023, "fiscal_quarter": [None, 3]})
    vendor = pd.DataFrame({"ticker": "ACME", "vendor": "yahoo",
                           "metric": ["revenue", "revenue", "operating_income"],
                           "frequency": ["quarterly", "quarterly", "annual"],
                           "period_end": pd.to_datetime(["2023-09-30", "2023-12-31", "2023-12-31"]).date,
                           "value": [100.0, 120.0, 50.0]})
    out = reconcile_statements(vendor, sec, pd.DataFrame(columns=["ticker", "date", "value"]))
    assert dict(zip(out["metric"] + "@" + out["vendor_period_end"].astype(str), out["status"])) == {
        "revenue@2023-09-30": "match", "revenue@2023-12-31": "q4_not_filed", "operating_income@2023-12-31": "vendor_only_metric"}


def test_is_share_split_rejects_spinoff_factors():
    from eval_lab.ingest_market import is_share_split

    assert all(is_share_split(r) for r in [2, 3, 4, 10, 20, 1.5, 0.5, 0.1, 1.25])
    assert not any(is_share_split(r) for r in [1.011, 1.032, 1.054, 1.061, 0.9535])


def test_street_eps_has_no_gaap_quarter_for_q4():
    from eval_lab.reconcile import reconcile_street_eps

    quarterly = pd.DataFrame({"ticker": "ACME", "metric": "eps_diluted", "fiscal_year": 2023,
                              "fiscal_quarter": [1, 2, 3], "value": [1.00, 1.10, 1.20],
                              "period_end": pd.to_datetime(["2022-12-31", "2023-04-01", "2023-07-01"])})
    earnings = pd.DataFrame({"ticker": "ACME", "eps_reported": [1.00, 1.35, 1.50],
                             "announced_at_et": pd.to_datetime(["2023-02-02 16:30", "2023-08-03 16:30", "2023-11-02 16:30"])})
    out = reconcile_street_eps(earnings, quarterly)
    assert out["status"].tolist() == ["match", "street_vs_gaap", "no_gaap_quarter"]


def test_merge_facts_combines_predecessor_and_successor_filers():
    from eval_lab.ingest_fundamentals import company_fundamentals, merge_facts

    old = {"entityName": "Old Co", "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        _obs(100.0, "2023-01-01", "2023-12-31", "2024-02-20")]}}}}}
    new = {"entityName": "New HoldCo", "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        _obs(120.0, "2024-01-01", "2024-12-31", "2025-02-20")]}}}}}
    spec = {"revenue": {"kind": "duration", "unit": "USD", "concepts": ["us-gaap:Revenues"]}}
    assert company_fundamentals(new | {"facts": {}}, "X", spec, "end_year").empty
    df = company_fundamentals(merge_facts([old, new]), "X", spec, "end_year")
    assert df.set_index("fiscal_year")["value"].to_dict() == {2023: 100.0, 2024: 120.0}


@pytest.mark.parametrize("end, convention, label", [
    ("2021-01-03", "end_year", 2020),    # J&J 53-week year
    ("2023-12-31", "end_year", 2023),
    ("2025-01-26", "end_year", 2025),    # NVDA
    ("2025-02-02", "start_year", 2024),  # Home Depot
    ("2024-09-28", "end_year", 2024),    # Apple
])
def test_fiscal_year_label(end, convention, label):
    from eval_lab.ingest_fundamentals import fiscal_year_label

    assert fiscal_year_label(pd.Timestamp(end), convention) == label


def test_same_period_dated_differently_is_one_period():
    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        _obs(1940.0, "2014-11-03", "2015-11-01", "2015-12-10"),   # Deere's FY2015 as first filed
        _obs(1940.0, "2014-11-01", "2015-10-31", "2016-12-09"),   # ...and in the next 10-K
    ]}}}}}
    spec = {"kind": "duration", "unit": "USD", "concepts": ["us-gaap:NetIncomeLoss"]}
    df = extract_metric(facts, "DE", "net_income", spec, "end_year")
    assert len(df) == 1 and df.iloc[0]["period_end"] == pd.Timestamp("2015-10-31") and df.iloc[0]["n_versions"] == 1


def test_derived_q4_uses_as_reported_figures_when_full_year_is_restated():
    from eval_lab.ingest_fundamentals import company_fundamentals, quarterly_fundamentals

    revenue = [
        _obs(400.0, "2022-10-01", "2023-09-30", "2023-11-03"),                 # FY as first reported
        _obs(360.0, "2022-10-01", "2023-09-30", "2024-11-01"),                 # recast for a divestiture
        _obs(285.0, "2022-10-01", "2023-07-01", "2023-08-04", form="10-Q"),    # 9-month YTD, never recast
    ]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": revenue}}}}}
    metrics = {"revenue": {"kind": "duration", "unit": "USD", "concepts": ["us-gaap:Revenues"]}}
    annual = company_fundamentals(facts, "ACME", metrics, "end_year")
    q4 = quarterly_fundamentals(facts, "ACME", metrics, annual).query("fiscal_quarter == 4").iloc[0]
    assert q4["value"] == 115.0   # 400 - 285, not 360 - 285


def test_revenue_takes_the_largest_concept_when_a_subtotal_outranks_the_total():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [_obs(28.4e9, "2012-07-01", "2013-06-30", "2014-08-08")]}},
        "SalesRevenueNet": {"units": {"USD": [_obs(84.2e9, "2012-07-01", "2013-06-30", "2013-08-08")]}},
    }}}
    spec = {"kind": "duration", "unit": "USD", "select": "largest",
            "concepts": ["us-gaap:Revenues", "us-gaap:SalesRevenueNet"]}
    row = extract_metric(facts, "PG", "revenue", spec, "end_year").iloc[0]
    assert (row["value"], row["source_concept"]) == (84.2e9, "us-gaap:SalesRevenueNet")


# ---------------------------------------------------------------- sentiment pipeline

def test_company_matching_is_word_bounded_and_short_tickers_need_cashtags():
    from eval_lab import companies

    assert companies.mentions("Apple shares rise", "AAPL") and companies.mentions("Buying $AAPL today", "AAPL")
    assert not companies.mentions("Pineapple prices rise", "AAPL")
    assert not companies.mentions("very good day, so we sold", "V") and not companies.mentions("so it goes", "SO")
    assert companies.mentions("$V hits a record", "V") and companies.mentions("Visa earnings beat", "V")
    assert companies.link("Alphabet Stock") == "GOOGL" and companies.link("Southern Company") == "SO"
    assert companies.link("Broadcom") is None
    assert companies.cashtags("Watching $NVDA and $V") == ["NVDA", "V"]


def test_session_close_follows_daylight_saving():
    from eval_lab.market_time import session_close_utc

    closes = session_close_utc(["2026-01-15", "2026-07-15"])
    assert [c.hour for c in closes] == [21, 20]


def test_event_returns_pick_the_reacting_session():
    from eval_lab.market_time import event_returns

    dates = pd.bdate_range("2026-07-13", periods=4)                      # Mon..Thu, summer: close = 20:00 UTC
    prices = pd.concat([
        pd.DataFrame({"ticker": "A", "date": dates, "adj_close": [100, 110, 121, 121]}),
        pd.DataFrame({"ticker": "SPY", "date": dates, "adj_close": [100, 100, 100, 100]}),
    ])
    events = pd.DataFrame({"ticker": ["A", "A"], "at": pd.to_datetime(["2026-07-14 13:00", "2026-07-14 20:30"])})
    out = event_returns(events, prices, "SPY")
    assert out["reaction_date"].dt.strftime("%m-%d").tolist() == ["07-14", "07-15"]   # before vs after the close
    assert out["reaction_abnormal"].round(3).tolist() == [0.1, 0.1]
    assert out["next_session_abnormal"].round(3).tolist() == [0.1, 0.0]


def test_mark_duplicates_keeps_the_earliest_copy_within_the_window():
    from eval_lab.news_sentiment import mark_duplicates

    times = pd.Series(pd.to_datetime(["2026-08-02 00:00", "2026-08-01 00:00", "2026-08-01 12:00", "2026-08-20 00:00"]))
    same, other = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    vectors = np.vstack([same, same, other, same])
    # doc 0 repeats doc 1 a day later; doc 3 says the same thing but weeks later
    assert mark_duplicates(times, vectors, threshold=0.85, days=3).tolist() == [1, -1, -1, -1]


def test_link_entities_dedupes_and_ignores_unknown_companies():
    from eval_lab.news_sentiment import link_entities

    ents = [{"text": "Nvidia"}, {"text": "Broadcom"}, {"text": "NVIDIA"}, {"text": "Google"}]
    assert link_entities(ents) == ["NVDA", "GOOGL"]


def test_fiqa_scores_map_to_classes():
    from eval_lab.sentiment_benchmark import fiqa_class

    assert [fiqa_class(s) for s in [-0.53, -0.05, 0.0, 0.09, 0.39]] == ["negative", "neutral", "neutral", "neutral", "positive"]


def test_ranking_uses_only_held_out_datasets():
    from eval_lab.sentiment_benchmark import rank

    results = pd.DataFrame({
        "scorer": ["a", "a", "a", "b", "b", "b"],
        "dataset": ["twitter_financial_news", "fiqa", "financial_phrasebank"] * 2,
        "macro_f1": [0.6, 0.6, 0.99, 0.65, 0.65, 0.7], "cohen_kappa": [0.4, 0.4, 0.98, 0.5, 0.5, 0.6],
    })
    assert rank(results).index.tolist() == ["b", "a"]   # a's 0.99 on its own training data doesn't count


def test_probability_columns_map_to_labels_and_scores():
    from eval_lab.sentiment_models import _from_probs

    out = _from_probs(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
    assert out["label"].tolist() == ["negative", "positive"]
    assert out["score"].round(2).tolist() == [-0.6, 0.7]


def test_google_news_items_are_parsed_and_placeholder_times_marked():
    from eval_lab.ingest_news import fetch_google, query_for, to_frame

    rss = b"""<rss><channel>
      <item><title>Apple Stock Falls - investopedia.com</title><link>https://n.example/1</link>
        <pubDate>Mon, 10 Aug 2026 07:00:00 GMT</pubDate><source url="https://www.investopedia.com">investopedia.com</source></item>
      <item><title>Apple gains on iPhone demand - Yahoo Finance</title><link>https://n.example/2</link>
        <pubDate>Fri, 25 Sep 2026 12:52:14 GMT</pubDate><source url="https://finance.yahoo.com">Yahoo Finance</source></item>
    </channel></rss>"""

    class Session:
        def get(self, url, params, timeout):
            assert "after:2026-08-01" in params["q"] and "before:2026-08-09" in params["q"]
            return type("R", (), {"content": rss, "raise_for_status": lambda self: None})()

    from datetime import datetime
    articles = fetch_google(Session(), query_for("AAPL", "google"), datetime(2026, 8, 1), datetime(2026, 8, 8))
    df = to_frame(articles, "AAPL", "google")
    assert df["title"].tolist() == ["Apple Stock Falls", "Apple gains on iPhone demand"]
    assert df["time_precision"].tolist() == ["day", "exact"]
    assert df["domain"].tolist() == ["investopedia.com", "finance.yahoo.com"]
    assert "sourcelang" not in query_for("AAPL", "google") and "sourcelang:english" in query_for("AAPL", "gdelt")
