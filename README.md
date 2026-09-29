# fin-eval-lab

A data-quality and evaluation lab for financial data. It grew out of
[PosiTech](https://github.com/arshiya-k/PosiTech), a stock-investment site I built
that scored news, Reddit and Twitter posts with VADER sentiment. This project asks
the question PosiTech never did: **how good is this data, and how would you know?**
It then extends the question to market data, SEC filings and a data vendor.

It has six parts, all driven by config files:

| Part | What it does |
|---|---|
| **Data layer** | Daily prices, corporate actions and ~1,050 earnings announcements (Yahoo Finance), annual and quarterly fundamentals (SEC EDGAR XBRL, 10-K and 10-Q), and PosiTech's scraped posts, normalized into DuckDB. |
| **Data quality** | Completeness, restatement and split detection, accounting identity and quarter-sum checks, and scraper reproducibility, plus a price anomaly detector whose precision and recall are measured on injected errors and whose flags are explained by earnings. |
| **Reconciliation** | Yahoo Finance's income statements vs. the SEC filings, and analysts' "street" EPS vs. GAAP EPS, with every difference classified by cause. |
| **Sentiment audit** | A stratified sample of PosiTech's posts: stored vs. re-run VADER, off-topic rate by source, and VADER checked against next-day abnormal returns. A rubric-driven browser annotation tool adds human labels as the reference when available. |
| **Sentiment, rebuilt** | The 2022 pipeline redone with current methods: a benchmark of nine scorers on professionally labeled data, entity-level relevance and sentiment, deduplication by meaning, and a market check on ~18,000 fresh headlines. |
| **Financial Q&A benchmark** | 160 templated questions with gold answers computed from the source data, each recording the plausible wrong answers and the trap it tests, sampled by stratified random sampling. |

## Findings so far

These come from running the data layer and profiler on the real data (`reports/`):

- **Most of PosiTech's news isn't about the stock it's tagged with.** 79% of news
  posts never mention their ticker or company name. The NewsAPI query
  `"AAPL AND stock"` returns generic market roundups. One post tagged GOOGL is titled
  *"Nvidia: Prepare To Be Underwhelmed."* The sentiment rubric's `relevance`
  dimension exists to measure this properly.
- **PosiTech's stored sentiment scores can't be reproduced from its own data.** For
  news, the app scored the article body but saved only the title. 66% of stored scores
  differ by more than 0.2 from VADER re-run on the saved text, and on the 150-post
  sample the two agree on the positive/neutral/negative class only 58% of the time
  (Cohen's κ = 0.34). On posts that actually mention their company, agreement rises
  to 75% (κ = 0.62), so the problem is concentrated in off-topic news.
- **VADER sentiment doesn't predict next-day moves.** Its direction matches the stock's
  next-day return vs. SPY 51% of the time, a coin flip (Spearman −0.04). On posts that
  mention the company it's 55%, from only 61 posts, which is not a usable signal.
- **Each scraper stored dates differently**: ISO 8601 for news, Unix epoch for Reddit,
  and Twitter's `Mon Dec 12 07:23:35 +0000 2022` format. Reddit posts span 2020–2022
  while everything else covers a few weeks in late 2022.
- **The price anomaly detector** (robust z-score on returns, ignoring market-wide
  shocks) catches 96% of injected errors with no false alarms at z ≥ 4. By error type,
  recall is 100% for one-day spikes, 96% for decimal slips and 92% for missed stock
  splits. On clean data it surfaces real events worth review, such as NKE −22% on
  2024-06-28 and META −21% on 2018-07-26.
- **Most extreme moves are earnings.** Of the 156 real moves above z = 6, 107 (69%)
  fall on the session that reacted to an earnings release, so the manual review queue
  shrinks to 49. The rest are news the calendar can't explain: J&J's talc report
  (Dec 2018), the bank rally after the 2024 election, UnitedHealth suspending guidance
  (May 2025), Pfizer's Paxlovid results (Nov 2021).
- **Release timing decides which session reacts.** Of 1,049 earnings releases, 724
  came before the open (the same session reacts) and 316 after the close (the *next*
  session reacts). Using the wrong session is the trap in the earnings-reaction questions.
- **Yahoo Finance vs. the SEC filings (2022–2026).** Where both sources carry a
  figure, they agree on:

  | Figure | Annual | Quarterly |
  |---|---|---|
  | Net income | 100% | 100% |
  | Diluted shares | 95% | 97% |
  | Diluted EPS | 93% | 96% |
  | Revenue | 88% | 88% |
  | Operating income | 59% | 68% |

  The differences are definitional, not errors:
  - Yahoo computes its own "operating income" for companies that don't report one
    (Exxon, Chevron, ConocoPhillips, IBM, Nike, Pfizer).
  - Where companies do report one, Yahoo normalizes away one-off charges (Coca-Cola
    +13%). Air Products' Q4 FY2025 operating income is $17M in the filings, after
    ~$2.3B of impairments that year, but ~$800M at Yahoo.
  - Revenue gaps are definitional too. Exxon and Chevron (−3%) include equity-affiliate
    and other income in filed revenue; Yahoo doesn't. JPMorgan and Deere show similar
    small gaps.
- **One systematic vendor anomaly.** Yahoo reports Honeywell's EPS at exactly 2× and
  its share count at exactly ½ of the filings, in every year. No split in Yahoo's own
  history explains it, so the reconciliation flags it as `unrecorded_basis_change` for
  review.
- **The vendor records spin-offs as "splits".** Yahoo's split history includes ratios
  like 1.061 (Honeywell/Solstice, 2025), 1.046 (IBM/Kyndryl, 2021) and 1.054
  (Pfizer/Viatris, 2020). These are price adjustments for distributed shares, and they
  don't change per-share figures. They're now stored as `spinoff_adjustment` so they
  don't trigger split logic.
- **Street EPS usually isn't GAAP EPS.** Of the earnings announcements matched to a
  GAAP quarter, 260 agree and 472 differ. In 72% of those differences the "street"
  (adjusted) figure is higher, by a median of 18%. Another 62 differ only because
  Yahoo restates old EPS for later splits (Netflix 10-for-1, Amazon, Nvidia), and
  237 are fourth quarters with no GAAP quarterly EPS to compare.
- **The SEC data needed cleaning too.** Profiling the raw XBRL caught:
  - J&J's 53-week fiscal 2020 (ended Jan 3, 2021) being labeled 2021.
  - Deere dating the same fiscal year Oct 31 in one filing and Nov 1 in the next.
  - Tags missing from the fallback chain: utilities' total operating revenue,
    Caterpillar's and Southern's net income, Exxon's combined share count.
  - P&G's 2014 10-K tagging a ~$28B subtotal as `Revenues` next to $84B of
    `SalesRevenueNet`, so revenue now takes the largest candidate.
  - Q4s derived from two different revenue definitions (ConocoPhillips FY2010: −$74B).
  - Restatement vintages. After a spin-off, later filings recast the full year but
    never the original 10-Qs (J&J's first-reported Q1–Q2 2023 still include Kenvue),
    so Q4s are derived and quarter sums checked within one vintage. That cut
    quarter-sum mismatches from 60 to 18, and only 2 (both under 1%) fall in the
    benchmark years.

  Each is handled in code or `concepts.yaml`, with a test.
- **The vendor relabels fiscal periods.** Yahoo Finance snaps period ends to month-end:
  Apple's fiscal year ends on the last Saturday of September, but Yahoo labels it
  Sept 30. The same goes for Nvidia, Home Depot, Deere, J&J, Coca-Cola and Pfizer:
  17% of matched figures carry a different period-end date (median 2 days). Joining on
  exact dates would silently drop them, so reconciliation matches within ±10 days and
  reports the offset.

## Sentiment, rebuilt with current methods

PosiTech used VADER (2014), a general-purpose word list, on keyword-searched
posts. The rebuild replaces each weak step and measures the effect.

### 1. Which scorer? A benchmark on professionally labeled data

`make sentiment-benchmark` scores nine methods on three labeled datasets:

- Twitter Financial News: 2,388 held-out tweets
- FiQA: 351 headlines and posts, each labeled for sentiment *toward a named company*
- Financial PhraseBank: 2,264 sentences all annotators agreed on

| Scorer | Tweets | FiQA | PhraseBank | Held-out macro-F1 | κ |
|---|---|---|---|---|---|
| **Zero-shot DeBERTa, entity-targeted** | 0.757 | 0.559 | 0.972 | **0.658** | 0.52 |
| **Financial-RoBERTa-large** | 0.645 | 0.653 | 0.912 | **0.649** | 0.54 |
| DistilRoBERTa-fin | 0.706 | 0.525 | 0.996\* | 0.616 | 0.47 |
| FinBERT | 0.668 | 0.496 | 0.963\* | 0.582 | 0.42 |
| FinSense-ModernBERT | 0.666 | 0.460 | 0.996\* | 0.563 | 0.39 |
| FinBERT-tone | 0.661 | 0.411 | 0.897 | 0.536 | 0.36 |
| Twitter-RoBERTa (general) | 0.613 | 0.400 | 0.575 | 0.506 | 0.31 |
| **VADER (PosiTech's)** | 0.447 | 0.374 | 0.487 | **0.411** | **0.16** |
| Loughran-McDonald dictionary | 0.447 | 0.317 | 0.472 | 0.382 | 0.15 |

\* The model was fine-tuned on this dataset, so the score is inflated and excluded
from the ranking. Models score 0.996 on their own training data but 0.46–0.71
elsewhere, which is why the ranking uses only the two datasets no model saw.

- The best current models agree with professional labels **3.4× better than VADER**
  (κ 0.54 vs. 0.16).
- The entity-targeted zero-shot model tracks FiQA's per-company scores best
  (Spearman 0.73), because it asks *"is this good news for Nvidia?"* rather than
  scoring the whole text. For "Nvidia slides as Google gains" it scores Nvidia −1.0
  and Google +0.99. Whole-text models give that headline a single label.

Everything runs locally. Half precision and length-sorted batching speed up the
largest models 1.2–1.55× with identical labels.

### 2. The rebuilt pipeline

`make news` collects 12 weeks of headlines for all 30 companies, and
`make news-sentiment` processes them alongside PosiTech's posts:

| Step | 2022 PosiTech | Rebuild |
|---|---|---|
| Relevance | keyword search | GLiNER entity recognition linked to tickers ("16 Apple Desserts" and "Visa requirements tightened" are rejected), plus cashtags |
| Duplicates | exact text | sentence embeddings: the same story syndicated within 3 days is counted once |
| Scoring | VADER on the whole text | best classifier on the text, and zero-shot sentiment toward each recognized company |
| Stored text | article body scored, title saved | the scored text is the stored text |
| Timing | mixed formats | UTC, daylight-saving-aware market close, timestamp precision recorded |

Funnel for the fresh news:
- 18,806 search hits, of which 16,030 match a keyword and **14,747** are recognized
  as the company.
- 627 extra company mentions found inside other companies' headlines.
- 13% removed as syndicated duplicates.
- **13,359 unique company mentions** left.

VADER agrees with the best classifier on only 49% of them. Typical disagreements:

- VADER positive, zero-shot negative: "IBM Lowers Its Growth Outlook as Sales of
  Data-Center Mainframes Sink 42%", "Amazon Stock Drops 1.8% as $3 Trillion Rally Cools".
- VADER negative, zero-shot positive: "JPMorgan aggressively double-upgrades another
  AI stock", "Iraq backs ConocoPhillips gas deal".

### 3. Market check: sentiment describes moves, it doesn't predict them

Abnormal return vs. SPY, measured per company-day, because headlines about the same
company and day aren't independent:

| | Company-days | VADER | Fin-RoBERTa-large | Zero-shot, entity |
|---|---|---|---|---|
| Same-session correlation | 1,674 | 0.05 | 0.21 | **0.25** |
| Same-session direction hit rate (per headline) | | 51% | 58% | **61%** |
| Positive minus negative headlines (per headline) | | 0.8 pp | 1.6 pp | **2.1 pp** |
| **Earnings days:** same-session correlation | 29 | 0.42 | 0.71 | **0.78** |
| Next-session correlation | 1,644 | −0.03 | −0.01 | 0.02 |

- **The modern models read price-moving news about 5× better than VADER.** On
  earnings days the zero-shot scores line up with the stock's reaction at 0.78
  (only 29 events, so treat that as indicative).
- **None of them predicts the next session.** Much of the same-session
  correlation is reverse causality: headlines like "Amazon Stock Drops 1.8%" are
  written about the move.
- **Why the company-day view matters:** per headline, the next-session numbers
  looked weakly significant (Spearman 0.036, p < 0.001). That signal disappears at
  the company-day level and among headlines with exact timestamps. It came from
  clustered headlines and date-only timestamps putting news in the wrong session.
  Reporting it as predictive would have been a look-ahead error.

## Quickstart

```bash
make setup
make data          # needs SEC_USER_AGENT, see below
make profile
make reconcile
make tasks         # Q&A benchmark -> reports/qa_benchmark.csv
make sentiment-benchmark   # ~10 min on a laptop GPU; downloads ~2 GB of models once
make news                  # ~25 min (paced requests)
make news-sentiment        # ~15 min
make test
```

SEC's fair-access policy requires a User-Agent that identifies you:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

### Sentiment audit

```bash
make sentiment          # stratified sample + VADER baselines
make sentiment-report   # stored vs. re-run VADER, market check, off-topic rate
```

Human labels are optional. To add them:

```bash
python -m eval_lab.annotation export sentiment --n 50
make annotate           # http://localhost:8765
python -m eval_lab.annotation import ~/Downloads/<batch>__<name>.json
make sentiment-report && make agreement
```

The annotator is a single HTML file driven by the rubric YAML, with keyboard-first
entry, autosave and per-item timing. Items are shuffled and VADER's score is hidden so
it can't anchor the human label. With two annotators, `make agreement` reports their
inter-annotator κ alongside VADER's.

## How it works

### The rubric is the single source of truth

`eval_lab/config/rubrics/sentiment.yaml` defines each dimension once: its question,
scale, anchors and flags. These are:

- relevance: is the post actually about the tagged stock?
- sentiment toward the stock, on a −2 to +2 scale
- whether the post makes a forward-looking claim
- content type
- flags such as sarcasm or truncated text

The annotation UI and the agreement metrics both come from that file. Agreement uses
quadratic-weighted κ for ordinal scales, so 4-vs-5 costs less than 1-vs-5, and plain κ
for categories. Changing the rubric is one edit.

### Q&A tasks record the likely wrong answers

Each template stores the gold answer and the plausible wrong ones, so an answer from
any source (an analyst, a model, a vendor feed) can be scored on *why* it's wrong:

| Template | Trap it tests | Recorded wrong answers |
|---|---|---|
| Revenue lookup | fiscal calendars (Apple ends in Sept, Home Depot labels by start year), restatements | prior / next year, first-reported value |
| Diluted EPS, as reported | stock splits after the period | split-adjusted EPS |
| Revenue growth, operating margin | percent vs. fraction, net vs. operating | net margin, prior-year growth |
| Q4 revenue | Q4 is never filed on its own | full year, 9-month YTD, Q3 |
| Trailing P/E at fiscal year-end | point-in-time prices | split-adjusted price ÷ reported EPS |
| Calendar-year total return | dividends | price-only return |
| Earnings-day abnormal return | release before the open vs. after the close | wrong session, raw instead of abnormal return |
| Operating margin drivers (free text) | faithfulness | reference figures recorded for manual grading |

Every gold answer was spot-checked against published figures. For example, Apple's
FY2023 revenue is $383,285M, Alphabet's derived Q4 2024 revenue is $96,469M, and
Netflix rose 10.6% vs. SPY on Jan 24, 2024, the session after its after-close release.

Tasks are sampled with stratified random sampling: a fixed quota per template, spread
proportionally across sectors, with at least one task per sector. Quotas are set per
template so the ~900 earnings events can't crowd out the other market questions.
`reports/qa_sampling_summary.csv` lists population vs. sample share and the design
weight for each stratum.

### Quarterly data and the missing Q4

Companies file 10-Qs for Q1–Q3 only. Q4 exists only inside the 10-K's full-year
figure, so for additive metrics (revenue, operating and net income) it's derived as
full year minus the 9-month year-to-date figure and flagged `is_derived`. Q4 EPS is
deliberately **not** derived: EPS isn't additive, because the share count changes
every quarter. The profiler checks that four quarters sum to the full year, and the
Q&A set asks for Q4 revenue, where full-year, 9-month and Q3 figures are all recorded
as likely wrong answers.

### Reconciling a vendor against the filings

`reconcile` matches each Yahoo Finance figure to the SEC period ending nearest to it,
then classifies the difference:

| Status | Meaning |
|---|---|
| `match` | within 0.5% (EPS within a cent) |
| `matches_first_reported` | vendor shows the original figure; the filings have since restated it |
| `split_basis` | per-share or share-count figure restated for a recorded split |
| `unrecorded_basis_change` | per-share figure off by a clean ratio (2×, 10×) that no recorded split explains |
| `vendor_only_metric` | the company doesn't report this line; the vendor computes its own |
| `q4_not_filed` | a fourth quarter, which companies never file on its own |
| `matches_adjacent_period` | vendor value belongs to the prior or next period, a period mapping error |
| `scale` | off by a power of 1,000 |
| `street_vs_gaap` | earnings-calendar EPS (analysts' adjusted figure) vs. GAAP diluted EPS |
| `no_gaap_quarter` | a Q4 announcement, for which no GAAP quarterly EPS is filed |
| `unexplained` | a real difference to investigate, usually a definition difference (e.g. bank revenue) |
| `missing_in_sec` / `missing_in_vendor` | coverage gaps |

## Reports (Tableau-ready)

Every step writes tidy CSVs to `reports/`, one row per entity, ready to use as
Tableau or Qlik data sources:

| File | Grain | Dashboard idea |
|---|---|---|
| `data_quality.csv` | one issue | DQ scorecard by check × severity × ticker |
| `price_detector_pr.csv` | threshold × error kind | precision/recall vs. threshold curve |
| `price_anomalies.csv` | one flagged day | review queue of extreme moves, by `explained_by` |
| `reconciliation.csv` | one vendor vs. SEC comparison | differences by status, metric and company; period-end offsets |
| `reconciliation_summary.csv` | metric × frequency × status | reconciliation scorecard |
| `sentiment_items.csv` | one post | stored vs. re-run VADER, company mention, next-day return |
| `sentiment_off_topic.csv` | source | off-topic share by source |
| `sentiment_metrics.csv` | labeler × subset | accuracy / F1 / κ; sentiment vs. next-day return |
| `agreement.csv` | labeler pair × dimension | inter-annotator agreement |
| `sentiment_benchmark.csv` | scorer × dataset | macro-F1 / κ / recall by class, contamination flag, speed |
| `sentiment_benchmark_items.csv` | one prediction | confusion matrices, error review |
| `news_funnel.csv` | source | search hits → relevant → unique |
| `news_items.csv` | one document × company | scores from each model, relevance flags, abnormal returns |
| `news_market_check.csv` | source × subset × window × scorer | correlation, hit rate, spread, company-day correlation |
| `news_relevance_examples.csv` | one example | where keyword and entity relevance disagree |
| `qa_benchmark.csv` | one question | question mix by template, sector and trap |
| `qa_sampling_summary.csv` | template × sector | population vs. sample share, design weights |

## Layout

```
eval_lab/
  config/            universe, XBRL concepts, sampling, rubrics/
  sql/               source and eval schemas
  annotation/        index.html annotator + exported batches/
  ingest_*.py        market + earnings, annual + quarterly fundamentals, posts, news
  companies.py       company-name matching and entity-to-ticker linking
  market_time.py     session close times (DST-aware) and event-window returns
  sentiment_models.py    one interface over VADER, Loughran-McDonald, transformers, zero-shot
  sentiment_benchmark.py scorer comparison on labeled data
  news_sentiment.py      rebuilt pipeline + market check
  profile_data.py    data quality + anomaly detector
  reconcile.py       vendor vs. SEC, street vs. GAAP EPS
  sentiment_eval.py  sample / report
  build_qa_tasks.py  templates, gold answers, stratified sample
  annotation.py      export / import human labels
  agreement.py       pairwise kappa across all labelers
data/positech/       posts inherited from PosiTech
tests/
```

## Limitations

- PosiTech's posts cover about a month (plus some older Reddit threads) for four
  tickers. The sentiment-vs-return check shows the method; the sample is far too small
  to say anything about predictive power.
- Market-cap terciles are relative to this 30-company universe, so "small" still means
  a large-cap company.
- Google News gives only a date (no time) for items older than about ten days, so
  most headlines can't be placed before or after the close. The market check
  reports exact-timestamp headlines separately. GDELT has exact times but blocked
  this network after a short burst, so `--provider gdelt` is supported but wasn't
  used for the results above.
- The entity recognizer is deliberately conservative. It misses some headlines
  a keyword would catch ("Alphabet's Stock Has Reached…"), in exchange for
  rejecting "Apple desserts"-type matches; `news_relevance_examples.csv` lists
  both kinds. It also doesn't understand negation: "It's Not Apple or Microsoft"
  still counts as an Apple mention.
- Transformer inputs are capped at 256 tokens, so long Reddit posts are scored on
  their opening.
- Yahoo Finance is convenient but isn't a vendor-grade source. The profiler's split and
  stale-price checks exist partly for that reason, and its statements only go back
  about four years and five quarters, which limits the reconciliation window.
- The earnings calendar is scraped. A few releases have no time of day, and one
  company has a 182-day gap in 2018–19; the profiler flags both.
