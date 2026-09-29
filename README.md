# positech-v2

**Does news sentiment line up with stock prices?** In 2022 I built
[PosiTech](https://github.com/arshiya-k/PosiTech), a stock-investment site that scored
news, Reddit and Twitter posts for sentiment, at a time when sentiment wasn't
recommended as a market signal. This project answers the question with current
methods: it picks the best open-source sentiment models on professionally labeled
data, runs them on ~18,000 fresh headlines for 30 companies, and tests the scores
against actual stock returns.

Everything runs locally in Python, SQL (DuckDB) and open-source models, with no API keys.

## Results

### Sentiment aligns with same-day price moves, but doesn't predict the next day

Each headline's sentiment toward its company was compared with the stock's abnormal
return (its return minus SPY's) in the trading session that could react. Headlines
about the same company and day are averaged into one data point. Numbers below are for
the best model:

| | Correlation | Direction right | Company-days |
|---|---|---|---|
| **Same session**, all days | **0.25** | 61% | 1,674 |
| Same session, stock moved ±3% vs. market | **0.69** | 79% | 149 |
| Same session, earnings days | **0.79** | 77% | 29 |
| Same session, excluding earnings days | 0.24 | 59% | 1,645 |
| **Next session**, all days | **0.02** (p = 0.39) | 52% | 1,644 |

- **It aligns, especially when something big happens.** Big moves and earnings reactions
  line up strongly with headline sentiment. Ordinary days line up weakly.
- **It doesn't predict.** The next session is a coin flip, so as a trading signal the
  2022 advice still holds. Markets absorb public news within the session.
- **Much of the alignment is the news describing the move.** Headlines like "Oil Stocks
  Fall Today" are written about the price change. Among headlines with exact timestamps,
  the same-session correlation drops to 0.10.
- **A false signal was ruled out.** Per headline, the next session looked significant
  (p < 0.001). The effect vanished per company-day and among exact timestamps: it came
  from duplicate coverage and date-only timestamps putting news in the wrong session.

The same pattern holds for Financial-RoBERTa-large (0.21 same session, −0.01 next).
Useful for *explaining and monitoring* moves, not for *predicting* them.

### Choosing the models

Seven open-source models were scored on professionally labeled financial text (Twitter
Financial News, FiQA, Financial PhraseBank). The ranking uses only the two datasets no
model was trained on:

| Model | Held-out macro-F1 | κ |
|---|---|---|
| **Zero-shot DeBERTa, sentiment toward each company** | **0.66** | 0.52 |
| **Financial-RoBERTa-large** | **0.65** | **0.54** |
| DistilRoBERTa-fin | 0.62 | 0.47 |
| FinBERT | 0.58 | 0.42 |
| FinSense-ModernBERT | 0.56 | 0.39 |
| FinBERT-tone | 0.54 | 0.36 |
| Twitter-RoBERTa (general-purpose) | 0.51 | 0.31 |

- **Contaminated benchmarks:** models fine-tuned on PhraseBank score 0.996 on it but
  0.46–0.71 on data they never saw. Those scores are flagged and excluded.
- **Why zero-shot:** it scores sentiment *toward a specific company*, so
  "Nvidia slides as Google gains" is −1.0 for Nvidia and +0.99 for Google.

### The pipeline, on 18,806 headlines

- **Relevance by entity recognition, not keywords:** 1,300 keyword matches rejected
  ("16 Apple Desserts…"), and 627 extra company mentions found.
- **Duplicates removed by meaning:** 13% were the same story syndicated elsewhere,
  leaving **13,358 unique company mentions**.

### 2022 vs. now

A quick comparison with the original PosiTech pipeline (VADER, keyword search). The
last three rows run the old scorer on the same labeled data and headlines as the new one:

| | 2022 PosiTech | Now |
|---|---|---|
| Scraped news actually about the tagged company | 18% | filtered by entity recognition |
| Agreement with professional labels (κ) | 0.16 | 0.54 |
| Same-session correlation with price moves | 0.05 | 0.25 |
| Next-session prediction | none | none |

The original also scored article bodies but saved only titles, so its stored scores
couldn't be reproduced (58% agreement with a re-run).

### The market data was checked too

The market test is only as good as its price data, so the project also profiles and
reconciles it:

- **Yahoo Finance vs. SEC filings:**

  | Figure | Agreement |
  |---|---|
  | Net income | 100% |
  | EPS | 93–96% |
  | Revenue | 88% |
  | Operating income | 59–68% |

  The operating-income gap is definitional: Yahoo computes its own figure.
- **Vendor issues found:**
  - Honeywell's per-share figures are 2× off at Yahoo.
  - Spin-offs are recorded as stock splits.
- **SEC data issues fixed:**
  - J&J's 53-week year mislabeled.
  - Deere dating one fiscal year two ways.
  - Missing XBRL tags.
  - Restatement versions: quarter-sum mismatches cut from 60 to 2.
- **Price anomaly detector:** catches 96% of planted errors with no false alarms.
  69% of real extreme moves are earnings days.

## Quickstart

```bash
make setup
export SEC_USER_AGENT="Your Name your.email@example.com"   # SEC requires one
make data                  # prices, earnings, SEC filings, PosiTech posts
make sentiment-benchmark   # ~10 min; downloads ~2 GB of models once
make news                  # ~25 min; 12 weeks of headlines for 30 companies
make news-sentiment        # ~15 min; rebuilt pipeline + market check
make sentiment-report      # 2022 vs. now: audit of the original PosiTech scores
make profile reconcile     # data quality, Yahoo vs. SEC
make test
```

Results land in `reports/` as CSVs, ready for Tableau.

## How the rebuilt pipeline works

| Step | 2022 | Now |
|---|---|---|
| Relevance | keyword search | GLiNER entity recognition linked to tickers, plus cashtags |
| Duplicates | exact text | sentence embeddings (same story within 3 days) |
| Scoring | a 2014 word list on the whole text | best benchmark classifier + zero-shot sentiment toward each company |
| Stored text | body scored, title saved | the scored text is the stored text |
| Timing | three date formats | UTC, DST-aware market close, timestamp precision recorded |

Model choices and thresholds live in `eval_lab/config/sentiment_models.yaml`, and the
company universe in `eval_lab/config/universe.yaml`.

## Layout

```
eval_lab/
  config/                 universe, models, XBRL concepts, sampling, rubric
  sentiment_models.py     one interface over all scorers (transformers, zero-shot, and baselines)
  sentiment_benchmark.py  scorer comparison on labeled data
  ingest_news.py          headlines (Google News RSS or GDELT)
  news_sentiment.py       rebuilt pipeline + market check
  sentiment_eval.py       2022 vs. now: audit of the original PosiTech scores
  ingest_*.py             prices, earnings, SEC filings, PosiTech posts
  profile_data.py         data quality + price anomaly detector
  reconcile.py            Yahoo Finance vs. SEC filings
  build_qa_tasks.py       160-question financial benchmark with known traps
  annotation/             browser tool for human labels (optional)
data/positech/            posts from the original PosiTech
tests/
```

## Limitations

- **Timestamps:** Google News gives only a date for items older than ~10 days. GDELT has
  exact times but rate-limits heavily; `--provider gdelt` is supported.
- **Earnings sample:** the earnings-day results rest on 29 events.
- **Entity recognizer:** conservative. It misses some headlines ("Alphabet's Stock Has
  Reached…") and doesn't understand negation.
- **Long posts:** transformer inputs are capped at 256 tokens.
- **Vendor data:** Yahoo Finance isn't a vendor-grade source, which is partly why it's
  reconciled against the filings.
