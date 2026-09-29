# positech-v2

A rebuild of the sentiment analysis in [PosiTech](https://github.com/arshiya-k/PosiTech),
a stock-investment site I built in 2022 that scored news, Reddit and Twitter posts with
VADER. This version audits the original, benchmarks current methods against
professionally labeled data, reruns the pipeline on ~18,000 fresh headlines, and checks
the results against the market. It also profiles and reconciles the market and SEC
data the market check depends on.

Everything runs locally in Python, SQL (DuckDB) and open-source models, with no API keys.

## Results

### 1. The original pipeline didn't hold up

- **82% of news posts were off-topic.** The NewsAPI query `"AAPL AND stock"` returned
  generic market roundups. One post tagged GOOGL is titled *"Nvidia: Prepare To Be Underwhelmed."*
- **Stored scores couldn't be reproduced.** The app scored article bodies but saved only
  titles. Re-running VADER agrees with the stored class 58% of the time (κ = 0.34).
- **VADER had no predictive value.** Its direction matched the next day's move vs. SPY
  51% of the time.

### 2. Current models read financial text far better

Nine scorers were tested on Twitter Financial News, FiQA and Financial PhraseBank.
The ranking uses only the two datasets no model was trained on:

| Scorer | Held-out macro-F1 | κ |
|---|---|---|
| Zero-shot DeBERTa, sentiment toward each company | **0.66** | 0.52 |
| Financial-RoBERTa-large | **0.65** | **0.54** |
| FinBERT | 0.58 | 0.42 |
| Twitter-RoBERTa (general-purpose) | 0.51 | 0.31 |
| **VADER (original)** | 0.41 | **0.16** |
| Loughran-McDonald finance dictionary | 0.38 | 0.15 |

- Agreement with professional labels is **3.4× VADER's** (κ 0.54 vs. 0.16).
- Models fine-tuned on PhraseBank score 0.996 on it but 0.46–0.71 elsewhere. Those
  scores are flagged and left out of the ranking.

### 3. The rebuilt pipeline, on 18,806 fresh headlines

- **Relevance by entity recognition, not keywords:** 1,300 keyword matches rejected
  ("16 Apple Desserts…"), and 627 extra company mentions found.
- **Duplicates removed by meaning:** 13% of headlines were syndicated copies, leaving
  **13,359 unique company mentions**.
- **Sentiment per company:** "Nvidia slides as Google gains" scores Nvidia −1.0 and
  Google +0.99.
- VADER agrees with the best model on only 49% of headlines. For example, it calls
  "IBM Lowers Its Growth Outlook as Sales… Sink 42%" positive.

### 4. Sentiment describes moves; it doesn't predict them

Correlation with abnormal return vs. SPY, measured per company-day:

| | VADER | Fin-RoBERTa-large | Zero-shot |
|---|---|---|---|
| Same session (1,674 company-days) | 0.05 | 0.21 | **0.25** |
| Same session, earnings days (29) | 0.42 | 0.71 | **0.78** |
| Next session (1,644) | −0.03 | −0.01 | 0.02 |

- **Same session:** the modern models explain moves about 5× better.
- **Next session:** none of them predicts anything. A small per-headline "signal"
  disappeared once headlines were grouped per company-day and restricted to exact
  timestamps. It came from duplicate coverage and date-only timestamps.

### 5. The market data needed checking too

- **Yahoo Finance vs. SEC filings:**

  | Figure | Agreement |
  |---|---|
  | Net income | 100% |
  | EPS | 93–96% |
  | Revenue | 88% |
  | Operating income | 59–68% |

  The operating-income gap is definitional: Yahoo computes its own and strips out
  one-off charges.
- **Vendor issues found:**
  - Honeywell's per-share figures are exactly 2× off at Yahoo.
  - Spin-offs are recorded as stock "splits" (1.061, 1.046).
  - Period ends are snapped to month-end.
- **SEC data issues found and fixed:**
  - J&J's 53-week year mislabeled.
  - Deere dating one fiscal year two ways.
  - Missing XBRL tags.
  - P&G's subtotal outranking its total revenue.
  - Restatements after spin-offs (quarter-sum mismatches cut from 60 to 2).
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
make sentiment-report      # audit of the original PosiTech scores
make profile reconcile     # data quality, Yahoo vs. SEC
make test
```

Results land in `reports/` as CSVs, ready for Tableau.

## How the rebuilt pipeline works

| Step | 2022 | Now |
|---|---|---|
| Relevance | keyword search | GLiNER entity recognition linked to tickers, plus cashtags |
| Duplicates | exact text | sentence embeddings (same story within 3 days) |
| Scoring | VADER on the whole text | best benchmark classifier + zero-shot sentiment toward each company |
| Stored text | body scored, title saved | the scored text is the stored text |
| Timing | three date formats | UTC, DST-aware market close, timestamp precision recorded |

Model choices and thresholds live in `eval_lab/config/sentiment_models.yaml`, and the
company universe in `eval_lab/config/universe.yaml`.

## Layout

```
eval_lab/
  config/                 universe, models, XBRL concepts, sampling, rubric
  sentiment_models.py     one interface over VADER, dictionaries, transformers, zero-shot
  sentiment_benchmark.py  scorer comparison on labeled data
  ingest_news.py          headlines (Google News RSS or GDELT)
  news_sentiment.py       rebuilt pipeline + market check
  sentiment_eval.py       audit of the original PosiTech scores
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
