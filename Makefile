PY ?= .venv/bin/python

.PHONY: setup data profile reconcile tasks sentiment sentiment-report sentiment-benchmark news news-sentiment agreement annotate test

setup:            ## create the virtualenv and install dependencies
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

data:             ## pull prices + earnings, SEC annual/quarterly fundamentals (needs SEC_USER_AGENT), PosiTech posts
	$(PY) -m eval_lab.ingest_market
	$(PY) -m eval_lab.ingest_fundamentals
	$(PY) -m eval_lab.ingest_posts

profile:          ## data-quality report + price anomaly detector precision/recall
	$(PY) -m eval_lab.profile_data

reconcile:        ## compare Yahoo Finance statements and street EPS with the SEC filings
	$(PY) -m eval_lab.reconcile

tasks:            ## build the financial Q&A benchmark (questions with gold answers and known traps)
	$(PY) -m eval_lab.build_qa_tasks

sentiment:        ## sample posts for annotation; after importing labels, compare VADER with humans and the market
	$(PY) -m eval_lab.sentiment_eval sample

sentiment-report:
	$(PY) -m eval_lab.sentiment_eval report

sentiment-benchmark: ## compare VADER, finance dictionaries and transformer models on professionally labeled data
	$(PY) -m eval_lab.sentiment_benchmark

news:             ## collect ~12 weeks of headlines for every company (google; use --provider gdelt for exact timestamps)
	$(PY) -m eval_lab.ingest_news --provider google

news-sentiment:   ## rebuilt pipeline: entity relevance, dedupe, modern scoring, market check (news + PosiTech posts)
	$(PY) -m eval_lab.news_sentiment

agreement:        ## pairwise kappa between all labelers
	$(PY) -m eval_lab.agreement

annotate:         ## serve the annotation tool at http://localhost:8765
	@echo "Annotation tool: http://localhost:8765"
	$(PY) -m http.server 8765 --directory eval_lab/annotation

test:
	$(PY) -m pytest -q
