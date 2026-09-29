"""The rebuilt sentiment pipeline, run on fresh news headlines (see
eval_lab.ingest_news) and on PosiTech's original posts so the two can be compared.

    python -m eval_lab.news_sentiment [--sources news positech]

Stages, each fixing something the 2022 pipeline got wrong:

  1. documents  The text that gets scored is the text that gets stored, with UTC
                timestamps. (PosiTech scored article bodies but saved titles.)
  2. relevance  GLiNER recognizes company and stock entities, and each is linked to
                a ticker; cashtags ($AAPL) count too. A document counts for a company
                only if it's recognized as an entity, not because a keyword matched.
                ("16 Apple Desserts" is not about AAPL; "Visa requirements
                tightened" is not about V.)
  3. dedupe     Syndicated copies of one story (same company, sentence-embedding
                similarity >= 0.85, within 3 days) collapse into the earliest copy,
                so one story isn't counted forty times.
  4. scoring    The benchmark's best whole-text classifier, and a zero-shot model
                asked whether the text is good or bad news *for that specific company*.
  5. market     Abnormal return vs SPY in the session that reacted to the document
                (does the score describe price-moving news?) and in the session
                after it (does it predict anything?): overall, on earnings days, and
                for documents with an exact timestamp only (date-only timestamps
                can put a headline in the wrong session).

Writes reports/news_funnel.csv, news_items.csv, news_market_check.csv and
news_relevance_examples.csv.
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind

from eval_lab import companies, sentiment_models
from eval_lab.db import DATA, NEWS_DB, REPORTS, load_config, news_db, source_db, write_report
from eval_lab.market_time import event_returns

CACHE = DATA / "cache"


# ---------------------------------------------------------------- cache

def cached(name: str, keys: pd.Series, compute) -> pd.DataFrame:
    """Row-level cache keyed by a string: compute only keys not seen before.
    `compute(missing_keys)` returns a frame with one row per key, in order."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.parquet"
    store = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["key"])
    missing = pd.Index(keys.unique()).difference(store["key"])
    if len(missing):
        fresh = compute(list(missing)).assign(key=list(missing))
        store = pd.concat([store, fresh], ignore_index=True)
        store.to_parquet(path, index=False)
    return store.set_index("key").loc[keys.to_numpy()].reset_index(drop=True)


def text_key(*parts) -> str:
    return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()


# ---------------------------------------------------------------- 1. documents

def load_documents(source: str) -> pd.DataFrame:
    """doc_id, text, published_at (naive UTC), time_precision, query_tickers (list),
    source (news provider, or "positech"), plus provenance columns."""
    if source == "news":
        if not NEWS_DB.exists():
            return pd.DataFrame()
        rows = news_db(read_only=True).sql("""
            SELECT provider || ':' || article_id AS doc_id, provider AS source, any_value(title) AS text,
                   min(seen_at) AS published_at, any_value(time_precision) AS time_precision,
                   list(DISTINCT query_ticker) AS query_tickers, any_value(domain) AS domain, any_value(url) AS url
            FROM news_articles GROUP BY provider, article_id
        """).df()
    else:
        rows = source_db().sql("""
            SELECT post_id AS doc_id, 'positech' AS source, text, posted_at AS published_at, 'exact' AS time_precision,
                   [ticker] AS query_tickers, source AS domain, link AS url
            FROM posts WHERE NOT is_duplicate
        """).df()
    rows["query_tickers"] = rows["query_tickers"].map(list)
    return rows[rows["text"].str.strip().str.len() > 0].reset_index(drop=True)


# ---------------------------------------------------------------- 2. relevance

def link_entities(entities: list[dict]) -> list[str]:
    """Tickers for the recognized company names, in order of first mention."""
    tickers = []
    for e in entities:
        ticker = companies.link(e["text"])
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def recognize_companies(texts: pd.Series, cfg: dict) -> pd.DataFrame:
    """Recognized company/stock spans per text (cached), and the tickers they link to.
    Linking runs fresh each time, so changing names in universe.yaml needs no re-run."""
    def compute(keys):
        import warnings

        from gliner import GLiNER

        warnings.filterwarnings("ignore", module="gliner")  # long posts are truncated to the model's window
        model = GLiNER.from_pretrained(cfg["ner_model"]).to(sentiment_models.device())
        by_key = dict(zip(keys_series, texts))
        batch = [by_key[k] for k in keys]
        entities = []
        for i in range(0, len(batch), 64):
            entities += model.inference(batch[i:i + 64], cfg["ner_labels"], threshold=cfg["ner_threshold"])
        return pd.DataFrame({"spans": [json.dumps([e["text"] for e in es]) for es in entities]})

    keys_series = texts.map(lambda t: text_key(cfg["ner_model"], cfg["ner_labels"], cfg["ner_threshold"], t))
    spans = cached("ner", keys_series, compute)["spans"].map(json.loads).set_axis(texts.index)
    linked = [list(dict.fromkeys([*link_entities([{"text": s} for s in ss]), *companies.cashtags(t)]))
              for ss, t in zip(spans, texts)]
    return pd.DataFrame({"spans": spans, "linked": linked}, index=texts.index)


def build_mentions(docs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per (document, company): every company the search was for, plus every
    company recognized in the text."""
    ents = recognize_companies(docs["text"], cfg)
    rows = []
    for doc, spans, linked in zip(docs.itertuples(), ents["spans"], ents["linked"]):
        for ticker in dict.fromkeys([*doc.query_tickers, *linked]):
            rows.append({
                "doc_id": doc.doc_id, "source": doc.source, "ticker": ticker, "text": doc.text,
                "published_at": doc.published_at, "time_precision": doc.time_precision, "domain": doc.domain, "url": doc.url,
                "searched_for": ticker in doc.query_tickers,
                "relevant_keyword": companies.mentions(doc.text, ticker),
                "relevant_ner": ticker in linked,
                "recognized_companies": ", ".join(spans),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. dedupe

def mark_duplicates(times: pd.Series, embeddings: np.ndarray, threshold: float, days: int) -> np.ndarray:
    """For documents about one company: index of the earlier document each one
    duplicates, or -1. Embeddings must be L2-normalized."""
    order = np.argsort(times.to_numpy(), kind="stable")
    t = times.to_numpy()[order].astype("datetime64[s]").astype(np.int64)
    sims = embeddings[order] @ embeddings[order].T
    kept = np.zeros(len(order), dtype=bool)
    duplicate_of = np.full(len(order), -1)
    for i in range(len(order)):
        # earlier documents that were kept, are recent enough, and say the same thing
        match = np.flatnonzero(kept[:i] & (t[i] - t[:i] <= days * 86400) & (sims[i, :i] >= threshold))
        if len(match):
            duplicate_of[i] = order[match[0]]
        else:
            kept[i] = True
    out = np.full(len(order), -1)
    out[order] = duplicate_of
    return out


def dedupe(mentions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    relevant = mentions[mentions["relevant_ner"]]
    if relevant.empty:
        return mentions.assign(duplicate_of=None, is_unique=False)

    def compute(keys):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg["embedding_model"], device=sentiment_models.device())
        by_key = dict(zip(key_series, relevant["text"]))
        vectors = model.encode([by_key[k] for k in keys], batch_size=128, normalize_embeddings=True)
        return pd.DataFrame({"embedding": list(vectors)})

    key_series = relevant["text"].map(lambda t: text_key(cfg["embedding_model"], t))
    vectors = np.vstack(cached("embeddings", key_series, compute)["embedding"].to_numpy())
    vectors = pd.DataFrame(vectors, index=relevant.index)
    duplicate_of = pd.Series(None, index=mentions.index, dtype=object)
    for ticker, group in relevant.groupby("ticker"):
        dup = mark_duplicates(group["published_at"], vectors.loc[group.index].to_numpy(),
                              cfg["duplicate_similarity"], cfg["duplicate_days"])
        duplicate_of.loc[group.index] = [group["doc_id"].iloc[d] if d >= 0 else None for d in dup]
    return mentions.assign(duplicate_of=duplicate_of, is_unique=mentions["relevant_ner"] & duplicate_of.isna())


# ---------------------------------------------------------------- 4. scoring

def pick_classifier(cfg: dict) -> str:
    if cfg["classifier"] != "auto":
        return cfg["classifier"]
    path = REPORTS / "sentiment_benchmark.csv"
    if not path.exists():
        raise SystemExit("Run `python -m eval_lab.sentiment_benchmark` first, or set pipeline.classifier.")
    results = pd.read_csv(path)
    kinds = {k: v["kind"] for k, v in sentiment_models.config()["scorers"].items()}
    results = results[results["scorer"].map(kinds) == "classifier"]
    return results.sort_values("held_out_macro_f1", ascending=False)["scorer"].iloc[0]


def score_column(name: str, texts: pd.Series, targets: pd.Series | None = None) -> pd.DataFrame:
    targets = targets if targets is not None else pd.Series("", index=texts.index)
    keys = pd.Series([text_key(name, t, g) for t, g in zip(texts, targets)], index=texts.index)
    lookup = pd.DataFrame({"text": texts, "target": targets, "key": keys}).drop_duplicates("key").set_index("key")

    def compute(missing):
        rows = lookup.loc[missing]
        out, _ = sentiment_models.score(name, rows["text"], None if targets.eq("").all() else rows["target"])
        return out[["label", "score"]]

    return cached(f"scores_{name}", keys, compute).set_index(texts.index)


# ---------------------------------------------------------------- 5. market

BIG_MOVE = 0.03   # abnormal return of at least 3% vs the benchmark

def market_metrics(items: pd.DataFrame, scorers: list[str]) -> pd.DataFrame:
    rows = []
    earnings_mask = items["on_earnings_day"].fillna(False).astype(bool)
    exact_mask = items["time_precision"].eq("exact")
    for source, s in items.groupby("source"):
        subsets = [("all", pd.Series(True, index=s.index)), ("exact_time", exact_mask.loc[s.index]),
                   ("earnings_days", earnings_mask.loc[s.index]), ("non_earnings_days", ~earnings_mask.loc[s.index]),
                   ("big_moves", None)]
        for subset, mask in subsets:
            for window in ["reaction", "next_session"]:
                if subset == "big_moves":
                    # Days the stock moved at least BIG_MOVE vs the market in this window. This
                    # conditions on the outcome, so it describes alignment; it can't test prediction.
                    mask = s[f"{window}_abnormal"].abs() >= BIG_MOVE
                for scorer in scorers:
                    d = s[mask][[f"{scorer}_score", f"{scorer}_label", f"{window}_abnormal", "ticker", "reaction_date"]].dropna()
                    if len(d) < 10:
                        continue
                    score, label, ret = d[f"{scorer}_score"], d[f"{scorer}_label"], d[f"{window}_abnormal"]
                    directional = d[label != "neutral"]
                    pos, neg = ret[label == "positive"], ret[label == "negative"]
                    daily = d.groupby(["ticker", "reaction_date"]).agg(score=(f"{scorer}_score", "mean"), ret=(f"{window}_abnormal", "first"))
                    rows.append({
                        "source": source, "subset": subset, "window": window, "scorer": scorer, "n": len(d),
                        "spearman": spearmanr(score, ret).statistic,
                        "direction_hit_rate": (np.sign(directional[f"{scorer}_score"]) == np.sign(directional[f"{window}_abnormal"])).mean(),
                        "mean_abnormal_bps_positive": pos.mean() * 1e4, "mean_abnormal_bps_negative": neg.mean() * 1e4,
                        "spread_bps": (pos.mean() - neg.mean()) * 1e4,
                        "spread_p_value": ttest_ind(pos, neg, equal_var=False).pvalue if len(pos) > 2 and len(neg) > 2 else np.nan,
                        "n_ticker_days": len(daily),
                        "spearman_ticker_day": spearmanr(daily["score"], daily["ret"]).statistic if len(daily) >= 10 else np.nan,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- run

def run(source: str, cfg: dict, prices: pd.DataFrame, earnings: pd.DataFrame, benchmark: str, classifier: str) -> tuple[pd.DataFrame, dict]:
    docs = load_documents(source)
    if docs.empty:
        print(f"{source}: no documents (for news, run `python -m eval_lab.ingest_news` first)")
        return pd.DataFrame(), {}
    print(f"{source}: {len(docs)} documents; recognizing companies...")
    mentions = dedupe(build_mentions(docs, cfg), cfg)
    searched = mentions[mentions["searched_for"]]
    funnel = {
        "source": source if source == "positech" else f"news ({', '.join(sorted(docs['source'].unique()))})",
        "documents": len(docs), "exact_timestamps": int(docs["time_precision"].eq("exact").sum()), "company_searches_matched": len(searched),
        "keyword_relevant": int(searched["relevant_keyword"].sum()),
        "entity_relevant": int(searched["relevant_ner"].sum()),
        "extra_companies_recognized": int((~mentions["searched_for"] & mentions["relevant_ner"]).sum()),
        "unique_after_dedupe": int(mentions["is_unique"].sum()),
    }

    items = mentions[mentions["is_unique"]].copy()
    print(f"{source}: {len(items)} unique company mentions; scoring...")
    names = items["ticker"].map(companies.display_name)
    for scorer, targets in [(classifier, None), (cfg["entity_scorer"], names)]:
        scored = score_column(scorer, items["text"], targets)
        items[f"{scorer}_label"], items[f"{scorer}_score"] = scored["label"], scored["score"]

    events = pd.DataFrame({"ticker": items["ticker"], "at": items["published_at"]}, index=items.index)
    items = items.join(event_returns(events, prices, benchmark))
    reaction_days = set(zip(earnings["ticker"], pd.to_datetime(earnings["reaction_date"])))
    items["on_earnings_day"] = [(t, d) in reaction_days if pd.notna(d) else None for t, d in zip(items["ticker"], items["reaction_date"])]
    funnel["with_market_data"] = int(items["reaction_abnormal"].notna().sum())
    return pd.concat([mentions[~mentions["is_unique"]], items], ignore_index=True), funnel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", default=["news", "positech"], choices=["news", "positech"])
    args = parser.parse_args()

    cfg = sentiment_models.config()["pipeline"]
    classifier = pick_classifier(cfg)
    scorers = [classifier, cfg["entity_scorer"]]
    src = source_db()
    prices = src.sql("SELECT ticker, date, adj_close FROM prices").df()
    earnings = src.sql("SELECT ticker, reaction_date FROM earnings WHERE reaction_date IS NOT NULL").df()
    benchmark = load_config("universe.yaml")["benchmark"]
    print(f"scorers: {', '.join(scorers)}")

    all_items, funnels = [], []
    for source in args.sources:
        items, funnel = run(source, cfg, prices, earnings, benchmark, classifier)
        if funnel:
            all_items.append(items)
            funnels.append(funnel)
    if not all_items:
        raise SystemExit("Nothing to analyze.")
    items = pd.concat(all_items, ignore_index=True)
    unique = items[items["is_unique"]]

    examples = items[items["searched_for"] & (items["relevant_keyword"] != items["relevant_ner"])]
    examples = examples.assign(case=np.where(examples["relevant_keyword"], "keyword yes, entity no", "entity yes, keyword no"))
    examples = examples.groupby(["source", "case"]).head(25)[["source", "case", "ticker", "text", "recognized_companies"]]

    metrics = market_metrics(unique, scorers)
    write_report(pd.DataFrame(funnels), "news_funnel.csv")
    write_report(items.drop(columns=["url"]), "news_items.csv")
    write_report(metrics, "news_market_check.csv")
    write_report(examples, "news_relevance_examples.csv")

    print("\nfunnel:\n" + pd.DataFrame(funnels).set_index("source").T.to_string())
    if len(metrics):
        cols = ["source", "subset", "window", "scorer", "n", "spearman", "direction_hit_rate", "spread_bps", "spread_p_value"]
        print("\nmarket check:\n" + metrics[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
