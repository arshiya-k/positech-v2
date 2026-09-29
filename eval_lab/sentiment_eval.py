"""Sentiment benchmark over PosiTech's scraped posts.

    python -m eval_lab.sentiment_eval sample    stratified sample + VADER baseline labels
    python -m eval_lab.sentiment_eval report    stored vs re-run VADER, next-day market moves, off-topic posts

Human labels from the annotation tool (see eval_lab.annotation) are optional:
when they exist, they become the reference VADER is measured against.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from eval_lab import companies
from eval_lab.market_time import event_returns
from eval_lab.db import eval_db, load_config, replace_rows, source_db, write_report
from eval_lab.labels import SENTIMENT_CLASSES, load_labels, now, vader_class, wide, write_labels
from eval_lab.rubric import Rubric
from eval_lab.sampling import representativeness, stratified_sample



def cmd_sample(_args) -> None:
    cfg, seed = load_config("sampling.yaml")["sentiment"], load_config("sampling.yaml")["seed"]
    posts = source_db().sql("SELECT * FROM posts").df()
    if posts.empty:
        raise SystemExit("No posts: run `python -m eval_lab.ingest_posts` first.")
    if cfg["exclude_duplicates"]:
        posts = posts[~posts["is_duplicate"]]
    posts["vader_class"] = posts["vader_recomputed"].map(vader_class)
    sample = stratified_sample(posts, cfg["n"], cfg["strata"], seed)

    con = eval_db()
    items = sample.rename(columns={"post_id": "item_id"})[["item_id", "source", "ticker", "vader_class"]].assign(sampled_at=now())
    con.execute("DELETE FROM sentiment_items")
    replace_rows(con, "sentiment_items", items, "item_id")

    rubric = Rubric.load("sentiment")
    # Baseline labels belong to one sample; clear the previous sample's before writing.
    con.execute("DELETE FROM labels WHERE rubric_id = ? AND labeler_type = 'baseline'", [rubric.id])
    for post in sample.itertuples():
        # Baselines: VADER re-run on the stored text, and the score PosiTech itself saved.
        write_labels(con, post.post_id, rubric, "vader", "baseline", {"sentiment_3class": vader_class(post.vader_recomputed)})
        write_labels(con, post.post_id, rubric, "vader:positech", "baseline", {"sentiment_3class": vader_class(post.vader_stored)})
    write_report(representativeness(posts, sample, cfg["strata"]), "sentiment_sampling_summary.csv")
    print(f"sampled {len(sample)} of {len(posts)} posts")
    print(sample.groupby(["source", "vader_class"]).size().unstack(fill_value=0).to_string())


def next_day_abnormal_returns(posts: pd.DataFrame, prices: pd.DataFrame, benchmark: str) -> pd.Series:
    """Return from the last close at or before each post to the next close, minus
    the benchmark's return over the same two closes."""
    events = pd.DataFrame({"ticker": posts["ticker"], "at": posts["posted_at"]}, index=posts.index)
    returns = event_returns(events, prices, benchmark)
    return returns.get("reaction_abnormal", pd.Series(index=posts.index, dtype=float)).rename("abnormal_return_next_day")


def compare(reference: pd.Series, candidate: pd.Series, labels: list) -> dict:
    both = pd.concat([reference, candidate], axis=1).dropna()
    if len(both) < 5:
        return {"n": len(both)}
    y_ref, y_hat = both.iloc[:, 0], both.iloc[:, 1]
    return {
        "n": len(both),
        "accuracy": accuracy_score(y_ref, y_hat),
        "macro_f1": f1_score(y_ref, y_hat, labels=labels, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_ref, y_hat, labels=labels),
    }


def mentions_company(posts: pd.DataFrame, universe: dict) -> pd.Series:
    """Keyword stand-in for the rubric's relevance label: does the post mention its
    ticker or company name at all?"""
    return pd.Series([companies.mentions(t, k) for t, k in zip(posts["text"], posts["ticker"])], index=posts.index)


def cmd_report(_args) -> None:
    """Human labels are optional. Without them the report still compares PosiTech's
    stored VADER scores with a re-run, measures each score against next-day market
    moves, and estimates off-topic posts with a keyword check."""
    universe = load_config("universe.yaml")
    con, src = eval_db(), source_db()
    items = con.sql("SELECT * FROM sentiment_items").df()
    if items.empty:
        raise SystemExit("No sample yet: run `python -m eval_lab.sentiment_eval sample` first.")
    posts = src.sql("SELECT * FROM posts").df().set_index("post_id").loc[items["item_id"]]
    labels = load_labels(con, "sentiment")
    humans = sorted(labels.loc[labels["labeler_type"] == "human", "labeler"].unique())
    reference = humans[0] if humans else "vader"
    reference_kind = "human" if humans else "VADER re-run on the stored text (no human labels)"

    three = wide(labels, "sentiment_3class")
    relevance = wide(labels, "relevance").apply(pd.to_numeric)
    five = wide(labels, "sentiment").apply(pd.to_numeric)
    mentions = mentions_company(posts, universe)
    if humans:
        relevant, relevance_source = (relevance[reference] >= 2).reindex(posts.index).fillna(False), f"{reference} relevance >= 2"
    else:
        relevant, relevance_source = mentions, "post mentions its ticker or company name"

    metrics = []
    for labeler in three.columns.drop(reference, errors="ignore"):
        for subset, mask in [("all", slice(None)), ("relevant_only", relevant.reindex(three.index).fillna(False).astype(bool))]:
            metrics.append({"labeler": labeler, "reference": reference, "reference_kind": reference_kind, "subset": subset,
                            **compare(three.loc[mask, reference], three.loc[mask, labeler], SENTIMENT_CLASSES)})

    per_item = posts[["source", "ticker", "posted_at", "text", "vader_stored", "vader_recomputed"]].copy()
    per_item["mentions_company"] = mentions
    per_item = per_item.join(three.add_prefix("sentiment3_")).join(five.add_prefix("sentiment5_")).join(relevance.add_prefix("relevance_"))
    prices = src.sql("SELECT ticker, date, adj_close FROM prices").df()
    if not prices.empty:
        per_item = per_item.join(next_day_abnormal_returns(posts, prices, universe["benchmark"]))
        move = per_item["abnormal_return_next_day"]
        scores = {**{f"{k} (5-pt)": five[k] for k in five.columns},
                  "vader": posts["vader_recomputed"], "vader:positech": posts["vader_stored"]}
        for labeler, score in scores.items():
            for subset, mask in [("all", pd.Series(True, index=per_item.index)), ("relevant_only", relevant.reindex(per_item.index).fillna(False))]:
                both = pd.concat([score.reindex(per_item.index), move], axis=1)[mask.astype(bool)].dropna()
                directional = both[both.iloc[:, 0] != 0]
                metrics.append({
                    "labeler": labeler, "subset": subset, "reference": "next-day abnormal return", "n": len(both),
                    "spearman_vs_return": spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic if len(both) >= 5 else np.nan,
                    "direction_hit_rate": (np.sign(directional.iloc[:, 0]) == np.sign(directional.iloc[:, 1])).mean() if len(directional) else np.nan,
                })

    off_topic = pd.DataFrame({"posts": posts.groupby("source").size(),
                              "no_company_mention": (~mentions).groupby(posts["source"]).mean()})
    if humans and reference in relevance:
        off_topic["judged_off_topic"] = (relevance[reference] == 1).groupby(posts["source"]).mean()
    write_report(per_item.reset_index(names="item_id"), "sentiment_items.csv")
    write_report(pd.DataFrame(metrics), "sentiment_metrics.csv")
    write_report(off_topic.reset_index(), "sentiment_off_topic.csv")
    print(f"reference: {reference} [{reference_kind}]; relevant = {relevance_source}\n")
    print(pd.DataFrame(metrics).round(3).to_string(index=False))
    print("\noff-topic posts by source:\n" + off_topic.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample").set_defaults(func=cmd_sample)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
