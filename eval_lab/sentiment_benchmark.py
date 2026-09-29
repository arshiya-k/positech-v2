"""Compare sentiment scorers on financial text labeled by people.

    python -m eval_lab.sentiment_benchmark [--scorers vader finbert ...]

Datasets (downloaded from Hugging Face on first run, cached in data/lab/benchmarks):

    twitter_financial_news  2,388 held-out finance tweets labeled bearish / bullish / neutral (MIT)
    fiqa                    351 headlines and posts, each scored for sentiment toward a named
                            target company (FiQA 2018, test + validation splits)
    financial_phrasebank    2,264 news sentences all annotators agreed on (CC BY-NC-SA 3.0)

Several finance models were fine-tuned on Financial PhraseBank, so their score
there is inflated. Those results are reported with `contaminated = true` and the
ranking uses only the two datasets no scorer was trained on.

Writes reports/sentiment_benchmark.csv (scorer x dataset metrics) and
reports/sentiment_benchmark_items.csv (every prediction).
"""
import argparse
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score

from eval_lab import sentiment_models
from eval_lab.db import DATA, write_report
from eval_lab.labels import SENTIMENT_CLASSES

CACHE = DATA / "benchmarks"
HELD_OUT = ["twitter_financial_news", "fiqa"]
FIQA_NEUTRAL_BAND = 0.1   # FiQA scores are continuous in [-1, 1]; |score| < 0.1 counts as neutral


def fiqa_class(score: float) -> str:
    return "positive" if score >= FIQA_NEUTRAL_BAND else "negative" if score <= -FIQA_NEUTRAL_BAND else "neutral"


def _twitter() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation").to_pandas()
    labels = {"0": "negative", "1": "positive", "2": "neutral"}   # bearish, bullish, neutral
    return pd.DataFrame({"text": ds["text"], "label": ds["label"].astype(str).map(labels), "source_type": "tweet"})


def _fiqa() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("TheFinAI/fiqa-sentiment-classification")
    df = pd.concat([ds["test"].to_pandas(), ds["valid"].to_pandas()], ignore_index=True)
    score = df["score"].astype(float)
    return pd.DataFrame({"text": df["sentence"], "label": score.map(fiqa_class), "target": df["target"],
                         "gold_score": score, "source_type": df["type"]})


def _phrasebank() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("Sentences_AllAgree.txt"))
        lines = z.read(name).decode("latin-1").splitlines()
    rows = [line.rsplit("@", 1) for line in lines if "@" in line]
    return pd.DataFrame({"text": [t.strip() for t, _ in rows], "label": [l.strip() for _, l in rows], "source_type": "news_sentence"})


LOADERS = {"twitter_financial_news": _twitter, "fiqa": _fiqa, "financial_phrasebank": _phrasebank}


def load_benchmarks() -> dict[str, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, loader in LOADERS.items():
        path = CACHE / f"{name}.parquet"
        if not path.exists():
            loader().to_parquet(path, index=False)
        out[name] = pd.read_parquet(path)
    return out


def metrics(gold: pd.Series, pred: pd.Series) -> dict:
    recalls = recall_score(gold, pred, labels=SENTIMENT_CLASSES, average=None, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=SENTIMENT_CLASSES, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(gold, pred, labels=SENTIMENT_CLASSES),
        **{f"recall_{c}": r for c, r in zip(SENTIMENT_CLASSES, recalls)},
    }


def rank(results: pd.DataFrame) -> pd.DataFrame:
    """Order scorers by mean macro-F1 on the held-out datasets."""
    held = results[results["dataset"].isin(HELD_OUT)]
    return (held.groupby("scorer")[["macro_f1", "cohen_kappa"]].mean()
            .rename(columns=lambda c: f"held_out_{c}").sort_values("held_out_macro_f1", ascending=False))


def main() -> None:
    cfg = sentiment_models.config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorers", nargs="*", default=list(cfg["scorers"]))
    args = parser.parse_args()

    benchmarks = load_benchmarks()
    print("datasets: " + ", ".join(f"{k} ({len(v)})" for k, v in benchmarks.items()))
    results, items = [], []
    for scorer in args.scorers:
        spec = cfg["scorers"][scorer]
        sentiment_models.score(scorer, ["warm-up"])  # load weights before timing
        for dataset, df in benchmarks.items():
            targets = df["target"] if "target" in df else None
            pred, rate = sentiment_models.score(scorer, df["text"], targets)
            row = {"scorer": scorer, "dataset": dataset, "contaminated": dataset in spec.get("trained_on", []),
                   **metrics(df["label"], pred["label"]), "texts_per_second": rate}
            if "gold_score" in df:
                row["spearman_vs_gold_score"] = spearmanr(pred["score"], df["gold_score"]).statistic
            results.append(row)
            items.append(pd.concat([df[["text", "label", "source_type"]].rename(columns={"label": "gold"}),
                                    pred.rename(columns={"label": "pred"})], axis=1).assign(scorer=scorer, dataset=dataset))
            print(f"  {scorer:20} {dataset:24} macro-F1 {row['macro_f1']:.3f}  kappa {row['cohen_kappa']:.3f}"
                  f"{'  (trained on this data)' if row['contaminated'] else ''}")

    results = pd.DataFrame(results)
    ranking = rank(results)
    results = results.merge(ranking, left_on="scorer", right_index=True)
    write_report(results.sort_values(["held_out_macro_f1", "dataset"], ascending=[False, True]), "sentiment_benchmark.csv")
    write_report(pd.concat(items, ignore_index=True), "sentiment_benchmark_items.csv")

    shown = results.assign(cell=[f"{f:.3f}{'*' if c else ''}" for f, c in zip(results["macro_f1"], results["contaminated"])])
    table = shown.pivot(index="scorer", columns="dataset", values="cell")[list(LOADERS)]
    table = table.join(ranking.round(3)).sort_values("held_out_macro_f1", ascending=False)
    print("\nmacro-F1 by dataset (* = scorer was trained on that dataset, so excluded from the ranking):")
    print(table.to_string())


if __name__ == "__main__":
    main()
