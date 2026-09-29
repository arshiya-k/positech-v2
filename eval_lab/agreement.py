"""Pairwise agreement between every pair of labelers (human annotators and VADER baselines)
on every rubric dimension. Ordinal scales use quadratic-weighted Cohen's kappa,
so a 4-vs-5 disagreement costs less than 1-vs-5; categories use plain kappa.

    python -m eval_lab.agreement
"""
from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from eval_lab.db import eval_db, write_report
from eval_lab.labels import SENTIMENT_CLASSES, load_labels, wide
from eval_lab.rubric import Rubric

MIN_OVERLAP = 5


def dimension_specs(rubric: Rubric) -> list[tuple[str, bool, list]]:
    specs = [(d["key"], rubric.is_ordinal(d["key"]), rubric.allowed_values(d["key"])) for d in rubric.dimensions]
    if rubric.id == "sentiment":
        specs.append(("sentiment_3class", False, SENTIMENT_CLASSES))
    return specs


def agreement_table(labels: pd.DataFrame, rubric: Rubric) -> list[dict]:
    rows = []
    for dimension, ordinal, allowed in dimension_specs(rubric):
        table = wide(labels, dimension)
        for a, b in combinations(table.columns, 2):
            both = table[[a, b]].dropna()
            if len(both) < MIN_OVERLAP:
                continue
            x, y = (both[a].astype(int), both[b].astype(int)) if ordinal else (both[a], both[b])
            rows.append({
                "rubric": rubric.id, "dimension": dimension, "labeler_a": a, "labeler_b": b, "n": len(both),
                "exact_agreement": (x == y).mean(),
                "kappa": cohen_kappa_score(x, y, labels=allowed, weights="quadratic" if ordinal else None),
                "kappa_type": "quadratic-weighted" if ordinal else "unweighted",
            })
    return rows


def main() -> None:
    con = eval_db()
    rows = []
    for name in ["sentiment"]:
        rubric = Rubric.load(name)
        labels = load_labels(con, rubric.id)
        if not labels.empty:
            rows += agreement_table(labels, rubric)
    if not rows:
        raise SystemExit(f"No labeler pairs with at least {MIN_OVERLAP} shared items yet.")
    out = pd.DataFrame(rows)
    path = write_report(out, "agreement.csv")
    print(out.round(3).to_string(index=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
