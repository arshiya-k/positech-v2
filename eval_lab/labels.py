"""Reading and writing rows of the `labels` table, which holds every label
from every labeler (human annotators and baselines) in one long format."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from eval_lab.rubric import Rubric

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def write_labels(con, item_id: str, rubric: Rubric, labeler: str, labeler_type: str, values: dict,
                 flags=(), rationale: str | None = None, seconds: float | None = None) -> None:
    """Replace this labeler's labels for one item."""
    con.execute("DELETE FROM labels WHERE item_id = ? AND labeler = ? AND rubric_id = ?", [item_id, labeler, rubric.id])
    rows = [
        (item_id, rubric.item_kind, rubric.id, labeler, labeler_type, key, str(value),
         ",".join(flags), rationale, seconds, now())
        for key, value in values.items()
        if value is not None
    ]
    con.executemany("INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def labeled_items(con, rubric_id: str, labeler: str) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT DISTINCT item_id FROM labels WHERE rubric_id = ? AND labeler = ?", [rubric_id, labeler]).fetchall()}


def vader_class(score: float) -> str:
    """VADER's standard cut-offs for its compound score."""
    return "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"


def load_labels(con, rubric_id: str) -> pd.DataFrame:
    """Long-format labels, with `sentiment_3class` derived from the 5-point
    sentiment scale so human labels can be compared with VADER."""
    df = con.execute("SELECT * FROM labels WHERE rubric_id = ?", [rubric_id]).df()
    if rubric_id == "sentiment" and not df.empty:
        five_point = df[df["dimension"] == "sentiment"].copy()
        five_point["value"] = np.sign(five_point["value"].astype(int)).map({-1: "negative", 0: "neutral", 1: "positive"})
        five_point["dimension"] = "sentiment_3class"
        df = pd.concat([df, five_point], ignore_index=True)
    return df


def wide(labels: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """item_id x labeler table of one dimension's values."""
    sub = labels[labels["dimension"] == dimension]
    return sub.pivot_table(index="item_id", columns="labeler", values="value", aggfunc="last")
