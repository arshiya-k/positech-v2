"""Export annotation batches for the browser tool, and import finished labels.

    python -m eval_lab.annotation export sentiment [--n 50]
    python -m eval_lab.annotation import path/to/labels_export.json

Items are shuffled so annotators can't infer the sampling strata from order,
and VADER's score isn't shown so it can't anchor the human label.
Open the tool with `make annotate` and load a batch from eval_lab/annotation/batches.
"""
import argparse
import json
from datetime import datetime

import pandas as pd

from eval_lab import companies
from eval_lab.db import BATCHES, eval_db, source_db
from eval_lab.labels import write_labels
from eval_lab.rubric import Rubric


def sentiment_items(n: int, seed: int) -> list[dict]:
    items = eval_db().sql("SELECT item_id FROM sentiment_items").df()["item_id"]
    posts = source_db().sql("SELECT * FROM posts").df().set_index("post_id").loc[items]
    posts = posts.sample(n=min(n, len(posts)), random_state=seed)
    return [{
        "item_id": post_id,
        "context": {"ticker": p.ticker, "company": companies.display_name(p.ticker), "source": p.source},
        "fields": [
            {"label": "Ticker", "value": p.ticker},
            {"label": "Source", "value": p.source},
            {"label": "Posted (UTC)", "value": f"{p.posted_at:%Y-%m-%d %H:%M}"},
            {"label": "Link", "value": p.link, "link": True},
            {"label": "Post", "value": p.text, "long": True},
        ],
    } for post_id, p in posts.iterrows()]


def cmd_export(args) -> None:
    rubric = Rubric.load(args.rubric)
    items = sentiment_items(args.n, args.seed)
    if not items:
        raise SystemExit("Nothing to export yet.")
    batch_id = f"{rubric.id}_{datetime.now():%Y%m%d_%H%M%S}"
    path = BATCHES / f"{batch_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"batch_id": batch_id, "rubric": rubric.to_ui(), "items": items}, indent=2, default=str))
    print(f"{len(items)} items -> {path}")


def cmd_import(args) -> None:
    data = json.loads(open(args.path).read())
    rubric = Rubric.load(data["rubric_id"])
    annotator = data["annotator"].strip().lower().replace(" ", "_")
    con = eval_db()
    for label in data["labels"]:
        write_labels(con, label["item_id"], rubric, f"human:{annotator}", "human", label["values"],
                     label.get("flags", []), label.get("note") or None, label.get("seconds"))
    seconds = [l["seconds"] for l in data["labels"] if l.get("seconds")]
    median = sorted(seconds)[len(seconds) // 2] if seconds else 0
    print(f"imported {len(data['labels'])} labels from {annotator} (median {median:.0f}s per item)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("rubric", choices=["sentiment"])
    export.add_argument("--n", type=int, default=50)
    export.add_argument("--seed", type=int, default=0)
    export.set_defaults(func=cmd_export)
    imp = sub.add_parser("import")
    imp.add_argument("path")
    imp.set_defaults(func=cmd_import)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
