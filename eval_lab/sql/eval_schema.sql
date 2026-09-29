-- Evaluation data: benchmark tasks with gold answers, and labels.

CREATE TABLE IF NOT EXISTS qa_tasks (
    task_id         VARCHAR,
    task_type       VARCHAR,   -- factual_lookup | calculation | market_calc | reasoning
    template_id     VARCHAR,
    ticker          VARCHAR,
    sector          VARCHAR,
    cap_tercile     VARCHAR,
    fiscal_year     INTEGER,
    prompt          VARCHAR,
    gold_value      DOUBLE,    -- NULL for reasoning tasks (graded by rubric only)
    unit            VARCHAR,
    tol_type        VARCHAR,   -- rel | abs
    tolerance       DOUBLE,
    alt_values      VARCHAR,   -- JSON: plausible wrong answers, used to classify errors
    plausible_low   DOUBLE,    -- range a production monitor could check without the gold answer
    plausible_high  DOUBLE,
    trap_tags       VARCHAR,   -- comma-separated: fiscal_calendar, split_adjustment, restated, ...
    reference       VARCHAR    -- JSON: the source facts behind the gold answer
);

CREATE TABLE IF NOT EXISTS sentiment_items (
    item_id     VARCHAR,   -- = posts.post_id
    source      VARCHAR,
    ticker      VARCHAR,
    vader_class VARCHAR,   -- negative | neutral | positive
    sampled_at  TIMESTAMP
);

-- One row per (item, labeler, dimension), for human and baseline labelers alike.
CREATE TABLE IF NOT EXISTS labels (
    item_id      VARCHAR,
    item_kind    VARCHAR,   -- sentiment_post
    rubric_id    VARCHAR,
    labeler      VARCHAR,   -- e.g. vader, vader:positech, human:alex
    labeler_type VARCHAR,   -- human | baseline
    dimension    VARCHAR,
    value        VARCHAR,
    flags        VARCHAR,   -- comma-separated rubric flags
    rationale    VARCHAR,
    seconds      DOUBLE,    -- annotation time (humans)
    created_at   TIMESTAMP
);
