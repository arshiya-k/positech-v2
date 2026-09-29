-- Source data: market, fundamentals, earnings, vendor statements and posts.

CREATE TABLE IF NOT EXISTS companies (
    ticker                VARCHAR,
    cik                   VARCHAR,
    name                  VARCHAR,
    sector                VARCHAR,
    fiscal_year_end_month INTEGER,
    fy_convention         VARCHAR,   -- end_year | start_year (see config/universe.yaml)
    in_positech           BOOLEAN
);

-- Annual 10-K values, one row per (ticker, metric, period_end).
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker               VARCHAR,
    metric               VARCHAR,
    fiscal_year          INTEGER,    -- the company's own fiscal-year label
    period_start         DATE,       -- NULL for balance-sheet (instant) metrics
    period_end           DATE,
    value                DOUBLE,     -- latest filed value (restated / split-adjusted)
    first_reported_value DOUBLE,     -- value as originally reported
    n_versions           INTEGER,    -- distinct values across filings; >1 means restated or split-adjusted
    unit                 VARCHAR,
    source_concept       VARCHAR,
    accn                 VARCHAR,
    filed                DATE
);

-- Quarterly values from 10-Qs. Q4 is never filed on its own: for additive
-- metrics it is derived from the 10-K (is_derived = true); Q4 EPS is left out.
CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
    ticker               VARCHAR,
    metric               VARCHAR,
    fiscal_year          INTEGER,
    fiscal_quarter       INTEGER,
    period_start         DATE,
    period_end           DATE,
    value                DOUBLE,
    first_reported_value DOUBLE,
    n_versions           INTEGER,
    unit                 VARCHAR,
    source_concept       VARCHAR,
    accn                 VARCHAR,
    filed                DATE,
    is_derived           BOOLEAN,
    derivation           VARCHAR
);

CREATE TABLE IF NOT EXISTS prices (
    ticker           VARCHAR,
    date             DATE,
    open             DOUBLE,
    high             DOUBLE,
    low              DOUBLE,
    close            DOUBLE,   -- split-adjusted (Yahoo convention)
    adj_close        DOUBLE,   -- split- and dividend-adjusted, use for total return
    close_unadjusted DOUBLE,   -- price as actually traded that day
    volume           BIGINT
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    ticker      VARCHAR,
    date        DATE,
    action_type VARCHAR,   -- split | spinoff_adjustment | dividend
    value       DOUBLE     -- split ratio (4.0 = 4-for-1), Yahoo's spin-off price factor, or cash dividend per share
);

-- Earnings announcements from Yahoo Finance. The EPS here is the "street"
-- figure analysts forecast, usually adjusted (non-GAAP), so it can differ from
-- the GAAP diluted EPS in fundamentals_quarterly.
CREATE TABLE IF NOT EXISTS earnings (
    ticker          VARCHAR,
    announced_at_et TIMESTAMP,   -- New York time
    timing          VARCHAR,     -- before_open | during_market | after_close | unknown
    reaction_date   DATE,        -- first session that could trade on the news
    eps_estimate    DOUBLE,
    eps_reported    DOUBLE,
    surprise_pct    DOUBLE
);

-- Income-statement figures as a data vendor (Yahoo Finance) reports them, for
-- reconciliation against the SEC filings.
CREATE TABLE IF NOT EXISTS vendor_fundamentals (
    ticker     VARCHAR,
    vendor     VARCHAR,
    frequency  VARCHAR,   -- annual | quarterly
    metric     VARCHAR,
    period_end DATE,      -- as the vendor labels it
    value      DOUBLE
);

-- PosiTech's scraped news / Reddit / Twitter posts, normalized.
CREATE TABLE IF NOT EXISTS posts (
    post_id          VARCHAR,
    source           VARCHAR,
    ticker           VARCHAR,
    posted_at        TIMESTAMP,   -- UTC
    text             VARCHAR,
    link             VARCHAR,
    vader_stored     DOUBLE,      -- score saved by the original PosiTech scraper
    vader_recomputed DOUBLE,      -- VADER re-run on the stored text
    is_retweet       BOOLEAN,
    is_duplicate     BOOLEAN,
    n_chars          INTEGER
);
