"""Paths, connections and small DuckDB helpers shared by the eval lab."""
from pathlib import Path

import duckdb
import pandas as pd
import yaml

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
CONFIG = PKG / "config"
POSITECH_DATA = REPO / "data" / "positech"  # scraped posts inherited from the PosiTech project
DATA = REPO / "data" / "lab"
REPORTS = REPO / "reports"
BATCHES = PKG / "annotation" / "batches"

SOURCE_DB = DATA / "source.duckdb"  # market, fundamentals, posts
EVAL_DB = DATA / "eval.duckdb"      # benchmark tasks with gold answers, and labels
NEWS_DB = DATA / "news.duckdb"      # GDELT headlines; its own file so a long fetch doesn't lock the others


def _connect(path: Path, read_only: bool, config: dict | None = None) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only, config=config or {})


def source_db() -> duckdb.DuckDBPyConnection:
    con = _connect(SOURCE_DB, read_only=False)
    con.execute((PKG / "sql" / "source_schema.sql").read_text())
    return con


def news_db(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return _connect(NEWS_DB, read_only=read_only)


def eval_db() -> duckdb.DuckDBPyConnection:
    con = _connect(EVAL_DB, read_only=False)
    con.execute((PKG / "sql" / "eval_schema.sql").read_text())
    return con


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text())


def replace_rows(con, table: str, df: pd.DataFrame, key_col: str, keys=None) -> None:
    """Delete rows whose `key_col` is in `keys` (default: the keys in `df`), then insert `df` by column name."""
    keys = [str(k) for k in (pd.unique(df[key_col]) if keys is None else keys)]
    if keys:
        con.execute(f"DELETE FROM {table} WHERE list_contains(?, {key_col})", [keys])
    if len(df):
        con.register("_incoming", df)
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming")
        con.unregister("_incoming")


def append_rows(con, table: str, df: pd.DataFrame) -> None:
    if len(df):
        con.register("_incoming", df)
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming")
        con.unregister("_incoming")


def report_path(*parts: str) -> Path:
    path = REPORTS.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_report(df: pd.DataFrame, *parts: str) -> Path:
    path = report_path(*parts)
    df.to_csv(path, index=False)
    return path
