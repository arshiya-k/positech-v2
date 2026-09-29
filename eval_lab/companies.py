"""Recognizing the universe's companies in text."""
import re
from functools import cache

from eval_lab.db import load_config


@cache
def universe() -> dict:
    return {c["ticker"]: c for c in load_config("universe.yaml")["companies"]}


def display_name(ticker: str) -> str:
    return universe()[ticker]["names"][0]


@cache
def pattern(ticker: str) -> re.Pattern:
    """Matches the company's names as whole words, its cashtag ($AAPL), and, for
    tickers of three or more letters, the bare ticker in capitals. One- and
    two-letter tickers (V, SO, DE) only match as cashtags; bare, they'd match
    ordinary words."""
    names = "|".join(re.escape(n) for n in universe()[ticker]["names"])
    ticker_part = rf"\${ticker}\b" + (rf"|\b{ticker}\b" if len(ticker) >= 3 else "")
    return re.compile(rf"(?i:(?<![\w&])(?:{names})(?![\w&]))|{ticker_part}")


def mentions(text: str, ticker: str) -> bool:
    return bool(pattern(ticker).search(text or ""))


def link(span: str) -> str | None:
    """The ticker a recognized entity refers to, if it's one of ours. The entity
    only has to contain the name: models often return "Alphabet Stock" or
    "Apple Inc." rather than the bare name."""
    for ticker in universe():
        if pattern(ticker).search(span):
            return ticker
    return None


def cashtags(text: str) -> list[str]:
    """Tickers written as cashtags ($AAPL), which are unambiguous without context."""
    return [t for t in universe() if re.search(rf"\${re.escape(t)}\b", text or "")]
