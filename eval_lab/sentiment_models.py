"""One interface over every sentiment scorer in config/sentiment_models.yaml.

Each scorer takes a list of texts (and, optionally, one target company per text)
and returns a frame with one row per text:

    label        negative | neutral | positive
    score        -1 (most negative) .. +1 (most positive)
    p_negative, p_neutral, p_positive   class probabilities (NaN for word-list scorers)

Only the zero-shot scorer uses targets: it asks whether the text is good or bad
news *for that company*, which is what separates "Nvidia slides as Google gains"
into one negative and one positive mention.
"""
import time
from functools import cache

import numpy as np
import pandas as pd

from eval_lab.db import load_config
from eval_lab.labels import SENTIMENT_CLASSES, vader_class

PROBS = [f"p_{c}" for c in SENTIMENT_CLASSES]


def config() -> dict:
    return load_config("sentiment_models.yaml")


def device() -> str:
    import torch

    wanted = config()["device"]
    if wanted != "auto":
        return wanted
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def half_precision() -> bool:
    """fp16 roughly doubles throughput on the Apple GPU / CUDA; on CPU it's slower."""
    return config().get("half_precision", True) and device() in ("mps", "cuda")


def length_sorted_batches(texts: list, size: int):
    """Yield (positions, texts) batches of similar length, so padding wastes little work.
    Callers put results back in the original order using the positions."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    for i in range(0, len(order), size):
        idx = order[i:i + size]
        yield idx, [texts[j] for j in idx]


def _frame(labels, scores, probs=None) -> pd.DataFrame:
    out = pd.DataFrame({"label": list(labels), "score": np.asarray(scores, dtype=float)})
    for i, col in enumerate(PROBS):
        out[col] = probs[:, i] if probs is not None else np.nan
    return out


def _from_probs(probs: np.ndarray) -> pd.DataFrame:
    """probs columns are ordered negative, neutral, positive."""
    return _frame(np.array(SENTIMENT_CLASSES)[probs.argmax(axis=1)], probs[:, 2] - probs[:, 0], probs)


class Vader:
    def __init__(self, spec: dict):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()

    def score(self, texts, targets=None) -> pd.DataFrame:
        compound = [self.analyzer.polarity_scores(t)["compound"] for t in texts]
        return _frame([vader_class(c) for c in compound], compound)


class LoughranMcDonald:
    """Counts finance-specific positive and negative words. Polarity is
    (pos - neg) / (pos + neg); a text with no sentiment words is neutral."""

    def __init__(self, spec: dict):
        import pysentiment2

        self.lm = pysentiment2.LM()

    def score(self, texts, targets=None) -> pd.DataFrame:
        labels, scores = [], []
        for text in texts:
            s = self.lm.get_score(self.lm.tokenize(text))
            pos, neg = int(s["Positive"]), int(s["Negative"])
            scores.append((pos - neg) / (pos + neg) if pos + neg else 0.0)
            labels.append("positive" if pos > neg else "negative" if neg > pos else "neutral")
        return _frame(labels, scores)


class Classifier:
    """A fine-tuned sequence classifier from Hugging Face."""

    def __init__(self, spec: dict):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(spec["model"])
            model = AutoModelForSequenceClassification.from_pretrained(spec["model"])
        except ValueError:  # finbert-tone's config has no model_type, so load it as the BERT it is
            from transformers import BertForSequenceClassification, BertTokenizerFast

            self.tokenizer = BertTokenizerFast.from_pretrained(spec["model"])
            model = BertForSequenceClassification.from_pretrained(spec["model"])
        self.model = (model.half() if half_precision() else model).to(device()).eval()
        id2label = spec.get("labels") or model.config.id2label
        names = [str(id2label[i]).lower() for i in range(len(id2label))]
        missing = set(SENTIMENT_CLASSES) - set(names)
        if missing:
            raise ValueError(f"{spec['model']} has labels {names}; set `labels` in sentiment_models.yaml")
        self.order = [names.index(c) for c in SENTIMENT_CLASSES]

    def score(self, texts, targets=None) -> pd.DataFrame:
        cfg = config()
        probs = np.zeros((len(texts), 3))
        with self.torch.inference_mode():
            for idx, chunk in length_sorted_batches(list(texts), cfg["batch_size"]):
                batch = self.tokenizer(chunk, padding=True, truncation=True,
                                       max_length=cfg["max_tokens"], return_tensors="pt").to(self.model.device)
                logits = self.model(**batch).logits.float()
                probs[idx] = logits.softmax(-1)[:, self.order].cpu().numpy()
        return _from_probs(probs)


class ZeroShot:
    """Natural-language-inference model asked, for each class, whether the text
    entails a hypothesis like "This is good news for Apple." The three entailment
    probabilities are normalized into class probabilities."""

    def __init__(self, spec: dict):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model"])
        model = AutoModelForSequenceClassification.from_pretrained(spec["model"])
        self.model = (model.half() if half_precision() else model).to(device()).eval()
        self.entail = {v.lower(): k for k, v in self.model.config.id2label.items()}["entailment"]

    def score(self, texts, targets=None) -> pd.DataFrame:
        cfg = config()
        targets = targets if targets is not None else [None] * len(texts)
        pairs = [
            (text, self.spec["hypotheses"][c].format(target=target or self.spec["default_target"]))
            for text, target in zip(texts, targets)
            for c in SENTIMENT_CLASSES
        ]
        entail = np.zeros(len(pairs))
        with self.torch.inference_mode():
            for idx, chunk in length_sorted_batches([p + h for p, h in pairs], cfg["batch_size"]):
                chunk = [pairs[j] for j in idx]
                batch = self.tokenizer([p for p, _ in chunk], [h for _, h in chunk], padding=True, truncation="only_first",
                                       max_length=cfg["max_tokens"], return_tensors="pt").to(self.model.device)
                entail[idx] = self.model(**batch).logits.float().softmax(-1)[:, self.entail].cpu().numpy()
        p = entail.reshape(-1, 3)
        return _from_probs(p / p.sum(axis=1, keepdims=True))


KINDS = {"vader": Vader, "lm_dictionary": LoughranMcDonald, "classifier": Classifier, "zero_shot": ZeroShot}


@cache
def load(name: str):
    spec = config()["scorers"][name]
    return KINDS[spec["kind"]](spec)


def score(name: str, texts, targets=None) -> tuple[pd.DataFrame, float]:
    """Score texts with a named scorer. Returns (scores, texts per second)."""
    scorer = load(name)
    started = time.perf_counter()
    out = scorer.score(list(texts), None if targets is None else list(targets))
    elapsed = time.perf_counter() - started
    return out, len(texts) / elapsed if elapsed else np.nan
