import re
import numpy as np


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Light cleaning for a student project.
    I tried aggressive stemming early on, but it removed too much context
    for short reviews, so I kept this minimal pass.
    """
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s!?]", " ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def extra_features(texts, ratings):
    """
    Handcrafted signals inspired by review spam literature.
    Keeping these simple makes it easy to explain in viva.
    """
    feats = []
    for text, rating in zip(texts, ratings):
        length = len(text)
        word_count = len(text.split())
        exclamations = text.count("!")
        uppercase_ratio = (
            sum(1 for c in text if c.isupper()) / max(1, len(text))
        )
        feats.append([length, word_count, exclamations, uppercase_ratio, rating])
    return np.array(feats, dtype=np.float32)


def extra_feature_transformer(df):
    """
    Kept as a top-level function so the trained pipeline is pickle-safe.
    sklearn can struggle to reload local functions.
    """
    return extra_features(df["clean_text"].tolist(), df["rating"].tolist())
