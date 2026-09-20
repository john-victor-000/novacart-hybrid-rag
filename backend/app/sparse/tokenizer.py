"""Lightweight tokenization for sparse retrieval."""

import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DASH_PATTERN = re.compile(r"[‐‑‒–—−]")


def tokenize(text: str) -> list[str]:
    """Normalize text while preserving complete hyphenated identifiers."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = DASH_PATTERN.sub("-", normalized)

    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(normalized):
        tokens.append(token)
        if "-" in token:
            tokens.extend(part for part in token.split("-") if part)
    return tokens
