"""Text helpers."""

from __future__ import annotations

import unicodedata


def ascii_label(text: str) -> str:
    """Closest ASCII rendering of ``text``.

    OpenCV's built-in fonts cannot draw accented letters, so "Renée" is drawn as
    "Renee" on the preview window. Characters with no ASCII base become "?".
    """
    decomposed = unicodedata.normalize("NFKD", text)
    kept = []
    for char in decomposed:
        if unicodedata.combining(char):
            continue
        kept.append(char if char.isascii() and char.isprintable() else "?")
    return "".join(kept)
