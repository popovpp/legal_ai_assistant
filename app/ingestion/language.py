import re

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str, threshold: float = 0.3) -> str:
    """Return 'ru' if Cyrillic share is above threshold, else 'en'."""
    sample = text[:5000]
    total_letters = len(_CYRILLIC.findall(sample)) + len(_LATIN.findall(sample))
    if total_letters == 0:
        return "en"
    cyrillic_share = len(_CYRILLIC.findall(sample)) / total_letters
    return "ru" if cyrillic_share >= threshold else "en"
