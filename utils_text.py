import re
import unicodedata
from typing import Iterable, List


_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    # Keep original accents for embeddings by default; exposed if needed
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return normalize_whitespace(text)


def safe_join(parts: Iterable[str], sep: str = " ") -> str:
    return sep.join([p for p in parts if p])


# Lightweight French stopword list (clean UTF-8)
FRENCH_STOPWORDS = set(
    [
        "de",
        "la",
        "le",
        "les",
        "des",
        "du",
        "un",
        "une",
        "et",
        "en",
        "à",
        "a",
        "aux",
        "au",
        "pour",
        "par",
        "avec",
        "sans",
        "sur",
        "dans",
        "ou",
        "d'",
        "l'",
        "se",
        "ses",
        "son",
        "sa",
        "leurs",
        "leur",
        "nos",
        "notre",
        "vos",
        "votre",
        "ce",
        "cet",
        "cette",
        "ces",
        "qui",
        "que",
        "qu'",
        "est",
        "être",
        "sont",
    ]
)


# Remove common bullet/formatting noise and collapse whitespace
_PUNCT_RE = re.compile(r"[\t\r\n\-\*•●▪◦·∙]+|\s{2,}")

# Token split: keep letters (incl. accents), digits, apostrophes and hyphens
_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ'’-]+", flags=re.UNICODE)


def extract_keyphrases(
    text: str,
    max_phrases: int = 20,
    min_words: int = 1,
    max_words: int = 5,
) -> List[str]:
    """Lightweight keyphrase extraction for French without external models.

    - Cleans bullets and excess whitespace
    - Tokenizes on non-word boundaries (keeps accents and apostrophes)
    - Drops short/stopword-only phrases
    - Produces multi-word candidates by merging adjacent non-stopword tokens
    """
    if not text:
        return []

    text = normalize_whitespace(_PUNCT_RE.sub(" ", text))
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    if not tokens:
        return []

    phrases: List[str] = []
    buf: List[str] = []
    for tok in tokens:
        low = tok.lower()
        is_stop = low in FRENCH_STOPWORDS or len(low) <= 1
        if is_stop:
            if buf:
                phrases.append(" ".join(buf))
                buf = []
        else:
            buf.append(tok)
            if len(buf) >= max_words:
                phrases.append(" ".join(buf))
                buf = []
    if buf:
        phrases.append(" ".join(buf))

    # Filter by word count and deduplicate preserving order
    def word_count(p: str) -> int:
        return len([w for w in p.split() if w])

    out: List[str] = []
    seen = set()
    for p in phrases:
        wc = word_count(p)
        if wc < min_words or wc > max_words:
            continue
        key = p.lower()
        if key not in seen:
            out.append(p)
            seen.add(key)
        if len(out) >= max_phrases:
            break
    return out

