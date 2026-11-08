"""Keyword analysis: how ``index/bm25`` and ``index/postgres-fts`` read text.

A text becomes **terms** in four steps:

1. **Words**: NFKC-normalized, case-folded runs of Unicode letters and digits. An apostrophe
   inside a word keeps it whole (``don't``, ``guest's``), so contractions meet the stop list
   and the stemmer sees possessives; any other punctuation, a hyphen included, separates words.
2. **Stop words** go: the Snowball project's English stop list (snowballstem.org, BSD-3-Clause),
   pronouns, articles, auxiliaries, their contractions and the commonest prepositions.
3. **Stems**: the Snowball English stemmer (Porter2, ``snowballstemmer``), which also drops a
   possessive ``'s``.
4. What's left, in order, repeats included: a document's term counts, or a query's terms.

``bm25_idf`` and the ``BM25_*`` constants are Okapi BM25 as Robertson and Zaragoza give it ("The
Probabilistic Relevance Framework: BM25 and Beyond", 2009), with the inverse document frequency
in the form that never goes negative, ``ln(1 + (N - n + 0.5) / (n + 0.5))``.

``query_words`` is for engines that stem for themselves, such as Postgres's ``english``
configuration: a query's distinct words without stop words, split at apostrophes so each is a
plain run of letters and digits, safe in a ``to_tsquery`` expression.
"""

import math
import re
import unicodedata
from functools import lru_cache

import snowballstemmer

ANALYZER = "unicode-snowball-english-v1"

# Okapi BM25's usual parameters: term-frequency saturation and length normalization.
BM25_K1 = 1.2
BM25_B = 0.75

# A word: letters and digits in any script, with apostrophes only between them.
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
# Apostrophes as typed or typeset (U+2019, U+2018, U+02BC, U+FF07) read as the ASCII one.
_APOSTROPHES = {0x2019: "'", 0x2018: "'", 0x02BC: "'", 0xFF07: "'"}

# The Snowball project's English stop list (snowballstem.org/algorithms/english/stop.txt), the
# words it lists rather than comments out. "us", "will", "can", "may", "might", "must" and
# "shall" aren't stop words there: each is also a common noun or verb.
STOP_WORDS = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
    she her hers herself it its itself they them their theirs themselves what which who whom
    this that these those am is are was were be been being have has had having do does did
    doing would should could ought i'm you're he's she's it's we're they're i've you've we've
    they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't
    aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't
    shouldn't can't cannot couldn't mustn't let's that's who's what's here's there's when's
    where's why's how's a an the and but if or because as until while of at by for with about
    against between into through during before after above below to from up down in out on
    off over under again further then once here there when where why how all any both each
    few more most other some such no nor not only own same so than too very
    """.split()  # noqa: SIM905 - the list reads better as text
)

_stemmer = snowballstemmer.stemmer("english")


def words(text: str) -> list[str]:
    """A text's words, normalized and case-folded, before stop words and stemming."""
    normalized = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).casefold()
    return _WORD_RE.findall(normalized)


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    return _stemmer.stemWord(word)


def terms(text: str) -> list[str]:
    """A text's terms: its words without stop words, stemmed, in order."""
    return [stem(word) for word in words(text) if word not in STOP_WORDS]


def query_words(text: str) -> list[str]:
    """A query's distinct words without stop words, in order, split at apostrophes: for an
    engine that applies its own stemmer and stop list."""
    found: dict[str, None] = {}
    for word in words(text):
        if word in STOP_WORDS:
            continue
        for part in word.split("'"):
            if part:
                found.setdefault(part, None)
    return list(found)


def bm25_idf(documents: int, containing: int) -> float:
    """BM25's inverse document frequency of a term found in ``containing`` of ``documents``."""
    return math.log1p((documents - containing + 0.5) / (containing + 0.5))
