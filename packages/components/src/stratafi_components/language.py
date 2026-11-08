"""Language identification by function words, for ``clean/language``.

Each language's most frequent function words (articles, pronouns, prepositions, auxiliaries)
make up a large share of any running text in it, and few of them are shared across languages.
``identify`` counts the share of a text's words that are function words of each language and
picks the highest, when it is clear enough. It is deterministic, needs no model or download, and
is good at what cleaning needs: telling a paragraph's language, not a phrase's. Texts shorter
than ``min_words`` are left unknown (``None``), so headings and labels are never judged.

Languages: English, Spanish, French, German, Portuguese, Italian and Dutch.
"""

import re
from collections import Counter

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _vocabulary(*lines: str) -> frozenset[str]:
    return frozenset(word for line in lines for word in line.split(" ") if word)


FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": _vocabulary(
        "the of and to a in is that for it as was with be by on not he this are or his from",
        "at which but have an they you were their has been one we its all would there will",
        "can if who our more any may must should shall your these",
    ),
    "es": _vocabulary(
        "de la que el en y a los se del las un por con no una su para es al lo como más pero",
        "sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay",
        "donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos",
        "e esto",
    ),
    "fr": _vocabulary(
        "de la le et les des en un du une que est pour qui dans par plus pas au sur ne se ce",
        "il sont avec son aux ou elle nous vous leur cette mais ont été être sa ses où",
    ),
    "de": _vocabulary(
        "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als",
        "auch es an werden aus er hat dass sie nach wird bei einer um am sind noch wie einem",
        "über einen so zum war haben nur oder aber vor zur bis mehr durch man",
    ),
    "pt": _vocabulary(
        "de a o que e do da em um para é com não uma os no se na por mais as dos como mas foi",
        "ao ele das tem à seu sua ou ser quando muito há nos já está eu também só pelo pela",
        "até isso ela entre era depois sem mesmo aos ter seus quem nas me esse eles",
    ),
    "it": _vocabulary(
        "di e il la che in a per un è del non sono una le si con da lo al gli dei ha come ma",
        "anche nel più alla della delle questo essere ci se sua suo loro tra cui dal degli",
        "molto quando io",
    ),
    "nl": _vocabulary(
        "de en van het een in is dat op te zijn met voor niet aan er die maar om ook als dan",
        "bij nog wordt door naar uit tot of over kan worden deze hun wel geen meer zo",
    ),
}
LANGUAGES = tuple(FUNCTION_WORDS)


def words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


def scores(text: str) -> dict[str, float]:
    """Each language's share of the text's words that are its function words."""
    found = words(text)
    if not found:
        return dict.fromkeys(LANGUAGES, 0.0)
    counts = Counter(found)
    return {
        language: sum(count for word, count in counts.items() if word in vocabulary) / len(found)
        for language, vocabulary in FUNCTION_WORDS.items()
    }


def identify(
    text: str, *, min_words: int = 8, min_share: float = 0.12, margin: float = 0.04
) -> str | None:
    """The text's language, or None when it's too short or no language is clearly ahead."""
    if len(words(text)) < min_words:
        return None
    ranked = sorted(scores(text).items(), key=lambda pair: (-pair[1], pair[0]))
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < min_share or top - second < margin:
        return None
    return best
