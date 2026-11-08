"""Relevance: mapping gold evidence onto a run's chunks.

Gold labels point at documents and evidence text, never at chunks, so every run maps them onto
its own chunks. Chunk and evidence text are normalized the same way first (``normalize``,
``tokens``): NFKC (which turns ligatures such as ``ﬁ`` into ``fi``), soft hyphens dropped, words
split across a hyphenated line break joined, case-folded, whitespace collapsed, and punctuation
stripped at the edges.

Three match rules are reported side by side (``RULES``):

- ``knowledge-hit-v1``: the retrieval YAML format's own rule. A chunk from an expected
  ``doc_key`` and version is a hit when its section is the expected section, or its text
  contains the evidence phrase (both compared normalized).
- ``evidence-coverage-v1``: a grade in [0, 1], the share of the evidence's tokens found in order
  in the chunk (their longest common subsequence), within a window of about twice the evidence's
  length, so words scattered across a long chunk don't add up. Tolerant of parser whitespace,
  hyphenation and ligatures. A chunk of the expected document version covers the evidence at a
  grade of at least ``threshold`` (0.7); the grade feeds nDCG. Evidence with no phrase falls
  back to its section.
- ``doc-v1``: any chunk from an expected document version.

**Findability** (``findability``) asks the same of a whole parsed or cleaned document: can its
evidence be found at all? Evidence that can't means the parser or a cleaning rule lost it, so
findability measures parse quality with no extra labels.

The LCS uses the bit-parallel algorithm of Allison and Dix as Hyyrö gives it ("Bit-parallel LCS
length computation revisited", 2004): one machine-word step per text token. Nothing here logs.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from stratafi_core.evalsets import EvidenceRef

KNOWLEDGE_HIT = "knowledge-hit-v1"
EVIDENCE_COVERAGE = "evidence-coverage-v1"
DOC = "doc-v1"
RULES = (KNOWLEDGE_HIT, EVIDENCE_COVERAGE, DOC)
DEFAULT_THRESHOLD = 0.7

# A word split across a line break with a hyphen ("ac-\ncount"), in any of the hyphens parsers
# leave; and a raw token that runs across such breaks.
_HYPHENS = "-\u2010\u2011"  # hyphen-minus, hyphen, non-breaking hyphen
_BREAK_RE = re.compile(rf"(\w)[{_HYPHENS}][ \t]*\r?\n[ \t]*(\w)")
_RAW_TOKEN_RE = re.compile(rf"(?:\S*\w[{_HYPHENS}][ \t]*\r?\n[ \t]*)*\S+")
_EDGES_RE = re.compile(r"^[\W_]+|[\W_]+$")
SOFT_HYPHEN = "­"


def normalize(text: str) -> str:
    """NFKC, joined hyphenated breaks, case-folded, whitespace collapsed, edge punctuation
    stripped: for comparing phrases and sections."""
    text = unicodedata.normalize("NFKC", text).replace(SOFT_HYPHEN, "")
    text = _BREAK_RE.sub(r"\1\2", text)
    return _EDGES_RE.sub("", " ".join(text.casefold().split()))


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """Normalized tokens with the character span each came from in ``text``."""
    spans = []
    for match in _RAW_TOKEN_RE.finditer(text):
        for part in normalize(match.group()).split():
            token = _EDGES_RE.sub("", part)
            if token:
                spans.append((token, match.start(), match.end()))
    return spans


def tokens(text: str) -> list[str]:
    return [token for token, _, _ in token_spans(text)]


def lcs_length(pattern: Sequence[str], text: Sequence[str]) -> int:
    """The length of the longest common subsequence of two token sequences."""
    if not pattern or not text:
        return 0
    masks: dict[str, int] = {}
    for index, token in enumerate(pattern):
        masks[token] = masks.get(token, 0) | (1 << index)
    full = (1 << len(pattern)) - 1
    row = full
    for token in text:
        matched = row & masks.get(token, 0)
        row = ((row + matched) | (row - matched)) & full
    return len(pattern) - row.bit_count()


def window_size(length: int) -> int:
    """How many text tokens an evidence of ``length`` tokens may spread over."""
    return max(2 * length, length + 8)


def _windows(length: int, text_length: int) -> range:
    """Window starts: every half evidence length, the last one reaching the text's end."""
    stride = max(1, length // 2)
    last = max(0, text_length - window_size(length))
    return range(0, last + stride, stride)


def best_window(evidence: Sequence[str], text: Sequence[str]) -> tuple[float, int, int]:
    """The best coverage of ``evidence`` in any window of ``text``, and that window's token
    range. Windows overlap by half the evidence's length, so a match spread over up to 1.5
    times its length always fits in one."""
    if not evidence or not text:
        return 0.0, 0, 0
    bound = sum((Counter(evidence) & Counter(text)).values())
    if bound == 0:
        return 0.0, 0, 0
    size = window_size(len(evidence))
    best, best_start = 0, 0
    for start in _windows(len(evidence), len(text)):
        found = lcs_length(evidence, text[start : start + size])
        if found > best:
            best, best_start = found, start
            if best == len(evidence) or best == bound:
                break
    return best / len(evidence), best_start, min(best_start + size, len(text))


def coverage(evidence: Sequence[str], text: Sequence[str]) -> float:
    """``evidence-coverage-v1``'s grade: the share of the evidence's tokens found in order in
    one window of the text."""
    return best_window(evidence, text)[0]


def highlights(evidence: str, text: str) -> list[tuple[int, int]]:
    """Character spans in ``text`` where the evidence's tokens were matched: the matched tokens
    of the best window, for showing a passage with its evidence marked."""
    wanted = tokens(evidence)
    spans = token_spans(text)
    grade, start, end = best_window(wanted, [token for token, _, _ in spans])
    if grade == 0:
        return []
    window = spans[start:end]
    # Walk the LCS back with a small table: the window is at most about twice the evidence.
    rows, cols = len(wanted), len(window)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            table[i][j] = (
                table[i + 1][j + 1] + 1
                if wanted[i] == window[j][0]
                else max(table[i + 1][j], table[i][j + 1])
            )
    marked: list[tuple[int, int]] = []
    i = j = 0
    while i < rows and j < cols:
        if wanted[i] == window[j][0]:
            span = (window[j][1], window[j][2])
            if marked and marked[-1][1] >= span[0]:
                marked[-1] = (marked[-1][0], max(marked[-1][1], span[1]))
            elif not marked or marked[-1] != span:
                marked.append(span)
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return marked


# --- passages and judgements ----------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk, as relevance sees it: its document version, section and text."""

    chunk_id: str
    doc_key: str | None
    version: str | None
    section: str | None
    text: str


def same_version(passage: Passage, ref: EvidenceRef) -> bool:
    return passage.doc_key == ref.doc_key and (
        ref.version is None or passage.version == ref.version
    )


@dataclass(frozen=True)
class Judgement:
    """What one passage covers of one item's evidence, by each rule: the evidence items
    covered, and (``evidence-coverage-v1``) each one's grade."""

    covered: dict[str, frozenset[int]]
    grades: tuple[float, ...]

    def relevant(self, rule: str) -> bool:
        return bool(self.covered[rule])


@dataclass
class Matcher:
    """Maps one item's expected evidence onto passages, by every rule. Judgements are kept per
    chunk, since a chunk recurs across checkpoints."""

    expected: Sequence[EvidenceRef]
    threshold: float = DEFAULT_THRESHOLD
    _phrases: list[str | None] = field(init=False)
    _sections: list[str | None] = field(init=False)
    _tokens: list[list[str]] = field(init=False)
    _judged: dict[str, Judgement] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._phrases = [normalize(ref.evidence) if ref.evidence else None for ref in self.expected]
        self._sections = [normalize(ref.section) if ref.section else None for ref in self.expected]
        self._tokens = [tokens(ref.evidence) if ref.evidence else [] for ref in self.expected]

    def judge(self, passage: Passage) -> Judgement:
        if passage.chunk_id in self._judged:
            return self._judged[passage.chunk_id]
        hits, covered, documents, grades = set(), set(), set(), []
        text = normalize(passage.text)
        section = normalize(passage.section) if passage.section else None
        text_tokens: list[str] | None = None
        for index, ref in enumerate(self.expected):
            if not same_version(passage, ref):
                grades.append(0.0)
                continue
            documents.add(index)
            section_match = section is not None and section == self._sections[index]
            phrase = self._phrases[index]
            if section_match or (phrase is not None and phrase in text):
                hits.add(index)
            if self._tokens[index]:
                if text_tokens is None:
                    text_tokens = tokens(passage.text)
                grade = coverage(self._tokens[index], text_tokens)
            else:
                grade = 1.0 if section_match else 0.0
            grades.append(grade)
            if grade >= self.threshold:
                covered.add(index)
        judgement = Judgement(
            {
                KNOWLEDGE_HIT: frozenset(hits),
                EVIDENCE_COVERAGE: frozenset(covered),
                DOC: frozenset(documents),
            },
            tuple(grades),
        )
        self._judged[passage.chunk_id] = judgement
        return judgement


# --- findability --------------------------------------------------------------------------


@dataclass(frozen=True)
class Findability:
    findable: bool
    grade: float


def findability(
    ref: EvidenceRef, document_text: str, *, threshold: float = DEFAULT_THRESHOLD
) -> Findability:
    """Whether evidence survives in its document's parsed or cleaned text: the phrase (or, with
    none, the section heading) contained once normalized, or covered at ``threshold``."""
    wanted = ref.evidence or ref.section
    if not wanted:
        return Findability(True, 1.0)
    if normalize(wanted) in normalize(document_text):
        return Findability(True, 1.0)
    grade = coverage(tokens(wanted), tokens(document_text))
    return Findability(grade >= threshold, round(grade, 6))


def document_findability(
    expected: Sequence[EvidenceRef],
    texts: Mapping[tuple[str, str | None], str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Findability | None]:
    """Each evidence item's findability in its document's text, by (doc_key, version); None
    where the run has no text for that document."""
    found: list[Findability | None] = []
    for ref in expected:
        text = texts.get((ref.doc_key, ref.version))
        if text is None and ref.version is None:
            text = next((t for (key, _), t in texts.items() if key == ref.doc_key), None)
        found.append(findability(ref, text, threshold=threshold) if text is not None else None)
    return found
