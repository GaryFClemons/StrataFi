"""Grounded chat: the prompt that turns retrieved passages into a cited answer.

``build_messages`` numbers the passages from 1 and asks the model to cite them in square
brackets; ``citations_used`` reads those numbers back out of an answer. Without passages (a chat
with no knowledge source, for comparing models alone) the conversation goes to the model as is,
under a plain system prompt.

The prompt's version is part of every answer's record (``PROMPT_VERSION``): change the wording
and bump it, so answers from two prompts are never compared as if they were one.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

PROMPT_VERSION = "grounded-chat-v1"

GROUNDED_SYSTEM = (
    "You are a careful assistant that answers questions using only the numbered sources the "
    "user provides.\n"
    "- Support every fact with the number of its source in square brackets, like [1] or "
    "[2][3]. Never cite a number that isn't a source.\n"
    "- If the sources don't answer the question, say you couldn't find it in the documents. "
    "Never guess or use outside knowledge.\n"
    "- Quote numbers, limits, dates and names exactly as the sources state them.\n"
    "- Be concise: answer first, then add only the details that matter."
)
PLAIN_SYSTEM = "You are a helpful, concise assistant."

# The most history the model sees, besides the question: older turns are dropped first.
MAX_HISTORY_MESSAGES = 8
MAX_PASSAGE_CHARS = 2400

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


@dataclass(frozen=True)
class Passage:
    """One retrieved passage as the prompt shows it."""

    text: str
    title: str | None = None
    heading_path: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        parts = [part for part in (self.title, " > ".join(self.heading_path)) if part]
        return " — ".join(parts)


def format_sources(passages: Sequence[Passage]) -> str:
    blocks = []
    for number, passage in enumerate(passages, start=1):
        text = passage.text.strip()
        if len(text) > MAX_PASSAGE_CHARS:
            text = text[:MAX_PASSAGE_CHARS].rstrip() + " …"
        header = f"[{number}] {passage.label}".rstrip()
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_messages(
    conversation: Sequence[tuple[str, str]],
    passages: Sequence[Passage] | None,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """The model's messages for ``conversation`` (``(role, content)`` pairs, the last one the
    user's question). With passages, the question is sent with them, numbered; with
    ``passages=None`` the chat has no knowledge source."""
    if not conversation or conversation[-1][0] != "user":
        raise ValueError("the conversation must end with the user's question")
    grounded = passages is not None
    system = system_prompt or (GROUNDED_SYSTEM if grounded else PLAIN_SYSTEM)
    *history, (_, question) = conversation
    messages = [{"role": "system", "content": system}]
    messages += [
        {"role": role, "content": content} for role, content in history[-MAX_HISTORY_MESSAGES:]
    ]
    if grounded:
        sources = format_sources(passages) if passages else "(no sources were found)"
        question = f"Sources:\n\n{sources}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": question})
    return messages


def citations_used(answer: str, passages: int) -> list[int]:
    """The source numbers an answer cites, in order of first use, ignoring numbers that name no
    passage."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= passages and number not in seen:
            seen.append(number)
    return seen
