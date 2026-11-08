"""Check the Halcyon Coffee demo corpus's eval set against its corpus.

    uv run --no-sync python fixtures/demo-corpus/check_eval_set.py

1. ``retrieval.yaml`` loads with ``stratafi_core.evalsets.load_eval_set`` and resolves with
   ``resolve_eval_set``, and every asker comes from the set's own ``askers`` table.
2. ``corpus/manifest.yaml`` loads with ``stratafi_components.manifest.load_corpus_manifest``.
   Every file it lists exists, every Markdown file in ``corpus/`` is listed, and every expected,
   near_misses, avoid and must_not_return reference names a document (and version) in it, with
   that document's ``source_uri``.
3. Every ``section`` is a heading of its document, as its Markdown has it (so it's the section
   label the document's chunks carry), and every ``evidence`` phrase is a verbatim substring, at
   most 100 characters long, of that section's own text (up to the next heading).
4. ACL and dates, as the retriever's filter sees them on ``as_of``: each asker can read every
   document it expects or is pointed to as a near miss (so no staff question expects a
   managers- or executives-only document), and those versions are in effect. Every ``avoid``
   names a version that isn't in effect, and every ``must_not_return`` names a document the
   asker can't see that day.
5. Each item's type, the prefix of its ID, follows its rules, and the counts by type are printed.

It exits with 1 and a list of problems, or 0 when the set is clean. Importing it does nothing;
``main`` does the work.
"""

import argparse
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from stratafi_components.manifest import ManifestDocument, load_corpus_manifest
from stratafi_components.retrieval import EntryMeta, visible_until
from stratafi_core.evalsets import (
    Asker,
    DocRef,
    EvalItem,
    EvalSet,
    EvidenceRef,
    asker_of,
    load_eval_set,
    resolve_eval_set,
)
from stratafi_core.specs import SpecInvalid

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
MANIFEST = CORPUS / "manifest.yaml"
EVAL_SET = HERE / "retrieval.yaml"
SET_NAME = "halcyon-retrieval"
MAX_EVIDENCE = 100
TYPES = ("lookup", "scenario", "table", "version", "acl", "abstain")
ASKER_ORDER = ("staff", "manager", "exec")
COPY_RE = re.compile(r"^(?P<base>.+)--as-(?P<asker>[a-z][a-z0-9_-]*)$")
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")

Documents = Mapping[tuple[str, str], "Document"]


# --- the corpus ---------------------------------------------------------------------------


@dataclass
class Document:
    entry: ManifestDocument
    end: date | None
    sections: dict[str, str] = field(default_factory=dict)

    def in_effect(self, day: date) -> bool:
        """Visible on ``day`` by status and dates, as the retriever's ``current_only`` sees it."""
        return (
            self.entry.status == "published"
            and self.entry.effective_from <= day
            and (self.end is None or day < self.end)
        )

    def readable_by(self, asker: Asker) -> bool:
        acl = self.entry.acl
        if asker.user is not None and asker.user in acl.users:
            return True
        return not set(acl.groups).isdisjoint(asker.groups)


def split_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Each heading's own text, as the Markdown has it: its lines up to the next heading of any
    level, so a table row keeps its pipes. The level-1 title isn't a section. Also the headings
    that appear more than once."""
    parts: dict[str, list[str]] = {}
    repeated: list[str] = []
    current: str | None = None
    for line in text.replace("\r\n", "\n").split("\n"):
        heading = HEADING_RE.match(line)
        if heading is None:
            if current is not None:
                parts[current].append(line)
            continue
        current = heading.group(2) if len(heading.group(1)) > 1 else None
        if current is not None:
            if current in parts:
                repeated.append(current)
            parts.setdefault(current, [])
    return {heading: "\n".join(lines).strip() for heading, lines in parts.items()}, repeated


def load_corpus(problems: list[str]) -> dict[tuple[str, str], Document] | None:
    """The manifest's documents by (doc_key, version), each with its sections and the day its
    version stops being in effect (the retriever's version chain, ``visible_until``)."""
    try:
        manifest = load_corpus_manifest(MANIFEST.read_text(encoding="utf-8"))
    except SpecInvalid as error:
        problems += [f"manifest.yaml: {issue.path}: {issue.code}" for issue in error.issues]
        return None
    ends = visible_until(
        EntryMeta.of(
            document_id=entry.path,
            name=entry.path,
            doc_key=entry.doc_key,
            doc_version=entry.version,
            collection=entry.collection,
            effective_from=entry.effective_from,
            effective_to=entry.effective_to,
            metadata={"status": entry.status},
        )
        for entry in manifest.documents
    )

    documents: dict[tuple[str, str], Document] = {}
    for entry in manifest.documents:
        where = f"manifest.yaml: {entry.path}"
        if (entry.doc_key, entry.version) in documents:
            problems.append(f"{where}: {entry.doc_key} v{entry.version} is listed twice")
            continue
        if entry.source_uri != f"corpus/{entry.path}":
            problems.append(f"{where}: source_uri isn't corpus/{entry.path}")
        document = Document(entry, ends.get(entry.path, entry.effective_to))
        path = CORPUS / entry.path
        if path.is_file():
            document.sections, repeated = split_sections(path.read_text(encoding="utf-8"))
            for heading in repeated:
                problems.append(f"{where}: the heading {heading!r} appears more than once")
        else:
            problems.append(f"{where}: the file doesn't exist")
        documents[(entry.doc_key, entry.version)] = document

    listed = {entry.path for entry in manifest.documents}
    for path in sorted(CORPUS.rglob("*.md")):
        relative = path.relative_to(CORPUS).as_posix()
        if relative not in listed:
            problems.append(f"corpus/{relative}: not listed in manifest.yaml")
    return documents


def corpus_words() -> int:
    return sum(len(path.read_text(encoding="utf-8").split()) for path in CORPUS.rglob("*.md"))


# --- the eval set -------------------------------------------------------------------------


def load_set(problems: list[str]) -> tuple[EvalSet, EvalSet, dict[str, int]] | None:
    """The set as written, the set as ``resolve_eval_set`` stores it, and its split counts."""
    try:
        source = load_eval_set(EVAL_SET.read_text(encoding="utf-8"), "yaml", name=SET_NAME)
        resolved, report = resolve_eval_set(source)
    except SpecInvalid as error:
        problems += [f"retrieval.yaml: {issue.path}: {issue.code}" for issue in error.issues]
        return None
    own = set(source.askers)
    if isinstance(source.defaults.asker, str) and source.defaults.asker not in own:
        problems.append("retrieval.yaml: defaults.asker isn't in the set's askers table")
    for item in source.items:
        if isinstance(item.asker, str) and item.asker not in own:
            problems.append(f"{item.id}: the asker {item.asker!r} isn't in the set's askers table")
        elif item.asker is None and source.defaults.asker is None:
            problems.append(f"{item.id}: no asker, and the set has no default")
    if source.as_of is None:
        problems.append("retrieval.yaml: as_of is missing, so dates can't be checked")
    return source, resolved, report.splits


def check_evidence(where: str, ref: EvidenceRef, documents: Documents, problems: list[str]):
    """Checks 2 and 3 for one evidence reference; returns its document, if it has one."""
    if ref.version is None:
        problems.append(f"{where}: {ref.doc_key} names no version")
        return None
    document = documents.get((ref.doc_key, ref.version))
    if document is None:
        problems.append(f"{where}: {ref.doc_key} v{ref.version} isn't in the manifest")
        return None
    if ref.source_uri is None:
        problems.append(f"{where}: no source_uri")
    elif ref.source_uri != document.entry.source_uri:
        problems.append(f"{where}: source_uri isn't the manifest's {document.entry.source_uri}")
    elif not (HERE / ref.source_uri).is_file():
        problems.append(f"{where}: {ref.source_uri} doesn't exist")
    if ref.section is None:
        problems.append(f"{where}: no section")
    elif ref.section not in document.sections:
        problems.append(f"{where}: {ref.section!r} isn't a heading of {document.entry.path}")
    if ref.evidence is None:
        problems.append(f"{where}: no evidence phrase")
    else:
        if len(ref.evidence) > MAX_EVIDENCE:
            problems.append(f"{where}: the evidence is {len(ref.evidence)} characters (max 100)")
        body = document.sections.get(ref.section or "")
        if body is not None and ref.evidence not in body:
            problems.append(f"{where}: the evidence isn't verbatim in section {ref.section!r}")
    return document


def named(ref: DocRef, documents: Documents) -> list[Document]:
    return [
        document
        for (key, version), document in documents.items()
        if key == ref.doc_key and (ref.version is None or version == ref.version)
    ]


def on_a_table_row(ref: EvidenceRef, documents: Documents) -> bool:
    document = documents.get((ref.doc_key, ref.version or ""))
    body = document.sections.get(ref.section or "", "") if document else ""
    return bool(ref.evidence) and any(
        line.lstrip().startswith("|") and ref.evidence in line for line in body.splitlines()
    )


def kind_of(item: EvalItem) -> str:
    return item.id.split("-", 1)[0]


def check_type(
    item: EvalItem,
    asker: Asker,
    documents: Documents,
    items: Mapping[str, EvalItem],
    askers: Mapping[str, Asker],
    problems: list[str],
) -> None:
    """Check 5: what each type of item must have."""
    kind = kind_of(item)
    if kind not in TYPES:
        problems.append(f"{item.id}: the ID doesn't start with a type ({', '.join(TYPES)})")
    elif kind == "lookup" and not item.expected:
        problems.append(f"{item.id}: a lookup expects evidence")
    elif kind == "scenario" and len(item.expected) < 2:
        problems.append(f"{item.id}: a scenario expects 2 or more evidence sections")
    elif kind == "table" and not (
        item.expected and all(on_a_table_row(ref, documents) for ref in item.expected)
    ):
        problems.append(f"{item.id}: a table question's evidence sits on a table row")
    elif kind == "abstain" and (item.expected or not item.near_misses):
        problems.append(f"{item.id}: an abstention expects nothing and lists its near misses")
    elif kind == "version":
        if item.expected:
            if not {ref.doc_key for ref in item.avoid} & {ref.doc_key for ref in item.expected}:
                problems.append(f"{item.id}: a version trap avoids another version it expects")
        elif not any(ref.version for ref in item.must_not_return):
            problems.append(f"{item.id}: a version trap with no answer names a version to keep out")
    elif kind == "acl":
        copy = COPY_RE.match(item.id)
        if copy is None:
            staff = askers.get("staff")
            readable = [
                documents[(ref.doc_key, ref.version or "")].readable_by(staff)
                for ref in item.expected
                if staff is not None and (ref.doc_key, ref.version or "") in documents
            ]
            if not item.expected or any(readable):
                problems.append(f"{item.id}: an ACL question expects documents staff can't read")
            return
        base = items.get(copy.group("base"))
        if base is None:
            problems.append(f"{item.id}: there's no {copy.group('base')} to copy")
        elif base.query != item.query:
            problems.append(f"{item.id}: the query differs from {base.id}'s")
        if copy.group("asker") != asker.name:
            problems.append(f"{item.id}: asked by {asker.name}, not {copy.group('asker')}")
        if item.expected or not item.must_not_return:
            problems.append(f"{item.id}: an ACL copy expects nothing and names must_not_return")
        elif base is not None and {ref.doc_key for ref in item.must_not_return} != {
            ref.doc_key for ref in base.expected
        }:
            problems.append(f"{item.id}: must_not_return isn't {base.id}'s expected documents")


def check_item(
    item: EvalItem,
    day: date,
    documents: Documents,
    items: Mapping[str, EvalItem],
    askers: Mapping[str, Asker],
    problems: list[str],
) -> None:
    """Checks 2 to 5 for one resolved item."""
    asker = asker_of(item)
    who = asker.name or "the asker"
    for field_name in ("expected", "near_misses"):
        for index, ref in enumerate(getattr(item, field_name)):
            where = f"{item.id}: {field_name}[{index}]"
            document = check_evidence(where, ref, documents, problems)
            if document is None:
                continue
            if not document.readable_by(asker):
                groups = ", ".join(document.entry.acl.groups)
                problems.append(f"{where}: {who} can't read {ref.doc_key} (ACL: {groups})")
            if not document.in_effect(day):
                problems.append(f"{where}: {ref.doc_key} v{ref.version} isn't in effect on {day}")

    for index, ref in enumerate(item.avoid):
        where = f"{item.id}: avoid[{index}]"
        found = named(ref, documents)
        if not found:
            problems.append(f"{where}: {ref.doc_key} v{ref.version} isn't in the manifest")
        elif ref.version is None:
            problems.append(f"{where}: name the version to avoid")
        elif any(document.in_effect(day) for document in found):
            problems.append(f"{where}: {ref.doc_key} v{ref.version} is in effect on {day}")

    for index, ref in enumerate(item.must_not_return):
        where = f"{item.id}: must_not_return[{index}]"
        found = named(ref, documents)
        if not found:
            problems.append(f"{where}: {ref.doc_key} isn't in the manifest")
        for document in found:
            if document.readable_by(asker) and document.in_effect(day):
                version = document.entry.version
                problems.append(f"{where}: {who} may see {ref.doc_key} v{version} on {day}")

    check_type(item, asker, documents, items, askers, problems)


# --- the report ---------------------------------------------------------------------------


def _counts(counter: Mapping[str, int], order: Iterable[str] = ()) -> str:
    keys = [key for key in order if key in counter]
    keys += sorted(key for key in counter if key not in keys)
    return ", ".join(f"{key} {counter[key]}" for key in keys)


def check() -> tuple[list[str], list[str]]:
    """Run every check: the problems found, and the summary lines."""
    problems: list[str] = []
    documents = load_corpus(problems)
    loaded = load_set(problems)
    if documents is None or loaded is None or loaded[0].as_of is None:
        return problems, []
    source, resolved, splits = loaded
    day = source.as_of
    items = {item.id: item for item in resolved.items}
    for item in resolved.items:
        check_item(item, day, documents, items, resolved.askers, problems)

    kinds = Counter(kind_of(item) for item in resolved.items)
    unknown = sorted(set(kinds) - set(TYPES))
    askers = Counter(asker_of(item).name or "(unnamed)" for item in resolved.items)
    collections = Counter(document.entry.collection for document in documents.values())
    summary = [
        f"corpus: {len(documents)} documents ({_counts(collections)}), {corpus_words():,} words",
        f"retrieval.yaml: {len(resolved.items)} items, tenant {source.tenant}, as_of {day}, "
        f"k {resolved.defaults.k}",
        f"askers: {_counts(askers, ASKER_ORDER)}",
        f"splits (resolve_eval_set, seed 0): {_counts(splits, ('dev', 'test'))}",
        f"evidence checked: {sum(len(i.expected) for i in resolved.items)} expected, "
        f"{sum(len(i.near_misses) for i in resolved.items)} near misses",
        f"traps: {sum(1 for i in resolved.items if i.avoid)} items with avoid, "
        f"{sum(1 for i in resolved.items if i.must_not_return)} with must_not_return, "
        f"{sum(1 for i in resolved.items if not i.expected)} expecting nothing",
        "by type:",
        *(f"  {kind:<9}{kinds[kind]:>4}" for kind in TYPES),
        *(f"  {kind:<9}{kinds[kind]:>4}  (unknown type)" for kind in unknown),
        f"  {'total':<9}{sum(kinds.values()):>4}",
    ]
    return problems, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the demo corpus's eval set.")
    parser.parse_args(argv)
    problems, summary = check()
    for line in summary:
        print(line)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\nOK: every check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
