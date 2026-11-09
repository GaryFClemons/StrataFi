"""Corpus entries as the run planner reads them."""

import uuid

from stratafi_store.corpora import Entry, planner_entries


def entry(name: str, sha256: str = "a" * 64) -> Entry:
    return Entry(
        name=name,
        document_id=uuid.uuid4(),
        sha256=sha256,
        size_bytes=1,
        media_type="text/plain",
        traits={},
    )


def test_the_first_name_is_kept_whatever_order_the_database_returned() -> None:
    """An en_US collation sorts wholesale_copy.txt before wholesale.txt, and a C one after it.
    Either way, the copy is the duplicate."""
    original, copy = entry("wholesale.txt"), entry("wholesale_copy.txt")
    for rows in ([original, copy], [copy, original]):
        assert [(found.name, found.duplicate) for found in planner_entries(rows)] == [
            ("wholesale.txt", False),
            ("wholesale_copy.txt", True),
        ]


def test_different_bytes_are_not_duplicates() -> None:
    rows = [entry("b.txt", "b" * 64), entry("a.txt", "a" * 64)]
    assert [(found.name, found.duplicate) for found in planner_entries(rows)] == [
        ("a.txt", False),
        ("b.txt", False),
    ]
