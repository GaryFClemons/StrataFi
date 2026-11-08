"""Email unpacking (``unpack/email``): .eml and .msg bodies and attachments become
members, with the archive limits."""

from pathlib import Path

import pytest

from stratafi_components.archives import ArchiveLimits, ArchiveRefused
from stratafi_components.detect import EML, MSG, detect_bytes
from stratafi_components.intake import stage
from stratafi_components.messages import extract_eml, extract_msg
from stratafi_testing.documents import Block, docx_bytes, eml_bytes, msg_bytes, zip_bytes

DOCX = docx_bytes("Fees", [Block("paragraph", "A wire costs 25 dollars.")])


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_an_eml_yields_its_body_and_attachments(tmp_path: Path) -> None:
    data = eml_bytes(
        "Callback rule",
        "Confirm every wire request by calling the number on file.",
        {"fees.docx": ("application/octet-stream", DOCX), "notes.txt": ("text/plain", b"Note.")},
    )
    members = list(extract_eml(_write(tmp_path, "m.eml", data), tmp_path / "out"))
    assert [m.name for m in members] == ["message.txt", "fees.docx", "notes.txt"]
    body = members[0].path.read_text()
    assert body.startswith("Subject: Callback rule\nDate: Fri, 30 Oct 2026 09:00:00 +0000\n\n")
    assert "calling the number on file" in body
    assert "policy@example.org" not in body  # addresses stay out
    assert members[1].path.read_bytes() == DOCX
    assert members[1].size == len(DOCX)


def test_an_html_only_eml_yields_an_html_body(tmp_path: Path) -> None:
    data = eml_bytes("Notice", "", html="<html><body><p>Branch hours change.</p></body></html>")
    members = list(extract_eml(_write(tmp_path, "m.eml", data), tmp_path / "out"))
    assert [m.name for m in members] == ["message.html"]
    assert b"Branch hours" in members[0].path.read_bytes()


def test_an_msg_yields_its_body_and_attachments(tmp_path: Path) -> None:
    data = msg_bytes("Wire policy", "Call back first.", {"fees.docx": DOCX, "../x/n.txt": b"n"})
    members = list(extract_msg(_write(tmp_path, "m.msg", data), tmp_path / "out"))
    assert [m.name for m in members] == ["message.txt", "fees.docx", "x/n.txt"]
    assert members[0].path.read_text() == "Subject: Wire policy\n\nCall back first.\n"
    assert members[1].path.read_bytes() == DOCX


def test_detection_knows_both_message_types(tmp_path: Path) -> None:
    eml = detect_bytes(eml_bytes("S", "Body."), "note.eml", tmp_path)
    assert (eml.media_type, eml.is_message) == (EML, True)
    msg = detect_bytes(msg_bytes("S", "Body."), "note.msg", tmp_path)
    assert (msg.media_type, msg.kind, msg.is_message) == (MSG, "ole", True)
    # A .msg whose bytes aren't a compound file is a mismatch, and isn't unpacked.
    fake = detect_bytes(b"Subject: hi\n\nplain", "note.msg", tmp_path)
    assert fake.mismatch
    assert not fake.is_message


def test_an_attached_archive_refuses_the_message(tmp_path: Path) -> None:
    data = eml_bytes("Bundle", "See attached.", {"all.zip": ("application/zip", zip_bytes({}))})
    with pytest.raises(ArchiveRefused) as refused:
        list(extract_eml(_write(tmp_path, "m.eml", data), tmp_path / "out"))
    assert refused.value.reason == "archive_nested"


def test_message_limits_hold(tmp_path: Path) -> None:
    attachments = {f"a{i}.txt": ("text/plain", b"x" * 100) for i in range(5)}
    path = _write(tmp_path, "m.eml", eml_bytes("Many", "Body.", attachments))
    with pytest.raises(ArchiveRefused) as refused:
        list(extract_eml(path, tmp_path / "a", ArchiveLimits(max_entries=3)))
    assert refused.value.reason == "archive_entry_limit"
    with pytest.raises(ArchiveRefused) as refused:
        list(extract_eml(path, tmp_path / "b", ArchiveLimits(max_expanded_bytes=300)))
    assert refused.value.reason == "archive_expansion_limit"


def test_malformed_messages_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ArchiveRefused) as refused:
        list(extract_msg(_write(tmp_path, "m.msg", b"not ole"), tmp_path / "out"))
    assert refused.value.reason == "message_malformed"


def test_intake_stages_a_message_as_a_tree(tmp_path: Path) -> None:
    data = eml_bytes("Fees", "Body text.", {"fees.docx": ("application/octet-stream", DOCX)})
    path = _write(tmp_path, "inbox.eml", data)
    staged = stage(path, "mail/inbox.eml", sha256="0" * 64, size=len(data), scratch=tmp_path)
    assert [member.name for member in staged.members] == [
        "mail/inbox.eml/message.txt",
        "mail/inbox.eml/fees.docx",
    ]
    assert staged.members[1].detection.media_type.endswith("wordprocessingml.document")
    assert staged.members[1].member_path == "fees.docx"
    off = stage(
        path, "inbox.eml", sha256="0" * 64, size=len(data), scratch=tmp_path, unpack_email=False
    )
    assert off.members == []
