"""The canary helper's own tests: a leak anywhere must fail the check."""

import logging
import sys

import pytest

from stratafi_testing.canary import (
    CANARY_PREFIX,
    CanaryCheck,
    CanaryLeak,
    new_canary,
    record_extras,
)

logger = logging.getLogger("stratafi.tests.canary")


def test_canaries_are_unique_and_recognizable() -> None:
    values = {new_canary() for _ in range(100)}
    assert len(values) == 100
    assert all(value.startswith(CANARY_PREFIX) and value.isupper() for value in values)


def test_quiet_code_passes(canary: CanaryCheck) -> None:
    logger.info("parsed a document", extra={"document_id": "d-1", "elements": 12})
    print("progress: 12 elements")
    canary.assert_not_logged()
    canary.assert_absent("a response body", b"an event payload")


def test_a_canary_in_a_log_message_fails(canary: CanaryCheck) -> None:
    logger.warning("chunk text: %s", canary.value)
    with pytest.raises(CanaryLeak):
        canary.assert_not_logged()


def test_a_canary_in_an_extra_field_fails(canary: CanaryCheck) -> None:
    logger.info("embedded a chunk", extra={"chunk_text": canary.value})
    with pytest.raises(CanaryLeak):
        canary.assert_not_logged()


def test_a_canary_nested_in_an_extra_field_fails(canary: CanaryCheck) -> None:
    logger.info("searched", extra={"request": {"query": canary.value}})
    with pytest.raises(CanaryLeak):
        canary.assert_not_logged()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_a_canary_written_to_an_output_stream_fails(canary: CanaryCheck, stream: str) -> None:
    print(f"query was {canary.value}", file=getattr(sys, stream))
    with pytest.raises(CanaryLeak):
        canary.assert_not_logged()


def test_output_streams_are_kept_between_checks(canary: CanaryCheck) -> None:
    print(canary.value)
    with pytest.raises(CanaryLeak):
        canary.assert_not_logged()
    with pytest.raises(CanaryLeak):
        canary.assert_not_logged()


def test_a_canary_in_checked_text_fails(canary: CanaryCheck) -> None:
    with pytest.raises(CanaryLeak, match="#1"):
        canary.assert_absent("clean", canary.value.encode())


def test_the_failure_message_never_repeats_the_canary(canary: CanaryCheck) -> None:
    logger.error(canary.value)
    with pytest.raises(CanaryLeak) as failure:
        canary.assert_not_logged()
    assert canary.value not in str(failure.value)


def test_exception_messages_are_not_read(canary: CanaryCheck) -> None:
    """The production formatter never writes exception messages, so they can't leak."""
    try:
        raise ValueError(canary.value)
    except ValueError:
        logger.exception("parse failed")
    canary.assert_not_logged()


def test_the_production_formatter_is_used(
    caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]
) -> None:
    """A formatter that writes a field the message doesn't carry is checked too."""
    caplog.set_level(logging.INFO)
    check = CanaryCheck(new_canary(), caplog, capfd, format_record=lambda record: check.value)
    logger.info("nothing here")
    with pytest.raises(CanaryLeak):
        check.assert_not_logged()


def test_record_extras_lists_only_caller_fields() -> None:
    record = logging.makeLogRecord({"msg": "m", "document_id": "d-1"})
    assert record_extras(record) == {"document_id": "d-1"}
