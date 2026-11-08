"""JSON logging and the "no text fields" policy, with a logging canary."""

import io
import json
import logging

import pytest

from stratafi_core.logs import (
    REDACTED_FIELD,
    REDACTED_TEXT,
    JsonFormatter,
    configure_logging,
    is_text_field,
    log_context,
    safe_fields,
    safe_value,
)
from stratafi_testing.canary import CanaryCheck

logger = logging.getLogger("stratafi.tests.logs")


@pytest.fixture
def stream() -> io.StringIO:
    out = io.StringIO()
    handler = configure_logging("test-service", level="DEBUG", stream=out)
    yield out
    logging.getLogger().removeHandler(handler)


def lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_each_record_is_one_json_object(stream: io.StringIO) -> None:
    logger.info("parsed %s elements", 12, extra={"document_id": "d-1", "pages": 3})
    (record,) = lines(stream)
    assert record["service"] == "test-service"
    assert record["level"] == "info"
    assert record["logger"] == "stratafi.tests.logs"
    assert record["msg"] == "parsed 12 elements"
    assert (record["document_id"], record["pages"]) == ("d-1", 3)
    assert record["ts"].endswith("+00:00")


def test_free_text_arguments_are_redacted(stream: io.StringIO) -> None:
    logger.info("query was %s; key %s", "what is the overdraft fee", "ws/w1/artifacts/abc")
    assert lines(stream)[0]["msg"] == f"query was {REDACTED_TEXT}; key ws/w1/artifacts/abc"


def test_text_fields_are_dropped_whatever_they_hold(stream: io.StringIO) -> None:
    logger.info(
        "chunked",
        extra={"chunk_text": "single", "query": "rates", "title": "Policy", "chunk_count": 4},
    )
    record = lines(stream)[0]
    assert record["chunk_text"] == record["query"] == record["title"] == REDACTED_FIELD
    assert record["chunk_count"] == 4


def test_values_that_are_not_scalars_become_type_names(stream: io.StringIO) -> None:
    logger.info("state", extra={"payload": {"text": "secret words"}, "ids": ["a", "b"]})
    record = lines(stream)[0]
    assert (record["payload"], record["ids"]) == ("dict", "list")


def test_exceptions_keep_their_type_and_frames_but_not_their_message(
    stream: io.StringIO,
) -> None:
    try:
        raise ValueError("the document said something private")
    except ValueError:
        logger.exception("parse failed")
    record = lines(stream)[0]
    assert record["exc_type"] == "builtins.ValueError"
    assert record["exc_frames"]
    assert "private" not in stream.getvalue()


def test_exception_arguments_become_type_names(stream: io.StringIO) -> None:
    logger.warning("retrying after %s", RuntimeError("upstream said: private words"))
    assert lines(stream)[0]["msg"] == "retrying after RuntimeError"


def test_log_context_adds_ids(stream: io.StringIO) -> None:
    with log_context(request_id="r-1", workspace_id="w-1"), log_context(job_id="j-1"):
        logger.info("inside")
    logger.info("outside")
    inside, outside = lines(stream)
    assert (inside["request_id"], inside["workspace_id"], inside["job_id"]) == ("r-1", "w-1", "j-1")
    assert "request_id" not in outside


def test_unknown_context_fields_are_refused() -> None:
    with pytest.raises(ValueError, match="query"), log_context(query="no"):
        pass


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("text", True),
        ("query_text", True),
        ("chunk", True),
        ("error_message", True),
        ("document_id", False),
        ("chunk_count", False),
        ("query_hash", False),
        ("title_len", False),
        ("pages", False),
    ],
)
def test_text_field_names(name: str, text: bool) -> None:
    assert is_text_field(name) is text


def test_safe_values() -> None:
    assert safe_value(None) is None
    assert safe_value(3.5) == 3.5
    assert safe_value("r-0123456789abcdef") == "r-0123456789abcdef"
    assert safe_value("two words") == REDACTED_TEXT
    assert safe_value("x" * 201) == REDACTED_TEXT
    assert safe_value(b"bytes") == "bytes"
    assert safe_fields({"text": "a", "n": 1}) == {"text": REDACTED_FIELD, "n": 1}


def test_configuring_twice_keeps_one_handler() -> None:
    first = configure_logging("a", stream=io.StringIO())
    second = configure_logging("b", stream=io.StringIO())
    handlers = [h for h in logging.getLogger().handlers if isinstance(h.formatter, JsonFormatter)]
    assert handlers == [second]
    assert first is not second
    logging.getLogger().removeHandler(second)


def test_the_logging_canary(canary: CanaryCheck) -> None:
    """Code that logs only metadata passes; the formatter keeps the canary out of the output."""
    sentence = f"the rate table says {canary.value} applies"
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.setFormatter(JsonFormatter("canary"))
    logger.addHandler(handler)
    try:
        logger.info("chunk %s", sentence)
        logger.info("chunked", extra={"text": canary.value, "body_text": sentence})
        try:
            raise ValueError(sentence)
        except ValueError:
            logger.exception("failed")
    finally:
        logger.removeHandler(handler)
    canary.assert_absent(out.getvalue())


def test_stored_metadata_must_hold_no_text() -> None:
    from stratafi_core.logs import TextFieldError, check_metadata

    check_metadata({"elements": 3, "pages": [1, 2], "tokens": {"p50": 512}, "language": "en"})
    check_metadata({"document_id": "d-1", "chunk_count": 4})
    with pytest.raises(TextFieldError, match="title"):
        check_metadata({"title": "Policy"})
    with pytest.raises(TextFieldError, match=r"headings\.0") as error:
        check_metadata({"headings": ["Rates and fees"]})
    assert "Rates" not in str(error.value)
    with pytest.raises(TextFieldError, match=r"stats\.query"):
        check_metadata({"stats": {"query": "x"}})
