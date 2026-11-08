"""Cron schedules: fields, steps, names, the day rule, and time zones."""

from datetime import UTC, datetime

import pytest

from stratafi_core.cron import CronError, parse


def at(text: str, after: datetime, zone: str = "UTC") -> datetime:
    return parse(text).next_after(after, zone)


def test_fields_steps_lists_and_names() -> None:
    start = datetime(2026, 10, 31, 10, 7, tzinfo=UTC)  # a Saturday
    assert at("*/15 * * * *", start) == datetime(2026, 10, 31, 10, 15, tzinfo=UTC)
    assert at("5,50 9-11 * * *", start) == datetime(2026, 10, 31, 10, 50, tzinfo=UTC)
    assert at("0 8 * * mon-fri", start) == datetime(2026, 11, 2, 8, 0, tzinfo=UTC)
    assert at("0 0 1 jan *", start) == datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
    assert at("@hourly", start) == datetime(2026, 10, 31, 11, 0, tzinfo=UTC)
    assert at("0 0 * * 7", start) == datetime(2026, 11, 1, 0, 0, tzinfo=UTC)  # 7 is Sunday


def test_the_time_is_strictly_after() -> None:
    start = datetime(2026, 10, 31, 10, 15, tzinfo=UTC)
    assert at("15 10 * * *", start) == datetime(2026, 11, 1, 10, 15, tzinfo=UTC)


def test_restricted_day_fields_match_either() -> None:
    # The 15th, or any Monday: cron's rule when both day fields are restricted.
    start = datetime(2026, 11, 3, 0, 0, tzinfo=UTC)  # a Tuesday
    assert at("0 12 15 * mon", start) == datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
    assert at("0 12 15 * *", start) == datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def test_a_time_zone_and_daylight_saving() -> None:
    chicago = "America/Chicago"
    # 06:17 in Chicago is 11:17 UTC in summer time, and 12:17 UTC once it ends at 02:00 on
    # Sunday, Nov 1, 2026.
    assert at("17 6 * * *", datetime(2026, 10, 30, 12, 0, tzinfo=UTC), chicago) == datetime(
        2026, 10, 31, 11, 17, tzinfo=UTC
    )
    assert at("17 6 * * *", datetime(2026, 10, 31, 12, 0, tzinfo=UTC), chicago) == datetime(
        2026, 11, 1, 12, 17, tzinfo=UTC
    )
    # 01:30 happens twice on Nov 1 in Chicago; it fires once.
    first = at("30 1 * * *", datetime(2026, 11, 1, 0, 0, tzinfo=UTC), chicago)
    assert first == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    assert at("30 1 * * *", first, chicago) == datetime(2026, 11, 2, 7, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("* * * *", "five_fields"),
        ("60 * * * *", "out_of_range"),
        ("5-1 * * * *", "invalid_range"),
        ("*/0 * * * *", "invalid_step"),
        ("x * * * *", "invalid_value"),
        ("0 0 31 2 *", "never_fires"),
        ("", "invalid_schedule"),
    ],
)
def test_bad_schedules(text: str, code: str) -> None:
    with pytest.raises(CronError) as raised:
        parse(text)
    assert raised.value.code == code


def test_an_unknown_zone() -> None:
    with pytest.raises(CronError, match="unknown_time_zone"):
        at("@daily", datetime(2026, 1, 1, tzinfo=UTC), "Nowhere/Else")
