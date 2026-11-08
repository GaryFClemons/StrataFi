"""Cron schedules for test suites.

A schedule is five fields, as cron writes them: minute (0-59), hour (0-23), day of the month
(1-31), month (1-12 or ``jan``-``dec``) and day of the week (0-7 or ``sun``-``sat``, where 0
and 7 are Sunday). A field is ``*``, a value, a range ``a-b``, any of these with a step
(``*/15``, ``8-18/2``, ``5/10``), or a comma-separated list of them. ``@hourly``, ``@daily``
(``@midnight``), ``@weekly``, ``@monthly`` and ``@yearly`` (``@annually``) stand for their usual
fields.

As in cron, when both day fields are restricted a day matches either of them. Times are wall
clock times in a time zone: ``next_after`` steps through local time and gives the answer in UTC.
A time that a daylight-saving change skips fires right after the change; a time that happens
twice fires once.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MACROS = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
DAYS = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")
# (low, high, names whose index is low + position)
FIELDS = (
    (0, 59, ()),
    (0, 23, ()),
    (1, 31, ()),
    (1, 12, MONTHS),
    (0, 7, DAYS),
)
MAX_LENGTH = 200
_PART_RE = re.compile(r"(?P<range>\*|[a-z0-9]+(?:-[a-z0-9]+)?)(?:/(?P<step>[0-9]{1,2}))?")
# A schedule whose fields never meet (``0 0 31 2 *``) is refused rather than searched forever.
SEARCH_DAYS = 366 * 8


class CronError(ValueError):
    """A schedule can't be read. ``code`` is short and safe to show."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _value(text: str, low: int, high: int, names: tuple[str, ...]) -> int:
    if text.isdigit():
        value = int(text)
    elif text in names:
        value = low + names.index(text)
    else:
        raise CronError("invalid_value")
    if not low <= value <= high:
        raise CronError("out_of_range")
    return value


def _field(text: str, low: int, high: int, names: tuple[str, ...]) -> tuple[frozenset[int], bool]:
    """The values a field allows, and whether it's unrestricted (``*``)."""
    values: set[int] = set()
    for part in text.split(","):
        match = _PART_RE.fullmatch(part)
        if match is None:
            raise CronError("invalid_field")
        step = int(match["step"]) if match["step"] else 1
        if step < 1:
            raise CronError("invalid_step")
        span = match["range"]
        if span == "*":
            start, end = low, high
        elif "-" in span:
            first, last = span.split("-")
            start, end = _value(first, low, high, names), _value(last, low, high, names)
            if start > end:
                raise CronError("invalid_range")
        else:
            start = _value(span, low, high, names)
            end = high if match["step"] else start
        values.update(range(start, end + 1, step))
    return frozenset(values), text == "*"


@dataclass(frozen=True)
class Schedule:
    text: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]  # 0 is Sunday
    any_day: bool  # the day of the month is *
    any_weekday: bool  # the day of the week is *

    def _day_matches(self, moment: datetime) -> bool:
        weekday = (moment.weekday() + 1) % 7  # Python's Monday is 0; cron's Sunday is 0
        in_month = moment.day in self.days
        in_week = weekday in self.weekdays
        if self.any_day and self.any_weekday:
            return True
        if self.any_day:
            return in_week
        if self.any_weekday:
            return in_month
        return in_month or in_week

    def next_after(self, after: datetime, zone: str = "UTC") -> datetime:
        """The first time after ``after`` (aware) that the schedule fires, in UTC."""
        tz = time_zone(zone)
        local = after.astimezone(tz).replace(tzinfo=None, second=0, microsecond=0)
        moment = local + timedelta(minutes=1)
        limit = moment + timedelta(days=SEARCH_DAYS)
        while moment < limit:
            if moment.month not in self.months:
                year = moment.year + (moment.month == 12)
                moment = datetime(year, moment.month % 12 + 1, 1)  # noqa: DTZ001 - local wall time
                continue
            if not self._day_matches(moment):
                moment = (moment + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if moment.hour not in self.hours:
                moment = (moment + timedelta(hours=1)).replace(minute=0)
                continue
            if moment.minute not in self.minutes:
                moment += timedelta(minutes=1)
                continue
            fired = moment.replace(tzinfo=tz).astimezone(UTC)
            if fired > after:
                return fired
            moment += timedelta(minutes=1)
        raise CronError("never_fires")


def time_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise CronError("unknown_time_zone") from None


def parse(text: str) -> Schedule:
    """A schedule from its five fields or a macro. Raises ``CronError``."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_LENGTH:
        raise CronError("invalid_schedule")
    source = MACROS.get(text.strip().lower(), text.strip().lower())
    parts = source.split()
    if len(parts) != 5:
        raise CronError("five_fields")
    fields = [_field(part, *spec) for part, spec in zip(parts, FIELDS, strict=True)]
    weekdays = frozenset(day % 7 for day in fields[4][0])
    schedule = Schedule(
        text=text.strip(),
        minutes=fields[0][0],
        hours=fields[1][0],
        days=fields[2][0],
        months=fields[3][0],
        weekdays=weekdays,
        any_day=fields[2][1],
        any_weekday=fields[4][1],
    )
    # Fields that can never meet, such as the 31st of February, fail here and not later.
    schedule.next_after(datetime(2026, 1, 1, tzinfo=UTC))
    return schedule
