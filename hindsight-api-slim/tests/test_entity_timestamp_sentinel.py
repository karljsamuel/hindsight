"""The year-0001 sentinel must never reach Oracle timestamp columns.

Oracle stores 0001-01-01 happily, but python-oracledb cannot read it back --
the Julian-day conversion lands outside Python's representable datetime range --
so one such value aborts every entity read for its bank. It is refused at each
boundary where a date can enter the system, not merely filtered on write.
"""
from datetime import UTC, datetime

from hindsight_api.engine.consolidation.consolidator import _aggregate_source_fields
from hindsight_api.engine.db.ops import nullable_year1_timestamp_sql, safe_entity_event_date
from hindsight_api.engine.retain.fact_extraction import _parse_datetime
from hindsight_api.engine.retain.orchestrator import parse_datetime_flexible


def test_safe_entity_event_date_only_rejects_year_one():
    assert safe_entity_event_date(datetime.min) is None
    assert safe_entity_event_date(datetime(1, 1, 2, tzinfo=UTC)) is None
    assert safe_entity_event_date(None) is None
    assert safe_entity_event_date("0001-01-01") is None  # type: ignore[arg-type]

    historical = datetime(1920, 4, 3, tzinfo=UTC)
    assert safe_entity_event_date(historical) is historical
    modern = datetime(2026, 10, 5, tzinfo=UTC)
    assert safe_entity_event_date(modern) is modern


def test_parse_datetime_flexible_refuses_year_one_from_either_type():
    # The retain entry point: an ISO string, or an already-parsed datetime.
    assert parse_datetime_flexible("0001-01-01T00:00:00+00:00") is None
    assert parse_datetime_flexible("0001-01-01") is None
    assert parse_datetime_flexible(datetime.min) is None
    assert parse_datetime_flexible(datetime(1, 12, 31, tzinfo=UTC)) is None


def test_parse_datetime_flexible_keeps_valid_dates():
    valid = parse_datetime_flexible("2026-10-05T12:00:00+00:00")
    assert valid == datetime(2026, 10, 5, 12, tzinfo=UTC)
    # Naive input is treated as UTC rather than refused.
    naive = parse_datetime_flexible("2026-10-05T12:00:00")
    assert naive is not None and naive.tzinfo is not None
    # Historical dates are real dates, not sentinels.
    historical = parse_datetime_flexible("1905-03-02T00:00:00+00:00")
    assert historical is not None and historical.year == 1905


def test_llm_date_parser_refuses_year_one():
    assert _parse_datetime("0001-01-01T00:00:00Z") is None
    assert _parse_datetime("0001-01-01T00:00:00+00:00") is None
    assert _parse_datetime("not a date") is None
    parsed = _parse_datetime("2026-10-05T12:00:00+00:00")
    assert parsed is not None and parsed.year == 2026


def test_entity_timestamp_projection_hides_only_year_one():
    sql = nullable_year1_timestamp_sql("e.last_seen")
    assert sql == "CASE WHEN TO_CHAR(e.last_seen, 'YYYY') = '0001' THEN NULL ELSE e.last_seen END"


def test_aggregate_source_fields_ignores_sentinel_but_keeps_historical_dates():
    historical = datetime(1920, 4, 3, tzinfo=UTC)
    sentinel = datetime.min.replace(tzinfo=UTC)
    sources = [
        {"event_date": sentinel, "occurred_start": sentinel,
         "occurred_end": sentinel, "mentioned_at": sentinel, "tags": []},
        {"event_date": historical, "occurred_start": historical,
         "occurred_end": historical, "mentioned_at": historical, "tags": []},
    ]

    result = _aggregate_source_fields(sources)

    # A sentinel must not win min() and erase the real 1920 timestamp; valid
    # pre-1970 dates remain valid rather than being clamped to an arbitrary floor.
    assert result.event_date == historical
    assert result.occurred_start == historical
    assert result.occurred_end == historical
    assert result.mentioned_at == historical


def test_aggregate_source_fields_returns_none_if_all_dates_are_sentinels():
    sentinel = datetime.min.replace(tzinfo=UTC)
    result = _aggregate_source_fields([
        {"event_date": sentinel, "occurred_start": sentinel,
         "occurred_end": sentinel, "mentioned_at": sentinel, "tags": []}
    ])
    assert result.event_date is None
    assert result.occurred_start is None
    assert result.occurred_end is None
    assert result.mentioned_at is None
