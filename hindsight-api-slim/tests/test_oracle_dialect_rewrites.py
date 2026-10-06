"""Dialect-rewrite regressions.

Each case here was a live defect: either invalid SQL reaching Oracle, or a
silently wrong answer with no error raised. They are asserted at the rewriter
level because that is where the defect lives and where a fix is falsifiable.
"""
import pytest

from hindsight_api.engine.db import oracle as ora


def rw(sql: str) -> str:
    return ora._rewrite_pg_to_oracle(sql).query


# --- date_trunc ---------------------------------------------------------
# Was regex `date_trunc\(\s*'(\w+)'\s*,\s*(.+?)\s*\)`, whose non-greedy group
# stopped at the first close paren and cut nested expressions in half.


def test_date_trunc_nested_call_is_balanced():
    out = rw("SELECT date_trunc('day', GREATEST(a, LEAST(b, c))) FROM t")
    assert out == "SELECT TRUNC(CAST(GREATEST(a, LEAST(b, c)) AS DATE), 'DD') FROM t"
    assert out.count("(") == out.count(")")


def test_date_trunc_coalesce_with_at_time_zone():
    out = rw("SELECT date_trunc('day', COALESCE(a, b) AT TIME ZONE 'UTC') FROM t")
    assert out == "SELECT TRUNC(CAST(COALESCE(a, b) AS DATE), 'DD') FROM t"
    assert "AT TIME ZONE" not in out


def test_date_trunc_keeps_literal_out_of_output():
    # Regression: slicing up to the paren left "date_trunc" glued to the result.
    out = rw("SELECT date_trunc('hour', created_at) FROM t")
    assert "date_trunc" not in out.lower()
    assert out == "SELECT TRUNC(CAST(created_at AS DATE), 'HH24') FROM t"


@pytest.mark.parametrize(
    "interval,fmt",
    [("year", "YYYY"), ("quarter", "Q"), ("month", "MM"), ("week", "IW"),
     ("day", "DD"), ("hour", "HH24"), ("minute", "MI"), ("second", "SS")],
)
def test_date_trunc_interval_maps_to_its_own_bucket(interval, fmt):
    out = rw(f"SELECT date_trunc('{interval}', c) FROM t")
    assert f"'{fmt}'" in out


def test_date_trunc_quarter_is_not_silently_day():
    # Was dict.get(interval, "DD"): quarter returned a DAY bucket, no error.
    assert "'Q'" in rw("SELECT date_trunc('quarter', c) FROM t")
    assert "'DD'" not in rw("SELECT date_trunc('quarter', c) FROM t")


def test_date_trunc_second_is_not_silently_day():
    assert "'SS'" in rw("SELECT date_trunc('second', c) FROM t")


def test_date_trunc_unknown_interval_is_refused():
    # Refusing beats a mismatched bucket for a caller-supplied parameter.
    with pytest.raises(ora.OracleRewriteError):
        rw("SELECT date_trunc('fortnight', c) FROM t")


# --- RETURNING ----------------------------------------------------------
# Was `\bRETURNING\s+(?!(?:CLOB|BLOB|VARCHAR2|JSON)\b)(.+)`, which matched a
# JSON function's own RETURNING type. The lookahead had to enumerate every
# possible type and missed NUMBER, which this rewriter itself emits.


def test_json_returning_type_is_not_treated_as_statement_returning():
    res = ora._rewrite_pg_to_oracle("UPDATE t SET x = 1 WHERE tags @> $3 RETURNING id")
    assert res.returning_cols == ["id"]
    assert "RETURNING NUMBER) INTO" not in res.query
    assert res.query.rstrip().endswith("RETURNING id INTO :ret_0")


def test_json_mergepatch_returning_clob_is_preserved():
    res = ora._rewrite_pg_to_oracle("UPDATE t SET m = m || $1::jsonb WHERE id = $2")
    assert "RETURNING CLOB)" in res.query


def test_plain_statement_returning_still_rewritten():
    out = rw("UPDATE t SET m = m WHERE id = $1 RETURNING id")
    assert out.endswith("RETURNING id INTO :ret_0")


def test_multiple_returning_columns_get_one_bind_each():
    res = ora._rewrite_pg_to_oracle("UPDATE t SET m = m WHERE id = $1 RETURNING id, metadata")
    assert res.returning_cols == ["id", "metadata"]
    assert ":ret_0" in res.query and ":ret_1" in res.query


# --- LIMIT / OFFSET with FOR UPDATE -------------------------------------
# Was `WHERE ROWNUM <= n ... ORDER BY`, which Oracle evaluates ROWNUM before
# ORDER BY, so claims took an arbitrary n rows -- measured returning the n
# NEWEST pending operations instead of the n OLDEST.


def test_for_update_uses_fetch_first_not_rownum():
    out = rw(
        "SELECT o.id FROM ops o WHERE o.status='pending' "
        "ORDER BY o.created_at LIMIT $1 FOR UPDATE SKIP LOCKED"
    )
    assert "ROWNUM" not in out
    assert "ORDER BY o.created_at FETCH FIRST :1 ROWS ONLY FOR UPDATE SKIP LOCKED" in out


def test_for_update_preserves_order_by_before_fetch():
    out = rw("SELECT id FROM t ORDER BY created_at LIMIT 5 FOR UPDATE SKIP LOCKED")
    assert out.index("ORDER BY") < out.index("FETCH FIRST")


def test_limit_with_offset_and_for_update():
    out = rw("SELECT id FROM t ORDER BY c LIMIT 10 OFFSET 20 FOR UPDATE")
    assert "OFFSET :2 ROWS FETCH FIRST :1 ROWS ONLY" in out or \
           "OFFSET 20 ROWS FETCH FIRST 10 ROWS ONLY" in out
    assert "ROWNUM" not in out


def test_plain_limit_without_for_update_unchanged():
    out = rw("SELECT id FROM t ORDER BY c LIMIT 7")
    assert "FETCH FIRST 7 ROWS ONLY" in out
    assert "ROWNUM" not in out
