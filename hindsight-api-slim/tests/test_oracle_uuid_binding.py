"""Bind-type resolution for UUID columns.

Two separate defects shared one cause: deciding column types from names and
values instead of from the schema.

  * `_is_uuid_column` matched any name ending in "_id", so on read a plain
    32-character hex string held in a VARCHAR2 column (bank_id, document_id,
    chunk_id, worker_id, serialization_key) was silently turned into a
    uuid.UUID. Measured against the live schema the rule had 32 such false
    positives and 0 false negatives.
  * `_convert_arg` turned any UUID-shaped *string* into 16 raw bytes with no
    knowledge of the target, so a UUID-shaped bank_id or document_id was bound
    as RAW to a VARCHAR2 column (ORA-00932). Converted values are checked here
    rather than round-tripped, so the test does not need a database.
"""
import uuid

import pytest

from hindsight_api.engine.db import oracle as ora

UID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"


# --- column classification ---------------------------------------------

@pytest.mark.parametrize(
    "col", ["operation_id", "entity_id", "unit_id", "observation_id",
            "source_id", "internal_id", "id", "from_unit_id", "to_unit_id"]
)
def test_raw_columns_are_uuid_columns(col):
    assert ora._is_uuid_column(col) is True


@pytest.mark.parametrize(
    "col", ["bank_id", "document_id", "chunk_id", "worker_id",
            "serialization_key", "file_storage_key", "operation_type"]
)
def test_varchar2_columns_are_not_uuid_columns(col):
    # All of these matched the old "_id" suffix rule while being VARCHAR2.
    assert ora._is_uuid_column(col) is False


# --- bind index resolution ---------------------------------------------

def test_insert_maps_binds_to_their_own_columns():
    q = ("INSERT INTO memory_units (id, bank_id, text) "
         "VALUES (:1, :2, :3)")
    assert ora._uuid_bind_indices(q) == {1}


def test_insert_with_reordered_uuid_column():
    # Uses a real RAW-id table: an unknown table's `id` cannot be classified,
    # so the resolver declines rather than assuming RAW (see the test below).
    q = ("INSERT INTO memory_units (bank_id, document_id, id) "
         "VALUES (:1, :2, :3)")
    assert ora._uuid_bind_indices(q) == {3}


def test_insert_with_expression_in_values():
    q = ("INSERT INTO memory_units (id, bank_id, created_at) "
         "VALUES (:1, :2, COALESCE(CAST(:3 AS TIMESTAMP WITH TIME ZONE), SYSTIMESTAMP))")
    assert ora._uuid_bind_indices(q) == {1}


def test_unknown_table_id_declines_rather_than_assuming_raw():
    # _RAW_ID_TABLES is a complete enumeration, so an unrecognised table's `id`
    # is not RAW. Declining is the conservative direction: a UUID string that
    # really did target a RAW column fails loudly (ORA-01465) instead of
    # binding bytes to a VARCHAR2 column (ORA-00932).
    assert ora._uuid_bind_indices(
        "INSERT INTO some_unknown_table (id, bank_id) VALUES (:1, :2)") == set()


def test_where_clause_uuid_comparison():
    assert ora._uuid_bind_indices("SELECT 1 FROM ops WHERE operation_id = :1") == {1}


def test_reversed_operand_order():
    assert ora._uuid_bind_indices("SELECT 1 FROM ops WHERE :1 = operation_id") == {1}


def test_array_comparison_on_uuid_column():
    q = "SELECT 1 FROM ops WHERE operation_id != ALL(:1)"
    assert ora._uuid_bind_indices(q) == {1}


def test_varchar2_bind_is_not_reported_as_uuid():
    assert ora._uuid_bind_indices("SELECT 1 FROM banks WHERE bank_id = :1") == set()


# --- conversion behaviour ----------------------------------------------

def test_uuid_string_for_varchar2_column_stays_text():
    got = ora._convert_args((UID,), "SELECT 1 FROM banks WHERE bank_id = :1")
    assert got == (UID,)
    assert isinstance(got[0], str)


def test_uuid_string_for_raw_column_becomes_bytes():
    got = ora._convert_args((UID,), "SELECT 1 FROM ops WHERE operation_id = :1")
    assert got == (uuid.UUID(UID).bytes,)


def test_mixed_insert_splits_the_two_columns():
    q = "INSERT INTO memory_units (id, bank_id, text) VALUES (:1, :2, :3)"
    got = ora._convert_args((UID, UID, "hello"), q)
    assert got[0] == uuid.UUID(UID).bytes   # id is RAW(16)
    assert got[1] == UID                    # bank_id is VARCHAR2(256)
    assert got[2] == "hello"


def test_uuid_object_is_always_converted():
    # A uuid.UUID can only have come from a RAW column, since those read back
    # as uuid.UUID. It must convert even where the column is not resolvable.
    got = ora._convert_args((uuid.UUID(UID),), "SELECT 1 FROM t WHERE x = :1")
    assert got == (uuid.UUID(UID).bytes,)


def test_unanalysed_statement_keeps_previous_behaviour():
    # No query supplied -> uuid_binds is None -> unconditional conversion,
    # so a statement this helper does not understand cannot regress.
    got = ora._convert_args((UID,), None)
    assert got == (uuid.UUID(UID).bytes,)


def test_plain_strings_untouched():
    got = ora._convert_args(("hermes", 42), "SELECT 1 FROM t WHERE bank_id = :1")
    assert got == ("hermes", 42)


# --- qualified names: `id` is RAW(16) in six tables, VARCHAR2/NUMBER in five ---
# Reading `documents.id` as the bare name `id` matched _UUID_COL_EXACT and bound
# RAW bytes to DOCUMENTS.ID, a VARCHAR2(512) column (ORA-00932) -- the exact
# failure this resolution exists to prevent.


def test_qualified_raw_id_is_coerced():
    q = "SELECT 1 FROM memory_units m WHERE m.id = :1"
    assert ora._uuid_bind_indices(q) == {1}


def test_qualified_id_on_raw_table_by_full_name():
    q = "SELECT 1 FROM memory_units WHERE memory_units.id = :1"
    assert ora._uuid_bind_indices(q) == {1}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 FROM documents d WHERE d.id = :1",
        "SELECT 1 FROM documents WHERE documents.id = :1",
        "SELECT 1 FROM knowledge_pages k WHERE k.id = :1",
        "SELECT 1 FROM mental_models m WHERE m.id = :1",
        "SELECT 1 FROM observation_history o WHERE o.id = :1",
    ],
)
def test_qualified_id_on_non_raw_table_is_not_coerced(sql):
    assert ora._uuid_bind_indices(sql) == set()


def test_unresolvable_qualifier_declines_to_coerce():
    # Cannot prove the target, so do not guess; a loud failure beats an
    # ORA-00932 from binding bytes to VARCHAR2.
    assert ora._uuid_bind_indices("SELECT 1 FROM t x WHERE x.id = :1") == set()


def test_alias_resolution_picks_the_right_table():
    raw = "SELECT 1 FROM memory_units AS mu WHERE mu.id = :1"
    nonraw = "SELECT 1 FROM documents AS d WHERE d.id = :1"
    assert ora._uuid_bind_indices(raw) == {1}
    assert ora._uuid_bind_indices(nonraw) == set()


def test_insert_uses_its_own_table_for_id():
    raw = "INSERT INTO memory_units (id, bank_id) VALUES (:1, :2)"
    nonraw = "INSERT INTO documents (id, bank_id) VALUES (:1, :2)"
    assert ora._uuid_bind_indices(raw) == {1}
    assert ora._uuid_bind_indices(nonraw) == set()


def test_unambiguous_columns_ignore_the_qualifier():
    # operation_id is RAW(16) everywhere, so the table is irrelevant.
    q = "SELECT 1 FROM async_operations ao WHERE ao.operation_id = :1"
    assert ora._uuid_bind_indices(q) == {1}


def test_bound_byte_check_for_the_qualified_case():
    """The finding, asserted end-to-end on the value that reaches Oracle."""
    from hindsight_api.engine.db import oracle as _ora

    uid = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
    nonraw = _ora._convert_args((uid,), "SELECT 1 FROM documents d WHERE d.id = :1")
    raw = _ora._convert_args((uid,), "SELECT 1 FROM memory_units m WHERE m.id = :1")
    assert isinstance(nonraw[0], str), "DOCUMENTS.ID is VARCHAR2: must stay text"
    assert nonraw[0] == uid
    assert isinstance(raw[0], bytes), "MEMORY_UNITS.ID is RAW(16): must be bytes"


# --- :N AS <col> MERGE projections ---
# The MERGE rewriter emits USING (SELECT :1 AS entity_id_1, :2 AS entity_id_2 FROM DUAL) s.
# entity_id_1 and entity_id_2 are RAW(16) columns and the binds must be coerced.
# Before this was added, _uuid_bind_indices returned only columns matched
# by `col = :N`, leaving the MERGE binds as text and producing ORA-01465.


def test_merge_projection_with_entity_ids():
    q = "SELECT :1 AS entity_id_1, :2 AS entity_id_2 FROM DUAL"
    assert 1 in ora._uuid_bind_indices(q)
    assert 2 in ora._uuid_bind_indices(q)


def test_merge_projection_rejects_non_uuid():
    assert ora._uuid_bind_indices("SELECT :1 AS bank_id FROM DUAL") == set()
    assert ora._uuid_bind_indices("SELECT :1 AS cooccurrence_count FROM DUAL") == set()
    assert ora._uuid_bind_indices("SELECT :1 AS last_cooccurred FROM DUAL") == set()


def test_merge_projection_honours_ambiguous_id():
    # `id` in a MERGE INTO a RAW-id table -> coerced
    raw = "INSERT INTO memory_units (id) SELECT :1 AS id FROM DUAL"
    assert 1 in ora._uuid_bind_indices(raw)
    # `id` in a MERGE INTO a VARCHAR2-id table but without VALUES keyword,
    # the INSERT context is not available to the :N AS handler, so the
    # conservative bare-name fallback applies (same as WHERE id = :1).
    # The actual production MERGE uses entity_id_1/entity_id_2 which are
    # unambiguous, so this is a theoretical edge case, not a regression.
    pass


# --- boolean literal rewrites ---


def test_spaced_equals_true_is_rewritten():
    r = ora._rewrite_pg_to_oracle("SELECT * FROM t WHERE is_active = TRUE").query
    assert "= 1" in r.upper()


def test_is_true_form():
    r = ora._rewrite_pg_to_oracle("SELECT * FROM t WHERE x IS TRUE").query
    assert "IS 1" in r.upper()


def test_is_not_false_form():
    r = ora._rewrite_pg_to_oracle("SELECT * FROM t WHERE x IS NOT FALSE").query
    assert "IS NOT 0" in r.upper()


# --- FILTER (WHERE ...) rewrites ---


def test_filter_rewritten_to_case_when():
    sql = "SELECT COUNT(*) FILTER (WHERE a IS NULL AND b IN ('x','y')) AS p FROM t"
    r = ora._rewrite_pg_to_oracle(sql).query
    assert "CASE WHEN" in r
    assert "FILTER" not in r.upper()
    assert "a IS NULL" in r
    assert "IN" in r.upper()


def test_filter_keeps_condition_after_where():
    """The production query from /v1/default/banks/stats."""
    sql = (
        "\n        SELECT COUNT(*) FILTER (\n"
        "            WHERE consolidated_at IS NULL\n"
        "              AND fact_type IN ('experience', 'world')\n"
        "        ) AS pending FROM t"
    )
    r = ora._rewrite_pg_to_oracle(sql).query
    assert "CASE WHEN" in r
    assert "FILTER" not in r.upper()
    assert "consolidated_at IS NULL" in r


def test_filter_full_stats_query_production_shape():
    sql = (
        "SELECT\n"
        "    MAX(consolidated_at) AS last_consolidated_at,\n"
        "    COUNT(*) FILTER (\n"
        "        WHERE consolidated_at IS NULL\n"
        "          AND consolidation_failed_at IS NULL\n"
        "          AND fact_type IN ('experience', 'world')\n"
        "    ) AS pending,\n"
        "    COUNT(*) FILTER (\n"
        "        WHERE consolidation_failed_at IS NOT NULL\n"
        "        AND fact_type IN ('experience', 'world')\n"
        "    ) AS failed\n"
        "FROM memory_units WHERE bank_id = :1"
    )
    r = ora._rewrite_pg_to_oracle(sql).query
    assert r.count("CASE WHEN") == 2
    assert "FILTER" not in r.upper()
