"""Remove sentinel entity timestamps that break Oracle timestamp decoding.

A per-entity ``event_date`` extracted by the LLM can be ``datetime.min``
(0001-01-01). Because ``datetime.min`` is truthy it slipped past the
``event_date or now()`` guard in ``bulk_insert_entities`` and was written
straight into the NOT NULL ``first_seen`` / ``last_seen`` columns.

Both databases accept the value. Oracle cannot *use* it: any statement that has
to materialise it fails with

    ORA-01843: An invalid month was specified

(usually surfaced as ORA-12801 through a parallel query worker), and
python-oracledb fails to read it back the same way --

    ValueError: year -1 is out of range

One bad row is enough to abort every entity read for the bank, so recall,
``resolve_entities`` and therefore all of consolidation and ``batch_retain``
fail together.

The writer side is fixed by ``safe_entity_event_date`` (see engine/db/ops.py).
This revision clears the rows that were already written.

Oracle: DELETE, not UPDATE
--------------------------
An UPDATE of the timestamp column is impossible on these rows -- the old value
must be materialised to build the update. Verified on the affected schema:

    UPDATE entities SET first_seen = <epoch>
        WHERE TO_CHAR(first_seen,'YYYY') = '0001'      -> ORA-12801
    UPDATE entities SET first_seen = <epoch> WHERE ROWNUM = 1
                                                        -> ORA-01843
    BEGIN FOR r IN (...) LOOP UPDATE ... END LOOP; END; -> ORA-01843
    DELETE FROM entities WHERE TO_CHAR(first_seen,'YYYY') = '0001'
                                                        -> OK, 9 rows

Deleting is also the semantically right call. Every affected row is an orphan:
mention_count 0 and no ``unit_entities`` rows -- junk left by a sentinel that
never became a real mention. The mention_count guard makes that explicit, so a
deployment where a sentinel *did* become referenced keeps the row rather than
silently dropping a live entity.

The predicate uses TO_CHAR deliberately: rendering the year to text never
materialises the value, so it avoids the conversion fault. Comparisons against a
typed literal do hit it, which is why
``WHERE first_seen < TIMESTAMP WITH TIME ZONE '...'`` cannot be used at all
(ORA-03048) and the CAST form still fails (ORA-12801). ``EXTRACT(YEAR ...)``
also fails, and silently matches nothing.

PostgreSQL: UPDATE, not DELETE
------------------------------
PostgreSQL represents 0001-01-01 fine, so there the value is repairable and an
UPDATE preserves the entity even when it is referenced. Same predicate,
deliberately different repair -- the backends genuinely differ here.

Revision ID: s4t5u6v7w8x9
Revises: e5b1c7d3a902
Create Date: 2026-10-05
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "s4t5u6v7w8x9"
down_revision: str | Sequence[str] | None = "e5b1c7d3a902"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# A row carrying the sentinel is an orphan by construction: the sentinel was
# written by an extraction that never became a real mention. Refuse to touch
# anything with a mention, so a referenced entity can never be dropped here.
_UNREFERENCED = "mention_count <= 0"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for PostgreSQL multi-tenant migration runs."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"UPDATE {schema}entities SET first_seen = TIMESTAMPTZ '1970-01-01 00:00:00+00' "
        f"WHERE TO_CHAR(first_seen, 'YYYY') = '0001' AND {_UNREFERENCED}"
    )
    op.execute(
        f"UPDATE {schema}entities SET last_seen = TIMESTAMPTZ '1970-01-01 00:00:00+00' "
        f"WHERE TO_CHAR(last_seen, 'YYYY') = '0001' AND {_UNREFERENCED}"
    )


def _pg_downgrade() -> None:
    # Not reversible: the original per-entity dates are unrecoverable, and
    # guessing them back would be worse than leaving the epoch floor.
    pass


def _oracle_upgrade() -> None:
    # Oracle migrations run with CURRENT_SCHEMA set to each tenant, so the table
    # name intentionally stays unqualified here.
    op.execute(
        "DELETE FROM entities "
        "WHERE TO_CHAR(first_seen, 'YYYY') = '0001' "
        "AND TO_CHAR(last_seen, 'YYYY') = '0001' "
        f"AND {_UNREFERENCED}"
    )


def _oracle_downgrade() -> None:
    # The removed rows held no real data; there is nothing to restore.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)