"""Oracle-specific memories store implementation.

This module provides Oracle-native search and recall using VECTOR_DISTANCE
with proper array.array("f") binding for embeddings, and Oracle Text for BM25.
Most methods delegate to PostgresMemories since Oracle uses the same schema.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..retain.types import EmbeddingLike, EntityResolutionResult
from ..schema import fq_store_table, fq_store_table_explicit
from ..search.tags import TagGroup, TagsMatch
from .base import (
    AttachmentRef,
    BankContentCounts,
    DeletedDocument,
    DeletePredicate,
    DocumentBase,
    DocumentChunkState,
    DocumentSourceUnits,
    DocumentTags,
    EntityPrunePassResult,
    EntityResolverHandle,
    ExistingChunk,
    MemoriesExtension,
    MemoryLocation,
    MemoryPatch,
    MemoryScopeWatermark,
    ObservationChunkIds,
    RecallArms,
    RelabelResult,
    RelinkPassResult,
    ScanPage,
    SemanticBm25Result,
    StoredMemory,
    TypedMemoryScope,
)
from .pg import admin as pg_admin
from .pg import banks as pg_banks
from .pg import consolidation as pg_consolidation
from .pg import counts, curation, documents, engine_curation, graph, reads, writes
from .pg import expand as pg_expand
from .pg import links as pg_links
from .pg import retain as pg_retain
from .pg import transfer as pg_transfer
from .postgres import PostgresMemories

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..consolidation.consolidator import _TemporalBounds
    from ..embeddings import Embeddings
    from ..response_models import MemoryFact
    from ..retain.types import CausalRelation, ProcessedFact
    from ..search.graph_retrieval import GraphRetriever
    from ..transfer.export import _LoadedExport, _UnitLocation
    from ..transfer.importer import FactLifecycle, _ObservationOutcome
    from ..transfer.schema import TransferObservation
    from .pg.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


def _is_unique_violation(exc: BaseException) -> bool:
    """True if `exc` is an Oracle unique-constraint violation (ORA-00001).

    Used by the document upsert to fall back from INSERT to UPDATE. Checks the
    Oracle error code on the exception and on any DatabaseError in its args,
    unwrapping chained exceptions, before falling back to the code string.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        codes = [getattr(cur, "code", None)]
        codes += [getattr(a, "code", None) for a in (getattr(cur, "args", ()) or ())]
        if any(str(c) == "ORA-00001" for c in codes):
            return True
        cur = cur.__cause__ or cur.__context__
    return "ORA-00001" in str(exc)


class OracleMemories(PostgresMemories):
    """Oracle-specific memories store using native VECTOR and Oracle Text.

    Inherits from PostgresMemories since Oracle uses the same tables and most
    SQL is compatible. Only overrides search/recall methods that need
    Oracle-specific syntax (VECTOR_DISTANCE, CONTAINS, different param style).
    """

    name = "oracle"

    # ------------------------------------------------------------------ recall
    async def recall_unified(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        query_text: str,
        limit: int,
        temporal_window: "tuple[datetime, datetime] | None" = None,
        temporal_semantic_threshold: float = 0.1,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_semantic: float | None = None,
        min_keyword: float | None = None,
        enable_text_search: bool = True,
        enable_graph: bool = True,
    ) -> "dict[str, RecallArms]":
        """Run every recall arm for Oracle using native VECTOR_DISTANCE and CONTAINS."""
        import asyncio

        from ..db_utils import acquire_with_retry
        from ..search.retrieval import get_default_graph_retriever

        pool = conn

        graph_seed_min_similarity = None
        retriever = None
        if enable_graph:
            from ...config import get_config

            graph_seed_min_similarity = get_config().graph_seed_min_similarity
            retriever = get_default_graph_retriever()

        # Semantic + BM25 (+ temporal) share ONE connection
        async with acquire_with_retry(pool) as db_conn:
            semantic_bm25 = await self.search(
                conn=db_conn,
                bank_id=bank_id,
                fact_types=fact_types,
                query_embedding=query_embedding,
                query_text=query_text,
                limit=limit,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
                created_after=created_after,
                created_before=created_before,
                min_semantic=min_semantic,
                min_keyword=min_keyword,
                graph_seed_min_similarity=graph_seed_min_similarity,
                enable_text_search=enable_text_search,
            )

            temporal_by_ft: dict[str, list] = {}
            if temporal_window is not None:
                start_date, end_date = temporal_window
                temporal_by_ft = await self.temporal_search(
                    conn=db_conn,
                    bank_id=bank_id,
                    fact_types=fact_types,
                    query_embedding=query_embedding,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                    semantic_threshold=temporal_semantic_threshold,
                    tags=tags,
                    tags_match=tags_match,
                    tag_groups=tag_groups,
                    created_after=created_after,
                    created_before=created_before,
                )

        # Graph per fact_type in parallel
        graph_by_ft: dict[str, list] = {ft: [] for ft in fact_types}
        if enable_graph:
            assert retriever is not None
            graph_tasks = [
                retriever.retrieve(
                    pool=pool,
                    query_embedding_str=query_embedding,
                    budget=limit,
                    bank_id=bank_id,
                    fact_type=ft,
                    preselected_semantic_seeds=semantic_bm25.get(ft, SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None)).semantic[:limit],
                )
                for ft in fact_types
            ]
            graph_results = await asyncio.gather(*graph_tasks)
            for ft, results in zip(fact_types, graph_results):
                graph_by_ft[ft] = results.results

        # Assemble RecallArms
        result: dict[str, RecallArms] = {}
        for ft in fact_types:
            arms = semantic_bm25.get(ft, SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None))
            result[ft] = RecallArms(
                semantic=arms.semantic[:limit],
                bm25=arms.bm25[:limit],
                graph=graph_by_ft.get(ft, []),
                temporal=temporal_by_ft.get(ft) or [],
            )
        return result

    # ------------------------------------------------------------------ Oracle-specific search
    async def search(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        query_text: str,
        limit: int,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_semantic: float | None = None,
        min_keyword: float | None = None,
        graph_seed_min_similarity: float | None = None,
        enable_text_search: bool = True,
    ) -> dict[str, SemanticBm25Result]:
        """Oracle-native semantic + BM25 search using VECTOR_DISTANCE and CONTAINS."""
        from .pg.recall import (
            build_bm25_query_text,
            tokenize_query,
            build_tags_where_clause_simple,
            build_tag_groups_where_clause,
            GRAPH_SEED_LIMIT,
        )
        from ...config import get_config
        from ..sql import create_sql_dialect

        result_dict = {ft: SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None) for ft in fact_types}

        config = get_config()
        tokens = tokenize_query(query_text) if enable_text_search else []

        sem_min = min_semantic if min_semantic is not None else config.semantic_min_similarity
        bm25_min = min_keyword if min_keyword is not None else config.bm25_min_score

        graph_seed_threshold = (
            graph_seed_min_similarity
            if graph_seed_min_similarity is not None and sem_min <= graph_seed_min_similarity
            else None
        )
        semantic_fetch = max(limit, GRAPH_SEED_LIMIT if graph_seed_threshold is not None else 0)

        cols = (
            "id, text, context, event_date, occurred_start, occurred_end, mentioned_at, "
            "fact_type, document_id, chunk_id, tags, metadata, proof_count"
        )
        table = fq_store_table("memory_units")

        dialect = create_sql_dialect("oracle")

        # Parameter layout for Oracle: :1 = query_emb, :2 = bank_id, :3+ = others
        _include_bm25 = bool(tokens)
        bm25_text_param: str = ""
        tags_param_idx = 3
        tags_clause = build_tags_where_clause_simple(tags, tags_param_idx, match=tags_match)

        tag_groups_param_start = tags_param_idx + (1 if tags else 0)
        built = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)
        groups_clause = built.sql
        groups_params = built.params

        _next_idx = tag_groups_param_start + len(groups_params)
        updated_range_clause = ""
        updated_range_params: list[Any] = []
        if created_after is not None:
            updated_range_params.append(created_after)
            updated_range_clause += f" AND updated_at > :{_next_idx}"
            _next_idx += 1
        if created_before is not None:
            updated_range_params.append(created_before)
            updated_range_clause += f" AND updated_at < :{_next_idx}"
            _next_idx += 1

        # --- Semantic UNION ALL arms ---
        arms = [
            dialect.build_semantic_arm(
                table=table,
                cols=cols,
                fact_type=ft,
                embedding_param=":1",
                bank_id_param=":2",
                fetch_limit=semantic_fetch,
                min_similarity=sem_min,
                tags_clause=tags_clause,
                groups_clause=groups_clause,
                extra_where=updated_range_clause,
            )
            for ft in fact_types
        ]

        # --- BM25 UNION ALL arms ---
        if _include_bm25:
            text_ext = config.text_search_extension
            bm25_text_param = await build_bm25_query_text(
                conn,
                dialect,
                tokens=tokens,
                query_text=query_text,
                table="memory_units",
                language=config.text_search_extension_native_language,
                config=config,
            )
            # Every BM25 arm binds the same limit and the same query text, so
            # they all share one pair of parameter indices. Giving each arm its
            # own pair (:3/:4, then :5/:6, ...) made the SQL reference
            # placeholders that were never appended -- params are built once
            # below, not per arm -- and Oracle rejected the statement with
            # DPY-4010 "a bind variable replacement value for placeholder :N
            # was not provided" as soon as more than one fact_type was
            # requested.
            bm25_limit_param = f":{_next_idx}"
            bm25_text_param_idx = f":{_next_idx + 1}"
            for i, ft in enumerate(fact_types):
                arms.append(
                    dialect.build_bm25_arm(
                        table=table,
                        cols=cols,
                        fact_type=ft,
                        bank_id_param=":2",
                        limit_param=bm25_limit_param,
                        text_param=bm25_text_param_idx,
                        tags_clause=tags_clause,
                        groups_clause=groups_clause,
                        arm_index=i,
                        text_search_extension=text_ext,
                        bm25_language=config.text_search_extension_native_language,
                        bm25_min_score=bm25_min,
                        pg_search_function_schema="",
                        pg_search_tokenizer="",
                        max_query_terms=config.bm25_max_query_terms,
                        extra_where=updated_range_clause,
                    )
                )
            _next_idx += 2

        query = "\nUNION ALL\n".join(arms)

        # Build params: :1 = query_emb, :2 = bank_id, then tags, tag groups, the
        # updated_at range, and finally the BM25 limit/text pair -- in exactly
        # that order. The indices above are allocated in that sequence, so
        # appending the BM25 pair before the tags would bind every value to the
        # wrong placeholder as soon as a tag filter or date range was used.
        params: list = [query_embedding, bank_id]
        if tags:
            params.append(tags)
        params.extend(groups_params)
        params.extend(updated_range_params)
        if _include_bm25:
            params.append(limit)  # bm25_limit_param
            params.append(bm25_text_param)  # bm25_text_param_idx

        try:
            rows = await conn.fetch(query, *params)
        except Exception as e:
            # Oracle Text CONTAINS can fail - fallback to semantic-only
            err_str = str(e)
            if _include_bm25 and ("DRG-10599" in err_str or "ORA-30600" in err_str or "ORA-29902" in err_str):
                logger.warning("Oracle Text CONTAINS failed (%s), falling back to semantic-only search", err_str[:120])
                # Rebuild semantic-only with correct param indices
                fb_tags_idx = 3
                fb_tags_clause = build_tags_where_clause_simple(tags, fb_tags_idx, match=tags_match)
                fb_groups_start = fb_tags_idx + (1 if tags else 0)
                built = build_tag_groups_where_clause(tag_groups, fb_groups_start)
                fb_groups_clause = built.sql
                fb_next_idx = fb_groups_start + len(groups_params)
                fb_updated_clause = ""
                if created_after is not None:
                    fb_updated_clause += f" AND updated_at > :{fb_next_idx}"
                    fb_next_idx += 1
                if created_before is not None:
                    fb_updated_clause += f" AND updated_at < :{fb_next_idx}"
                    fb_next_idx += 1
                fb_arms = [
                    dialect.build_semantic_arm(
                        table=table,
                        cols=cols,
                        fact_type=ft,
                        embedding_param=":1",
                        bank_id_param=":2",
                        fetch_limit=semantic_fetch,
                        min_similarity=sem_min,
                        tags_clause=fb_tags_clause,
                        groups_clause=fb_groups_clause,
                        extra_where=fb_updated_clause,
                    )
                    for ft in fact_types
                ]
                fb_query = "\nUNION ALL\n".join(fb_arms)
                fb_params: list = [query_embedding, bank_id]
                if tags:
                    fb_params.append(tags)
                fb_params.extend(groups_params)
                fb_params.extend(updated_range_params)
                rows = await conn.fetch(fb_query, *fb_params)
            else:
                raise

        # Group results
        from ..search.types import RetrievalResult

        semantic_candidates: dict[str, list[RetrievalResult]] = {ft: [] for ft in fact_types}
        for r in rows:
            row = dict(r)
            source = row.pop("source")
            ft = row.get("fact_type")
            if ft not in result_dict:
                continue
            if source == "semantic":
                if len(semantic_candidates[ft]) < semantic_fetch:
                    semantic_candidates[ft].append(RetrievalResult.from_db_row(row))
            else:
                result_dict[ft].bm25.append(RetrievalResult.from_db_row(row))

        for ft, candidates in semantic_candidates.items():
            result_dict[ft].semantic.extend(candidates[:limit])
            if graph_seed_threshold is not None:
                result_dict[ft].graph_seeds = [
                    r for r in candidates[:limit]
                    if (r.similarity or 0) >= graph_seed_threshold
                ][:GRAPH_SEED_LIMIT]

        return result_dict

    async def temporal_search(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        start_date: datetime,
        end_date: datetime,
        limit: int,
        semantic_threshold: float = 0.1,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> dict[str, list]:
        """Oracle-native temporal search with VECTOR_DISTANCE."""
        from .pg.recall import (
            build_tags_where_clause_simple,
            build_tag_groups_where_clause,
        )
        from ...config import get_config
        from ..sql import create_sql_dialect
        from ..search.types import RetrievalResult

        config = get_config()
        dialect = create_sql_dialect("oracle")

        table = fq_store_table("memory_units")
        cols = (
            "id, text, context, event_date, occurred_start, occurred_end, mentioned_at, "
            "fact_type, document_id, chunk_id, tags, metadata, proof_count"
        )

        result_dict: dict[str, list] = {ft: [] for ft in fact_types}

        tags_param_idx = 3
        tags_clause = build_tags_where_clause_simple(tags, tags_param_idx, match=tags_match)

        tag_groups_param_start = tags_param_idx + (1 if tags else 0)
        built = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)
        groups_clause = built.sql
        groups_params = built.params

        _next_idx = tag_groups_param_start + len(groups_params)
        updated_range_clause = ""
        updated_range_params: list[Any] = []
        if created_after is not None:
            updated_range_params.append(created_after)
            updated_range_clause += f" AND updated_at > :{_next_idx}"
            _next_idx += 1
        if created_before is not None:
            updated_range_params.append(created_before)
            updated_range_clause += f" AND updated_at < :{_next_idx}"
            _next_idx += 1

        # Temporal window clause
        temporal_clause = f" AND event_date >= :{_next_idx} AND event_date <= :{_next_idx + 1}"
        temporal_params = [start_date, end_date]
        _next_idx += 2

        for ft in fact_types:
            arm = dialect.build_semantic_arm(
                table=table,
                cols=cols,
                fact_type=ft,
                embedding_param=":1",
                bank_id_param=":2",
                fetch_limit=limit,
                min_similarity=semantic_threshold,
                tags_clause=f"{tags_clause}{groups_clause}{updated_range_clause}{temporal_clause}",
                groups_clause="",
                extra_where="",
            )

            params: list = [query_embedding, bank_id]
            if tags:
                params.append(tags)
            params.extend(groups_params)
            params.extend(updated_range_params)
            params.extend(temporal_params)

            rows = await conn.fetch(arm, *params)
            for r in rows:
                result_dict[ft].append(RetrievalResult.from_db_row(dict(r)))

        return result_dict

    # ------------------------------------------------------------------ Oracle-specific document upsert
    async def upsert_document_row(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        document_id: str,
        original_text: str | None,
        content_hash: str,
        retain_params: dict | None,
        document_tags: list[str] | None,
        preserved_created_at: datetime | None,
    ) -> None:
        """Oracle-specific document upsert using MERGE."""
        # Oracle MERGE equivalent of PostgreSQL ON CONFLICT
        # original_text, retain_params, tags can be very large, bind as CLOB
        doc_table = "documents"
        params = [
            document_id,  # :1 id
            bank_id,      # :2 bank_id
            original_text,  # :3 original_text (CLOB)
            content_hash,  # :4 content_hash
            json.dumps(retain_params) if retain_params else None,  # :5 retain_params (CLOB)
            document_tags or [],  # :6 tags (CLOB)
            preserved_created_at,  # :7 preserved_created_at
        ]
        # original_text is an unbounded CLOB, and Oracle cannot bind a CLOB larger
        # than 32767 bytes through a MERGE's `SELECT :N FROM DUAL` source --
        # the DUAL projection forces VARCHAR2 typing and the bind fails with
        # ORA-01461 at bind position 3. Verified against the test schema: the
        # same MERGE binds fine at 32767 bytes and fails at 32768, while a plain
        # INSERT of the identical 256KB value succeeds. So do the upsert as
        # INSERT-then-UPDATE, which both accept arbitrarily large CLOBs.
        #
        # Do not "fix" this by adding CAST(:N AS CLOB) to the MERGE either --
        # CAST of a CLOB inside `SELECT ... FROM DUAL` raises ORA-22849.
        insert_sql = f"""
            INSERT INTO {doc_table} (
                id, bank_id, original_text, content_hash,
                retain_params, tags, created_at, updated_at
            ) VALUES (
                :1, :2, :3, :4,
                :5, :6,
                COALESCE(CAST(:7 AS TIMESTAMP WITH TIME ZONE), SYSTIMESTAMP),
                SYSTIMESTAMP
            )
        """
        update_sql = f"""
            UPDATE {doc_table} SET
                original_text = :3,
                content_hash = :4,
                retain_params = :5,
                tags = :6,
                updated_at = SYSTIMESTAMP
            WHERE id = :1 AND bank_id = :2
        """
        try:
            await conn.execute(insert_sql, *params)
        except Exception as exc:
            # ORA-00001: unique constraint violated -> row already exists
            if not _is_unique_violation(exc):
                raise
            # update_sql does not bind :7 (preserved_created_at is insert-only),
            # and python-oracledb rejects a param the SQL never references.
            await conn.execute(update_sql, *params[:6])

    # ------------------------------------------------------------------ addressed reads (delegate to pg)
    async def get_memories(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[StoredMemory]:
        return await reads.get_memories(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

    async def scan_memories(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_types: list[str] | None = None,
        limit: int = 100,
        page_token: str = "",
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
        document_id: str | None = None,
        metadata_equals: dict[str, str] | None = None,
        skip: int = 0,
        include_edges: bool = False,
    ) -> ScanPage:
        return await reads.scan_memories(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=limit,
            page_token=page_token,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            document_id=document_id,
            metadata_equals=metadata_equals,
            skip=skip,
            include_edges=include_edges,
        )

    # Delegate remaining methods to pg implementation
    # ... (other methods use same SQL, just different dialect)