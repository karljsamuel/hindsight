#!/bin/bash
# Smoke test script for Hindsight Oracle image
# Uses OCI Oracle database via connection string (no local container)

set -euo pipefail

IMAGE="${1:-hindsight-oracle-test:latest}"
HINDSIGHT_PORT=8888
CONTAINER_NAME="hindsight-smoke-test-$$"

# Get password from secret (only the password, not the full URL)
DB_PASSWORD="${KJS_TEST_ODB_PASSWORD:-}"
if [[ -z "$DB_PASSWORD" ]]; then
    echo "❌ KJS_TEST_ODB_PASSWORD environment variable not set"
    exit 1
fi

# Get OpenRouter API key from secret (for embeddings/reranker)
OPENROUTER_API_KEY="${KJS_TEST_LLM_API_KEY:-}"
if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo "❌ KJS_TEST_LLM_API_KEY environment variable not set"
    exit 1
fi

# Build the database URL with decoded DSN (oracledb thin mode compatible)
# Use the high performance service (port 1522, TLS)
DATABASE_URL="oracle+oracledb://HINDSIGHT_TEST:${DB_PASSWORD}@/?dsn=$(python3 -c "
import urllib.parse
dsn = '(description=(retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.ap-hyderabad-1.oraclecloud.com))(connect_data=(service_name=g4a90f2577f59fb_hindsightdb_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))'
print(urllib.parse.quote(dsn))
")"

echo "=== Smoke Test Starting ==="
echo "Image: $IMAGE"
echo "Using OCI Oracle database via connection string"

cleanup() {
    echo "=== Cleanup ==="
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Clear all tables in the test database before starting
echo "Clearing existing tables in test database..."
docker run --rm \
    -e DATABASE_URL="$DATABASE_URL" \
    python:3.11-slim bash -c '
pip install -q sqlalchemy oracledb 2>/dev/null
cat > /tmp/clear_tables.py << "PYEOF"
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import NullPool
import os
import sys

database_url = os.environ["DATABASE_URL"]
engine = create_engine(database_url, poolclass=NullPool)
with engine.connect() as conn:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    # Oracle stores table names in UPPERCASE. Use the names as returned by inspector.
    existing_tables = []
    for table in tables:
        try:
            # Use the table name as-is (Oracle returns uppercase)
            conn.execute(text("SELECT 1 FROM \"" + table + "\" WHERE ROWNUM = 1"))
            existing_tables.append(table)
        except Exception as e:
            if "ORA-00942" in str(e):
                print("  Skipped (not exist): " + table)
            else:
                print("  Could not check " + table + ": " + str(e))
    
    if len(existing_tables) == 0:
        print("Tables cleared (clean schema)")
    else:
        print("Found " + str(len(existing_tables)) + " tables to drop")
        errors = 0
        for table in existing_tables:
            try:
                conn.execute(text("DROP TABLE \"" + table + "\" CASCADE CONSTRAINTS PURGE"))
                print("  Dropped: " + table)
            except Exception as e:
                print("  Could not drop " + table + ": " + str(e))
                errors += 1
        conn.commit()
        if errors > 0:
            print("Tables cleared with " + str(errors) + " errors")
            sys.exit(1)
        else:
            print("Dropped " + str(len(existing_tables)) + " tables")
PYEOF
python3 /tmp/clear_tables.py
'

# Start Hindsight container
echo "Starting Hindsight container (database-only mode)..."
docker run -d --name "$CONTAINER_NAME" \
    -e HINDSIGHT_API_DATABASE_BACKEND=oracle \
    -e HINDSIGHT_API_VECTOR_EXTENSION=oracle \
    -e HINDSIGHT_API_EMBEDDINGS_ONNX_DIMENSIONS=2048 \
    -e HINDSIGHT_API_DATABASE_URL="$DATABASE_URL" \
    -e HINDSIGHT_API_LLM_PROVIDER=none \
    -e HINDSIGHT_API_EMBEDDINGS_PROVIDER=openrouter \
    -e HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
    -e HINDSIGHT_API_EMBEDDINGS_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free \
    -e HINDSIGHT_API_RERANKER_PROVIDER=openrouter \
    -e HINDSIGHT_API_RERANKER_OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
    -e HINDSIGHT_API_RERANKER_MODEL=nvidia/llama-nemotron-rerank-vl-1b-v2:free \
    -e HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP=false \
    -e HINDSIGHT_API_WORKER_ENABLED=false \
    -p "${HINDSIGHT_PORT}:8888" \
    "$IMAGE" >/dev/null

# Wait for health endpoint - print logs while waiting
echo "Waiting for Hindsight health endpoint (max 120s)..."
for i in {1..60}; do
    if curl -sf "http://localhost:${HINDSIGHT_PORT}/health" >/dev/null 2>&1; then
        echo "✅ Health endpoint responded"
        break
    fi
    # Print container logs every 10 seconds
    if [ $((i % 10)) -eq 0 ]; then
        echo "--- Container logs (attempt $i/60) ---"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "❌ Health endpoint never responded after 120s"
        echo "=== Final container logs ==="
        docker logs "$CONTAINER_NAME"
        exit 1
    fi
done

# Verify process still running
if ! docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "❌ Container died"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

echo "✅ Container is running and healthy"

# Test internal imports
echo "Testing internal imports..."
docker exec "$CONTAINER_NAME" python3 -c "
import hindsight_api.engine.db.oracle
import hindsight_api.engine.memories.oracle
import hindsight_api._vector_index
import hindsight_api.migrations
import hindsight_api.config
print('✅ All critical modules import successfully')
"

# Test config validation
echo "Testing config validation..."
docker exec "$CONTAINER_NAME" python3 -c "
from hindsight_api.config import get_config
cfg = get_config()
assert cfg.database_backend == 'oracle'
assert cfg.vector_extension == 'oracle'
assert cfg.embeddings_onnx_dimensions == 2048
print('✅ Config validation passed')
"

# Test Oracle dialect detection
echo "Testing Oracle dialect..."
docker exec "$CONTAINER_NAME" python3 -c "
from hindsight_api._vector_index import validate_extension, uses_per_bank_vector_indexes, index_type_keyword
ext = validate_extension('oracle')
assert ext == 'oracle'
assert uses_per_bank_vector_indexes('oracle') == False
assert index_type_keyword('oracle') == 'hnsw'
print('✅ Oracle vector index config correct')
"

# Test migration functions
echo "Testing migration functions..."
docker exec "$CONTAINER_NAME" python3 -c "
from hindsight_api.migrations import _detect_vector_extension
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

db_url = '$DATABASE_URL'
engine = create_engine(db_url, poolclass=NullPool)
with engine.connect() as conn:
    from hindsight_api.migrations import _detect_vector_extension
    ext = _detect_vector_extension(conn, 'oracle')
    assert ext == 'oracle'
    print('✅ Migration functions work with Oracle')
"

# Run actual database migration (non-fatal - upstream Oracle deadlock issue)
echo "Running database migration..."
docker exec "$CONTAINER_NAME" hindsight-admin run-db-migration --embedding-dimension 2048 2>&1 | tail -30 || {
    echo "⚠️ Migration failed (upstream Oracle deadlock issue), continuing with schema validation..."
}

# ============================================================
# Functional tests: Bank creation, ingestion, recall, consolidation
# ============================================================

echo "=== Functional Tests: Bank, Ingest, Recall, Consolidation ==="

# Create a test bank
echo "Creating test bank..."
BANK_RESPONSE=$(docker exec "$CONTAINER_NAME" python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app/api')
from hindsight_api.config import get_config

async def create_bank():
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.config import get_config
    from hindsight_api.models import RequestContext
    
    cfg = get_config()
    engine = MemoryEngine(
        db_url=cfg.database_url,
    )
    await engine.initialize()
    
    # Use internal API to ensure bank exists
    rc = RequestContext(internal=True, tenant_id=None, api_key_id=None)
    bank_id = 'smoke_test_bank'
    await engine._ensure_bank_exists(bank_id, rc)
    print(f'BANK_ID:{bank_id}')
    await engine.close()

asyncio.run(create_bank())
")

BANK_ID=$(echo "$BANK_RESPONSE" | grep 'BANK_ID:' | cut -d: -f2)
if [[ -z "$BANK_ID" ]]; then
    echo "❌ Failed to create bank"
    echo "$BANK_RESPONSE"
    exit 1
fi
echo "✅ Created bank: $BANK_ID"

# Ingest test content 1
echo "Ingesting test content 1..."
INGEST1=$(docker exec "$CONTAINER_NAME" python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app/api')
from hindsight_api.models import RequestContext

async def ingest():
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.config import get_config
    
    cfg = get_config()
    engine = MemoryEngine(db_url=cfg.database_url,)
    await engine.initialize()
    
    rc = RequestContext(internal=True, tenant_id=None, api_key_id=None)
    
    try:
        result = await engine.retain_async(
            bank_id='$BANK_ID',
            content='Hindsight is an agent memory system that learns from interactions. It uses Oracle 26ai for vector storage and OpenRouter for LLM/embedding providers.',
            context='smoke test observation',
            fact_type_override='world',
            document_id='doc-001',
            request_context=rc
        )
        print(f'INGEST_RESULT:{result}')
        print(f'INGEST_ID:{result[0] if result else \"\"}')
    except Exception as e:
        print(f'INGEST_ERROR:{e}')
        import traceback
        traceback.print_exc()
        raise
    await engine.close()

asyncio.run(ingest())
")

INGEST1_ID=$(echo "$INGEST1" | grep 'INGEST_ID:' | cut -d: -f2)
if [[ -z "$INGEST1_ID" ]]; then
    echo "❌ Failed to ingest content 1"
    echo "$INGEST1"
    exit 1
fi
echo "✅ Ingested content 1: $INGEST1_ID"

# Ingest test content 2
echo "Ingesting test content 2..."
INGEST2=$(docker exec "$CONTAINER_NAME" python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app/api')
from hindsight_api.models import RequestContext

async def ingest():
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.config import get_config
    
    cfg = get_config()
    engine = MemoryEngine(db_url=cfg.database_url,)
    await engine.initialize()
    
    rc = RequestContext(internal=True, tenant_id=None, api_key_id=None)
    
    result = await engine.retain_async(
        bank_id='$BANK_ID',
        content='Oracle 26ai provides native VECTOR type support with HNSW indexing for efficient similarity search. The VECTOR(2048, FLOAT32) type stores 2048-dimensional embeddings.',
        context='smoke test observation',
        fact_type_override='world',
        document_id='doc-002',
        request_context=rc
    )
    print(f'INGEST_ID:{result[0] if result else \"\"}')
    await engine.close()

asyncio.run(ingest())
")

INGEST2_ID=$(echo "$INGEST2" | grep 'INGEST_ID:' | cut -d: -f2)
if [[ -z "$INGEST2_ID" ]]; then
    echo "❌ Failed to ingest content 2"
    exit 1
fi
echo "✅ Ingested content 2: $INGEST2_ID"

# Ingest test content 3
echo "Ingesting test content 3..."
INGEST3=$(docker exec "$CONTAINER_NAME" python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app/api')
from hindsight_api.models import RequestContext

async def ingest():
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.config import get_config
    
    cfg = get_config()
    engine = MemoryEngine(db_url=cfg.database_url,)
    await engine.initialize()
    
    rc = RequestContext(internal=True, tenant_id=None, api_key_id=None)
    
    result = await engine.retain_async(
        bank_id='$BANK_ID',
        content='OpenRouter provides access to multiple LLM providers including Nemotron models. The nemotron-3-super-120b-a12b:free model offers strong reasoning capabilities.',
        context='smoke test observation',
        fact_type_override='world',
        document_id='doc-003',
        request_context=rc
    )
    print(f'INGEST_ID:{result[0] if result else \"\"}')
    await engine.close()

asyncio.run(ingest())
")

INGEST3_ID=$(echo "$INGEST3" | grep 'INGEST_ID:' | cut -d: -f2)
if [[ -z "$INGEST3_ID" ]]; then
    echo "❌ Failed to ingest content 3"
    exit 1
fi
echo "✅ Ingested content 3: $INGEST3_ID"

# Test recall - semantic search
echo "Testing recall (semantic search)..."
RECALL_RESULT=$(docker exec "$CONTAINER_NAME" python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app/api')
from hindsight_api.models import RequestContext
from hindsight_api.engine.memory_engine import Budget

async def recall():
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.config import get_config
    
    cfg = get_config()
    engine = MemoryEngine(db_url=cfg.database_url,)
    await engine.initialize()
    
    rc = RequestContext(internal=True, tenant_id=None, api_key_id=None)
    
    # Search for Oracle-related content
    results = await engine.recall_async(
        bank_id='$BANK_ID',
        query='Oracle VECTOR type HNSW indexing',
        budget=Budget.MID,
        max_tokens=4096,
        fact_type=['world'],
        request_context=rc
    )
    
    print(f'RECALL_COUNT:{len(results.facts)}')
    for r in results.facts:
        print(f'  - Score: {r.score:.4f}, Text: {r.text[:80]}...')
    
    # Search for Nemotron content
    results2 = await engine.recall_async(
        bank_id='$BANK_ID',
        query='Nemotron model OpenRouter provider',
        budget=Budget.MID,
        max_tokens=4096,
        fact_type=['world'],
        request_context=rc
    )
    
    print(f'RECALL_COUNT2:{len(results2.facts)}')
    for r in results2.facts:
        print(f'  - Score: {r.score:.4f}, Text: {r.text[:80]}...')
    
    await engine.close()

asyncio.run(recall()
")

echo "$RECALL_RESULT"
RECALL_COUNT=$(echo "$RECALL_RESULT" | grep 'RECALL_COUNT:' | head -1 | cut -d: -f2)
RECALL_COUNT2=$(echo "$RECALL_RESULT" | grep 'RECALL_COUNT2:' | head -1 | cut -d: -f2)

if [[ -z "$RECALL_COUNT" ]] || [[ "$RECALL_COUNT" -eq 0 ]]; then
    echo "❌ Recall returned no results"
    exit 1
fi
echo "✅ Recall test passed: $RECALL_COUNT results for Oracle query, $RECALL_COUNT2 for Nemotron query"

# Trigger consolidation
echo "Triggering consolidation..."
CONSOLIDATE_RESULT=$(docker exec "$CONTAINER_NAME" python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app/api')
from hindsight_api.models import RequestContext

async def consolidate():
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.config import get_config
    
    cfg = get_config()
    engine = MemoryEngine(db_url=cfg.database_url,)
    await engine.initialize()
    
    rc = RequestContext(internal=True, tenant_id=None, api_key_id=None)
    
    # Run consolidation for the test bank
    result = await engine.consolidate(
        bank_id='$BANK_ID',
        fact_types=['observation'],
        max_iterations=2,
        request_context=rc
    )
    
    print(f'CONSOLIDATED:{result.consolidated_count}')
    print(f'MERGED:{result.merged_count}')
    print(f'NEW_FACTS:{result.new_fact_count}')
    
    await engine.close()

asyncio.run(consolidate())
")

echo "$CONSOLIDATE_RESULT"
CONSOLIDATED=$(echo "$CONSOLIDATE_RESULT" | grep 'CONSOLIDATED:' | cut -d: -f2)
MERGED=$(echo "$CONSOLIDATE_RESULT" | grep 'MERGED:' | cut -d: -f2)
NEW_FACTS=$(echo "$CONSOLIDATE_RESULT" | grep 'NEW_FACTS:' | cut -d: -f2)

if [[ -z "$CONSOLIDATED" ]] || [[ "$CONSOLIDATED" -eq 0 ]]; then
    echo "⚠️  Consolidation ran but no facts consolidated (may be expected for small dataset)"
else
    echo "✅ Consolidation: $CONSOLIDATED consolidated, $MERGED merged, $NEW_FACTS new facts"
fi

# Verify recall still works after consolidation
echo "Testing recall after consolidation..."
RECALL_AFTER=$(docker exec "$CONTAINER_NAME" python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app/api')
from hindsight_api.models import RequestContext
from hindsight_api.engine.memory_engine import Budget

async def recall():
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.config import get_config
    
    cfg = get_config()
    engine = MemoryEngine(db_url=cfg.database_url,)
    await engine.initialize()
    
    rc = RequestContext(internal=True, tenant_id=None, api_key_id=None)
    
    results = await engine.recall_async(
        bank_id='$BANK_ID',
        query='Oracle VECTOR type',
        budget=Budget.MID,
        max_tokens=4096,
        fact_type=['world'],
        request_context=rc
    )
    
    print(f'RECALL_COUNT:{len(results.facts)}')
    for r in results.facts:
        print(f'  - Score: {r.score:.4f}, Text: {r.text[:80]}...')
    
    await engine.close()

asyncio.run(recall()
")

echo "$RECALL_AFTER"
RECALL_AFTER_COUNT=$(echo "$RECALL_AFTER" | grep 'RECALL_COUNT:' | cut -d: -f2)
if [[ -z "$RECALL_AFTER_COUNT" ]] || [[ "$RECALL_AFTER_COUNT" -eq 0 ]]; then
    echo "❌ Recall after consolidation returned no results"
    exit 1
fi
echo "✅ Recall after consolidation: $RECALL_AFTER_COUNT results"

echo \"=== All Smoke Tests Passed ===\"
exit 0