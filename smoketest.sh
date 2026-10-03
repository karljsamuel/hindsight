#!/bin/bash
# Smoke test script for Hindsight Oracle image
# Tests: database schema creation, migration, and validation (NO models/LLMs)

set -euo pipefail

IMAGE="${1:-hindsight-oracle-test:latest}"
ORACLE_PORT=1521
ORACLE_SID="FREE"
ORACLE_PDB="FREEPDB1"
ORACLE_PASSWORD="testpassword123"
HINDSIGHT_PORT=8888
CONTAINER_NAME="hindsight-smoke-test-$$"
ORACLE_CONTAINER_NAME="oracle-smoke-test-$$"
NETWORK_NAME="smoke-test-net-$$"

# Use Docker network for inter-container communication
ORACLE_DSN="(description=(address=(protocol=tcp)(host=${ORACLE_CONTAINER_NAME})(port=${ORACLE_PORT}))(connect_data=(service_name=${ORACLE_PDB})))"
DATABASE_URL="oracle+oracledb://ADMIN:${ORACLE_PASSWORD}@/?dsn=${ORACLE_DSN}"

echo "=== Smoke Test Starting ==="
echo "Image: $IMAGE"
echo "Oracle DSN: $ORACLE_DSN"

cleanup() {
    echo "=== Cleanup ==="
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker stop "$ORACLE_CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$ORACLE_CONTAINER_NAME" >/dev/null 2>&1 || true
    docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Create Docker network for inter-container communication
echo "Creating Docker network..."
docker network create "$NETWORK_NAME" >/dev/null

# Start Oracle Database Free container
echo "Starting Oracle Database Free container..."
docker run -d --name "$ORACLE_CONTAINER_NAME" \
    --network "$NETWORK_NAME" \
    -e ORACLE_PASSWORD="${ORACLE_PASSWORD}" \
    -p "${ORACLE_PORT}:1521" \
    container-registry.oracle.com/database/free:23.4.0.0 >/dev/null

# Wait for Oracle to be ready
echo "Waiting for Oracle to be ready (max 300s)..."
for i in {1..100}; do
    if docker exec "$ORACLE_CONTAINER_NAME" /opt/oracle/checkDBReady.sh >/dev/null 2>&1; then
        echo "✅ Oracle Database is ready"
        break
    fi
    sleep 3
    if [ $i -eq 100 ]; then
        echo "❌ Oracle Database never became ready"
        docker logs "$ORACLE_CONTAINER_NAME" | tail -50
        exit 1
    fi
done

# Create HINDSIGHT user and grant privileges
echo "Creating HINDSIGHT user..."
docker exec "$ORACLE_CONTAINER_NAME" bash -c "
sqlplus -s / as sysdba <<EOF
ALTER SESSION SET CONTAINER = FREEPDB1;
CREATE USER HINDSIGHT IDENTIFIED BY ${ORACLE_PASSWORD} DEFAULT TABLESPACE USERS QUOTA UNLIMITED ON USERS;
GRANT CONNECT, RESOURCE, CREATE TABLE, CREATE SEQUENCE, CREATE VIEW, CREATE PROCEDURE, CTXAPP TO HINDSIGHT;
ALTER USER HINDSIGHT DEFAULT ROLE ALL;
EXIT;
EOF
" >/dev/null 2>&1

echo "✅ HINDSIGHT user created"

# Update DSN to use the container name (resolves via Docker network)
ORACLE_DSN="(description=(address=(protocol=tcp)(host=${ORACLE_CONTAINER_NAME})(port=${ORACLE_PORT}))(connect_data=(service_name=${ORACLE_PDB})))"
DATABASE_URL="oracle+oracledb://HINDSIGHT:${ORACLE_PASSWORD}@/?dsn=${ORACLE_DSN}"

# Start Hindsight container on the same network
echo "Starting Hindsight container (database-only mode)..."
docker run -d --name "$CONTAINER_NAME" \
    --network "$NETWORK_NAME" \
    -e HINDSIGHT_API_DATABASE_BACKEND=oracle \
    -e HINDSIGHT_API_VECTOR_EXTENSION=oracle \
    -e HINDSIGHT_API_EMBEDDINGS_DIMENSION=2048 \
    -e HINDSIGHT_API_DATABASE_URL="${DATABASE_URL}" \
    -e HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP=false \
    -e HINDSIGHT_API_WORKER_ENABLED=false \
    -p "${HINDSIGHT_PORT}:8888" \
    "$IMAGE" >/dev/null

# Wait for health endpoint
echo "Waiting for Hindsight health endpoint (max 120s)..."
for i in {1..60}; do
    if curl -sf "http://localhost:${HINDSIGHT_PORT}/health" >/dev/null 2>&1; then
        echo "✅ Health endpoint responded"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "❌ Health endpoint never responded"
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
assert cfg.embeddings_dimension == 2048
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

# Test migration functions - use the actual container name
DB_URL="oracle+oracledb://HINDSIGHT:${ORACLE_PASSWORD}@/?dsn=(description=(address=(protocol=tcp)(host=${ORACLE_CONTAINER_NAME})(port=${ORACLE_PORT}))(connect_data=(service_name=${ORACLE_PDB})))"
echo "Testing migration functions..."
docker exec "$CONTAINER_NAME" python3 -c "
from hindsight_api.migrations import _detect_vector_extension
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

db_url = '${DATABASE_URL}'
engine = create_engine(db_url, poolclass=NullPool)
with engine.connect() as conn:
    from hindsight_api.migrations import _detect_vector_extension
    ext = _detect_vector_extension(conn, 'oracle')
    assert ext == 'oracle'
    print('✅ Migration functions work with Oracle')
"

# Run actual database migration
echo "Running database migration..."
docker exec "$CONTAINER_NAME" hindsight-admin run-db-migration --embedding-dimension 2048 2>&1 | tail -30

# Verify schema created correctly
echo "Verifying schema..."
docker exec "$CONTAINER_NAME" python3 -c "
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import NullPool

db_url = '${DATABASE_URL}'
engine = create_engine(db_url, poolclass=NullPool)
inspector = inspect(engine)
tables = inspector.get_table_names()
print(f'Tables created: {len(tables)}')
for t in sorted(tables):
    cols = inspector.get_columns(t)
    print(f'  {t}: {len(cols)} columns')
    # Check vector columns
    for col in cols:
        if 'VECTOR' in str(col['type']).upper():
            print(f'    VECTOR column: {col[\"name\"]} = {col[\"type\"]}')
"

echo \"=== All Smoke Tests Passed ===\"
exit 0