#!/bin/sh
# Build the index on first start if leaflets are mounted but no index exists yet, then serve the API.
set -e
if [ ! -f "$LEAFLET_INDEX_DIR/meta.json" ]; then
    echo "No index found in $LEAFLET_INDEX_DIR, building it from $LEAFLET_RAW_DIR ..."
    python -m leaflet_rag ingest
fi
exec python -m leaflet_rag serve --port 8000
