#!/usr/bin/env bash
# Day 13: start the demo inside one container (Hugging Face Space, or CI's test run).
#   1. bundle: use $RECSYS_BUNDLE if it already holds one (CI mounts it), else download $BUNDLE_REPO
#   2. API:    uvicorn on 127.0.0.1:8000 (internal only), wait until /health answers
#   3. UI:     Streamlit on 0.0.0.0:${PORT:-7860} (the public port), talks to the API
# If either process dies, the container exits -> the platform restarts it instead of serving a
# UI whose API is gone.
set -euo pipefail

BUNDLE="${RECSYS_BUNDLE:-$HOME/app/data/serving}"
export RECSYS_BUNDLE="$BUNDLE"
export RECSYS_API_URL="${RECSYS_API_URL:-http://127.0.0.1:8000}"

if [ ! -f "$BUNDLE/bundle.json" ]; then
    if [ -z "${BUNDLE_REPO:-}" ]; then
        echo "No bundle at $BUNDLE and BUNDLE_REPO is not set (Space -> Settings -> Variables)." >&2
        exit 1
    fi
    echo "Downloading bundle $BUNDLE_REPO -> $BUNDLE"
    python -m recsys.hub download --repo "$BUNDLE_REPO" --out "$BUNDLE"
fi

uvicorn recsys.api:app --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!

echo "Waiting for the API ..."
for _ in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null; then break; fi
    if ! kill -0 "$API_PID" 2> /dev/null; then echo "API exited during startup" >&2; exit 1; fi
    sleep 1
done
curl -sf http://127.0.0.1:8000/health || { echo "API not healthy after 180 s" >&2; exit 1; }
echo

streamlit run app/streamlit_app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false &
UI_PID=$!

# exit as soon as either process stops
wait -n "$API_PID" "$UI_PID" || true
echo "a process exited — stopping the container" >&2
exit 1
