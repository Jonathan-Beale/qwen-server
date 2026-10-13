#!/bin/bash
# vLLM Server Management Script

set -e
cd "$(dirname "$0")"

PROFILE="production"
SERVICE="vllm-server"
IMAGE="vllm/vllm-openai:v0.31.0"
MODEL_REPO="lued/Qwen3.8-27B-INT8-W8A16-MTP"
MODEL_DIR="models/Qwen3.8-27B-INT8-W8A16-MTP"
SERVED_NAME="qwen"
API_URL="http://localhost:8000"
SEARCH_URL="http://localhost:8888"
MAX_WAIT=1200  # seconds; loading 32GB of weights is slow, especially on the x1 PCIe slot

preflight() {
    local ok=1
    if ! docker compose version > /dev/null 2>&1; then
        echo "ERROR: 'docker compose' is not installed."; ok=0
    fi
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "ERROR: nvidia-smi cannot talk to the NVIDIA driver."; ok=0
    elif [ "$(nvidia-smi -L | wc -l)" -lt 2 ]; then
        echo "ERROR: fewer than 2 GPUs visible to the NVIDIA driver (is one still bound to vfio-pci?)."; ok=0
    fi
    if ! docker info 2> /dev/null | grep -q nvidia; then
        echo "ERROR: Docker has no NVIDIA runtime (nvidia-container-toolkit)."; ok=0
    fi
    if [ ! -f "$MODEL_DIR/config.json" ]; then
        echo "ERROR: model not found at $MODEL_DIR. Run: $0 download"; ok=0
    fi
    if [ "$ok" -eq 0 ]; then
        echo ""
        echo "Host prerequisites are installed by: sudo ./scripts/host-setup.sh"
        return 1
    fi
}

wait_for_healthy() {
    echo "Waiting for server to be healthy (model load can take several minutes)..."
    for i in $(seq 1 $MAX_WAIT); do
        if curl -s "$API_URL/health" > /dev/null 2>&1; then
            echo "Server is healthy!"
            return 0
        fi
        if [ -z "$(docker compose --profile "$PROFILE" ps -q --status running "$SERVICE" 2> /dev/null)" ]; then
            echo "ERROR: $SERVICE exited during startup. Last log lines:"
            docker compose --profile "$PROFILE" logs --tail 30 "$SERVICE"
            return 1
        fi
        sleep 1
    done
    echo "ERROR: Server did not become healthy within ${MAX_WAIT}s"
    return 1
}

case "$1" in
    start)
        preflight
        echo "Starting vLLM server and SearXNG..."
        docker compose --profile "$PROFILE" up -d
        wait_for_healthy
        ;;
    stop)
        echo "Stopping vLLM server and SearXNG..."
        docker compose --profile "$PROFILE" down
        ;;
    restart)
        preflight
        echo "Restarting (full stop/start cycle)..."
        docker compose --profile "$PROFILE" down
        docker compose --profile "$PROFILE" up -d
        wait_for_healthy
        ;;
    status)
        if curl -s "$API_URL/health" > /dev/null 2>&1; then
            echo "vLLM: running and healthy"
            curl -s "$API_URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('\n'.join(['  - ' + m['id'] for m in d['data']]))"
        else
            echo "vLLM: not running or not healthy"
        fi
        if curl -s "$SEARCH_URL/healthz" > /dev/null 2>&1; then
            echo "SearXNG: running"
        else
            echo "SearXNG: not running"
        fi
        docker compose --profile "$PROFILE" ps
        ;;
    logs)
        docker compose --profile "$PROFILE" logs -f "$SERVICE"
        ;;
    download)
        echo "Downloading $MODEL_REPO to $MODEL_DIR (~32GB)..."
        mkdir -p models
        docker run --rm --user "$(id -u):$(id -g)" \
            -e HOME=/tmp -e HF_HOME=/models/.cache \
            -v "$PWD/models:/models" --entrypoint hf "$IMAGE" \
            download "$MODEL_REPO" --local-dir "/$MODEL_DIR"
        ;;
    test)
        echo "Testing chat + tool calling..."
        curl -s "$API_URL/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"$SERVED_NAME"'",
            "messages": [{"role": "user", "content": "List the files in the current directory."}],
            "tools": [{"type": "function", "function": {
              "name": "run_command", "description": "Run a bash command",
              "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}],
            "reasoning_effort": "low",
            "max_tokens": 1024
          }' | python3 -c "
import sys, json
m = json.load(sys.stdin)['choices'][0]['message']
calls = m.get('tool_calls') or []
for c in calls:
    print('tool call:', c['function']['name'], c['function']['arguments'])
if not calls:
    print('FAIL: no tool call. Content:', m.get('content'))
    sys.exit(1)
print('OK')"
        echo "Testing web search..."
        curl -s "$SEARCH_URL/search?q=vllm&format=json" | python3 -c "
import sys, json
n = len(json.load(sys.stdin)['results'])
print('OK:' if n else 'FAIL:', n, 'results')
sys.exit(0 if n else 1)"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|download|test}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the vLLM server and SearXNG"
        echo "  stop     - Stop both"
        echo "  restart  - Full stop/start cycle"
        echo "  status   - Check health and list models"
        echo "  logs     - Follow vLLM server logs"
        echo "  download - Download the model weights into models/"
        echo "  test     - Quick test of tool calling and web search"
        exit 1
        ;;
esac
