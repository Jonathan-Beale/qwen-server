#!/bin/bash
# vLLM Server Management Script
# Ensures proper stop/start cycles to avoid stale state issues with BitsAndBytes quantization

set -e
cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.yml"
PROFILE="production"
SERVICE="vllm-server"
HEALTH_URL="http://localhost:8000/health"
MAX_WAIT=180  # seconds

wait_for_healthy() {
    echo "Waiting for server to be healthy..."
    for i in $(seq 1 $MAX_WAIT); do
        if curl -s "$HEALTH_URL" > /dev/null 2>&1; then
            echo "Server is healthy!"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Server did not become healthy within ${MAX_WAIT}s"
    return 1
}

case "$1" in
    start)
        echo "Starting vLLM server..."
        docker compose --profile "$PROFILE" up -d
        wait_for_healthy
        ;;
    stop)
        echo "Stopping vLLM server..."
        docker compose --profile "$PROFILE" down
        ;;
    restart)
        # IMPORTANT: Use full down/up cycle to avoid stale state issues
        # with BitsAndBytes quantization on WSL2
        echo "Restarting vLLM server (full stop/start cycle)..."
        echo "Stopping..."
        docker compose --profile "$PROFILE" down 2>&1 | grep -v "^time="
        echo "Starting..."
        docker compose --profile "$PROFILE" up -d 2>&1 | grep -v "^time="
        wait_for_healthy
        ;;
    status)
        if curl -s "$HEALTH_URL" > /dev/null 2>&1; then
            echo "Server is running and healthy"
            echo ""
            echo "Models available:"
            curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('\n'.join(['  - ' + m['id'] for m in d['data']]))"
        else
            echo "Server is not running or not healthy"
            docker compose --profile "$PROFILE" ps
        fi
        ;;
    logs)
        docker compose --profile "$PROFILE" logs -f "$SERVICE"
        ;;
    test)
        echo "Testing tool-calling model..."
        curl -s http://localhost:8000/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d '{
            "model": "tool-calling",
            "messages": [
              {"role": "system", "content": "You use XML tool calls. Example: <tool_call><tool_name>bash</tool_name><parameters><command>pwd</command></parameters></tool_call>"},
              {"role": "user", "content": "List files in current directory"}
            ],
            "max_tokens": 128,
            "temperature": 0.0,
            "repetition_penalty": 1.1
          }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the vLLM server"
        echo "  stop    - Stop the vLLM server"
        echo "  restart - Full stop/start cycle (required for BitsAndBytes)"
        echo "  status  - Check server health and list models"
        echo "  logs    - Follow server logs"
        echo "  test    - Quick test of tool-calling model"
        exit 1
        ;;
esac
