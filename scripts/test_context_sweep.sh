#!/bin/bash
# Test various max-model-len settings to find the maximum supported context

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_FILE="$PROJECT_DIR/training-data/context_length_results.json"

# Max model lengths to test (ascending order)
# Testing between 8192 and 8704 to find exact limit
LENGTHS_TO_TEST=(8192 8320 8448 8576 8704)

# Wiki page to test with
WIKI_TITLE="${WIKI_TITLE:-Albert Einstein}"

echo "=========================================="
echo "Context Length Sweep Test"
echo "=========================================="
echo "Wiki page: $WIKI_TITLE"
echo "Lengths to test: ${LENGTHS_TO_TEST[*]}"
echo "Results file: $RESULTS_FILE"
echo ""

# Initialize results file
echo '{"tests": [], "max_working": null}' > "$RESULTS_FILE"

update_max_model_len() {
    local len=$1
    echo "Updating docker-compose.yml with max-model-len=$len..."

    python3 << EOF
import re

with open("$PROJECT_DIR/docker-compose.yml", "r") as f:
    content = f.read()

# Replace the max-model-len value (line after --max-model-len)
pattern = r"(--max-model-len\n\s+- ')[0-9]+(')"
replacement = r"\g<1>${len}\2"
content = re.sub(pattern, replacement, content)

with open("$PROJECT_DIR/docker-compose.yml", "w") as f:
    f.write(content)

print(f"Updated max-model-len to $len")
EOF
}

restart_server() {
    echo "Restarting vLLM server..."
    cd "$PROJECT_DIR"
    ./vllm-server.sh restart
}

run_test() {
    local len=$1
    echo ""
    echo "Running test for max-model-len=$len..."

    # Capture output
    local output
    output=$(python3 "$SCRIPT_DIR/test_max_context.py" \
        --max-model-len "$len" \
        --wiki-title "$WIKI_TITLE" 2>&1) || true

    echo "$output"

    # Extract result JSON (everything after "RESULT SUMMARY" line, between the === lines)
    local result
    result=$(echo "$output" | awk '/RESULT SUMMARY/{found=1; next} found && /^{/{p=1} p{print} p && /^}$/{exit}')

    if [ -z "$result" ]; then
        result="{\"success\": false, \"max_model_len\": $len, \"error\": \"Could not parse output\"}"
    fi

    # Append to results file
    python3 << EOF
import json

with open("$RESULTS_FILE", "r") as f:
    data = json.load(f)

try:
    result = json.loads('''$result''')
except Exception as e:
    result = {"success": False, "max_model_len": $len, "error": f"JSON parse failed: {e}"}

data["tests"].append(result)

# Update max_working if this test succeeded
if result.get("success") and result.get("correct"):
    if data["max_working"] is None or $len > data["max_working"]:
        data["max_working"] = $len

with open("$RESULTS_FILE", "w") as f:
    json.dump(data, f, indent=2)
EOF

    # Return success/failure based on whether server started and test ran
    if echo "$result" | grep -q '"success": false'; then
        if echo "$result" | grep -qi "server\|start\|OOM\|memory\|cuda"; then
            return 1  # Server failure - stop testing higher values
        fi
    fi
    return 0
}

check_server_started() {
    # Check if the server is actually running with the expected max_model_len
    local expected_len=$1
    local actual_len

    actual_len=$(curl -s http://localhost:8000/tokenize -X POST \
        -H "Content-Type: application/json" \
        -d '{"model": "tool-calling", "prompt": "test"}' 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('max_model_len', 0))" 2>/dev/null || echo "0")

    if [ "$actual_len" = "$expected_len" ]; then
        echo "Confirmed: server running with max_model_len=$actual_len"
        return 0
    else
        echo "WARNING: Expected max_model_len=$expected_len but got $actual_len"
        return 1
    fi
}

# Main test loop
for len in "${LENGTHS_TO_TEST[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing max-model-len = $len"
    echo "=========================================="

    update_max_model_len "$len"

    if restart_server; then
        if check_server_started "$len"; then
            if ! run_test "$len"; then
                echo ""
                echo "Server startup or test failed at max-model-len=$len"
                echo "This indicates GPU memory limits - stopping sweep."
                break
            fi
        else
            echo "Server configuration mismatch - stopping sweep."
            break
        fi
    else
        echo "Server failed to start with max-model-len=$len"
        # Record failure
        python3 << EOF
import json

with open("$RESULTS_FILE", "r") as f:
    data = json.load(f)

data["tests"].append({
    "success": False,
    "max_model_len": $len,
    "error": "Server failed to start (likely OOM)"
})

with open("$RESULTS_FILE", "w") as f:
    json.dump(data, f, indent=2)
EOF
        echo "Stopping at max-model-len=$len (server startup failed)"
        break
    fi
done

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
cat "$RESULTS_FILE"

# Print summary
python3 << EOF
import json
with open("$RESULTS_FILE", "r") as f:
    data = json.load(f)

print("\n" + "="*50)
print("SUMMARY")
print("="*50)

if data["max_working"]:
    print(f"Maximum working context length: {data['max_working']} tokens")
else:
    print("No successful tests")

print("\nAll tests:")
for t in data["tests"]:
    status = "PASS" if t.get("success") and t.get("correct") else "FAIL"
    tokens = t.get("prompt_tokens", "N/A")
    print(f"  {t['max_model_len']:>6} tokens: {status} (used {tokens} tokens)")
EOF
