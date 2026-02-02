# Qwen Server

A local OpenAI-compatible LLM inference server for fine-tuned models, optimized for RTX 4070 (12GB VRAM).

## What This Is

qwen-server is an **LLM inference backend** that:
- Runs Qwen2-7B with vLLM
- Supports LoRA adapters for fine-tuned models
- Exposes an OpenAI-compatible API at `http://localhost:8000/v1`
- Provides training infrastructure for custom LoRA adapters

**Frontends** (like code-assistant, planning tools, etc.) connect to this server and:
- Send messages to the LLM
- Parse tool calls from responses
- Execute tools locally
- Manage their own conversation context

## Quick Start

### Start the Server

```bash
./vllm-server.sh start
```

### Check Status

```bash
./vllm-server.sh status
```

### Test

```bash
./vllm-server.sh test
```

## Server Management

| Command | Description |
|---------|-------------|
| `./vllm-server.sh start` | Start the server |
| `./vllm-server.sh stop` | Stop the server |
| `./vllm-server.sh restart` | Full stop/start cycle |
| `./vllm-server.sh status` | Check health and list models |
| `./vllm-server.sh logs` | Follow server logs |

**IMPORTANT:** Always use `vllm-server.sh` instead of direct docker commands to avoid BitsAndBytes quantization issues on WSL2.

---

## API Usage

The server exposes an OpenAI-compatible API at `http://localhost:8000`.

### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tool-calling",
    "messages": [
      {"role": "system", "content": "<YOUR_SYSTEM_PROMPT>"},
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.0,
    "repetition_penalty": 1.1
  }'
```

### Token Counting

Use the `/tokenize` endpoint to count tokens before sending requests. This is useful for staying within context limits:

```bash
curl http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tool-calling",
    "prompt": "Your text to tokenize"
  }'
```

Response:
```json
{
  "count": 5,
  "max_model_len": 8192,
  "tokens": [1426, 1467, 311, 47510, 553]
}
```

The `max_model_len` field confirms the server's context limit. Use this to:
- Validate prompt length before sending
- Trim content to fit within context window
- Calculate remaining tokens for response generation

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat-style inference |
| `/v1/completions` | POST | Raw completion inference |
| `/v1/models` | GET | List available models |
| `/tokenize` | POST | Count tokens in text |
| `/health` | GET | Server health check |

### Available Models

| Model | Description |
|-------|-------------|
| `Qwen/Qwen2-7B-Instruct` | Base model |
| `tool-calling` | LoRA adapter for coding tools |

---

## Training Custom LoRA Adapters

Each frontend can train its own LoRA adapter with a custom system prompt and dataset.

### 1. Prepare Training Data

Create a JSONL file with conversation examples:

```json
{"type": "multi_turn", "conversation": [
  {"role": "user", "content": "Your user message"},
  {"role": "assistant", "content": "Model response with <tool_call>...</tool_call>"},
  {"role": "tool_result", "content": "Tool output"},
  {"role": "assistant", "content": "Final response"}
]}
```

Or use existing datasets:
```bash
docker compose --profile training run --rm trainer python prepare_datasets.py
```

### 2. Define Your System Prompt

Edit `training-scripts/train_lora.py` and modify `SYSTEM_PROMPT` to match your frontend's needs:

```python
SYSTEM_PROMPT = """You are a [YOUR ASSISTANT TYPE] with access to tools...

Available tools:
- **your_tool**: Description. Parameters: param1, param2
...
"""
```

### 3. Train

```bash
# Stop server to free GPU
./vllm-server.sh stop

# Train (adjust paths and epochs as needed)
docker compose --profile training run --rm trainer python train_lora.py \
  --data_path /training-data/your-dataset.jsonl \
  --num_epochs 2

# Update symlink to new adapter
cd lora-adapters
ln -sf your-adapter-YYYYMMDD/checkpoint-XXX current

# Restart server
./vllm-server.sh start
```

### 4. Use from Frontend

Your frontend must use the **exact same system prompt** that was used during training:

```python
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "tool-calling",  # or your adapter name
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},  # Must match training!
        {"role": "user", "content": user_input}
    ]
})
```

---

## Tool Call Format

The default training uses XML-based tool calls:

```xml
<tool_call>
<tool_name>TOOL_NAME</tool_name>
<parameters>
<param1>value1</param1>
</parameters>
</tool_call>
```

Frontends parse these from the model response and execute tools locally.

### Standard Coding Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `bash` | `command` | Execute shell commands |
| `read_file` | `path` | Read file contents |
| `write_file` | `path`, `content` | Create/overwrite files |
| `edit_file` | `path`, `find`, `replace` | Find/replace edits |
| `list_files` | `path`, `pattern` | List directory contents |
| `search_files` | `pattern`, `path` | Search for text in files |

You can define **different tools** for different frontends by training with different datasets and system prompts.

---

## Directory Structure

```
qwen-server/
├── docker-compose.yml        # Production + training profiles
├── vllm-server.sh            # Server management
├── training-scripts/
│   ├── train_lora.py         # LoRA training (edit SYSTEM_PROMPT here)
│   ├── prepare_datasets.py   # Download public datasets
│   └── ...
├── training-data/            # Training datasets
├── lora-adapters/            # Trained adapters
│   └── current -> ...        # Symlink to active adapter
└── models/                   # Model cache
```

---

## Critical Requirements

### Training/Inference Consistency

| Setting | Must Match |
|---------|------------|
| Quantization | BitsAndBytes 4-bit for both |
| System Prompt | Exact same text |
| Max Length | 8192 tokens |

If you change any of these, you must retrain the adapter.

### WSL2 Notes

- `--enforce-eager` is required in docker-compose.yml
- Use `./vllm-server.sh restart` (not `docker compose restart`)

---

## Troubleshooting

### Model Not Using Tool Calls
- System prompt doesn't match training prompt
- Verify with `./vllm-server.sh test`

### Garbled Output
- Use `./vllm-server.sh restart` for full stop/start cycle
- Add `repetition_penalty: 1.1` to requests

### Out of Memory
- Server uses ~11.5GB VRAM
- Reduce `--max-model-len` if needed

---

## Performance

| Metric | Value |
|--------|-------|
| Context length | 8192 tokens (max supported on 12GB VRAM) |
| Latency | ~1-2s for short responses, ~8s at full context |
| VRAM usage | ~10.6GB at full context |
| GPU | NVIDIA RTX 4070 (12GB) |

### Context Length Testing

The maximum context length was determined by testing increasing values until OOM:

```bash
# Run the context sweep test
./scripts/test_context_sweep.sh
```

Results on RTX 4070 12GB:
- 8192 tokens: Works (uses ~10.6GB VRAM)
- 8320+ tokens: Fails to start (OOM)

The test uses Wikipedia content with anonymized subjects to verify the model can utilize the full context window.

---

## License

MIT
