# Qwen Server

A local OpenAI-compatible LLM inference server, optimized for RTX 4070 (12GB VRAM).

## Stack

| Component | Value |
|-----------|-------|
| Target model | Qwen2-7B-Instruct-AWQ (awq_marlin) |
| Draft model | Qwen2.5-0.5B (FP16, k=1 speculative decoding) |
| Runtime | vLLM v0.19.0, Docker |
| Throughput | ~91 tok/s |

## Quick Start

```bash
./vllm-server.sh start
./vllm-server.sh status
./vllm-server.sh restart
./vllm-server.sh stop
./vllm-server.sh logs
```

## API

OpenAI-compatible API at `http://localhost:8000/v1`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
  }'
```

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat inference |
| `POST /v1/completions` | Raw completion |
| `GET /v1/models` | List models |
| `POST /tokenize` | Count tokens |
| `GET /health` | Health check |

## Training LoRA Adapters

```bash
# Stop server to free GPU
./vllm-server.sh stop

# Train
docker compose --profile training run --rm trainer python train_lora.py \
  --data_path /training-data/your-dataset.jsonl \
  --num_epochs 2

# Point to new adapter
cd lora-adapters && ln -sf your-adapter-YYYYMMDD/checkpoint-XXX current

# Restart
./vllm-server.sh start
```

See `TRAINING_README.md` for full details.

## Directory Structure

```
qwen-server/
├── docker-compose.yml        # Server and trainer config
├── vllm-server.sh            # Server management
├── scripts/                  # Benchmark and utility scripts
├── training-scripts/         # LoRA training pipeline
├── training-data/            # Training datasets
├── lora-adapters/
│   ├── current -> ...        # Symlink to active adapter
│   └── tool-calling-*/       # Trained checkpoints
└── models/
    ├── Qwen2-7B-Instruct-AWQ/     # Target model
    └── Qwen2.5-0.5B-vocab152064/  # Draft model
```

## Performance

| Config | tok/s |
|--------|-------|
| Baseline (no spec decoding) | 82.4 |
| k=1 speculative (current) | **90.7** |

Hardware: RTX 4070 12GB, WSL2.
