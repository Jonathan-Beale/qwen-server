# Training LoRA Adapters

This guide covers training custom LoRA adapters for different frontends/applications.

## Overview

Each frontend application can train its own LoRA adapter with:
- Custom system prompt
- Custom tools
- Custom training dataset

Examples:
- **code-assistant**: coding tools (bash, read_file, write_file, etc.)
- **planning-app**: planning tools (create_task, list_tasks, etc.)
- **notes-app**: note tools (save_note, search_notes, etc.)

---

## Quick Start

### 1. Stop the Server

```bash
./vllm-server.sh stop
```

### 2. Prepare Training Data

Create a JSONL file with your conversation examples:

```json
{"type": "multi_turn", "conversation": [
  {"role": "user", "content": "Save a note about the meeting"},
  {"role": "assistant", "content": "I'll save that note.\n\n<tool_call>\n<tool_name>save_note</tool_name>\n<parameters>\n<title>Meeting Notes</title>\n<content>Discussion about Q1 goals</content>\n</parameters>\n</tool_call>"},
  {"role": "tool_result", "content": "Note saved: Meeting Notes"},
  {"role": "assistant", "content": "I've saved your note titled 'Meeting Notes'."}
]}
```

### 3. Define Your System Prompt

Edit `training-scripts/train_lora.py` and set your `SYSTEM_PROMPT`:

```python
SYSTEM_PROMPT = """You are a notes assistant with access to note-taking tools.

## Tool Usage

When you need to save or search notes, use XML format:

```xml
<tool_call>
<tool_name>TOOL_NAME</tool_name>
<parameters>
<param_name>value</param_name>
</parameters>
</tool_call>
```

Available tools:
- **save_note**: Save a note. Parameters: title, content
- **search_notes**: Search notes. Parameters: query
- **list_notes**: List all notes. Parameters: (none)
"""
```

### 4. Train

```bash
docker compose --profile training run --rm trainer python train_lora.py \
  --data_path /training-data/your-dataset.jsonl \
  --output_dir /lora-adapters/your-app-$(date +%Y%m%d) \
  --num_epochs 2
```

### 5. Deploy

```bash
# Update symlink
cd lora-adapters
ln -sf your-app-YYYYMMDD/checkpoint-XXX current

# Or for multiple adapters, edit docker-compose.yml:
# --lora-modules=coding=/lora-adapters/coding,notes=/lora-adapters/notes

# Restart server
./vllm-server.sh start
```

---

## Training Data Format

### Multi-Turn Conversations

```json
{
  "type": "multi_turn",
  "conversation": [
    {"role": "user", "content": "User request"},
    {"role": "assistant", "content": "Response with <tool_call>...</tool_call>"},
    {"role": "tool_result", "content": "Tool output"},
    {"role": "assistant", "content": "Final response"}
  ]
}
```

### Single-Turn Examples

```json
{
  "type": "single_turn",
  "instruction": "User request",
  "output": "<tool_call>...</tool_call>",
  "input": ""
}
```

---

## Using Public Datasets

Download and prepare function-calling datasets from HuggingFace:

```bash
docker compose --profile training run --rm trainer python prepare_datasets.py
```

This creates:
- `function-calling-combined.jsonl` - 65k examples
- `function-calling-10pct.jsonl` - 6.5k sample for faster iteration

---

## Training Parameters

```bash
python train_lora.py \
  --data_path /training-data/dataset.jsonl \
  --output_dir /lora-adapters/my-adapter \
  --num_epochs 2 \
  --batch_size 1 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_length 8192
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_epochs` | 3 | Training epochs |
| `--batch_size` | 1 | Batch size (limited by VRAM) |
| `--learning_rate` | 2e-4 | Learning rate |
| `--lora_r` | 16 | LoRA rank (8-64) |
| `--lora_alpha` | 32 | LoRA alpha (usually 2x rank) |
| `--max_length` | 8192 | Max sequence length (matches server) |

---

## Multiple Adapters

You can load multiple LoRA adapters and select at inference time:

### Configure in docker-compose.yml

```yaml
command:
  - --lora-modules=coding=/lora-adapters/coding,notes=/lora-adapters/notes,planning=/lora-adapters/planning
```

### Select at Inference

```python
# Use coding adapter
response = client.post("/v1/chat/completions", json={
    "model": "coding",
    "messages": [...]
})

# Use notes adapter
response = client.post("/v1/chat/completions", json={
    "model": "notes",
    "messages": [...]
})
```

---

## Critical Requirements

### Must Match Between Training and Inference

| Setting | Location |
|---------|----------|
| Quantization | BitsAndBytes 4-bit |
| System Prompt | Exact same text |
| Max Length | 8192 tokens |

**If any of these differ, the model won't work correctly.**

### Token Counting

Use vLLM's `/tokenize` endpoint to validate prompt lengths before sending:

```bash
curl http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "tool-calling", "prompt": "Your prompt text"}'
```

Response includes:
- `count`: Number of tokens in the prompt
- `max_model_len`: Server's context limit (8192)
- `tokens`: Array of token IDs

Use this to ensure `prompt_tokens + max_response_tokens <= max_model_len`.

### Frontend Responsibility

Each frontend must:
1. Store its system prompt (must match training)
2. Parse tool calls from model responses
3. Execute tools locally
4. Format tool results for the model
5. Track token usage to stay within context limits

---

## Troubleshooting

### Model Not Using Tools
- System prompt doesn't match training
- Dataset doesn't have enough tool-calling examples

### Garbled Output
- Use `./vllm-server.sh restart` (full stop/start)
- Add `repetition_penalty: 1.1` to requests

### Training Too Slow
- Use smaller dataset (10% sample)
- Reduce `--max_length`
- Reduce `--num_epochs`

### Out of Memory (Training)
- Reduce `--batch_size` to 1
- Reduce `--max_length`
- Increase `--gradient_accumulation`

### Out of Memory (Inference)
- The maximum `--max-model-len` depends on GPU VRAM
- Run `./scripts/test_context_sweep.sh` to find your limit
- RTX 4070 12GB: max 8192 tokens

---

## Files Reference

```
scripts/
├── test_max_context.py   # Test model with long context
└── test_context_sweep.sh # Find max supported context length

training-scripts/
├── train_lora.py         # Main training script (edit SYSTEM_PROMPT)
├── prepare_datasets.py   # Download public datasets
├── generate_training_data.py  # Generate synthetic data
└── test_adapter.py       # Test trained adapter

training-data/
├── function-calling-combined.jsonl  # Public dataset (65k)
├── function-calling-10pct.jsonl     # 10% sample (6.5k)
├── context_length_results.json      # Context sweep test results
└── [your-custom-dataset.jsonl]      # Your training data

lora-adapters/
├── current -> ...        # Symlink to active adapter
├── coding-YYYYMMDD/      # Coding assistant adapter
├── notes-YYYYMMDD/       # Notes app adapter
└── planning-YYYYMMDD/    # Planning app adapter
```
