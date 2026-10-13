# Qwen Server

A local OpenAI-compatible LLM server with an agent (browser UI or terminal) that can search the web and work on code, including this repository. Other machines connect with a one-command install.

## Stack

| Component | Value |
|-----------|-------|
| Model | Qwen3.8-27B, INT8 weights (`lued/Qwen3.8-27B-INT8-W8A16-MTP`), served as `qwen` |
| Hardware | 2x RTX 3090 (24GB each), tensor-parallel |
| Context | 262,144 tokens, one request at a time |
| Runtime | vLLM v0.31.0, Docker |
| Web search | SearXNG (self-hosted, no API key) |

## Server setup

`scripts/host-setup.sh` and the model settings in `docker-compose.yml` are specific to this server (Ubuntu 26.04, two RTX 3090s). A different server needs its own driver setup and a model sized to its GPUs.

One-time host setup (NVIDIA kernel modules, docker compose, NVIDIA container toolkit, release the first 3090 from VM passthrough):

```bash
sudo ./scripts/host-setup.sh && sudo reboot
nvidia-smi                    # should list two RTX 3090s
./vllm-server.sh download     # ~32GB into models/
```

## Quick Start

On the server:

```bash
./vllm-server.sh start          # vLLM + SearXNG; first load takes several minutes
./vllm-server.sh test           # tool calling + web search smoke test
python3 scripts/agent.py --ui   # chat in the browser (omit --ui for a terminal chat)
./vllm-server.sh status
./vllm-server.sh logs
./vllm-server.sh stop
```

Both containers restart on boot once started.

## Use it from another machine

The model stays on the server. Each client machine runs the agent locally, so the assistant reads and edits files on that machine, and reaches the model through an SSH tunnel. Clients need Python 3.10+ and an SSH client; nothing else is installed.

**Windows** (PowerShell), replacing `user@server`:

```powershell
scp user@server:qwen-server/client/install.ps1 $env:TEMP\install.ps1
powershell -ExecutionPolicy Bypass -File $env:TEMP\install.ps1 -Server user@server
```

This puts a **Qwen** icon on the desktop.

**Linux / macOS:**

```bash
ssh user@server cat qwen-server/client/install.sh | sh -s -- user@server
```

This installs a `qwen` command (and an applications-menu entry on Linux).

Both installers set up a passwordless SSH key if there isn't one (asking for the server password once), copy `agent.py` and `agent_ui.html` from the server, and create the launcher. Re-run to update. If the repo is not at `~/qwen-server` on the server, pass `-RemoteDir path` (Windows) or a second argument (Linux/macOS).

## Agent

`scripts/agent.py` gives the model these tools:

| Tool | Description |
|------|-------------|
| `web_search` | Search via the SearXNG instance |
| `fetch_url` | Read a web page as text |
| `list_files`, `read_file`, `search_files` | Inspect the workspace |
| `write_file`, `edit_file` | Modify files in the workspace |
| `run_command` | Run a shell command in the workspace (bash; PowerShell on Windows) |

`--ui` opens a browser UI served on `127.0.0.1` only; without it you get a terminal chat. `--ssh user@server` opens the tunnel to the server for the session.

The workspace is the folder the assistant works in. On the server it defaults to this repository, so the model can read and change its own setup. Change it with **Change folder** in the UI (remembered in `~/.qwen-agent.json`) or `--workspace path`.

File tools are confined to the workspace. `run_command` is not sandboxed: it runs as your user. The agent asks before every write and command unless you choose **Always allow** or pass `--yes`.

The UI keeps running after you close the tab so the next launch is instant; **Quit** stops it and closes the tunnel.

Pages are fetched over plain HTTP. If `brave-browser` or `chromium` is on `PATH`, the agent renders pages with it headlessly instead (`--no-browser` to disable).

Other options: `--effort low|medium|xhigh` (the model defaults to `xhigh`, which can overthink), `--model`, `--url`, `--searx-url`, `--port`.

`scripts/chat.py` is a plain chat without tools.

## API

OpenAI-compatible API at `http://localhost:8000/v1`, bound to localhost only. To expose it on the network, change the port mapping in `docker-compose.yml`; there is no authentication.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Tool calling (`tools`) and reasoning output are enabled. Omit sampling parameters to use the model's recommended defaults (temperature 1.0, top_p 0.95, top_k 20). Disable thinking per request with `"chat_template_kwargs": {"enable_thinking": false}`.

SearXNG's JSON API is at `http://localhost:8888/search?q=...&format=json`.

## Notes

- `--max-num-seqs` is 1 because concurrent requests can crash the engine with MTP speculative decoding on this model (vllm-project/vllm#50021).
- The second 3090 (`06:00.0`) currently negotiates a PCIe x1 link. That slows model loading and may limit tensor-parallel throughput; reseating the card or riser is worth trying.
- To give the first 3090 back to a VM: `sudo systemctl enable vfio-card1.service` and reboot. The model as configured needs both cards.

## Legacy: LoRA training

`training-scripts/`, `Dockerfile.training`, the `trainer` compose service, `TRAINING_README.md`, `benchmark_results.json`, and the remaining files in `scripts/` belong to the previous Qwen2-7B / RTX 4070 setup. They are not used by the current server and have not been ported to Qwen3.8.

## Directory Structure

```
qwen-server/
├── docker-compose.yml        # vLLM, SearXNG (and legacy trainer)
├── vllm-server.sh            # Server management
├── searxng/settings.yml      # Search backend config
├── client/
│   ├── install.ps1           # Client installer, Windows
│   └── install.sh            # Client installer, Linux / macOS
├── scripts/
│   ├── agent.py              # Chat agent with web + coding tools
│   ├── agent_ui.html         # Browser UI served by agent.py --ui
│   ├── chat.py               # Plain chat
│   └── host-setup.sh         # One-time host prerequisites
├── training-scripts/         # Legacy LoRA training pipeline
└── models/                   # Model weights (not in git)
```
