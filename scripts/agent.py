#!/usr/bin/env python3
"""Agent: chat with the local model, with web search and coding tools.

The model can search the web (SearXNG), read pages, and read/search/edit files
and run shell commands inside a workspace directory. Tools run on the machine
this script runs on; the model can be local or reached over an SSH tunnel.

Usage:
    python3 scripts/agent.py                        # terminal chat, workspace = this repo
    python3 scripts/agent.py --ui                   # browser UI
    python3 scripts/agent.py --ui --ssh user@host   # browser UI, model on another machine
    python3 scripts/agent.py --workspace ~/proj     # work on another project
    python3 scripts/agent.py --yes                  # don't ask before writes/commands

Stdlib only (Python 3.10+). On the server, start with ./vllm-server.sh start
"""

import argparse
import datetime
import hmac
import http.client
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = Path.home() / ".qwen-agent.json"
DEFAULT_URL = os.environ.get("LLM_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen")
DEFAULT_SEARX = os.environ.get("SEARXNG_URL", "http://localhost:8888")

WINDOWS = os.name == "nt"
SHELL = "PowerShell" if WINDOWS else "bash"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # don't flash consoles under pythonw
MAX_TOOL_OUTPUT = 40_000  # chars returned to the model per tool call
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"

DIM, BOLD, CYAN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m"

TOOLS = [
    {"name": "web_search",
     "description": "Search the web. Returns titles, URLs and snippets. Follow up with fetch_url to read a result.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"},
         "max_results": {"type": "integer", "description": "Default 8"}},
         "required": ["query"]}},
    {"name": "fetch_url",
     "description": "Fetch a web page and return its readable text. Use offset to continue a long page.",
     "parameters": {"type": "object", "properties": {
         "url": {"type": "string"},
         "offset": {"type": "integer", "description": "Character offset to start from. Default 0"}},
         "required": ["url"]}},
    {"name": "list_files",
     "description": "List files in the workspace (respects .gitignore). Optionally limit to a subdirectory.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Subdirectory relative to the workspace. Default '.'"}}}},
    {"name": "read_file",
     "description": "Read a text file from the workspace, with line numbers.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"},
         "offset": {"type": "integer", "description": "First line to read (1-based). Default 1"},
         "limit": {"type": "integer", "description": "Number of lines. Default 2000"}},
         "required": ["path"]}},
    {"name": "search_files",
     "description": "Search file contents in the workspace with a regular expression. Returns file:line:text.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"},
         "path": {"type": "string", "description": "Subdirectory or file to search. Default '.'"}},
         "required": ["pattern"]}},
    {"name": "write_file",
     "description": "Create or overwrite a file in the workspace with the given content.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path", "content"]}},
    {"name": "edit_file",
     "description": "Replace one exact occurrence of old_string with new_string in a workspace file. "
                    "old_string must match exactly and be unique in the file.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
         "required": ["path", "old_string", "new_string"]}},
    {"name": "run_command",
     "description": f"Run a {SHELL} command in the workspace directory and return its exit code and output. "
                    "Use for tests, builds, git, etc.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"},
         "timeout": {"type": "integer", "description": "Seconds. Default 120"}},
         "required": ["command"]}},
]
NEEDS_APPROVAL = {"write_file", "edit_file", "run_command"}


class ToolError(Exception):
    pass


class Cancelled(Exception):
    pass


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(**updates):
    config = {**load_config(), **updates}
    config = {k: v for k, v in config.items() if v is not None}
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if not WINDOWS:
        CONFIG_PATH.chmod(0o600)


def default_workspace() -> str:
    saved = load_config().get("workspace")
    if saved and Path(saved).is_dir():
        return saved
    if (REPO_ROOT / "vllm-server.sh").exists():
        return str(REPO_ROOT)
    return str(Path.cwd())


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "iframe"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "pre", "blockquote", "table", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self._skip, self._in_title = [], "", 0, False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return parser.title.strip(), text.strip()


class Tools:
    def __init__(self, workspace: Path, searx_url: str, browser: str | None):
        self.workspace = workspace.resolve()
        self.searx_url = searx_url.rstrip("/")
        self.browser = browser
        self.is_git = (self.workspace / ".git").exists() and shutil.which("git") is not None

    def _path(self, path: str) -> Path:
        p = (self.workspace / (path or ".")).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ToolError(f"path is outside the workspace: {path}")
        return p

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.workspace).as_posix() or "."

    def _walk(self, root: Path):
        """Yield files under root (or root itself), skipping hidden directories."""
        if root.is_file():
            yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                yield Path(dirpath) / name

    # -- web ---------------------------------------------------------------

    def web_search(self, query: str, max_results: int = 8) -> str:
        url = f"{self.searx_url}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise ToolError(f"search backend unreachable at {self.searx_url} ({e}); is SearXNG running?")
        results = data.get("results", [])[:max(1, min(int(max_results), 20))]
        if not results:
            failed = ", ".join(e[0] for e in data.get("unresponsive_engines", []))
            return "No results." + (f" Unresponsive engines: {failed}" if failed else "")
        return "\n\n".join(
            f"[{i}] {r.get('title', '')}\n{r.get('url', '')}\n{(r.get('content') or '').strip()}"
            for i, r in enumerate(results, 1))

    def fetch_url(self, url: str, offset: int = 0) -> str:
        if not re.match(r"https?://", url):
            raise ToolError("only http(s) URLs are supported")
        if self.browser:
            proc = subprocess.run(
                [self.browser, "--headless", "--disable-gpu", "--dump-dom", url],
                capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)
            if proc.returncode != 0 or not proc.stdout.strip():
                raise ToolError(f"browser failed: {proc.stderr.strip()[-500:]}")
            ctype, body = "text/html", proc.stdout
        else:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    ctype = resp.headers.get_content_type()
                    charset = resp.headers.get_content_charset() or "utf-8"
                    body = resp.read(5_000_000).decode(charset, errors="replace")
            except urllib.error.HTTPError as e:
                raise ToolError(f"HTTP {e.code} {e.reason}")
            except urllib.error.URLError as e:
                raise ToolError(f"fetch failed: {e.reason}")
        if "html" in ctype:
            title, text = html_to_text(body)
            text = f"# {title}\n\n{text}" if title else text
        elif ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
            text = body
        else:
            raise ToolError(f"unsupported content type: {ctype}")
        offset = max(0, int(offset))
        chunk = text[offset:offset + MAX_TOOL_OUTPUT]
        if offset + len(chunk) < len(text):
            chunk += f"\n\n[truncated: {len(text)} chars total, continue with offset={offset + len(chunk)}]"
        return chunk or "[empty page]"

    # -- files -------------------------------------------------------------

    def list_files(self, path: str = ".") -> str:
        root = self._path(path)
        if not root.is_dir():
            raise ToolError(f"not a directory: {path}")
        if self.is_git:
            out = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", self._rel(root)],
                cwd=self.workspace, capture_output=True, text=True, creationflags=NO_WINDOW).stdout
            files = sorted(set(out.splitlines()))
        else:
            files = []
            for p in self._walk(root):
                files.append(self._rel(p))
                if len(files) >= 5000:
                    files.append("[more files not shown]")
                    break
        return "\n".join(files) or "[no files]"

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            raise ToolError(f"not a UTF-8 text file: {path}")
        start = max(1, int(offset))
        chunk = lines[start - 1:start - 1 + max(1, int(limit))]
        out = "\n".join(f"{n:6d}\t{line}" for n, line in enumerate(chunk, start))
        end = start + len(chunk) - 1
        if end < len(lines):
            out += f"\n[showing lines {start}-{end} of {len(lines)}]"
        return out or "[empty file]"

    def search_files(self, pattern: str, path: str = ".") -> str:
        root = self._path(path)
        if self.is_git:
            proc = subprocess.run(
                ["git", "grep", "-nI", "--untracked", "-E", "-e", pattern, "--", self._rel(root)],
                cwd=self.workspace, capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)
            if proc.returncode > 1:
                raise ToolError(proc.stderr.strip())
            lines = proc.stdout.splitlines()
        else:
            try:
                regex = re.compile(pattern)
            except re.error as e:
                raise ToolError(f"invalid regex: {e}")
            lines = []
            for p in self._walk(root):
                try:
                    if p.stat().st_size > 2_000_000:
                        continue
                    text = p.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                lines += [f"{self._rel(p)}:{n}:{line}" for n, line in enumerate(text.splitlines(), 1)
                          if regex.search(line)]
                if len(lines) > 300:
                    break
        out = "\n".join(line[:300] for line in lines[:300])
        if len(lines) > 300:
            out += "\n[more matches not shown]"
        return out or "No matches."

    def write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {self._rel(p)}"

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        with open(p, encoding="utf-8", newline="") as f:
            text = f.read()
        if "\r\n" in text and "\r\n" not in old_string:  # keep the file's CRLF line endings
            old_string = old_string.replace("\n", "\r\n")
            new_string = new_string.replace("\n", "\r\n")
        count = text.count(old_string)
        if count != 1:
            raise ToolError(f"old_string matched {count} times; it must match exactly once")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace(old_string, new_string, 1))
        return f"Edited {self._rel(p)}"

    def run_command(self, command: str, timeout: int = 120) -> str:
        if WINDOWS:
            argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            argv = ["bash", "-c", command]
        try:
            proc = subprocess.run(argv, cwd=self.workspace, capture_output=True,
                                  text=True, errors="replace", timeout=max(1, int(timeout)),
                                  stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {timeout}s")
        return f"exit code: {proc.returncode}\n{proc.stdout}{proc.stderr}"


def clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} chars omitted ...]\n{text[-half:]}"


class Agent:
    """Model + tool loop. Output and approvals go through `ui` (TerminalUI or WebApp)."""

    def __init__(self, args, ui):
        self.args, self.ui = args, ui
        self.browser = None if args.no_browser else next(
            (b for b in ("brave-browser", "brave", "chromium", "chromium-browser") if shutil.which(b)), None)
        self.cancel = threading.Event()
        self._resp = None
        self._calls = 0
        self.set_workspace(Path(args.workspace))

    def set_workspace(self, workspace: Path):
        self.tools = Tools(workspace, self.args.searx_url, self.browser)
        self.reset()

    def reset(self):
        self.auto_approve = self.args.yes
        self.history = [{"role": "system", "content": self.system_prompt()}]

    def system_prompt(self) -> str:
        return (
            "You are a coding and research assistant running on the user's own machine.\n"
            f"Today is {datetime.date.today().isoformat()}. The machine runs {platform.system()}; "
            f"run_command uses {SHELL}.\n"
            f"Your workspace is {self.tools.workspace}; file tools take paths relative to it "
            "and run_command starts there.\n"
            "Use web_search and fetch_url for anything current or that you are unsure of, and cite the URLs you used.\n"
            "Before changing code, read the relevant files. Make focused edits with edit_file, "
            "then verify with run_command where possible.\n"
            "Be concise. Report what you did and what you could not verify.")

    def stop(self):
        """Abort the turn in progress (called from another thread)."""
        self.cancel.set()
        try:  # unblock a read that is waiting on the model
            self._resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

    def repair_history(self):
        """Give every tool call a result; the server rejects a history without them."""
        for i in range(len(self.history) - 1, 0, -1):
            message = self.history[i]
            if message["role"] == "user":
                return
            if message["role"] == "assistant":
                done = {m["tool_call_id"] for m in self.history[i + 1:]}
                for tc in message.get("tool_calls") or []:
                    if tc["id"] not in done:
                        self.history.append({"role": "tool", "tool_call_id": tc["id"],
                                             "content": "Error: cancelled by the user."})
                return

    def complete(self, payload: dict) -> dict:
        """POST a streaming chat completion; stream it to the UI; return the assistant message."""
        req = urllib.request.Request(f"{self.args.url}/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        content, calls = [], {}
        try:
            resp = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:1000]}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach the model server at {self.args.url}: {e.reason}")
        self._resp = resp
        try:
            with resp:
                for raw in resp:
                    if self.cancel.is_set():
                        raise Cancelled()
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(chunk["error"].get("message", str(chunk["error"])))
                    if not chunk.get("choices"):
                        continue
                    delta = chunk["choices"][0].get("delta") or {}
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                    if reasoning:
                        self.ui.reasoning(reasoning)
                    if delta.get("content"):
                        self.ui.content(delta["content"])
                        content.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        call = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        call["id"] = tc.get("id") or call["id"]
                        call["name"] += fn.get("name") or ""
                        call["arguments"] += fn.get("arguments") or ""
        except (OSError, ValueError, http.client.HTTPException):
            if self.cancel.is_set():
                raise Cancelled()
            raise
        finally:
            self._resp = None
            self.ui.message_end()
        if self.cancel.is_set():
            raise Cancelled()
        message = {"role": "assistant", "content": "".join(content)}
        if calls:
            message["tool_calls"] = [
                {"id": c["id"] or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for i, c in sorted(calls.items())]
        return message

    def call_tool(self, name: str, raw_args: str) -> str:
        self._calls += 1
        uid = f"t{self._calls}"
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            args, result = {"raw": raw_args}, f"Error: invalid tool arguments: {e}"
        else:
            result = None
        self.ui.tool_call(uid, name, args)
        if result is None:
            result = self._run_tool(uid, name, args)
        self.ui.tool_result(uid, result)
        return result

    def _run_tool(self, uid: str, name: str, args: dict) -> str:
        if name not in {t["name"] for t in TOOLS}:
            return f"Error: unknown tool {name}"
        if name in NEEDS_APPROVAL and not self.auto_approve:
            decision = self.ui.approve(uid, name)
            if decision == "always":
                self.auto_approve = True
            elif decision != "yes":
                return "Error: the user declined this action."
        try:
            return clip(str(getattr(self.tools, name)(**args)))
        except ToolError as e:
            return f"Error: {e}"
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def turn(self, user_input: str):
        self.cancel.clear()
        self.history.append({"role": "user", "content": user_input})
        for _ in range(self.args.max_steps):
            payload = {"model": self.args.model, "messages": self.history, "stream": True,
                       "tools": [{"type": "function", "function": t} for t in TOOLS]}
            if self.args.effort:
                payload["reasoning_effort"] = self.args.effort
            message = self.complete(payload)
            self.history.append(message)
            if not message.get("tool_calls"):
                return
            for tc in message["tool_calls"]:
                if self.cancel.is_set():
                    raise Cancelled()
                result = self.call_tool(tc["function"]["name"], tc["function"]["arguments"])
                self.history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        self.ui.notice(f"Stopped after {self.args.max_steps} tool steps; say 'continue' to go on.")


class TerminalUI:
    def __init__(self):
        self.thinking = self.wrote = False

    def reasoning(self, text):
        if not self.thinking:
            print(DIM, end="")
            self.thinking = True
        print(text, end="", flush=True)

    def content(self, text):
        if self.thinking:
            print(RESET + "\n")
            self.thinking = False
        print(text, end="", flush=True)
        self.wrote = True

    def message_end(self):
        if self.thinking:
            print(RESET)
        if self.wrote:
            print()
        self.thinking = self.wrote = False

    def tool_call(self, uid, name, args):
        preview = json.dumps(args, ensure_ascii=False)
        print(f"{CYAN}→ {name}{RESET} {preview if len(preview) <= 400 else preview[:400] + '…'}")

    def tool_result(self, uid, result):
        first = result.strip().splitlines()[0] if result.strip() else ""
        print(f"{DIM}  {first[:150]}{' …' if len(result) > len(first) else ''}{RESET}")

    def approve(self, uid, name) -> str:
        try:
            answer = input(f"{YELLOW}Allow {name}? [y]es / [n]o / [a]lways: {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return "no"
        return {"y": "yes", "a": "always"}.get(answer[:1], "no")

    def notice(self, text):
        print(f"{YELLOW}[{text}]{RESET}")


def run_terminal(args):
    agent = Agent(args, TerminalUI())
    print(f"Agent — model: {args.model}  workspace: {agent.tools.workspace}")
    print(f"Search: {agent.tools.searx_url}  page fetch: {agent.browser or 'direct HTTP'}")
    print("Commands: /clear resets the conversation, exit quits.\n")
    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == "/clear":
            agent.reset()
            print("[conversation cleared]")
            continue
        try:
            agent.turn(user_input)
        except (KeyboardInterrupt, Cancelled):
            print(f"{RESET}\n[interrupted]")
        except Exception as e:
            print(f"{RESET}\n[error: {e}]")
        agent.repair_history()
        print()


# -- browser UI ----------------------------------------------------------------

class WebApp:
    """Serves agent_ui.html and streams the conversation to it as server-sent events."""

    def __init__(self, args, tunnel_error: str | None):
        self.args = args
        self.tunnel_error = tunnel_error
        self.agent = Agent(args, self)
        self.token = secrets.token_urlsafe(24)
        self.cond = threading.Condition()
        self.events, self.generation = [], 0
        self.pending = {}  # approval uid -> [Event, decision]
        self.busy = False
        self.quit = threading.Event()

    def emit(self, **event):
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def info(self) -> dict:
        return {"model": self.args.model, "workspace": str(self.agent.tools.workspace),
                "shell": SHELL, "host": platform.node()}

    def status(self) -> dict:
        def reachable(url):
            try:
                with urllib.request.urlopen(url, timeout=3):
                    return True
            except (OSError, ValueError):
                return False
        return {"model": reachable(f"{self.args.url}/models"),
                "search": reachable(f"{self.args.searx_url.rstrip('/')}/healthz"),
                "tunnel_error": self.tunnel_error, "ssh": self.args.ssh}

    # UI interface used by Agent
    def reasoning(self, text):
        self.emit(type="reasoning", text=text)

    def content(self, text):
        self.emit(type="content", text=text)

    def message_end(self):
        self.emit(type="message_end")

    def tool_call(self, uid, name, args):
        self.emit(type="tool_call", id=uid, name=name, args=args)

    def tool_result(self, uid, result):
        self.emit(type="tool_result", id=uid, text=clip(result, 20_000))

    def notice(self, text):
        self.emit(type="notice", text=text)

    def approve(self, uid, name) -> str:
        waiter = [threading.Event(), "no"]
        self.pending[uid] = waiter
        self.emit(type="approval", id=uid)
        while not waiter[0].wait(0.2):
            if self.agent.cancel.is_set():
                break
        del self.pending[uid]
        self.emit(type="approval_done", id=uid, decision=waiter[1])
        return waiter[1]

    # actions from the browser
    def send(self, text: str) -> bool:
        with self.cond:
            if self.busy or not text.strip():
                return False
            self.busy = True
        threading.Thread(target=self._work, args=(text,), daemon=True).start()
        return True

    def _work(self, text: str):
        self.emit(type="user", text=text)
        self.emit(type="busy", value=True)
        try:
            self.agent.turn(text)
        except Cancelled:
            self.emit(type="notice", text="Stopped.")
        except Exception as e:
            self.emit(type="error", text=str(e))
        finally:
            self.agent.repair_history()
            self.busy = False
            self.emit(type="busy", value=False)

    def decide(self, uid: str, decision: str):
        waiter = self.pending.get(uid)
        if waiter and decision in ("yes", "no", "always"):
            waiter[1] = decision
            waiter[0].set()

    def clear(self, workspace: str | None = None) -> str | None:
        """Start a new conversation, optionally in another workspace. Returns an error message."""
        if self.busy:
            return "Stop the current reply first."
        if workspace is not None:
            path = Path(workspace.strip().strip('"')).expanduser()
            if not path.is_dir():
                return f"Not a folder: {path}"
            self.agent.set_workspace(path)
            save_config(workspace=str(self.agent.tools.workspace))
        else:
            self.agent.reset()
        with self.cond:
            self.events, self.generation = [], self.generation + 1
            self.cond.notify_all()
        return None


def make_handler(app: WebApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code: int, body, ctype="application/json"):
            data = (json.dumps(body) if ctype == "application/json" else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self, query: dict) -> bool:
            # The token stops other web pages in the browser from driving the agent.
            token = self.headers.get("X-Token") or query.get("token", [""])[0]
            return hmac.compare_digest(token.encode(), app.token.encode())

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if not self._authorized(urllib.parse.parse_qs(url.query)):
                return self._reply(403, "Open this page from the launcher (missing or wrong token).", "text/plain")
            if url.path == "/":
                self._reply(200, (SCRIPT_DIR / "agent_ui.html").read_text(encoding="utf-8"), "text/html")
            elif url.path == "/api/status":
                self._reply(200, app.status())
            elif url.path == "/api/events":
                self._events()
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized({}):
                return self._reply(403, {"error": "forbidden"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self._reply(400, {"error": "bad request"})
            path, error = self.path, None
            if path == "/api/send":
                error = None if app.send(str(body.get("text", ""))) else "Busy."
            elif path == "/api/approve":
                app.decide(str(body.get("id")), str(body.get("decision")))
            elif path == "/api/stop":
                app.agent.stop()
            elif path == "/api/clear":
                error = app.clear()
            elif path == "/api/workspace":
                error = app.clear(str(body.get("path", "")))
            elif path == "/api/quit":
                app.agent.stop()
                app.quit.set()
            else:
                return self._reply(404, {"error": "not found"})
            self._reply(409 if error else 200, {"error": error} if error else {"ok": True})

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            generation, sent = -1, 0
            try:
                while True:
                    with app.cond:
                        if generation == app.generation and sent >= len(app.events):
                            app.cond.wait(15)
                        if generation != app.generation:  # new connection or new conversation: replay
                            generation = app.generation
                            batch = [{"type": "reset", **app.info()}] + app.events
                        else:
                            batch = app.events[sent:]
                        sent = len(app.events)
                    data = "".join(f"data: {json.dumps(e)}\n\n" for e in batch) or ": keepalive\n\n"
                    self.wfile.write(data.encode())
                    self.wfile.flush()
            except OSError:
                pass

    return Handler


def run_web(args, tunnel_error: str | None):
    app = WebApp(args, tunnel_error)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/?token={app.token}"
    save_config(instance={"url": url})
    if sys.stdout:
        print(f"Agent UI: {url}\nCtrl-C to quit.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    threading.Thread(target=lambda: (app.quit.wait(), server.shutdown()), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        save_config(instance=None)


def reopen_running_ui(args) -> bool:
    """If a UI is already running, show it instead of starting a second one."""
    url = (load_config().get("instance") or {}).get("url")
    if not url:
        return False
    try:
        with urllib.request.urlopen(url.replace("/?", "/api/status?"), timeout=5):
            pass
    except (OSError, ValueError):
        return False
    if not args.no_open:
        webbrowser.open(url)
    return True


# -- SSH tunnel ----------------------------------------------------------------

def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def open_tunnel(target: str, ports: list[int]):
    """Forward local ports to the same ports on `target`. Returns (process, error)."""
    ports = [p for p in ports if not port_open(p)]
    if not ports:
        return None, None
    cmd = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30"]
    for port in ports:
        cmd += ["-L", f"{port}:localhost:{port}"]
    try:
        proc = subprocess.Popen(cmd + [target], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, creationflags=NO_WINDOW)
    except OSError as e:
        return None, f"could not run ssh: {e}"
    for _ in range(60):
        if proc.poll() is not None:
            return None, f"ssh {target} failed: {proc.stderr.read().strip() or 'exit code %d' % proc.returncode}"
        if all(port_open(p) for p in ports):
            return proc, None
        threading.Event().wait(0.25)
    proc.terminate()
    return None, f"ssh {target}: timed out opening the tunnel"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ui", action="store_true", help="Open the browser UI instead of the terminal chat")
    parser.add_argument("--ssh", metavar="USER@HOST",
                        help="Reach the model and search servers on another machine through an SSH tunnel")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"OpenAI-compatible base URL (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Served model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--workspace",
                        help="Directory the file and command tools operate in "
                             "(default: the one last chosen in the UI, else this repo, else the current directory)")
    parser.add_argument("--searx-url", default=DEFAULT_SEARX, help=f"SearXNG base URL (default: {DEFAULT_SEARX})")
    parser.add_argument("--effort", choices=["low", "medium", "xhigh"],
                        help="Reasoning effort (model default is xhigh)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max tool-call rounds per user turn")
    parser.add_argument("--yes", action="store_true", help="Don't ask before writing files or running commands")
    parser.add_argument("--no-browser", action="store_true",
                        help="Fetch pages over plain HTTP even if a headless Brave/Chromium is installed")
    parser.add_argument("--port", type=int, default=8765, help="Port for the browser UI (default: 8765)")
    parser.add_argument("--no-open", action="store_true", help="With --ui, don't open a browser window")
    args = parser.parse_args()
    args.workspace = args.workspace or default_workspace()
    if not Path(args.workspace).is_dir():
        sys.exit(f"workspace is not a directory: {args.workspace}")
    if args.ui and reopen_running_ui(args):
        return

    tunnel, tunnel_error = None, None
    if args.ssh:
        ports = [urllib.parse.urlparse(u).port for u in (args.url, args.searx_url)]
        tunnel, tunnel_error = open_tunnel(args.ssh, [p for p in ports if p])
    try:
        if args.ui:
            run_web(args, tunnel_error)
        else:
            if tunnel_error:
                print(f"{YELLOW}[{tunnel_error}]{RESET}")
            run_terminal(args)
    finally:
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    main()
