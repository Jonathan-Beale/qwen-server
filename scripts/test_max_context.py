#!/usr/bin/env python3
"""
Test maximum context length for the tool-calling model.

This script:
1. Fetches a Wikipedia page
2. Anonymizes the subject (replaces name with "this individual/person/subject")
3. Uses vLLM's /tokenize endpoint to count tokens
4. Trims excerpt to max_model_len - 100 tokens
5. Asks the model to identify the subject
"""

import argparse
import json
import re
import requests
import sys
import time

# System prompt from training (modified to not use web search)
SYSTEM_PROMPT = """You are a coding assistant with access to tools for file operations and command execution.

## Tool Usage

When you need to perform file or command operations, use the following XML format:

```xml
<tool_call>
<tool_name>TOOL_NAME</tool_name>
<parameters>
<param_name>value</param_name>
</parameters>
</tool_call>
```

Available tools:
- **bash**: Execute shell commands. Parameters: command
- **read_file**: Read file contents. Parameters: path
- **write_file**: Create/overwrite files. Parameters: path, content
- **edit_file**: Edit existing files. Parameters: path, find, replace
- **list_files**: List directory contents. Parameters: path, pattern
- **search_files**: Search for text in files. Parameters: pattern, path
- **summarize**: Create a context summary. Parameters: content

## Guidelines

1. **Break down complex tasks** into steps and announce what you're doing
2. **Use tools for file/command operations**, answer directly for questions
3. **Handle errors gracefully** - if something fails, investigate and fix
4. **Write production-quality code** - no placeholders or TODOs
5. **After creating files**, summarize what was created

## When to use tools vs answer directly

**Use tools for:**
- File operations: read, write, edit, list, search
- Commands: git, pytest, pip, npm, docker, etc.
- Creating projects or code files

**Answer directly for:**
- Explanations and definitions
- Math calculations
- General knowledge questions
- Code review comments

## IMPORTANT: Do NOT use any web search, external lookups, or internet access.
Simply answer based on the information provided in the conversation."""


def fetch_wikipedia_page(title: str) -> tuple[str, str]:
    """Fetch Wikipedia page content and return (subject_name, text)."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "format": "json"
    }
    headers = {
        "User-Agent": "ContextLengthTest/1.0 (testing tool; contact@example.com)"
    }

    response = requests.get(url, params=params, headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Wikipedia API returned {response.status_code}: {response.text[:200]}")

    try:
        data = response.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse Wikipedia response: {e}\nResponse: {response.text[:500]}")

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    if "extract" not in page:
        raise ValueError(f"Could not fetch page: {title}")

    return title, page["extract"]


def anonymize_text(text: str, subject_name: str) -> str:
    """Replace subject name with anonymized references."""
    # Split name into parts for comprehensive replacement
    name_parts = subject_name.split()

    # Replace full name first
    text = re.sub(re.escape(subject_name), "this individual", text, flags=re.IGNORECASE)

    # Replace first name and last name separately
    if len(name_parts) >= 2:
        first_name = name_parts[0]
        last_name = name_parts[-1]

        # Replace standalone first/last name (with word boundaries)
        text = re.sub(r'\b' + re.escape(first_name) + r'\b', "the subject", text, flags=re.IGNORECASE)
        text = re.sub(r'\b' + re.escape(last_name) + r'\b', "the person", text, flags=re.IGNORECASE)

    # Also replace possessive forms
    text = re.sub(r"this individual's", "their", text, flags=re.IGNORECASE)
    text = re.sub(r"the subject's", "their", text, flags=re.IGNORECASE)
    text = re.sub(r"the person's", "their", text, flags=re.IGNORECASE)

    return text


def count_tokens(text: str, model: str = "tool-calling", base_url: str = "http://localhost:8000") -> int:
    """Use vLLM's tokenize endpoint to count tokens."""
    url = f"{base_url}/tokenize"

    response = requests.post(url, json={
        "model": model,
        "prompt": text
    })

    if response.status_code != 200:
        raise RuntimeError(f"Tokenize failed: {response.status_code} - {response.text}")

    data = response.json()
    return data["count"]


def trim_to_tokens(text: str, target_tokens: int, model: str = "tool-calling",
                   base_url: str = "http://localhost:8000") -> str:
    """Trim text to approximately target_tokens using binary search."""
    current_tokens = count_tokens(text, model, base_url)

    if current_tokens <= target_tokens:
        return text

    # Binary search to find the right length
    words = text.split()
    low, high = 0, len(words)

    while low < high - 1:
        mid = (low + high) // 2
        trimmed = " ".join(words[:mid])
        tokens = count_tokens(trimmed, model, base_url)

        if tokens <= target_tokens:
            low = mid
        else:
            high = mid

    return " ".join(words[:low])


def build_prompt(system_prompt: str, excerpt: str) -> str:
    """Build the full prompt in Qwen chat format."""
    user_message = f"""Read the following excerpt and identify who it is about.

Excerpt:
{excerpt}

Based on the excerpt above, identify the subject of this text. Who is this individual?"""

    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""


def query_model(prompt: str, model: str = "tool-calling",
                base_url: str = "http://localhost:8000", max_tokens: int = 100) -> str:
    """Query the model and return the response."""
    url = f"{base_url}/v1/completions"

    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stop": ["<|im_end|>"]
    })

    if response.status_code != 200:
        raise RuntimeError(f"Query failed: {response.status_code} - {response.text}")

    data = response.json()
    return data["choices"][0]["text"].strip()


def test_context_length(max_model_len: int, wiki_title: str = "Albert Einstein",
                        base_url: str = "http://localhost:8000") -> dict:
    """Test if the model can handle a context of max_model_len - 100 tokens."""

    print(f"\n{'='*60}")
    print(f"Testing max_model_len = {max_model_len}")
    print(f"{'='*60}")

    # Fetch and anonymize Wikipedia content
    print(f"Fetching Wikipedia page: {wiki_title}")
    subject_name, raw_text = fetch_wikipedia_page(wiki_title)

    print(f"Anonymizing text (replacing '{subject_name}')")
    anonymized_text = anonymize_text(raw_text, subject_name)

    # Build the base prompt without excerpt to measure overhead
    base_prompt = build_prompt(SYSTEM_PROMPT, "")
    base_tokens = count_tokens(base_prompt, base_url=base_url)
    print(f"Base prompt tokens (system + user template): {base_tokens}")

    # Calculate target excerpt tokens
    # Leave room for: base_prompt + excerpt + 100 tokens for response + 5 buffer
    target_excerpt_tokens = max_model_len - base_tokens - 105
    print(f"Target excerpt tokens: {target_excerpt_tokens}")

    if target_excerpt_tokens <= 0:
        print(f"ERROR: max_model_len {max_model_len} too small for prompt overhead")
        return {"success": False, "error": "max_model_len too small", "max_model_len": max_model_len}

    # Trim excerpt to target length
    print("Trimming excerpt to target token count...")
    trimmed_excerpt = trim_to_tokens(anonymized_text, target_excerpt_tokens, base_url=base_url)

    # Build final prompt
    final_prompt = build_prompt(SYSTEM_PROMPT, trimmed_excerpt)
    final_tokens = count_tokens(final_prompt, base_url=base_url)
    print(f"Final prompt tokens: {final_tokens}")
    print(f"Tokens remaining for response: {max_model_len - final_tokens}")

    # Query the model
    print("\nQuerying model...")
    start_time = time.time()

    try:
        response = query_model(final_prompt, base_url=base_url)
        elapsed = time.time() - start_time

        print(f"\nModel response ({elapsed:.2f}s):")
        print("-" * 40)
        print(response)
        print("-" * 40)

        # Check if the model correctly identified the subject
        subject_lower = subject_name.lower()
        response_lower = response.lower()
        correct = subject_lower in response_lower or any(
            part.lower() in response_lower for part in subject_name.split()
        )

        print(f"\nCorrect identification: {'YES' if correct else 'NO'}")

        return {
            "success": True,
            "max_model_len": max_model_len,
            "prompt_tokens": final_tokens,
            "response": response,
            "correct": correct,
            "elapsed_seconds": elapsed
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        return {
            "success": False,
            "max_model_len": max_model_len,
            "prompt_tokens": final_tokens,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Test maximum context length")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max model length to test")
    parser.add_argument("--wiki-title", type=str, default="Albert Einstein",
                        help="Wikipedia page to use for testing")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000",
                        help="vLLM server base URL")
    parser.add_argument("--test-multiple", action="store_true",
                        help="Test multiple max-model-len values (requires restarting server)")
    args = parser.parse_args()

    if args.test_multiple:
        print("Multi-test mode requires restarting the server with different configs.")
        print("Use --max-model-len to test a single configuration.")
        sys.exit(1)

    result = test_context_length(
        max_model_len=args.max_model_len,
        wiki_title=args.wiki_title,
        base_url=args.base_url
    )

    print(f"\n{'='*60}")
    print("RESULT SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
