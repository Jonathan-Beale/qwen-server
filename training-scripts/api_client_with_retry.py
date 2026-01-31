#!/usr/bin/env python3
"""
API client with automatic retry on format errors.

Works with vLLM server or any OpenAI-compatible API.
Collects corrections for future training.
"""

import json
import time
import random
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from inference_with_retry import validate_response, ValidationResult, CorrectionsCollector


class APIClientWithRetry:
    """
    OpenAI-compatible API client with retry on format errors.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "qwen2-7b-instruct",
        max_retries: int = 3,
        collect_corrections: bool = True,
        corrections_path: str = "/training-data/corrections.jsonl",
        verbose: bool = True
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.verbose = verbose

        self.collector = CorrectionsCollector(corrections_path) if collect_corrections else None

        # Stats
        self.total_requests = 0
        self.retries_needed = 0
        self.unrecoverable_errors = 0

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        base_seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, ValidationResult, int]:
        """
        Send chat completion request with retry on format errors.

        Args:
            messages: List of {"role": str, "content": str} dicts
            max_tokens: Maximum tokens to generate
            base_seed: Base random seed (varied on retries)

        Returns:
            Tuple of (response_text, validation_result, attempts_used)
        """
        self.total_requests += 1

        if base_seed is None:
            base_seed = random.randint(0, 2**32 - 1)

        # Extract user prompt for logging
        user_prompt = next(
            (m["content"] for m in messages if m["role"] == "user"),
            str(messages)
        )

        attempts = []

        for attempt in range(self.max_retries + 1):
            seed = base_seed + attempt * 1000

            # Vary temperature on retries
            temperature = 0.0 if attempt == 0 else 0.1 + (attempt * 0.15)

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                **kwargs
            }

            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()

                content = result["choices"][0]["message"]["content"]

            except Exception as e:
                if self.verbose:
                    print(f"  [Attempt {attempt + 1}] API error: {e}")
                continue

            # Validate response
            validation = validate_response(content)
            attempts.append((content, validation, seed))

            if validation.is_valid:
                # Success!
                if attempt > 0:
                    self.retries_needed += 1

                    # Log correction
                    if self.collector:
                        first_content, first_validation, first_seed = attempts[0]
                        self.collector.add_correction(
                            prompt=user_prompt,
                            failed_response=first_content,
                            corrected_response=content,
                            error_type=first_validation.error_type or "unknown",
                            error_detail=first_validation.error_detail or "unknown",
                            attempt_number=attempt,
                            seed_used=seed
                        )

                    if self.verbose:
                        print(f"  [Retry {attempt}] Fixed: {attempts[0][1].error_type}")

                return content, validation, attempt + 1

            if self.verbose:
                print(f"  [Attempt {attempt + 1}] Format error: {validation.error_type}")

        # All retries failed
        self.unrecoverable_errors += 1

        if self.collector and attempts:
            first_content, first_validation, first_seed = attempts[0]
            self.collector.add_correction(
                prompt=user_prompt,
                failed_response=first_content,
                corrected_response=None,
                error_type=first_validation.error_type or "unknown",
                error_detail=first_validation.error_detail or "unknown",
                attempt_number=self.max_retries + 1,
                seed_used=first_seed
            )

        if attempts:
            return attempts[0][0], attempts[0][1], self.max_retries + 1
        else:
            return "", ValidationResult(is_valid=False, has_tool_call=False, error_type="api_error"), self.max_retries + 1

    def simple_query(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple query interface.

        Args:
            user_message: The user's message
            system_prompt: Optional system prompt

        Returns:
            Response text (retried if needed)
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_message})

        response, validation, attempts = self.chat_completion(messages)
        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = {
            "total_requests": self.total_requests,
            "retries_needed": self.retries_needed,
            "unrecoverable_errors": self.unrecoverable_errors,
            "success_rate": (self.total_requests - self.unrecoverable_errors) / max(self.total_requests, 1) * 100,
            "retry_rate": self.retries_needed / max(self.total_requests, 1) * 100
        }

        if self.collector:
            stats["corrections"] = self.collector.get_stats()

        return stats


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a coding assistant with access to tools for file operations and command execution.

## Tool Format

When you need to use a tool, use this exact XML format:

<tool_call>
<tool_name>TOOL_NAME</tool_name>
<parameters>
<param_name>value</param_name>
</parameters>
</tool_call>

## Available Tools

- **bash**: Execute shell commands
  Parameters: <command>your command here</command>

- **read_file**: Read file contents
  Parameters: <path>file/path</path>

- **write_file**: Create or overwrite files
  Parameters: <path>file/path</path>, <content>file content</content>

- **edit_file**: Edit existing files
  Parameters: <path>file/path</path>, <find>text to find</find>, <replace>replacement</replace>

## Guidelines

1. Use tools for file operations and commands
2. Answer directly for questions (no tools needed)
3. Break down complex tasks into steps
4. Write production-quality code (no placeholders)"""


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="API client with retry")
    parser.add_argument('--url', type=str, default="http://localhost:8000/v1",
                       help="API base URL")
    parser.add_argument('--model', type=str, default="qwen2-7b-instruct",
                       help="Model name")
    parser.add_argument('--query', type=str, help="Single query to run")
    parser.add_argument('--interactive', action='store_true', help="Interactive mode")
    parser.add_argument('--test', action='store_true', help="Run test suite")
    args = parser.parse_args()

    client = APIClientWithRetry(
        base_url=args.url,
        model=args.model,
        max_retries=3,
        verbose=True
    )

    if args.query:
        print(f"Query: {args.query}\n")
        response = client.simple_query(args.query, DEFAULT_SYSTEM_PROMPT)
        print(f"Response:\n{response}")
        print(f"\nStats: {json.dumps(client.get_stats(), indent=2)}")

    elif args.test:
        test_queries = [
            "Run git status",
            "Read main.py",
            "Create hello.py that prints hello world",
            "What is Python?",
            "Run git add . && git commit -m 'update'",
        ]

        print("Running test queries...\n")
        for query in test_queries:
            print(f"→ {query}")
            response, validation, attempts = client.chat_completion(
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ]
            )
            status = "✓" if validation.is_valid else "✗"
            print(f"  {status} Attempts: {attempts}, Valid: {validation.is_valid}")
            print(f"  Response: {response[:100]}...\n")

        print("\n" + "="*60)
        print("Stats:")
        print(json.dumps(client.get_stats(), indent=2))

        if client.collector:
            count = client.collector.save_training_examples()
            print(f"\nSaved {count} corrections for training")

    elif args.interactive:
        print("Interactive mode (type 'quit' to exit, 'stats' for statistics)\n")

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                continue
            if query.lower() == 'quit':
                break
            if query.lower() == 'stats':
                print(json.dumps(client.get_stats(), indent=2))
                continue

            response = client.simple_query(query, DEFAULT_SYSTEM_PROMPT)
            print(f"\nAssistant: {response}\n")

        # Save corrections on exit
        if client.collector:
            count = client.collector.save_training_examples()
            print(f"\nSaved {count} corrections for training")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
