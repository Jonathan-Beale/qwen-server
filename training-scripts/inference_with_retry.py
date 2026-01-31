#!/usr/bin/env python3
"""
Inference wrapper with automatic retry on format errors.

Features:
- Validates tool call XML format
- Retries with different seeds on format errors
- Collects failed examples for future training corrections
- Can be used standalone or integrated with vLLM server
"""

import json
import re
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

# =============================================================================
# VALIDATION
# =============================================================================

# Correct format patterns
VALID_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*<tool_name>(\w+)</tool_name>\s*<parameters>(.*?)</parameters>\s*</tool_call>',
    re.DOTALL
)

# Invalid format patterns to detect
INVALID_PATTERNS = [
    (re.compile(r'<api_call>', re.IGNORECASE), "api_call_format"),
    (re.compile(r'<function_call', re.IGNORECASE), "function_call_format"),
    (re.compile(r'<client>', re.IGNORECASE), "client_format"),
    (re.compile(r'<invoke\s', re.IGNORECASE), "invoke_format"),
    (re.compile(r'\w+\([^)]*\)\s*$', re.MULTILINE), "function_syntax"),  # func() syntax
    (re.compile(r'```\s*\n\s*<tool', re.IGNORECASE), "tool_in_codeblock"),
]

# Valid tools
VALID_TOOLS = {"bash", "read_file", "write_file", "edit_file", "list_files", "search_files", "summarize"}


@dataclass
class ValidationResult:
    """Result of validating a response."""
    is_valid: bool
    has_tool_call: bool
    tool_name: Optional[str] = None
    error_type: Optional[str] = None
    error_detail: Optional[str] = None


def validate_response(response: str) -> ValidationResult:
    """
    Validate that a response uses correct tool call format.

    Returns ValidationResult with details about any issues found.
    """
    # Check for invalid patterns first
    for pattern, error_type in INVALID_PATTERNS:
        if pattern.search(response):
            return ValidationResult(
                is_valid=False,
                has_tool_call=True,
                error_type=error_type,
                error_detail=f"Found invalid pattern: {error_type}"
            )

    # Check if response contains a tool call
    if "<tool_call>" not in response:
        # No tool call - this is valid (direct answer)
        return ValidationResult(is_valid=True, has_tool_call=False)

    # Validate tool call structure
    match = VALID_TOOL_CALL_PATTERN.search(response)
    if not match:
        return ValidationResult(
            is_valid=False,
            has_tool_call=True,
            error_type="malformed_xml",
            error_detail="Tool call XML is malformed or incomplete"
        )

    tool_name = match.group(1)

    # Check if tool name is valid
    if tool_name not in VALID_TOOLS:
        return ValidationResult(
            is_valid=False,
            has_tool_call=True,
            tool_name=tool_name,
            error_type="unknown_tool",
            error_detail=f"Unknown tool: {tool_name}"
        )

    # Validate parameters based on tool
    params_content = match.group(2)

    if tool_name == "bash":
        # Accept either <command>...</command> or bare command in parameters
        has_command_tag = "<command>" in params_content
        has_bare_command = params_content.strip() and not params_content.strip().startswith("<")

        if not has_command_tag and not has_bare_command:
            return ValidationResult(
                is_valid=False,
                has_tool_call=True,
                tool_name=tool_name,
                error_type="missing_param",
                error_detail="bash tool missing command"
            )

    if tool_name == "read_file" and "<path>" not in params_content:
        return ValidationResult(
            is_valid=False,
            has_tool_call=True,
            tool_name=tool_name,
            error_type="missing_param",
            error_detail="read_file tool missing <path> parameter"
        )

    if tool_name == "write_file":
        if "<path>" not in params_content:
            return ValidationResult(
                is_valid=False,
                has_tool_call=True,
                tool_name=tool_name,
                error_type="missing_param",
                error_detail="write_file tool missing <path> parameter"
            )
        if "<content>" not in params_content:
            return ValidationResult(
                is_valid=False,
                has_tool_call=True,
                tool_name=tool_name,
                error_type="missing_param",
                error_detail="write_file tool missing <content> parameter"
            )

    # All checks passed
    return ValidationResult(
        is_valid=True,
        has_tool_call=True,
        tool_name=tool_name
    )


# =============================================================================
# CORRECTIONS COLLECTOR
# =============================================================================

class CorrectionsCollector:
    """Collects failed examples for future training."""

    def __init__(self, output_path: str = "/training-data/corrections.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_corrections = []

    def add_correction(
        self,
        prompt: str,
        failed_response: str,
        corrected_response: Optional[str],
        error_type: str,
        error_detail: str,
        attempt_number: int,
        seed_used: int
    ):
        """Add a correction example."""
        correction = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "failed_response": failed_response,
            "corrected_response": corrected_response,
            "error_type": error_type,
            "error_detail": error_detail,
            "attempt_number": attempt_number,
            "seed_used": seed_used,
        }
        self.session_corrections.append(correction)

        # Append to file immediately
        with open(self.output_path, "a") as f:
            f.write(json.dumps(correction) + "\n")

    def get_training_examples(self) -> List[Dict[str, Any]]:
        """
        Convert collected corrections to training format.

        Creates examples that show the correct format given the same prompt.
        """
        training_examples = []

        for corr in self.session_corrections:
            if corr["corrected_response"]:
                training_examples.append({
                    "type": "single_turn",
                    "category": "correction",
                    "instruction": corr["prompt"],
                    "output": corr["corrected_response"],
                    "metadata": {
                        "original_error": corr["error_type"],
                        "was_correction": True
                    }
                })

        return training_examples

    def save_training_examples(self, output_path: str = "/training-data/corrections_training.jsonl"):
        """Save corrections in training format."""
        examples = self.get_training_examples()
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        return len(examples)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected corrections."""
        if not self.session_corrections:
            return {"total": 0}

        error_types = {}
        for corr in self.session_corrections:
            et = corr["error_type"]
            error_types[et] = error_types.get(et, 0) + 1

        successful_corrections = sum(
            1 for c in self.session_corrections if c["corrected_response"]
        )

        return {
            "total": len(self.session_corrections),
            "successful_corrections": successful_corrections,
            "error_types": error_types
        }


# =============================================================================
# RETRY INFERENCE
# =============================================================================

class RetryInference:
    """
    Wrapper for model inference with automatic retry on format errors.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_retries: int = 3,
        collect_corrections: bool = True,
        corrections_path: str = "/training-data/corrections.jsonl",
        verbose: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self.verbose = verbose

        self.collector = CorrectionsCollector(corrections_path) if collect_corrections else None

        # Track stats
        self.total_requests = 0
        self.retry_counts = {i: 0 for i in range(max_retries + 1)}
        self.format_errors = 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        base_seed: Optional[int] = None,
        **generate_kwargs
    ) -> Tuple[str, ValidationResult, int]:
        """
        Generate response with automatic retry on format errors.

        Returns:
            Tuple of (response, validation_result, attempts_used)
        """
        import torch

        self.total_requests += 1

        if base_seed is None:
            base_seed = random.randint(0, 2**32 - 1)

        attempts = []

        for attempt in range(self.max_retries + 1):
            # Use different seed for each attempt
            seed = base_seed + attempt * 1000
            torch.manual_seed(seed)

            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                # Vary temperature slightly on retries
                temperature = 0.1 + (attempt * 0.1) if attempt > 0 else 0.0
                do_sample = attempt > 0  # Greedy on first attempt, sample on retries

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generate_kwargs
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Validate
            validation = validate_response(response)
            attempts.append((response, validation, seed))

            if validation.is_valid:
                self.retry_counts[attempt] += 1

                # If this wasn't the first attempt, we have a correction
                if attempt > 0 and self.collector:
                    first_response, first_validation, first_seed = attempts[0]
                    self.collector.add_correction(
                        prompt=prompt,
                        failed_response=first_response,
                        corrected_response=response,
                        error_type=first_validation.error_type,
                        error_detail=first_validation.error_detail,
                        attempt_number=attempt,
                        seed_used=seed
                    )

                if self.verbose and attempt > 0:
                    print(f"  [Retry {attempt}] Fixed format error: {attempts[0][1].error_type}")

                return response, validation, attempt + 1

            if self.verbose:
                print(f"  [Attempt {attempt + 1}] Format error: {validation.error_type}")

        # All retries failed
        self.format_errors += 1
        self.retry_counts[self.max_retries] += 1

        # Log the failure
        if self.collector:
            first_response, first_validation, first_seed = attempts[0]
            self.collector.add_correction(
                prompt=prompt,
                failed_response=first_response,
                corrected_response=None,  # No correction found
                error_type=first_validation.error_type,
                error_detail=first_validation.error_detail,
                attempt_number=self.max_retries + 1,
                seed_used=first_seed
            )

        # Return the first attempt (or could return best attempt)
        return attempts[0][0], attempts[0][1], self.max_retries + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = {
            "total_requests": self.total_requests,
            "format_errors_unrecoverable": self.format_errors,
            "retry_distribution": self.retry_counts,
            "success_rate": (self.total_requests - self.format_errors) / max(self.total_requests, 1) * 100
        }

        if self.collector:
            stats["corrections"] = self.collector.get_stats()

        return stats


# =============================================================================
# STANDALONE TESTING
# =============================================================================

def test_validation():
    """Test the validation function."""
    test_cases = [
        # Valid cases
        ("<tool_call>\n<tool_name>bash</tool_name>\n<parameters>\n<command>git status</command>\n</parameters>\n</tool_call>", True),
        ("The answer is 42", True),  # Direct answer, no tool
        ("<tool_call>\n<tool_name>read_file</tool_name>\n<parameters>\n<path>main.py</path>\n</parameters>\n</tool_call>", True),

        # Invalid cases
        ("<api_call><name>bash</name></api_call>", False),
        ("<function_call name='bash'>git status</function_call>", False),
        ("bash(command='git status')", False),
        ("<tool_call><tool_name>bash</tool_name><parameters></parameters></tool_call>", False),  # Missing command
    ]

    print("Testing validation function...")
    for response, expected_valid in test_cases:
        result = validate_response(response)
        status = "✓" if result.is_valid == expected_valid else "✗"
        print(f"  {status} Valid={result.is_valid} (expected {expected_valid})")
        if not result.is_valid:
            print(f"      Error: {result.error_type} - {result.error_detail}")


def main():
    """Demo/test the retry inference system."""
    import argparse

    parser = argparse.ArgumentParser(description="Test retry inference")
    parser.add_argument('--test-validation', action='store_true', help="Test validation only")
    parser.add_argument('--adapter_path', type=str,
                       default='/lora-adapters/tool-calling-20260128-204910/checkpoint-400')
    parser.add_argument('--base_model', type=str,
                       default='/root/qwen-server/models/.cache/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c')
    args = parser.parse_args()

    if args.test_validation:
        test_validation()
        return

    # Load model
    print("Loading model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Create retry inference wrapper
    inference = RetryInference(
        model=model,
        tokenizer=tokenizer,
        max_retries=3,
        collect_corrections=True,
        verbose=True
    )

    # Test prompts
    SYSTEM_PROMPT = """You are a coding assistant with access to tools.

Use this XML format for tool calls:
<tool_call>
<tool_name>TOOL_NAME</tool_name>
<parameters>
<param>value</param>
</parameters>
</tool_call>

Available tools: bash, read_file, write_file, edit_file"""

    test_prompts = [
        "Run git status",
        "Create hello.py that prints hello",
        "What is Python?",
        "Run pytest && git add .",
    ]

    print("\n" + "="*60)
    print("Testing Retry Inference")
    print("="*60)

    for prompt in test_prompts:
        full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        print(f"\n→ {prompt}")
        response, validation, attempts = inference.generate(full_prompt)

        print(f"  Attempts: {attempts}, Valid: {validation.is_valid}")
        print(f"  Response: {response[:100]}...")

    # Print stats
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    stats = inference.get_stats()
    print(json.dumps(stats, indent=2))

    # Save corrections for training
    if inference.collector:
        count = inference.collector.save_training_examples()
        print(f"\nSaved {count} correction examples for training")


if __name__ == "__main__":
    main()
