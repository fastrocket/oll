#!/usr/bin/env python3
"""
Simple Ollama client for interacting with a locally running Ollama server.

Usage example:
    python ollama_client.py --prompt "Hello!" --temperature 0.7
    python ollama_client.py --coder --prompt "write a function to add two ints"
"""

import argparse
import ast
import json
import sys
from typing import Any, Dict, Optional, Tuple

import subprocess
from subprocess import TimeoutExpired

import requests

CODER_MODEL = "qwen2.5-coder:14b"
CODER_SYSTEM_PROMPT = (
    "You are the world's sharpest Python code generator. "
    "Respond with valid, ready-to-run Python source code only. "
    "Do not include explanations, comments, markdown, or backticks—only the code."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to a local Ollama model and print the response."
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Base URL for the Ollama server (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="dolphin3:latest",
        help="Name of the Ollama model to use (e.g., 'llama3', 'phi3', 'mistral').",
    )
    parser.add_argument(
        "--prompt",
        default="tell me a funny joke",
        help="Prompt text to send to the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; higher values increase randomness.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Restrict sampling to the top-k most likely tokens.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold; consider tokens whose cumulative probability is below this value.",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Penalty for repeated tokens (1.0 disables the penalty).",
    )
    parser.add_argument(
        "--repeat-last-n",
        type=int,
        default=64,
        help="How many previous tokens to consider when applying the repeat penalty.",
    )
    parser.add_argument(
        "--mirostat",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Use Mirostat sampling (0 disables, 1 original algorithm, 2 alternate).",
    )
    parser.add_argument(
        "--mirostat-eta",
        type=float,
        default=0.1,
        help="Learning rate for Mirostat sampling.",
    )
    parser.add_argument(
        "--mirostat-tau",
        type=float,
        default=5.0,
        help="Target surprise for Mirostat sampling.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON response instead of only the model output.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for both the HTTP request and optional execution.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to send with the request.",
    )
    parser.add_argument(
        "--coder",
        action="store_true",
        help=(
            f"Shortcut for --model {CODER_MODEL} with a system prompt that enforces pure Python output."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the returned Python code in a simple sandbox (use with caution).",
    )
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repeat_penalty": args.repeat_penalty,
        "repeat_last_n": args.repeat_last_n,
        "mirostat": args.mirostat,
        "mirostat_eta": args.mirostat_eta,
        "mirostat_tau": args.mirostat_tau,
        "num_predict": args.num_predict,
    }
    return options


def generate(
    host: str,
    model: str,
    prompt: str,
    options: Dict[str, Any],
    timeout: float,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": options,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_code_block(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence
    lines = lines[1:]
    # Remove optional language hint from opening fence handled above
    if lines and lines[0].lower().startswith("python"):
        lines = lines[1:]
    # Trim closing fence if present
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def ensure_pure_python(text: str) -> Tuple[str, Optional[str]]:
    code = _extract_code_block(text)
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return code, f"Generated code failed to parse as Python: {exc}"
    if "```" in code or code.strip().startswith("#"):
        return code, "Generated output may contain non-code elements."
    return code, None


def execute_python(code: str, timeout: float) -> None:
    """Execute code in an isolated Python subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except TimeoutExpired:
        print(f"Interpreter timed out after {timeout} seconds.", file=sys.stderr)
        return
    except OSError as exc:
        print(f"Failed to launch interpreter: {exc}", file=sys.stderr)
        return

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        print(f"Interpreter exited with status {result.returncode}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    model = args.model
    system_prompt = args.system_prompt

    if args.coder:
        model = CODER_MODEL
        system_prompt = CODER_SYSTEM_PROMPT if args.system_prompt is None else args.system_prompt

    print(
        f"[client] Sending request to {args.host} for model '{model}' "
        f"with timeout {args.timeout}s.",
        file=sys.stderr,
    )
    try:
        result = generate(
            host=args.host,
            model=model,
            prompt=args.prompt,
            options=build_options(args),
            timeout=args.timeout,
            system_prompt=system_prompt,
        )
    except requests.exceptions.RequestException as exc:
        print(f"Request to Ollama server failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"[client] Received response payload with keys: {', '.join(result.keys())}.",
            file=sys.stderr,
        )
        response_text: Optional[str] = result.get("response")
        if response_text is None:
            print("No response field found in Ollama output.", file=sys.stderr)
            print(json.dumps(result, indent=2))
            sys.exit(2)
        output = response_text.strip()
        warning: Optional[str] = None
        if args.coder:
            print("[client] Enforcing coder mode output validation.", file=sys.stderr)
            output, warning = ensure_pure_python(output)
            if warning:
                print(warning, file=sys.stderr)
        else:
            print(
                "[client] No coder validation requested; returning raw model response.",
                file=sys.stderr,
            )
        print(output)
        if args.execute:
            print(
                f"[client] Executing generated code with timeout {args.timeout}s.",
                file=sys.stderr,
            )
            execute_python(output, timeout=args.timeout)
        else:
            print(
                "[client] Execution not requested (use --execute to run the code).",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
