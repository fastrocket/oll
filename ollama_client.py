#!/usr/bin/env python3
"""
Simple Ollama client for interacting with a locally running Ollama server.

Usage example:
    python ollama_client.py --prompt "Hello!" --temperature 0.7
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional

import requests


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
        default=120.0,
        help="HTTP request timeout in seconds.",
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
) -> Dict[str, Any]:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": options,
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    try:
        result = generate(
            host=args.host,
            model=args.model,
            prompt=args.prompt,
            options=build_options(args),
            timeout=args.timeout,
        )
    except requests.exceptions.RequestException as exc:
        print(f"Request to Ollama server failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        response_text: Optional[str] = result.get("response")
        if response_text is None:
            print("No response field found in Ollama output.", file=sys.stderr)
            print(json.dumps(result, indent=2))
            sys.exit(2)
        print(response_text.strip())


if __name__ == "__main__":
    main()
