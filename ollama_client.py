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
from datetime import datetime
from pathlib import Path
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
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of iterations to run (default: %(default)s).",
    )
    parser.add_argument(
        "--log-file",
        default="ollama_iterations.log",
        help="Path to append JSONL iteration logs (default: %(default)s).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Maximum attempts per iteration to fix errors (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON response for each iteration instead of only the model output.",
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


def execute_python(code: str, timeout: float) -> Dict[str, Any]:
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
        msg = f"Interpreter timed out after {timeout} seconds."
        print(msg, file=sys.stderr)
        return {
            "timeout": True,
            "stdout": "",
            "stderr": msg,
            "returncode": None,
        }
    except OSError as exc:
        print(f"Failed to launch interpreter: {exc}", file=sys.stderr)
        return {
            "timeout": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": None,
        }

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        print(f"Interpreter exited with status {result.returncode}", file=sys.stderr)
    return {
        "timeout": False,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def build_refinement_prompt(
    base_prompt: str,
    iteration_number: int,
    previous_code: str,
    execution: Optional[Dict[str, Any]],
) -> str:
    parts = [
        "We are iteratively improving a Python program.",
        f"Original goal: {base_prompt}",
        f"Previous iteration #{iteration_number} program:",
        previous_code,
    ]

    if execution is not None:
        exec_lines = ["Execution results:"]
        if execution.get("timeout"):
            exec_lines.append("- Timed out under the current limit.")
        stdout = (execution.get("stdout") or "").strip()
        stderr = (execution.get("stderr") or "").strip()
        returncode = execution.get("returncode")
        no_output = execution.get("no_output")
        if stdout:
            exec_lines.append("STDOUT:")
            exec_lines.append(stdout)
        if stderr:
            exec_lines.append("STDERR:")
            exec_lines.append(stderr)
        if returncode not in (None, 0):
            exec_lines.append(f"Return code: {returncode}")
        if no_output:
            exec_lines.append(
                "Program produced no observable output. Ensure the next version prints or otherwise emits measurable results."
            )
        if len(exec_lines) == 1:
            exec_lines.append("No observable output.")
        parts.append("\n".join(exec_lines))

    parts.append(
        "Create a more interesting or improved Python program that advances the goal. "
        "If the previous solution is already interesting, refine it further. Respond with Python code only."
    )
    return "\n\n".join(parts)


def append_log_entry(log_handle, entry: Dict[str, Any]) -> None:
    log_handle.write(json.dumps(entry, ensure_ascii=False))
    log_handle.write("\n")
    log_handle.flush()


def build_retry_prompt(
    base_prompt: str,
    iteration_number: int,
    attempt_number: int,
    previous_code: str,
    warning: Optional[str],
    execution: Optional[Dict[str, Any]],
) -> str:
    parts = [
        "We are iteratively improving a Python program but the last attempt failed.",
        f"Original goal: {base_prompt}",
        f"Iteration #{iteration_number}, attempt #{attempt_number} produced this code:",
        previous_code,
    ]

    if warning:
        parts.append(f"Parser feedback: {warning}")

    if execution:
        exec_lines = ["Runtime feedback:"]
        if execution.get("timeout"):
            exec_lines.append("- Execution timed out under the current limit.")
        stdout = (execution.get("stdout") or "").strip()
        stderr = (execution.get("stderr") or "").strip()
        returncode = execution.get("returncode")
        no_output = execution.get("no_output")
        if stdout:
            exec_lines.append("STDOUT:")
            exec_lines.append(stdout)
        if stderr:
            exec_lines.append("STDERR:")
            exec_lines.append(stderr)
        if returncode not in (None, 0):
            exec_lines.append(f"Return code: {returncode}")
        if no_output:
            exec_lines.append(
                "Program produced no observable output. Ensure the next version prints or otherwise emits measurable results."
            )
        if len(exec_lines) == 1:
            exec_lines.append("No observable output.")
        parts.append("\n".join(exec_lines))

    parts.append(
        "Please produce a corrected and improved Python program that addresses the issues above. "
        "Respond with Python code only."
    )
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    if args.count < 1:
        print("--count must be at least 1.", file=sys.stderr)
        sys.exit(2)
    if args.json and args.count > 1:
        print("--json is only supported with --count 1.", file=sys.stderr)
        sys.exit(2)
    if args.retries < 1:
        print("--retries must be at least 1.", file=sys.stderr)
        sys.exit(2)

    model = args.model
    system_prompt = args.system_prompt

    if args.coder:
        model = CODER_MODEL
        system_prompt = CODER_SYSTEM_PROMPT if args.system_prompt is None else args.system_prompt

    log_path = Path(args.log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    should_execute = args.execute or args.count > 1
    if should_execute and not args.execute:
        print(
            "[client] Auto-enabling execution to capture outputs for iterative refinement.",
            file=sys.stderr,
        )

    base_prompt = args.prompt
    prompt = base_prompt
    final_output: Optional[str] = None
    final_warning: Optional[str] = None
    last_execution_details: Optional[Dict[str, Any]] = None

    with log_path.open("a", encoding="utf-8") as log_handle:
        for iteration in range(1, args.count + 1):
            attempt_prompt = prompt
            success = False
            for attempt in range(1, args.retries + 1):
                print(
                    f"[client] Iteration {iteration}/{args.count} attempt {attempt}/{args.retries} "
                    f"| requesting model '{model}' from {args.host} (timeout {args.timeout}s).",
                    file=sys.stderr,
                )
                print("[client] Prompt being sent to model:", file=sys.stderr)
                print(attempt_prompt, file=sys.stderr)
                try:
                    result = generate(
                        host=args.host,
                        model=model,
                        prompt=attempt_prompt,
                        options=build_options(args),
                        timeout=args.timeout,
                        system_prompt=system_prompt,
                    )
                except requests.exceptions.RequestException as exc:
                    print(f"Request to Ollama server failed: {exc}", file=sys.stderr)
                    success = False
                    break

                if args.json:
                    print(json.dumps(result, indent=2))

                print(
                    f"[client] Iteration {iteration} attempt {attempt} response keys: "
                    f"{', '.join(result.keys())}.",
                    file=sys.stderr,
                )
                response_text: Optional[str] = result.get("response")
                if response_text is None:
                    print("No response field found in Ollama output.", file=sys.stderr)
                    print(json.dumps(result, indent=2))
                    success = False
                    break

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
                print("[client] Model produced program:", file=sys.stderr)
                print(output, file=sys.stderr)

                execution_details: Optional[Dict[str, Any]] = None
                if should_execute:
                    print(
                        f"[client] Executing generated code with timeout {args.timeout}s.",
                        file=sys.stderr,
                    )
                    execution_details = execute_python(output, timeout=args.timeout)
                    if execution_details is not None:
                        stdout_clean = (execution_details.get("stdout") or "").strip()
                        stderr_clean = (execution_details.get("stderr") or "").strip()
                        no_output = not stdout_clean and not stderr_clean
                        execution_details["no_output"] = no_output
                        if no_output:
                            print(
                                "[client] Execution produced no observable output; requesting visible results next attempt.",
                                file=sys.stderr,
                            )
                else:
                    print(
                        "[client] Execution not requested (use --execute or set --count>1).",
                        file=sys.stderr,
                    )

                timestamp = datetime.utcnow().isoformat() + "Z"

                execution_failed = False
                if should_execute:
                    if execution_details is None:
                        execution_failed = True
                    else:
                        timeout_flag = execution_details.get("timeout")
                        returncode = execution_details.get("returncode")
                        no_output_flag = execution_details.get("no_output")
                        execution_failed = (
                            bool(timeout_flag)
                            or returncode is None
                            or returncode != 0
                            or bool(no_output_flag)
                        )

                success = warning is None and not execution_failed

                append_log_entry(
                    log_handle,
                    {
                        "timestamp": timestamp,
                        "iteration": iteration,
                        "attempt": attempt,
                        "prompt": attempt_prompt,
                        "response": response_text,
                        "normalized_code": output,
                        "warning": warning,
                        "execution": execution_details,
                        "success": success,
                    },
                )

                if success:
                    final_output = output
                    final_warning = warning
                    last_execution_details = execution_details
                    break

                if attempt == args.retries:
                    print(
                        "[client] Maximum retries reached for this iteration without success.",
                        file=sys.stderr,
                    )
                    break

                attempt_prompt = build_retry_prompt(
                    base_prompt=base_prompt,
                    iteration_number=iteration,
                    attempt_number=attempt,
                    previous_code=output,
                    warning=warning,
                    execution=execution_details,
                )

            if not success:
                print(
                    f"[client] Stopping after iteration {iteration} due to errors.",
                    file=sys.stderr,
                )
                break

            if iteration < args.count:
                prompt = build_refinement_prompt(
                    base_prompt=base_prompt,
                    iteration_number=iteration,
                    previous_code=final_output,
                    execution=last_execution_details,
                )

    if final_output is None:
        print("[client] No successful iterations were completed.", file=sys.stderr)
        sys.exit(1)

    if final_warning:
        print(final_warning, file=sys.stderr)
    if not args.json:
        print(final_output)


if __name__ == "__main__":
    main()
