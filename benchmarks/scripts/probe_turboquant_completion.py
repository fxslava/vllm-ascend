# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect raw completion tokens around a context boundary, using stdlib only.

Run on the serving host:
    python benchmarks/scripts/probe_turboquant_completion.py --lengths 2047 2048 2049 2176
    python benchmarks/scripts/probe_turboquant_completion.py --prompt-file turn3.txt

The first command uses synthetic, exactly token-sized completion prompts via
/tokenize. A prompt file is sent unchanged and should contain the rendered
completion prompt, including the conversation/template when reproducing chat.
Both modes call /v1/completions, expose special tokens, and save raw JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_LENGTHS = (2047, 2048, 2049, 2176)
FILLER = "В архиве лежали письма о путешествиях, встречах и старых городах. "
QUESTION = "\nРасскажи короткую историю на русском языке.\n"


def request_json(base_url, endpoint, payload=None, *, api_key=None, timeout=180):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + endpoint, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}") from error


def boundary_prompts(lengths, tokenize):
    """Keep the same question and an exact prompt length at every boundary."""
    suffix = tokenize(QUESTION)
    if min(lengths) <= len(suffix):
        raise ValueError("every requested length must exceed the question's token count")
    repeats = 64
    prefix = tokenize(FILLER * repeats)
    while len(prefix) < max(lengths):
        repeats *= 2
        prefix = tokenize(FILLER * repeats)
    return [(f"tokens-{length}", prefix[: length - len(suffix)] + suffix) for length in lengths]


def report(response):
    print(f"usage={response.get('usage')!r}")
    for choice in response.get("choices", []):
        text = choice.get("text", "")
        logprobs = choice.get("logprobs") or {}
        tokens = logprobs.get("tokens") or []
        ids = choice.get("token_ids")
        if ids is None and tokens and all(token.startswith("token_id:") for token in tokens):
            ids = [int(token.removeprefix("token_id:")) for token in tokens]
        print(f"finish_reason={choice.get('finish_reason')!r} stop_reason={choice.get('stop_reason')!r}")
        print(f"text_repr={text!r}")
        print(f"text_codepoints={[f'U+{ord(character):04X}' for character in text]}")
        print(f"token_ids={ids!r}")
        if ids is None:
            print("Server returned no exact IDs; raw response is saved. Do not infer IDs by re-tokenizing text.")
        probabilities = logprobs.get("token_logprobs") or []
        alternatives = logprobs.get("top_logprobs") or []
        for index, token in enumerate(tokens):
            probability = probabilities[index] if index < len(probabilities) else None
            top = alternatives[index] if index < len(alternatives) else None
            print(f"  token[{index}]={token!r} logprob={probability!r} top={top!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server root URL, without /v1")
    parser.add_argument("--model", help="Served model name; defaults to the first /v1/models entry")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prompt-file", type=Path, help="Send a rendered UTF-8 completion prompt unchanged")
    mode.add_argument(
        "--lengths", nargs="+", type=int, help=f"Exact synthetic prompt lengths; default {DEFAULT_LENGTHS}"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("turboquant-probe"))
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    if args.max_tokens <= 0 or (args.lengths is not None and min(args.lengths) <= 0):
        parser.error("token counts must be positive")

    def call(endpoint, payload=None):
        return request_json(args.base_url, endpoint, payload, api_key=args.api_key, timeout=args.timeout)

    model = args.model or call("/v1/models")["data"][0]["id"]
    if args.prompt_file:
        prompts = [("prompt-file", args.prompt_file.read_text(encoding="utf-8"))]
    else:
        prompts = boundary_prompts(
            args.lengths or DEFAULT_LENGTHS,
            lambda text: call("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"],
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, prompt in prompts:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": False,
            "logprobs": 5,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "skip_special_tokens": False,
        }
        destination = args.output_dir / label
        destination.with_suffix(".request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n--- {label} ---", flush=True)
        response = call("/v1/completions", payload)
        destination.with_suffix(".response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report(response)
        print(f"saved={destination.with_suffix('.response.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
