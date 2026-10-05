"""Interactive real-weight V2-Lite chat with a persistent expert cache.

Run from the repository root with ``python -m tools.dsv4_moe_runtime.chat``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transformers import AutoTokenizer

# Permit both direct script execution and python -m without installation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dsv4_moe_runtime.inference.dsv2_lite import V2LiteDecoder  # noqa: E402

DEFAULT_WEIGHTS = Path("F:/AI/models/DeepSeek-V2-Lite-Chat")
DEFAULT_SLOTS = 384
DEFAULT_CONTEXT_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class TurnStats:
    prompt_tokens: int
    generated_tokens: int
    seconds: float
    hits: int
    misses: int
    finish_reason: str
    trimmed_turns: int

    def display(self) -> str:
        requests = self.hits + self.misses
        hit_rate = self.hits / requests if requests else 0.0
        speed = self.generated_tokens / self.seconds if self.seconds else 0.0
        return (
            f"[prompt={self.prompt_tokens} generated={self.generated_tokens} "
            f"elapsed={self.seconds:.2f}s tokens/s={speed:.2f} "
            f"cache={hit_rate:.1%} ({self.hits} hits/{self.misses} misses) "
            f"finish={self.finish_reason} trimmed_turns={self.trimmed_turns}]"
        )


class ChatSession:
    """Conversation state; one decoder remains alive for the entire session."""

    def __init__(self, decoder, tokenizer, max_new_tokens: int):
        if not 0 < max_new_tokens < decoder.capacity:
            raise ValueError("max_new_tokens must be positive and smaller than context capacity")
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.messages: list[dict[str, str]] = []

    def prompt_tokens(self, messages: list[dict[str, str]]) -> list[int]:
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return list(tokens["input_ids"] if hasattr(tokens, "keys") else tokens)

    def reset(self) -> None:
        self.messages.clear()
        self.decoder.reset_sequence()

    def pool_stats(self) -> str:
        pool = self.decoder.pool
        stats = pool.stats
        requests = stats.hits + stats.loads
        rate = stats.hits / requests if requests else 0.0
        return (
            f"[pool={pool.resident_expert_count}/{pool.num_slots} routed experts "
            f"arena={pool.slot_arena.numel() / 2**30:.3f} GiB "
            f"cache={rate:.1%} hits={stats.hits} misses={stats.loads} "
            f"evictions={stats.evictions} forwarded_tokens={self.decoder.completed_tokens}]"
        )

    def respond(self, user_text: str, emit: Callable[[str], None]) -> tuple[str, TurnStats]:
        messages = [*self.messages, {"role": "user", "content": user_text}]
        trimmed = 0
        tokens = self.prompt_tokens(messages)
        # Reserve the full generation budget, then discard only complete turns.
        while len(tokens) + self.max_new_tokens > self.decoder.capacity and len(messages) > 1:
            del messages[:2]
            trimmed += 1
            tokens = self.prompt_tokens(messages)
        if not tokens or len(tokens) + self.max_new_tokens > self.decoder.capacity:
            raise ValueError("prompt exceeds context capacity; shorten it or increase --context-tokens")

        self.decoder.reset_sequence()
        before = (self.decoder.pool.stats.hits, self.decoder.pool.stats.loads)
        started = time.perf_counter()
        for token in tokens:
            logits = self.decoder.step(token)
        generated, printed = [], ""
        reason = "length"
        for position in range(self.max_new_tokens):
            # Greedy selection must reach the host to drive the next token.
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token == self.decoder.config["eos_token_id"]:
                reason = "eos"
                break
            text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # Byte fallback tokens may form only part of a UTF-8 character.
            # Wait for the completed character instead of printing U+FFFD.
            if not text.endswith("\ufffd"):
                emit(text[len(printed) :])
                printed = text
            if position + 1 < self.max_new_tokens:
                logits = self.decoder.step(token)
        text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        emit(text[len(printed) :])
        self.decoder.runtime.synchronize_device()
        elapsed = time.perf_counter() - started
        self.messages = [*messages, {"role": "assistant", "content": text}]
        stats = TurnStats(
            len(tokens),
            len(generated),
            elapsed,
            self.decoder.pool.stats.hits - before[0],
            self.decoder.pool.stats.loads - before[1],
            reason,
            trimmed,
        )
        return text, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--auto-slots", action="store_true", help="use the available VRAM budget for routed slots")
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prompt", action="append", help="non-interactive prompt; repeat for multiple turns")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.slots < 6:
        parser.error("--slots must hold at least six routed experts")
    if not 0 < args.max_new_tokens < args.context_tokens:
        parser.error("--max-new-tokens must be positive and smaller than --context-tokens")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    slot_plan = "automatic VRAM sizing" if args.auto_slots else f"{args.slots} routed slots"
    print(f"Loading {args.weights_dir} on {args.device} with {slot_plan}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.weights_dir, local_files_only=True)
    decoder = V2LiteDecoder(args.weights_dir, args.device, None if args.auto_slots else args.slots, args.context_tokens)
    session = ChatSession(decoder, tokenizer, args.max_new_tokens)
    print(
        f"Ready ({decoder.pool.num_slots} VRAM slots). /reset clears history; /stats shows the cache; /exit quits.",
        flush=True,
    )
    try:

        def respond(text: str) -> None:
            print("Assistant: ", end="", flush=True)
            try:
                _, stats = session.respond(text, lambda part: print(part, end="", flush=True))
            except KeyboardInterrupt:
                decoder.reset_sequence()
                print("\nGeneration interrupted; conversation history was preserved.")
                return
            print("\n" + stats.display(), flush=True)

        if args.prompt is not None:
            for text in args.prompt:
                print("User: " + text, flush=True)
                respond(text)
            return 0
        while True:
            try:
                text = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text == "/exit":
                break
            if text == "/reset":
                session.reset()
                print("History cleared; expert cache retained.")
            elif text == "/stats":
                print(session.pool_stats())
            elif text.startswith("/"):
                print("Commands: /reset, /stats, /exit")
            else:
                try:
                    respond(text)
                except ValueError as error:
                    print("\n" + str(error))
        return 0
    finally:
        decoder.close()


if __name__ == "__main__":
    raise SystemExit(main())
