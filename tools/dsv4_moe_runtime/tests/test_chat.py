"""Conversation, EOS, streaming and persistent-cache regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ..chat import ChatSession, build_parser


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return {"input_ids": [1] * len(messages)}

    def decode(self, tokens, **kwargs):
        return "".join({3: "H", 4: "i", 9: ""}[token] for token in tokens)


class FakeDecoder:
    def __init__(self, selected, capacity=10):
        self.selected = iter(selected)
        self.capacity = capacity
        self.completed_tokens = 0
        self.position = 0
        self.config = {"eos_token_id": 9}
        self.pool = SimpleNamespace(stats=SimpleNamespace(hits=0, loads=0))
        self.runtime = SimpleNamespace(synchronize_device=lambda: None)

    def reset_sequence(self):
        self.position = 0

    def step(self, token):
        value = next(self.selected)
        self.completed_tokens += 1
        self.position += 1
        self.pool.stats.hits += 156
        return SimpleNamespace(argmax=lambda dim: SimpleNamespace(item=lambda: value))


def test_eos_streaming_and_cache_survive_context_reset():
    decoder = FakeDecoder([3, 4, 9, 3, 9])
    session = ChatSession(decoder, FakeTokenizer(), 3)
    parts = []
    text, stats = session.respond("Hello", parts.append)
    assert text == "Hi" == "".join(parts)
    assert stats.finish_reason == "eos"
    assert stats.generated_tokens == 3
    assert stats.hits == 468 and stats.misses == 0
    assert session.messages == [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    pool = decoder.pool
    session.reset()
    assert session.messages == [] and decoder.position == 0
    assert decoder.completed_tokens == 3 and decoder.pool is pool
    session.respond("Hello again", lambda part: None)
    assert decoder.completed_tokens == 5


def test_length_limit_and_complete_turn_trimming():
    decoder = FakeDecoder([3, 4, 3, 4, 9], capacity=5)
    session = ChatSession(decoder, FakeTokenizer(), 3)
    _, stats = session.respond("first", lambda part: None)
    assert stats.finish_reason == "length"
    _, stats = session.respond("second", lambda part: None)
    assert stats.trimmed_turns == 1
    assert [message["content"] for message in session.messages] == ["second", "i"]


def test_oversized_prompt_keeps_existing_history_and_cache_untouched():
    class LongTokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return [1] * 10

    decoder = FakeDecoder([])
    session = ChatSession(decoder, LongTokenizer(), 3)
    with pytest.raises(ValueError, match="prompt exceeds"):
        session.respond("long", lambda part: None)
    assert decoder.completed_tokens == 0 and session.messages == []


def test_partial_utf8_character_is_not_printed_prematurely():
    class ByteTokenizer(FakeTokenizer):
        def decode(self, tokens, **kwargs):
            return "\ufffd" if tokens == [3] else "猫"

    session = ChatSession(FakeDecoder([3, 4, 9]), ByteTokenizer(), 3)
    parts = []
    text, _ = session.respond("hello", parts.append)
    assert text == "猫" and "".join(parts) == "猫"


def test_default_slots_and_noninteractive_prompts():
    args = build_parser().parse_args(["--prompt", "Hello", "--prompt", "Again"])
    assert args.slots == 384
    assert args.prompt == ["Hello", "Again"]
