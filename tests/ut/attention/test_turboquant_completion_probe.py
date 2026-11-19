# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free checks for the raw completion diagnostic probe."""

import json

import pytest

from benchmarks.scripts import probe_turboquant_completion as probe


def test_report_preserves_invisible_text_and_exact_ids(capsys):
    text = " \n\t\u200b"
    probe.report({"choices": [{"text": text, "token_ids": [220, 198, 197, 16067], "finish_reason": "length"}]})
    output = capsys.readouterr().out
    assert f"text_repr={text!r}" in output
    assert "U+200B" in output
    assert "token_ids=[220, 198, 197, 16067]" in output


def test_report_extracts_ids_from_logprobs_without_retokenizing(capsys):
    probe.report({"choices": [{"text": "", "logprobs": {"tokens": ["token_id:0", "token_id:151643"]}}]})
    assert "token_ids=[0, 151643]" in capsys.readouterr().out


def test_report_does_not_invent_ids_from_decoded_strings(capsys):
    probe.report({"choices": [{"text": " ", "logprobs": {"tokens": [" "]}}]})
    assert "token_ids=None" in capsys.readouterr().out


def test_boundary_prompts_have_exact_lengths_and_a_shared_question():
    prompts = probe.boundary_prompts([2047, 2048, 2049], lambda text: list(text.encode("utf-8")))
    suffix = list(probe.QUESTION.encode("utf-8"))
    assert [len(prompt) for _, prompt in prompts] == [2047, 2048, 2049]
    assert all(prompt[-len(suffix) :] == suffix for _, prompt in prompts)
    with pytest.raises(ValueError, match="exceed"):
        probe.boundary_prompts([1], lambda text: list(text.encode("utf-8")))


def test_prompt_file_is_sent_unchanged_and_response_is_saved(monkeypatch, tmp_path):
    prompt = "<|im_start|>user\nРасскажи историю.<|im_end|>\n<|im_start|>assistant\n"
    source = tmp_path / "turn3.txt"
    source.write_text(prompt, encoding="utf-8")
    response = {"usage": {"prompt_tokens": 2049}, "choices": [{"text": "\n", "token_ids": [198]}]}
    calls = []

    def request(base_url, endpoint, payload=None, **kwargs):
        calls.append((endpoint, payload))
        return response

    monkeypatch.setattr(probe, "request_json", request)
    output = tmp_path / "output"
    assert probe.main(["--model", "test-model", "--prompt-file", str(source), "--output-dir", str(output)]) == 0
    assert len(calls) == 1
    endpoint, payload = calls[0]
    assert endpoint == "/v1/completions" and payload["prompt"] == prompt
    assert payload["return_token_ids"] and payload["return_tokens_as_token_ids"]
    assert payload["logprobs"] == 5 and payload["skip_special_tokens"] is False
    assert json.loads((output / "prompt-file.response.json").read_text(encoding="utf-8")) == response
