# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/check_turboquant_sync.py"
SPEC = importlib.util.spec_from_file_location("check_turboquant_sync", SCRIPT)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)

SHARED = "common/fixture.h"
ADAPTER = "bindings/fixture.cpp"
SOURCE = b'const char* url = "https://example.com/*";\n'


@pytest.fixture
def trees(tmp_path):
    plugin, core = tmp_path / "plugin", tmp_path / "core"
    for root in (plugin, core):
        (root / SHARED).parent.mkdir(parents=True)
        (root / SHARED).write_bytes(SOURCE)
    (plugin / ADAPTER).parent.mkdir()
    (plugin / ADAPTER).write_bytes(SOURCE)
    record = {"sha256": hashlib.sha256(SOURCE).hexdigest()}
    inventory = {"shared": {SHARED: record}, "plugin_only": {ADAPTER: record}, "core_only": {}}
    for root in (plugin, core):
        (root / sync.MANIFEST).write_text(json.dumps(inventory))
    return plugin, core


def test_exact_shared_bytes_and_local_adapter(trees):
    assert sync.verify(*trees) == []


def test_same_tokens_different_line_endings(trees):
    plugin, core = trees
    (core / SHARED).write_bytes(SOURCE.replace(b"\n", b"\r\n"))
    assert sync.verify(plugin, core) == [f"core: hash differs for {SHARED}", f"shared bytes differ for {SHARED}"]


def test_same_drift_on_both_sides(trees):
    for root in trees:
        (root / SHARED).write_bytes(SOURCE + b"\n")
    assert sync.verify(*trees) == [f"plugin: hash differs for {SHARED}", f"core: hash differs for {SHARED}"]


def test_missing_and_untracked_sources(trees):
    plugin, core = trees
    (core / SHARED).unlink()
    (core / "unexpected.hxx").touch()
    assert sync.verify(plugin, core) == ["core: unexpected unexpected.hxx", f"core: missing {SHARED}"]


def test_pinned_comment_is_detected(trees):
    plugin, core = trees
    data = SOURCE + b"// comment\n"
    (plugin / ADAPTER).write_bytes(data)
    inventory = json.loads((plugin / sync.MANIFEST).read_text())
    inventory["plugin_only"][ADAPTER]["sha256"] = hashlib.sha256(data).hexdigest()
    for root in trees:
        (root / sync.MANIFEST).write_text(json.dumps(inventory))
    assert sync.verify(plugin, core) == [f"plugin: comments in {ADAPTER}"]


def test_digit_separators_and_raw_strings():
    assert sync.has_comment(b"auto value = 1'000; // comment\nauto other = 2'000;\n")
    assert not sync.has_comment(b'''auto s = R"delim(".../*...*/")delim";''')
    assert sync.has_comment(b'''auto s = R"delim(".../*...*/")delim"; // comment\n''')


def test_pinned_shared_source_cannot_escape_categories(trees):
    plugin, core = trees
    stray = "uncategorized.hxx"
    inventory = json.loads((plugin / sync.MANIFEST).read_text())
    inventory["shared"][stray] = inventory["shared"][SHARED]
    for root in trees:
        (root / stray).write_bytes(SOURCE)
        (root / sync.MANIFEST).write_text(json.dumps(inventory))
    assert sync.verify(plugin, core) == [f"shared: source outside its categories: {stray}"]
