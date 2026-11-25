# SPDX-License-Identifier: Apache-2.0
"""Check that the two TurboQuant source trees match their source inventory."""

import argparse
import hashlib
import json
import re
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "SOURCE_SYNC.json"
SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl"}
SHARED_CATEGORIES = ("common/", "ingress/native/", "decode/native/", "decode/e8/")
CPP_TEXT = re.compile(
    r'(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n\v\f]{0,16})\(.*?\)(?P=delimiter)"'
    r'|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''
    r"|(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[A-Za-z0-9_.]|'[A-Za-z0-9_])*"
    r"|/\*.*?\*/|//[^\n]*",
    re.DOTALL,
)


def has_comment(source: bytes) -> bool:
    text = source.decode("utf-8").replace("\\\n", "").replace("\\\r\n", "")
    return any(match.group().startswith(("//", "/*")) for match in CPP_TEXT.finditer(text))


def source_paths(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS | {".aka"}
    }


def verify(plugin: Path, core: Path) -> list[str]:
    plugin_manifest = (plugin / MANIFEST).read_bytes()
    if plugin_manifest != (core / MANIFEST).read_bytes():
        return ["source inventories differ"]
    manifest = json.loads(plugin_manifest)
    shared = manifest["shared"]
    diagnostics = []
    for label, paths, prefixes in (
        ("shared", shared, SHARED_CATEGORIES),
        ("plugin-only", manifest["plugin_only"], ("bindings/",)),
        ("core-only", manifest["core_only"], ("common/",)),
    ):
        diagnostics.extend(
            f"{label}: source outside its categories: {path}"
            for path in sorted(paths)
            if not path.startswith(prefixes)
        )
    for label, root in (("plugin", plugin), ("core", core)):
        expected = {**shared, **manifest[f"{label}_only"]}
        actual = source_paths(root)
        diagnostics.extend(f"{label}: unexpected {path}" for path in sorted(actual - expected.keys()))
        diagnostics.extend(f"{label}: missing {path}" for path in sorted(expected.keys() - actual))
        for path in sorted(actual & expected.keys()):
            data = (root / path).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected[path]["sha256"]:
                diagnostics.append(f"{label}: hash differs for {path}")
            if Path(path).suffix.lower() in SOURCE_EXTENSIONS and has_comment(data):
                diagnostics.append(f"{label}: comments in {path}")
    for path in shared:
        left, right = plugin / path, core / path
        if left.is_file() and right.is_file() and left.read_bytes() != right.read_bytes():
            diagnostics.append(f"shared bytes differ for {path}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, type=Path, help="standalone turboquant-core repository")
    args = parser.parse_args()
    plugin = PLUGIN_ROOT / "csrc/attention/turboquant"
    core = args.core / "src/turboquant"
    try:
        diagnostics = verify(plugin, core)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    if diagnostics:
        parser.exit(1, "\n".join(diagnostics) + "\n")
    print("TurboQuant source inventories, byte hashes and shared files match")


if __name__ == "__main__":
    main()
