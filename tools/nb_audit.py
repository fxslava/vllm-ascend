"""Classify every `non_blocking=True` H2D in vllm_ascend by the C5 criteria.

For each site: is the source host memory pinned, is it a persistent buffer that
a later step rewrites, and does the flag follow the pinning rather than being a
literal?

Reads source only. Run from the repo root.
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path("C:/vllm-ascend")
PKG = ROOT / "vllm_ascend"


def unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "<?>"


def classify(source_expr: str, flag_expr: str) -> str:
    """The C5 verdict for one site."""
    if flag_expr != "True":
        return "FLAG-DERIVED"
    if ".pin_memory()" in source_expr:
        return "PINNED-TEMP"
    if "pin_memory" in source_expr:
        return "PINNED-OTHER"
    # A persistent host buffer: reached through self/attribute rather than built
    # on the spot.
    head = source_expr.split("[")[0].split("(")[0]
    if head.startswith("self.") or head.startswith("common.") or "." in head:
        return "PERSISTENT?"
    return "LOCAL-TEMP"


def main() -> int:
    rows = []
    for path in sorted(PKG.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            flag = None
            for kw in node.keywords:
                if kw.arg == "non_blocking":
                    flag = unparse(kw.value)
            if flag is None:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else unparse(func)
            # The host source: `.to(dev)` copies from the object, `.copy_(src)`
            # from its first argument.
            if name == "copy_" and node.args:
                source_expr = unparse(node.args[0])
            elif isinstance(func, ast.Attribute):
                source_expr = unparse(func.value)
            else:
                source_expr = "<?>"
            rows.append(
                (
                    classify(source_expr, flag),
                    str(path.relative_to(ROOT)).replace("\\", "/"),
                    node.lineno,
                    name,
                    source_expr[:70],
                    flag,
                )
            )

    order = {"PERSISTENT?": 0, "LOCAL-TEMP": 1, "PINNED-OTHER": 2, "PINNED-TEMP": 3, "FLAG-DERIVED": 4}
    rows.sort(key=lambda r: (order[r[0]], r[1], r[2]))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row[0]] = counts.get(row[0], 0) + 1
    print("total sites:", len(rows))
    for verdict, n in sorted(counts.items(), key=lambda kv: order[kv[0]]):
        print(f"  {verdict:<13} {n}")
    print()
    print("PERSISTENT? sites -- a host buffer reached through an attribute, copied")
    print("non_blocking from unpinned memory. Each needs the 'rewritten later?' question:")
    for verdict, path, line, name, src, flag in rows:
        if verdict != "PERSISTENT?":
            continue
        print(f"  {path}:{line}  {name}({src})  non_blocking={flag}")
    print()
    print("FLAG-DERIVED sites (the C5 shape):")
    for verdict, path, line, name, src, flag in rows:
        if verdict == "FLAG-DERIVED":
            print(f"  {path}:{line}  {name}({src})  non_blocking={flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
