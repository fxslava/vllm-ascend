#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""The harness has to import on the Ascend image's Python, which is 3.9.

Nothing else in this suite notices: the machines that run it are newer, PEP 604
is valid syntax on all of them, and a ``str | TurnBody`` reads as ordinary
modern Python right up until the device image evaluates it and raises
``TypeError: unsupported operand type(s) for |: 'type' and 'type'``. That is an
import error, so it takes the whole run with it -- and it arrives at the worst
possible moment, on the far side of a model load.

The distinction these checks turn on is the one that is easy to get wrong.
``from __future__ import annotations`` (PEP 563) makes *annotations* strings, so
a 3.10-only union in a signature or a field is never evaluated and 3.9 is happy
with it. It does nothing at all for an *assignment*: a module-level type alias,
``isinstance(x, A | B)``, a ``TypeVar`` bound, a default argument, a class base
-- those are ordinary expressions, evaluated the moment the module is imported,
future import or not. That is exactly what
:data:`tq_longbench.families.TurnPart` was, and why the future import already at
the top of that module did not save it.

So two rules, checked separately:

* a PEP 604 union in a position Python evaluates eagerly is banned outright;
* a PEP 604 union in an annotation is allowed only where the module carries the
  future import.

Plus the grammar itself, since 3.10 syntax 3.9 cannot even parse (``match``)
fails earlier still, and the ruff pin -- without ``target-version = "py39"``
pyupgrade's UP rules are free to rewrite every ``Optional[X]`` the other way and
reintroduce the whole class in a single ``--fix``.

Runs under pytest and under ``python -m unittest`` anywhere: nothing here needs
torch, a device, or the corpus. It reads the sources, it does not import them.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The trees held to the 3.9 floor: the harness itself, and the tests that import it.
AUDITED_ROOTS = (
    REPO_ROOT / "tools" / "tq_longbench",
    REPO_ROOT / "tests" / "ut" / "_tools",
)

#: Modules outside those trees that the 3.9 host still imports, by way of the harness.
#: They are backend code rather than harness code, so they have no root of their own,
#: but a PEP 604 union in one of them fails the import just the same.
AUDITED_FILES = (
    REPO_ROOT / "vllm_ascend" / "attention" / "turboquant_rotation.py",
    REPO_ROOT / "vllm_ascend" / "attention" / "turboquant_layout.py",
    REPO_ROOT / "tests" / "ut" / "attention" / "turboquant_cpu_ops.py",
)

#: The lowest interpreter the Ascend runtime image offers (``/usr/local/lib/python3.9``).
TARGET_VERSION = (3, 9)

#: Names whose appearance either side of a ``|`` means the expression is a type
#: union rather than an integer's bitwise-or, a set union, or a 3.9-legal dict merge.
_TYPE_NAMES = frozenset(
    {
        "Any", "Optional", "Union", "List", "Dict", "Tuple", "Set", "Sequence",
        "Mapping", "MutableMapping", "Iterable", "Iterator", "Callable", "Type",
        "TypeVar", "Literal", "Final", "ClassVar", "Awaitable", "Coroutine",
        "int", "float", "str", "bytes", "bool", "list", "dict", "tuple", "set",
        "frozenset", "complex", "object", "type", "bytearray",
    }
)


def audited_sources():
    """Every module under :data:`AUDITED_ROOTS`, plus :data:`AUDITED_FILES`; caches excluded."""
    found = []
    for root in AUDITED_ROOTS:
        found.extend(p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts)
    found.extend(AUDITED_FILES)
    return found


def _is_type_expression(node) -> bool:
    """Does this operand read as a type, rather than a value ``|`` also accepts?

    Deliberately generous on the capitalised-``Name`` side: a false positive
    costs a ``Union[...]`` spelling somebody did not strictly need, a false
    negative costs an import failure on the device.
    """
    if isinstance(node, ast.Constant):
        return node.value is None
    if isinstance(node, ast.Name):
        return node.id in _TYPE_NAMES or node.id[:1].isupper()
    if isinstance(node, ast.Attribute):
        return node.attr in _TYPE_NAMES or node.attr[:1].isupper()
    if isinstance(node, ast.Subscript):
        return _is_type_expression(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_type_expression(node.left) and _is_type_expression(node.right)
    return False


def type_unions(node):
    """Every PEP 604 union in the subtree rooted at ``node``."""
    if node is None:
        return
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr) and _is_type_expression(sub):
            yield sub


def has_future_annotations(tree) -> bool:
    """Does this module defer its annotations?"""
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


class _UnionSites(ast.NodeVisitor):
    """Splits a module's PEP 604 unions by whether the interpreter evaluates them.

    Annotation positions -- :attr:`ast.AnnAssign.annotation`, an
    :class:`ast.arg`'s annotation, a function's ``returns`` -- are collected as
    *deferred*. Everything else reachable from a statement is *eager*: values,
    defaults, decorators, class bases, call arguments.
    """

    def __init__(self) -> None:
        self.eager = []
        self.deferred = []

    def _eagerly(self, node) -> None:
        self.eager.extend(type_unions(node))

    def _annotated(self, node) -> None:
        self.deferred.extend(type_unions(node))

    def visit_AnnAssign(self, node) -> None:
        self._annotated(node.annotation)
        self._eagerly(node.value)

    def visit_Assign(self, node) -> None:
        self._eagerly(node.value)

    def visit_FunctionDef(self, node) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self._visit_function(node)

    def _visit_function(self, node) -> None:
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
            if arg is not None:
                self._annotated(arg.annotation)
        self._annotated(node.returns)
        for value in (*args.defaults, *(d for d in args.kw_defaults if d), *node.decorator_list):
            self._eagerly(value)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ClassDef(self, node) -> None:
        for value in (*node.bases, *node.decorator_list, *(kw.value for kw in node.keywords)):
            self._eagerly(value)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Call(self, node) -> None:
        # isinstance/issubclass take their union eagerly even under PEP 563.
        for value in (*node.args, *(kw.value for kw in node.keywords)):
            self._eagerly(value)
        self.generic_visit(node)

    def generic_visit(self, node) -> None:
        if isinstance(node, (ast.Return, ast.Expr, ast.Subscript, ast.Tuple, ast.List)):
            self._eagerly(node)
        super().generic_visit(node)


def _sites(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _UnionSites()
    for stmt in tree.body:
        visitor.visit(stmt)
    return tree, visitor


def _where(path, nodes):
    rel = path.relative_to(REPO_ROOT).as_posix()
    return [f"{rel}:{node.lineno}: {ast.unparse(node)}" for node in nodes]


class TestPython39Compatibility(unittest.TestCase):
    def test_roots_are_populated(self):
        """A path typo would make every other check below pass vacuously."""
        sources = audited_sources()
        self.assertGreater(len(sources), 20, "audited roots came back nearly empty")
        names = {path.name for path in sources}
        self.assertIn("families.py", names)
        self.assertIn("run_longbench.py", names)
        for path in AUDITED_FILES:
            self.assertTrue(path.is_file(), f"{path} is audited but does not exist")
            self.assertIn(path, sources)

    def test_sources_parse_under_the_39_grammar(self):
        """3.10-only syntax -- ``match``, mostly -- fails before any union does."""
        failures = []
        for path in audited_sources():
            try:
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=TARGET_VERSION,
                )
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{exc.lineno}: {exc.msg}")
        self.assertEqual(failures, [], "not parseable by Python 3.9:\n" + "\n".join(failures))

    def test_no_eagerly_evaluated_pep604_unions(self):
        """The regression itself: ``TurnPart = str | TurnBody`` at module scope.

        A future import does not defer an assignment, so this is the form that
        reaches the device and raises. Spell these ``Union[...]``/``Optional[...]``.
        """
        offenders = []
        for path in audited_sources():
            _, sites = _sites(path)
            offenders.extend(_where(path, sites.eager))
        self.assertEqual(
            offenders,
            [],
            "PEP 604 union in a position Python 3.9 evaluates at import; "
            "use typing.Union/typing.Optional:\n" + "\n".join(offenders),
        )

    def test_annotation_unions_only_where_annotations_are_deferred(self):
        """PEP 604 in a signature is fine, but only behind the future import."""
        offenders = []
        for path in audited_sources():
            tree, sites = _sites(path)
            if sites.deferred and not has_future_annotations(tree):
                offenders.extend(_where(path, sites.deferred))
        self.assertEqual(
            offenders,
            [],
            "PEP 604 union in an annotation without `from __future__ import annotations`:\n"
            + "\n".join(offenders),
        )

    def test_future_import_precedes_every_other_statement(self):
        """``__future__`` is only honoured ahead of the first real statement."""
        offenders = []
        for path in audited_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for index, node in enumerate(tree.body):
                if not (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "__future__"
                    and any(alias.name == "annotations" for alias in node.names)
                ):
                    continue
                first = tree.body[0]
                after_docstring = (
                    index == 1
                    and isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                )
                if index != 0 and not after_docstring:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{node.lineno}: preceded by other statements")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_families_turn_part_is_a_typing_union(self):
        """The specific alias the device tripped over, pinned by name."""
        path = REPO_ROOT / "tools" / "tq_longbench" / "families.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "TurnPart" for t in node.targets)
        ]
        self.assertEqual(len(assignments), 1, "TurnPart is not assigned exactly once")
        self.assertEqual(ast.unparse(assignments[0].value), "Union[str, TurnBody]")

    def test_ruff_pins_the_target_version(self):
        """Without the pin, pyupgrade may rewrite Union[...] back into ``|``."""
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('target-version = "py39"', text)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
