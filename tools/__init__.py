"""Repository utility scripts and standalone tool packages.

This ``__init__.py`` exists so that ``tools.*`` resolves as a regular package
from the repository root: pytest and ``python -m`` then put the *repo root* on
``sys.path`` (never ``tools/`` itself), keeping ``tools/bisect/`` and other
stdlib-shadowing directories out of module resolution for
``tools.dsv4_moe_runtime`` and everything else under this tree.

Deliberately empty otherwise -- the scripts here are standalone.
"""
