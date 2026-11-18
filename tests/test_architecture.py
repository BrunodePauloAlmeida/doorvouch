"""Rules about which part of the code may depend on what.

The rule engine and the storage must run on the standard library alone, and no
module may need OpenCV, MediaPipe or NumPy just to be imported. That is what
keeps the logic testable on any machine and the camera-free commands usable
before the heavy packages are installed.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SOURCE / "doorvouch"
STANDARD_LIBRARY = sys.stdlib_module_names

# Which parts of the package each layer may import, besides the standard library.
ALLOWED = {
    "core": {"doorvouch.core", "doorvouch.errors"},
    "storage": {"doorvouch.core", "doorvouch.errors", "doorvouch.storage"},
}

STANDARD_LIBRARY_ONLY = [
    *sorted((PACKAGE / "core").glob("*.py")),
    *sorted((PACKAGE / "storage").glob("*.py")),
    PACKAGE / "config.py",
    PACKAGE / "errors.py",
    PACKAGE / "fps.py",
    PACKAGE / "models.py",
    PACKAGE / "text.py",
]


def imported_modules(path: Path, *, include_functions: bool) -> set[str]:
    """Absolute module names imported by ``path``.

    With ``include_functions=False`` only imports that run when the module is
    imported are returned; imports inside functions are deliberately lazy.
    """
    found: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name} uses a relative import"
            assert node.module is not None
            found.add(node.module)
        for child in ast.iter_child_nodes(node):
            is_function = isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
            if include_functions or not is_function:
                visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return found


def outside_the_standard_library(modules: set[str]) -> set[str]:
    return {
        module
        for module in modules
        if module.partition(".")[0] not in STANDARD_LIBRARY
        and module.partition(".")[0] != "doorvouch"
    }


def label(path: Path) -> str:
    return str(path.relative_to(PACKAGE))


@pytest.mark.parametrize("path", STANDARD_LIBRARY_ONLY, ids=label)
def test_the_logic_uses_only_the_standard_library(path: Path):
    modules = imported_modules(path, include_functions=True)
    assert outside_the_standard_library(modules) == set()


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=label)
def test_no_module_needs_third_party_packages_to_be_imported(path: Path):
    modules = imported_modules(path, include_functions=False)
    assert outside_the_standard_library(modules) == set()


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_inner_layers_do_not_depend_on_outer_ones(layer: str):
    for path in sorted((PACKAGE / layer).glob("*.py")):
        internal = {
            module
            for module in imported_modules(path, include_functions=True)
            if module.partition(".")[0] == "doorvouch"
        }
        stray = {
            module
            for module in internal
            if not any(
                module == allowed or module.startswith(allowed + ".") for allowed in ALLOWED[layer]
            )
        }
        assert stray == set(), f"{label(path)} imports {sorted(stray)}"


BLOCKED_IMPORTS = """
import importlib
import pkgutil
import sys

BLOCKED = {"cv2", "mediapipe", "numpy"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked in this test")
        return None


sys.meta_path.insert(0, Blocker())

import doorvouch

for module in pkgutil.walk_packages(doorvouch.__path__, "doorvouch."):
    if not module.name.endswith("__main__"):
        importlib.import_module(module.name)

from doorvouch.cli import main

status = main(["--home", sys.argv[1], "people"])
assert not BLOCKED & set(sys.modules), sorted(BLOCKED & set(sys.modules))
raise SystemExit(status)
"""


def test_every_module_imports_and_the_cli_works_without_third_party_packages(tmp_path: Path):
    environment = {**os.environ, "PYTHONPATH": str(SOURCE)}
    finished = subprocess.run(
        [sys.executable, "-c", BLOCKED_IMPORTS, str(tmp_path / "home")],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
    assert "Nobody is enrolled yet" in finished.stdout
