import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "goalpost"

NFL_TERMS = {"down", "yardline", "nflverse", "espn", "quarter", "touchdown", "field_goal", "punt"}


def _python_files(base):
    return list(base.rglob("*.py"))


def test_base_files_no_nfl_terms():
    base = SRC / "data" / "base"
    for path in _python_files(base):
        text = path.read_text()
        found = {t for t in NFL_TERMS if t in text.lower()}
        assert not found, f"{path.name} contains NFL terms: {found}"


def test_base_does_not_import_nfl():
    for sub in ["data", "representation", "simulation"]:
        base = SRC / sub / "base"
        for path in _python_files(base):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if ".nfl." in module or module.endswith(".nfl"):
                        assert False, f"{path} imports from {module}"
