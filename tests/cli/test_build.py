import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"


def test_build_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "goalpost.cli.build", "--help"],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert result.returncode == 0
    assert "Build GoalPost" in result.stdout
