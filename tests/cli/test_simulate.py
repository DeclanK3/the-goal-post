import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"


def test_simulate_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "goalpost.cli.simulate", "--help"],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert result.returncode == 0
    assert "Run GoalPost" in result.stdout
