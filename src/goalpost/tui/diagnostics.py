"""Bounded local TUI diagnostics; never record environment values or input fields."""

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG = logging.getLogger("goalpost.tui")


def redact(text: str) -> str:
    for name in ("ODDS_API", "ODDS_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[redacted]")
    return re.sub(r"(?i)(api_?key=|authorization:\s*bearer\s+)[^&\s]+", r"\1[redacted]", text)


class RedactingFormatter(logging.Formatter):
    def format(self, record):
        return redact(super().format(record))


def configure_logging() -> Path:
    directory = Path(
        os.environ.get(
            "GOALPOST_LOG_DIR",
            str(Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "goalpost"),
        )
    ).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tui.log"

    # Restrict new files, including files created after rotation.
    class PrivateHandler(RotatingFileHandler):
        def _open(self):
            fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.chmod(self.baseFilename, 0o600)
            return os.fdopen(fd, self.mode, encoding=self.encoding, errors=self.errors)

    handler = PrivateHandler(path, maxBytes=1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(message)s"))
    for previous in LOG.handlers[:]:
        LOG.removeHandler(previous)
        previous.close()
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    return path


class LogStream:
    """Keep action output out of the raw screen without buffering whole datasets."""

    def write(self, text):
        for line in text.splitlines():
            if line.strip():
                LOG.info("action output: %s", line[:4096])
        return len(text)

    def flush(self):
        pass
