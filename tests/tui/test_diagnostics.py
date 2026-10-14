import stat

from goalpost.tui.diagnostics import LOG, configure_logging


def test_traceback_redacts_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("GOALPOST_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ODDS_API", "test-secret-not-real")
    path = configure_logging()
    try:
        raise RuntimeError("https://example.invalid/?apiKey=test-secret-not-real&other=1")
    except RuntimeError:
        LOG.exception("action failed")
    content = path.read_text()
    assert "Traceback" in content and "RuntimeError" in content
    assert "test-secret-not-real" not in content
    assert "[redacted]" in content
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_logging_is_bounded_and_does_not_duplicate_handlers(monkeypatch, tmp_path):
    monkeypatch.setenv("GOALPOST_LOG_DIR", str(tmp_path))
    configure_logging()
    path = configure_logging()
    assert len(LOG.handlers) == 1 and LOG.propagate is False
    for _ in range(6):
        LOG.info("x" * (1024 * 1024 - 100))
    assert path.exists()
    assert len(list(tmp_path.glob("tui.log*"))) == 4
    assert all(p.stat().st_size <= 1024 * 1024 for p in tmp_path.glob("tui.log*"))
