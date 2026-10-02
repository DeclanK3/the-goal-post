"""Headless tests for TUI event handling."""
from unittest.mock import MagicMock, patch

import pytest
import ratatui as r


@pytest.fixture
def headless_terminal():
    t = r.Terminal()
    t.enable_raw()
    yield t
    t.disable_raw()


def test_j_key_navigates_down(headless_terminal):
    t = headless_terminal
    t.inject_key(r.KeyCode.Char, ord("j"), r.KeyMods(0))
    evt = t.next_event_typed(100)
    assert evt.kind == "key"
    assert evt.code == r.KeyCode.Char
    assert evt.ch == ord("j")


def test_k_key_navigates_up(headless_terminal):
    t = headless_terminal
    t.inject_key(r.KeyCode.Char, ord("k"), r.KeyMods(0))
    evt = t.next_event_typed(100)
    assert evt.kind == "key"
    assert evt.code == r.KeyCode.Char
    assert evt.ch == ord("k")


def test_arrow_down_navigates(headless_terminal):
    t = headless_terminal
    t.inject_key(r.KeyCode.Down, 0, r.KeyMods(0))
    evt = t.next_event_typed(100)
    assert evt.kind == "key"
    assert evt.code == r.KeyCode.Down


def test_arrow_up_navigates(headless_terminal):
    t = headless_terminal
    t.inject_key(r.KeyCode.Up, 0, r.KeyMods(0))
    evt = t.next_event_typed(100)
    assert evt.kind == "key"
    assert evt.code == r.KeyCode.Up


def test_enter_selects(headless_terminal):
    t = headless_terminal
    t.inject_key(r.KeyCode.Enter, 0, r.KeyMods(0))
    evt = t.next_event_typed(100)
    assert evt.kind == "key"
    assert evt.code == r.KeyCode.Enter


def test_q_quits(headless_terminal):
    t = headless_terminal
    t.inject_key(r.KeyCode.Char, ord("q"), r.KeyMods(0))
    evt = t.next_event_typed(100)
    assert evt.kind == "key"
    assert evt.code == r.KeyCode.Char
    assert evt.ch == ord("q")


def test_show_cursor_called_with_no_arguments():
    """Regression: show_cursor() must be called without arguments."""
    from goalpost.tui.main import run_tui

    mock_term = MagicMock()
    mock_term.next_event_typed.side_effect = [
        MagicMock(kind="key", code=r.KeyCode.Char, ch=ord("q")),
    ]

    with patch("goalpost.tui.main.r.Terminal", return_value=mock_term):
        run_tui()

    mock_term.enable_raw.assert_called_once()
    mock_term.disable_raw.assert_called_once()
    mock_term.show_cursor.assert_called()
    # Verify show_cursor was never called with arguments
    for call in mock_term.show_cursor.call_args_list:
        assert call.args == (), f"show_cursor called with args {call.args}"
        assert call.kwargs == {}, f"show_cursor called with kwargs {call.kwargs}"
    # clear() is called on every menu redraw plus once in finally
    assert mock_term.clear.call_count >= 1
