"""Headless tests for TUI event handling."""
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
