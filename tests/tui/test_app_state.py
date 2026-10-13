from goalpost.tui.app_state import AppState


def test_default_state():
    s = AppState()
    assert s.current_screen == "main"
    assert s.selected_menu == 0
    assert s.sources_dir is None
    assert s.status_message == ""


def test_state_mutation():
    s = AppState()
    s.selected_menu = 2
    s.status_message = "hello"
    assert s.selected_menu == 2
    assert s.status_message == "hello"
