from goalpost.tui.screens import draw_main_menu


def test_main_menu_contains_options():
    text = draw_main_menu(0)
    assert "Generate Dataset" in text
    assert "Run Simulation" in text
    assert "Inspect Game" in text
    assert "Settings" in text
    assert "Exit" in text


def test_main_menu_shows_selection():
    text = draw_main_menu(0)
    lines = text.splitlines()
    gen_line = [ln for ln in lines if "Generate Dataset" in ln][0]
    assert ">" in gen_line
