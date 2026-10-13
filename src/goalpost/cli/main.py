"""Main CLI entry point. Defaults to TUI; preserves --build and --simulate for scripting."""
import sys


def main():
    # If no args, launch interactive TUI
    if len(sys.argv) == 1:
        from goalpost.tui.main import run_tui
        run_tui()
        return

    # If first arg is --build or --simulate, delegate
    cmd = sys.argv[1]
    if cmd == "--build":
        # Remove --build from args so argparse works
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from goalpost.cli.build import main as build_main
        build_main()
    elif cmd == "--simulate":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from goalpost.cli.simulate import main as simulate_main
        simulate_main()
    else:
        from goalpost.tui.main import run_tui
        run_tui()


if __name__ == "__main__":
    main()
