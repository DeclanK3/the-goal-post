# GoalPost: NFL Transition Modeling

GoalPost is a sport-agnostic framework for extracting football play-by-play data into transition matrices, simulating game outcomes, and building VAE-ready datasets from paired team-context observations. The NFL-specific implementation uses [nflverse](https://github.com/nflverse/nflverse-data) play-by-play data as its source of truth.

## Architecture

The codebase is organized into generic (sport-agnostic) and sport-specific layers:

| Layer | Purpose | NFL Implementation |
|-------|---------|-------------------|
| `goalpost.data.base` | Generic data-source abstractions (`DataSource`, `Game`, `Play`, `Possession`) | `goalpost.data.nfl` |
| `goalpost.data.nfl` | NFL-specific data acquisition and parsing | `NFLDataSource`, `fetcher.py`, `schedule.py` |
| `goalpost.representation.base` | Generic extraction, transition building, imputation | `GameExtractor`, `TransitionBuilder`, `Imputer` |
| `goalpost.representation.nfl` | NFL-specific extraction and transition modeling | `NFLGameExtractor`, `NFLTransitionBuilder`, `VAEDatasetBuilder` |
| `goalpost.simulation.base` | Generic simulation engine and market models | `Simulator`, `Market` |
| `goalpost.simulation.nfl` | NFL-specific simulation and weekly scheduling | `weekly_runner` |
| `goalpost.cli` | Command-line entry points | `--build`, `--simulate` |
| `goalpost.tui` | Interactive terminal UI | `run_tui()` |

### Key design principles

- **Sport-agnostic core:** The generic layers (`base`) define abstractions that could support other sports. NFL specifics are confined to `goalpost.data.nfl` and `goalpost.representation.nfl`.
- **No synthetic data in production:** Production code never generates, substitutes, or silently falls back to synthetic data. Missing or unavailable real data produces a clear failure. Synthetic data is strictly confined to tests for mocking and fixtures.
- **Explicit data acquisition:** The real-data path downloads nflverse release assets directly. There is no hidden caching, no fallback to demo data, and no silent production of empty datasets.

## Install and test

Requires Python 3.12+:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

## CLI usage

### Build a VAE-ready dataset from nflverse data

Download real NFLverse play-by-play data and build the transition dataset in one command:

```sh
python -m goalpost.cli.build \
  --seasons 2023,2024,2025 \
  --output artifacts/dataset
```

This will:
1. Download `play_by_play_{season}.parquet` from nflverse releases for each requested season
2. Cache the raw files in `~/.cache/goalpost/nflverse/`
3. Extract games, build transition matrices, and generate paired context observations
4. Write `vae_dataset.parquet` and `vae_schema.json` to `--output`

You can also provide pre-existing local parquet files:

```sh
python -m goalpost.cli.build \
  --sources /path/to/local/parquets \
  --output artifacts/dataset
```

**One of `--seasons` or `--sources` is required; both together is an error.**

#### Validation

Downloaded nflverse data is validated against the required schema before any processing begins. If a season's release asset is missing or the schema is unexpected, the command fails with a clear `NFLverseError` — no synthetic or empty substitute is produced.

### Run simulations

```sh
python -m goalpost.cli.simulate \
  --dataset artifacts/dataset \
  --week 5 \
  --season 2024 \
  --n-sims 10000
```

### Interactive TUI

Launch the terminal UI with no arguments:

```sh
python -m goalpost
```

From the TUI you can:
- **Generate Dataset:** Provide a sources directory, or type `download` to fetch nflverse data directly
- **Run Simulation:** Select a built dataset directory, week, and season
- **Inspect Game:** Load and inspect any parquet file

## Data flow

```
User specifies seasons (--seasons)
           |
           v
    fetch_nflverse([2023, 2024, 2025])
           |
           v
    Cache: ~/.cache/goalpost/nflverse/nfl-{season}.parquet
           |
           v
    NFLDataSource.parse() -> List[Game]
           |
           v
    extract_game() -> NFLExtractedGame
           |
           v
    NFLTransitionBuilder.build() -> per-team transition matrices
           |
           v
    extract_paired_observations() -> PairedContextObservation[]
           |
           v
    VAEDatasetBuilder.build() -> pl.DataFrame
           |
           v
    Write: {output}/vae_dataset.parquet + vae_schema.json
```

## NFLverse source

The downloader reads directly from nflverse GitHub release assets:

```
https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet
```

Polars reads these Parquet files over HTTPS. No API keys, credentials, or environment variables are required. Each season is downloaded and validated independently; a failure for any season aborts the entire build with a clear error.

## Output artifacts

After a successful build, the output directory contains:

| File | Description |
|------|-------------|
| `vae_dataset.parquet` | Paired context observations per game, flattened transition features |
| `vae_schema.json` | Column manifest: context columns, home/away feature names, total count |

The dataset schema includes:
- Context columns: `game_id`, `season`, `week`, `date`, `home_team`, `away_team`, `context_index`, `time_bin`, `score_bin`, `home_score`, `away_score`, `home_available`, `away_available`
- Home features: `home_{state}_to_{state}` for each valid transition
- Away features: `away_{state}_to_{state}` for each valid transition

A missing or unavailable dataset must produce a clear failure; the production pipeline never fabricates or substitutes synthetic data.

## Tests and synthetic data policy

The test suite is fully offline. Synthetic data is used **only** inside tests for mocking and fixtures:

- `tests/cli/test_integration.py` — `_make_synthetic_parquet()` creates minimal rows for CLI integration tests
- `tests/data/nfl/test_fetcher.py` — mocked nflverse downloads validate the download path without making network requests
- `tests/representation/nfl/test_dataset.py` — `_make_observation()` creates toy observations for unit tests

**No synthetic or demo data is ever created, consumed, or substituted by production code.** The production `build` CLI and TUI require either a valid local sources directory or a successful nflverse download. Empty or missing inputs raise `FileNotFoundError` or `ValueError`; bad downloads raise `NFLverseError`.

## Known limitations

- The extraction supports NFL regulation play only. Overtime, preseason, and special-teams scoring details are not fully modeled.
- College football support exists in the architecture but is not currently wired to a live data source.
- The VAE dataset is a structured representation of transition probabilities; it does not itself train a VAE.
- Simulation outputs are exploratory and not validated betting probabilities.

## License

MIT

### TUI input and diagnostics

Type directly into fields, use Backspace to edit, Enter to advance, and Esc to
cancel. `q` exits only at the main menu, so paths containing q remain usable.
Ctrl-C exits and restores the terminal. Action errors return to the menu; fatal
terminal errors print the diagnostic path after cleanup. The TUI uses the same
build and simulation services as the CLI. Empty simulation results are errors.

Local tracebacks and action progress go to `~/.local/state/goalpost/tui.log`
(or `$XDG_STATE_HOME/goalpost/tui.log`). Set `GOALPOST_LOG_DIR` to override the
directory. Logs rotate at 1 MiB with three backups and redact Odds API secrets;
input fields and environment contents are not logged. Native terminal startup
failures are logged too. Ratatui requires its platform-native FFI library; a
missing backend is reported explicitly instead of disappearing into the screen.

The offline TUI tests exercise menu actions and cleanup using a terminal double;
they do not require a native library or mutate the test runner's terminal.
