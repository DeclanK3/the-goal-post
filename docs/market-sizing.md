# Astra export → stored odds → EV/Kelly table

This is the repository version of the existing workspace handoff bridge. It
consumes the standard `goalpost.transitions.weekly` export without rerunning
simulations. Compatibility with a different, uncommitted Astra CSV format still
requires a sample; the bridge rejects ambiguous inputs instead of guessing.

## Run

Use the project's normal environment (`python -m pip install -e '.[dev]'`).
No dependencies beyond the existing package are added.

```bash
python -m goalpost.markets --demo
python -m goalpost.markets --export /path/to/weekly-export --validate-only
python -m goalpost.markets --export /path/to/weekly-export \
  --data /path/to/export-dataset --database /path/to/betting.db \
  --output /path/to/new-output
```

The installed alias is `goalpost-market-sizing`. `--output` defaults to a new
UTC timestamped folder under `./artifacts/market-sizing`. Existing output
folders are refused. CLI failures exit nonzero and retain `failure.json` in a
newly created output folder; no prior comparison is substituted.

`--demo` uses packaged fictional teams and prices at a fixed January 1, 2000
clock, creates an isolated synthetic SQLite file under its output, and makes no
network calls. It cannot be combined with real export, dataset or database
inputs. `--validate-only` checks exports offline without opening a quote database.

A live comparison requires the exact model dataset used by the export and an
accessible **existing** quote database. It verifies all three dataset hashes,
then makes one public nflverse schedule read to check current completed-game
coverage and actual simulation donor history. Missing history, provenance,
source access or mismatched hashes blocks the comparison. Public schedule
retrieval does not rebuild model data and is not a paid odds request.

## Export contract

```text
weekly-export/
  report.json
  schedule-espn.json
  GAME_ID/
    total-pmf.csv
    home_minus_away-pmf.csv
```

Each CSV has `points,count,probability`. Points/counts are finite integers,
counts are nonnegative and sum to retained `n`, and probabilities equal
`count/n` and sum to one. Total points cannot be negative. Home margin is
**home minus away**: positive values favor home. Missing or malformed PMFs,
duplicate game IDs, path traversal and symlinks escaping the export are rejected.

`report.json` declares `regulation_only=true`, `betting_admitted=false`, and a
`results` list. Each result retains `game.game_id`, `game.home`, `game.away`,
`game.kickoff_utc`, `status`, `attempted`, `n`, and optional
`diagnostics.grace_completed_games`. Blocked/zero-completion matchups remain
visible. A simulated matchup must have positive retained paths. Approximate
completions must be between zero and retained paths.

The untouched `schedule-espn.json` must identify each home/away pair and kickoff
exactly once. The bridge checks orientation and timezone-aware kickoff against
that schedule; an arbitrary team-name dictionary is insufficient. The standard
export with the automatic ESPN schedule is supported. A custom `--schedule`
export without `schedule-espn.json` needs an explicit adapter and is not accepted.

Live use also requires `dataset_hashes` for `transition_set.json`,
`nfl_games.parquet`, `parsed_segments.parquet`, plus the original per-game
`provenance.teams` donor records. Send both metadata files unchanged alongside
the PMFs. CSVs alone support neither identity validation nor live pricing.

## Stored quote adapter

This command reads stored Odds API evidence; **it does not fetch Odds API prices**.
No API key, credentials, production database, alerts or messaging helper is
bundled. The caller supplies the database path; there are no machine-specific
paths. The SQLite connection uses `mode=ro`, `PRAGMA query_only=ON`, and one read
transaction. It requires this engine schema (additional columns are fine):

| Table | Required columns |
|---|---|
| `games` | `id, league, home_team, away_team, commence_time_utc, status` |
| `markets` | `id, game_id, market_type, participant, line` |
| `odds_snapshots` | `id, market_id, book, source, odds_american, taken_at, provider_updated_at, event_context_json` |
| `flags` | `key, value` (the NFL hold is read for disclosure only) |

Only `league='nfl'`, `book='fanduel'`, `source IS NULL` rows are compared.
Games must uniquely match both team names and kickoff within ten minutes, have
`status='scheduled'`, and be upcoming relative to the earlier kickoff. Main
`h2h`, `spreads`, `totals` families require an exact two-sided pair at the newest
family capture: opposing teams with negated spreads, identical total lines,
or moneyline sides with no line. If that latest pair is incomplete or ambiguous,
older pairs and alternate lines are excluded; they are never fallback prices.

Capture and provider-update timestamps must both be within 30 minutes for the
current table, precede kickoff, and not be more than 60 seconds in the future.
Missing provider timestamps are unpriced. Captured event context, when present,
must agree with the stored game. Missing context is explicitly labeled
`current_database_identity_only`, never quote-time identity verification.
API quote evidence is not an accepted-bet receipt.

## Tables and evidence

- `comparison/current-market-sizing.csv` and `.json`: fresh evaluated main-market
  contracts sorted by hypothetical EV, including negative rows; no minimum pick
  count or positive edge is forced.
- `comparison/ev-kelly.csv`: all evaluated latest complete pairs, including stale
  and unpriced rows.
- `comparison/matchup-summary.csv`: every matchup and its coverage/identity status,
  including failed simulations, missing families and missing prices.
- `comparison/quote-evidence.json`: latest snapshot per stored contract, retaining
  selected, superseded, alternate and incomplete-pair dispositions.
- `comparison/report.json` and `README.md`: counts, hold disclosure and limitations.
- `input-validation.json`: hashes and validation denominators;
  `data-freshness/`: live public-source evidence and model-history disposition.

Table fields include selection/line/American odds, win/push/loss probability,
hypothetical EV %, full/quarter-Kelly % of bankroll, quote IDs/ages, attempted,
retained, incomplete and approximate counts, PMF hash and missing-path bounds.
Report metadata binds the dataset, simulation, schedule and bridge code hashes.

`--bankroll` is optional and must be positive. Omit it for percentages only;
explicitly supplying it adds hypothetical quarter-Kelly dollar arithmetic.
Correlated single-contract Kelly fractions must not be summed as a portfolio.

## Arithmetic and limitations

For win probability W, push P, loss L, and net offered payout b:

```text
EV = b*W - L
full Kelly = max(0, EV / (b*(W+L)))
quarter Kelly = full Kelly / 4
```

All-push distributions receive zero allocation. EV uses the actual offered odds,
including vig. De-vigging a reference book is a separate comparison; imaginary
no-vig payouts do not represent the offer being priced.

The demo has six contracts. Fictional Example Home moneyline at -200 has W=70%,
regulation tie=10%, L=20%: EV=15%, full Kelly=33.3333%, quarter Kelly=8.3333%.
Its -3 spread retains 40% push mass. These are deliberately simple arithmetic
fixtures, not plausible NFL edge estimates. Offline tests check all six rows
against the original handoff's expected values.

**Research only:** the PMFs are uncalibrated, conditional on retained paths, and
regulation-only. Missing-path bounds are sensitivity bounds, not confidence
intervals or evidence of unbiasedness. Moneyline ties are void only for the
illustrative calculation. `actual_contract_ev` and `actual_contract_kelly` remain
null and `betting_admitted=false` while overtime-inclusive sportsbook contracts
are unmodeled. This integration does not change simulation, imputation, clock,
VAE, studies, production holds, collection, alerting or betting behavior.

Run offline bridge tests with `python -m pytest tests/markets -q`; the existing
repository CI also includes them in `python -m pytest -q`.
