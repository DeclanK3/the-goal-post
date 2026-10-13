> Scaffold design document. The active transition implementation now lives in `goalpost.transitions`; see the root README for its tensor format and elapsed-clock API. Former terminal-state/training examples referenced below are retired and this scaffold is not yet integrated with the replacement.

# GoalPost — Four-Layer Architecture

```
┌─────────────────────────────────────────┐
│  Layer 1: DATA                          │
│  └── DataSource (ABC)                   │
│      ├── NFLVerseSource                 │
│      ├── ESPNSource                     │
│      └── UnifiedDataSource              │
│                                         │
│  Extracts: raw play-by-play             │
│  Outputs: Game objects with possessions │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Layer 2: REPRESENTATION                │
│  └── TeamRepresentation (ABC)           │
│      ├── BayesianTeamUpdater             │
│      └── VAEEncoder (future)            │
│                                         │
│  Extracts: team embeddings from history │
│  Outputs: z vectors per team            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Layer 3: PREDICTION                    │
│  └── GameModel (ABC)                    │
│      └── NFLDNN (future)                │
│                                         │
│  Input: z_home, z_away, game context    │
│  Learns: predict TransitionModel from   │
│          team representations           │
│  Output: predicted TransitionModel for  │
│          upcoming game                  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Layer 4: SIMULATION                    │
│  └── Simulator (ABC)                    │
│      └── MonteCarloSimulator            │
│                                         │
│  Input: predicted TransitionModel     │
│  Output: score distribution             │
└─────────────────────────────────────────┘
```

## Key Flow

1. **Data** extracts transition matrices from historical games (labels)
2. **Representation** learns team embeddings from historical data
3. **Prediction** learns to map (z_A, z_B, context) → predicted transition matrix
4. **Simulation** uses predicted matrix to score upcoming game

## Architecture Components

### Layer 1: DATA
**Base Class:** `DataSource` — `fetch()`, `parse()`, `load()`

**NFL Implementations:**
- `NFLVerseSource` — nflverse historical (1999–2025), rich features
- `ESPNSource` — ESPN API live (2026+), same-day data
- `UnifiedDataSource` — routes between sources, tracks provenance for backfill

**NCAAB Example (future):**
- `StatsBroadcastNCAABSource` — XML game feeds

---

### Layer 2: REPRESENTATION
**Base Class:** `TeamRepresentation` — `fit()`, `encode()`, `update()`

**Purpose:** Learn a fixed-size vector z for each team that captures their offensive/defensive tendencies.

**Current Implementation:**
- `BayesianTeamUpdater` — Beta-Bernoulli conjugate priors, O(1) online updates

**Future:**
- `VAEEncoder` — variational autoencoder for deep representations
- `ContrastiveEncoder` — contrastive learning from similar teams

---

### Layer 3: PREDICTION
**Base Class:** `GameModel` — `fit()`, `predict_transition_model()`

**Purpose:** The supervised learning layer. Train on thousands of historical games where:
- **Input features:** (z_home, z_away, game_context)
- **Label:** The actual `TransitionModel` extracted from that game's play-by-play

**Learned mapping:** `(z_A, z_B, context) → predicted_transition_matrix`

**Current Status:** Not yet implemented. We are building the dataset now:
1. Extract `TransitionModel` from every historical game (labels)
2. Compute `TeamRepresentation` for each team (features)
3. Train `GameModel` to predict TransitionModel from representations

**Future Implementation:**
- `NFLDNN` — neural network that outputs predicted transition probabilities

**Why this matters:** Without this layer, we can only simulate games we've already seen. With this layer, we can predict how Team A will play against Team B next Sunday.

---

### Layer 4: SIMULATION
**Base Class:** `Simulator` — `simulate_game()`, `price_markets()`

**Purpose:** Take a predicted `TransitionModel` and simulate the upcoming game 10,000 times to get a score distribution.

**NFL Implementation:**
- `MonteCarloSimulator` — Monte Carlo forward simulation using predicted transition matrix

---

## Sport Expansion

| Sport | DataSource | PossessionExtractor | TransitionModel | TeamRepresentation | GameModel | Simulator |
|-------|-----------|---------------------|-----------------|-------------------|-----------|-----------|
| NFL | NFLVerseSource, ESPNSource | NFLDriveExtractor | NFLTransitionModel | BayesianTeamUpdater | NFLDNN | MonteCarloSimulator |
| NBA | NBAStatsSource | NBAPossessionExtractor | NBATransitionModel | (future) | (future) | (future) |
| MLB | MLBStatsSource | MLBInningExtractor | MLBTransitionModel | (future) | (future) | (future) |

## Current Status

**Working:**
- ✅ Layer 1: Data (NFLVerseSource, ESPNSource, UnifiedDataSource)
- ✅ Layer 2: Representation (BayesianTeamUpdater)
- ✅ TransitionModel extraction (NFLTransitionModel from single games)
- ✅ Simulator (MonteCarloSimulator)

**Next:**
- ⏳ Build dataset: extract thousands of (z_home, z_away, TransitionModel) tuples
- ⏳ Implement GameModel (NFLDNN) — the prediction layer
- ⏳ Train and evaluate GameModel on held-out games

## Dataset for Layer 3

Each training example:
```python
{
    "features": {
        "z_home": np.ndarray(8),      # Team A representation
        "z_away": np.ndarray(8),      # Team B representation  
        "context": {
            "is_home": bool,
            "week": int,
            "season": int,
        }
    },
    "label": {
        "play_transitions": {         # Actual extracted from game
            "1st_10": {"2nd_long": 0.39, "1st_10": 0.28, "2nd_medium": 0.15},
            "2nd_long": {"3rd_long": 0.48, "1st_10": 0.23, "3rd_short": 0.13},
            # ... etc
        },
        "drive_results": {
            "red_zone": {"td": 0.38, "fg": 0.30, "punt": 0.12},
            "own_40": {"punt": 0.39, "td": 0.19, "fg": 0.16},
            # ... etc
        }
    }
}
```

The GameModel learns to predict this label from the features.

## Downstream market comparison (implemented)

`goalpost.markets` is a separate consumer of the standard weekly simulation
export; it does not alter the four modeling layers. Its supported path is:

```
weekly PMFs + report/schedule metadata
    -> export validation + exact dataset/hash/history checks
    -> existing stored Odds API quotes (read-only SQLite)
    -> exact FanDuel main-market probabilities and research EV/Kelly tables
```

`bridge.py` owns the CLI, export binding and synthetic demo; `history.py` checks
public schedule coverage and actual simulation donors; `comparison.py` matches
upcoming events and selects the latest main-market pairs; `pricing.py` contains
PMF checks and pure win/push/loss and EV/Kelly arithmetic. The package includes no
odds collection client, sender, live admission controller or betting operation.
See [market sizing](docs/market-sizing.md) for the executable contract.
