# Simulator backtest

Held-out test season **2024**: 24 races, 1812 driver-checkpoint predictions (1000 simulations per checkpoint, 220 ms each). Knobs tuned on 2023 (22 races): pace uncertainty = 0.2 s/lap (shrinking as laps are observed), grid-pace prior = 0.05 s/lap per grid slot, observed-pace prior weight = 25 laps. Track parameters and degradation priors use only seasons before each race. Lower is better for both scores.

## Overall

| Model | Brier P(top 3) | Log loss P(top 3) | Brier P(win) |
|---|---:|---:|---:|
| Simulator | 0.061 | 0.205 | 0.028 |
| Order holds | 0.077 | 0.712 | 0.035 |
| Historical lookup | 0.058 | 0.201 | n/a |

## By race distance

| Checkpoint | Simulator | Historical lookup | Order holds |
|---|---:|---:|---:|
| Grid (lap 0) | 0.075 | 0.070 | 0.101 |
| 25% distance | 0.065 | 0.075 | 0.105 |
| 50% distance | 0.053 | 0.048 | 0.054 |
| 75% distance | 0.050 | 0.039 | 0.046 |

Brier score of P(top 3).

## Calibration (simulator P(top 3))

| Predicted bin | Predictions | Mean predicted | Observed frequency |
|---|---:|---:|---:|
| 0.0-0.1 | 1305 | 0.01 | 0.01 |
| 0.1-0.2 | 85 | 0.15 | 0.18 |
| 0.2-0.3 | 62 | 0.24 | 0.24 |
| 0.3-0.4 | 44 | 0.36 | 0.39 |
| 0.4-0.5 | 38 | 0.45 | 0.53 |
| 0.5-0.6 | 47 | 0.54 | 0.53 |
| 0.6-0.7 | 49 | 0.65 | 0.63 |
| 0.7-0.8 | 45 | 0.74 | 0.71 |
| 0.8-0.9 | 62 | 0.85 | 0.85 |
| 0.9-1.0 | 75 | 0.95 | 0.84 |

## Tuning grid (2023, Brier P(top 3))

| Pace uncertainty | Grid-pace prior | Prior laps | Brier |
|---:|---:|---:|---:|
| 0.2 | 0.05 | 8 | 0.0594 |
| 0.2 | 0.05 | 25 | 0.0576 |
| 0.2 | 0.1 | 8 | 0.0587 |
| 0.2 | 0.1 | 25 | 0.0579 |
| 0.35 | 0.05 | 8 | 0.0610 |
| 0.35 | 0.05 | 25 | 0.0600 |
| 0.35 | 0.1 | 8 | 0.0586 |
| 0.35 | 0.1 | 25 | 0.0582 |
| 0.5 | 0.05 | 8 | 0.0629 |
| 0.5 | 0.05 | 25 | 0.0630 |
| 0.5 | 0.1 | 8 | 0.0595 |
| 0.5 | 0.1 | 25 | 0.0594 |

Races: Bahrain Grand Prix, Saudi Arabian Grand Prix, Australian Grand Prix, Japanese Grand Prix, Chinese Grand Prix, Miami Grand Prix, Emilia Romagna Grand Prix, Monaco Grand Prix, Canadian Grand Prix, Spanish Grand Prix, Austrian Grand Prix, British Grand Prix, Hungarian Grand Prix, Belgian Grand Prix, Dutch Grand Prix, Italian Grand Prix, Azerbaijan Grand Prix, Singapore Grand Prix, United States Grand Prix, Mexico City Grand Prix, São Paulo Grand Prix, Las Vegas Grand Prix, Qatar Grand Prix, Abu Dhabi Grand Prix.

Reproduce: `make backtest` (needs the races in the local replay cache; see `scripts/backtest_sim.py`).
