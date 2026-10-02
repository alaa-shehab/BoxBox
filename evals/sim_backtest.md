# Simulator backtest

Held-out test season **2024**: 24 races, 1812 driver-checkpoint predictions (1000 simulations per checkpoint, 208 ms each). Knobs tuned on 2023 (22 races): pace uncertainty = 0.2 s/lap (shrinking as laps are observed), grid-pace prior = 0.05 s/lap per grid slot, observed-pace prior weight = 25 laps, pace shrink = 1.0, pass-probability scale = 0.5. Track parameters and degradation priors use only seasons before each race. Lower is better for both scores.

## Overall

| Model | Brier P(top 3) | Log loss P(top 3) | Brier P(win) |
|---|---:|---:|---:|
| Simulator | 0.059 | 0.203 | 0.026 |
| Order holds | 0.077 | 0.712 | 0.035 |
| Historical lookup | 0.058 | 0.201 | n/a |

## By race distance

| Checkpoint | Simulator | Historical lookup | Order holds |
|---|---:|---:|---:|
| Grid (lap 0) | 0.074 | 0.070 | 0.101 |
| 25% distance | 0.067 | 0.075 | 0.105 |
| 50% distance | 0.052 | 0.048 | 0.054 |
| 75% distance | 0.044 | 0.039 | 0.046 |

Brier score of P(top 3).

## Calibration (simulator P(top 3))

| Predicted bin | Predictions | Mean predicted | Observed frequency |
|---|---:|---:|---:|
| 0.0-0.1 | 1304 | 0.01 | 0.01 |
| 0.1-0.2 | 96 | 0.14 | 0.18 |
| 0.2-0.3 | 52 | 0.24 | 0.27 |
| 0.3-0.4 | 53 | 0.36 | 0.32 |
| 0.4-0.5 | 29 | 0.44 | 0.41 |
| 0.5-0.6 | 48 | 0.55 | 0.54 |
| 0.6-0.7 | 39 | 0.65 | 0.79 |
| 0.7-0.8 | 54 | 0.75 | 0.72 |
| 0.8-0.9 | 55 | 0.85 | 0.78 |
| 0.9-1.0 | 82 | 0.95 | 0.88 |

## Tuning grid (2023, Brier P(top 3))

| Pace uncertainty | Grid-pace prior | Prior laps | Pace shrink | Pass scale | Brier |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.05 | 25 | 0.5 | 0.5 | 0.0573 |
| 0.2 | 0.05 | 25 | 0.5 | 1.0 | 0.0608 |
| 0.2 | 0.05 | 25 | 0.75 | 0.5 | 0.0561 |
| 0.2 | 0.05 | 25 | 0.75 | 1.0 | 0.0589 |
| 0.2 | 0.05 | 25 | 1.0 | 0.5 | 0.0558 |
| 0.2 | 0.05 | 25 | 1.0 | 1.0 | 0.0581 |
| 0.2 | 0.1 | 25 | 0.5 | 0.5 | 0.0565 |
| 0.2 | 0.1 | 25 | 0.5 | 1.0 | 0.0586 |
| 0.2 | 0.1 | 25 | 0.75 | 0.5 | 0.0563 |
| 0.2 | 0.1 | 25 | 0.75 | 1.0 | 0.0581 |
| 0.2 | 0.1 | 25 | 1.0 | 0.5 | 0.0569 |
| 0.2 | 0.1 | 25 | 1.0 | 1.0 | 0.0582 |
| 0.35 | 0.05 | 25 | 0.5 | 0.5 | 0.0604 |
| 0.35 | 0.05 | 25 | 0.5 | 1.0 | 0.0650 |
| 0.35 | 0.05 | 25 | 0.75 | 0.5 | 0.0586 |
| 0.35 | 0.05 | 25 | 0.75 | 1.0 | 0.0622 |
| 0.35 | 0.05 | 25 | 1.0 | 0.5 | 0.0579 |
| 0.35 | 0.05 | 25 | 1.0 | 1.0 | 0.0606 |
| 0.35 | 0.1 | 25 | 0.5 | 0.5 | 0.0583 |
| 0.35 | 0.1 | 25 | 0.5 | 1.0 | 0.0613 |
| 0.35 | 0.1 | 25 | 0.75 | 0.5 | 0.0572 |
| 0.35 | 0.1 | 25 | 0.75 | 1.0 | 0.0592 |
| 0.35 | 0.1 | 25 | 1.0 | 0.5 | 0.0568 |
| 0.35 | 0.1 | 25 | 1.0 | 1.0 | 0.0586 |

Races: Bahrain Grand Prix, Saudi Arabian Grand Prix, Australian Grand Prix, Japanese Grand Prix, Chinese Grand Prix, Miami Grand Prix, Emilia Romagna Grand Prix, Monaco Grand Prix, Canadian Grand Prix, Spanish Grand Prix, Austrian Grand Prix, British Grand Prix, Hungarian Grand Prix, Belgian Grand Prix, Dutch Grand Prix, Italian Grand Prix, Azerbaijan Grand Prix, Singapore Grand Prix, United States Grand Prix, Mexico City Grand Prix, São Paulo Grand Prix, Las Vegas Grand Prix, Qatar Grand Prix, Abu Dhabi Grand Prix.

Reproduce: `make backtest` (needs the races in the local replay cache; see `scripts/backtest_sim.py`).
