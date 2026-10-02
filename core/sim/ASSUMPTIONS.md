# Simulator assumptions

The Monte Carlo simulator (`core/sim/model.py`) is meant to be honest and calibrated, not
a team-grade strategy tool. Each assumption is listed below with its value and source.
Every input comes from information available **at the moment of prediction**: the laps
already run in this race, plus races from **earlier seasons**.

## Inputs

| Input | Source / treatment | Where |
|---|---|---|
| Current order, gaps, laps down | The current `RaceState` tick (from `RaceFeed`) | `inputs.py` |
| Tyre compound, age, compounds used | `RaceState` history so far | `inputs.py` |
| Driver pace | Median of the last ≤10 clean laps (fuel-, compound- and age-corrected), relative to the field median, shrunk towards a grid prior. Weight of observed pace = n / (n + 8). | `inputs.py` |
| Grid pace prior | 0.10 s/lap per grid slot from the middle of the grid (tuned on 2023, see backtest) | `inputs.py` |
| Tyre degradation | Prior: fitted on the most recent *earlier* dry race at the circuit. Updated in-race with this race's clean laps: (40 × prior + n × observed) / (40 + n). Clamped to [0, 0.4] s/lap. | `degradation.py`, `history.py` |
| Pit loss | Median green-flag pit loss at the circuit in earlier seasons (in-lap + out-lap − 2 × clean lap); global median if none; 22 s default | `history.py` |
| Safety car / VSC probability | Share of earlier races at the circuit with an SC / VSC, shrunk towards the global rate with 4 pseudo-races | `history.py` |
| DNF hazard | Global DNFs per car-lap in earlier seasons (measured ~0.002, about 12% per car per race; the 0.0008 default is used only without history) | `history.py` |
| Overtaking difficulty | `pass_prob` = 0.35 × (circuit's on-track passes per car-lap ÷ global rate), shrunk with 1 pseudo-race (passing data is plentiful and strongly circuit-specific), clipped to [0.05, 0.8] | `history.py` |
| Mandatory compound rule | In a dry race, a car that hasn't used two dry compounds must stop before the flag | `model.py` |

## Degradation fit

- **Fuel correction:** a car gains 0.03 s/kg as it burns fuel. Start fuel is assumed to be 100 kg, burned linearly over the race distance. This is the usual public estimate; teams don't publish their real fuel loads.
- **Clean laps:** lap > 1, not an in- or out-lap, not deleted, no SC/VSC/red flag during the lap, no rain on slicks, and within 1.5 s of the stint median.
- **The fit:** a linear fit of lap time against tyre age per compound, *within each stint* (one intercept per driver-stint). Needs ≥ 4 laps per stint and ≥ 15 laps per compound; otherwise the defaults apply: SOFT 0.10, MEDIUM 0.07, HARD 0.05, INTERMEDIATE 0.08, WET 0.06 s/lap.
- **Known confound:** within a stint, tyre age and lap number rise together, so track evolution is folded into the slope. That's why drying wet races yield negative slopes. They are clamped to 0.
- **Fresh-tyre pace offsets vs MEDIUM:** SOFT −0.6 s/lap, HARD +0.4 s/lap. These are assumed typical values, not fitted.

## Race model (per lap)

| Assumption | Value |
|---|---|
| Lap time | reference lap + driver pace + compound offset + degradation × tyre age + N(0, 0.4 s) |
| Model's own pace uncertainty | per simulation and driver, N(0, σ_d), with σ_d = σ₀ √(k / (k + n_obs)): it shrinks as more of the driver's laps are observed (σ₀, k tuned on 2023) |
| Pace shrink | final pace deltas × λ ≤ 1 (tuned on 2023): short-run pace differences partly reflect traffic and strategy, not car speed |
| Pit strategy (economic) | Each lap, for the laps left on the current set (horizon H = tyre age + laps left), pick n ∈ {0, 1, 2} more stops minimising deg·H²/(2(n+1)) + n·stop cost. Stop when tyre age ≥ U(0.85, 1.15) × H/(n+1). Stop cost = pit loss × (SC 0.5 / VSC 0.6 / green 1) × track-position factor. |
| Track-position factor | 1 + (1 − p)·(0.35 / p), with p = the track's pass probability: about 1.65 at a typical track, about 7.6 at Monaco. Losing a position you can't regain costs far more than the pit-lane time. |
| Tyre wear used for strategy | fitted degradation + 0.03 s/lap track evolution. The fit is net of evolution, but fresh tyres also enjoy the faster track. |
| Tyre-life safety caps | SOFT 45, MEDIUM 60, HARD 80, INTER/WET 60 laps (Monaco 2024: 77 laps on one set of hards) |
| Compound after a stop | ≤ 18 laps left: SOFT; ≤ 30: MEDIUM; else HARD (then adjusted to satisfy the two-compound rule) |
| Pit stop | track pit loss + N(0, 0.8 s); × 0.5 under SC, × 0.6 under VSC |
| Cheap stop | emerges from the economics: under SC/VSC the stop cost halves, so more cars stop |
| Safety car | per-lap hazard from the race probability; lasts 3–6 laps; bunches the field to 0.8 s gaps; laps at 1.4 × reference; no overtaking |
| VSC | lasts 1–3 laps; laps at 1.35 × normal pace; gaps kept; no overtaking |
| Overtaking | a car that would get ahead on time only completes the pass if its *systematic* pace (driver pace + tyre offset + degradation) is better, with probability pass_prob × min(1, advantage / 0.5 s); otherwise it's held 0.3 s behind. Lap-to-lap noise alone never produces a pass, and position changes caused by pit stops are not passes. |
| DNF | per-lap hazard; a retired car is classified behind all finishers, ordered by laps completed |
| Rain scenario | slicks lose 8 s/lap after rain arrives; each car switches to intermediates with probability 0.6 per lap |

## Known limitations

- **Driver behaviour:** no team orders, no defensive driving, no tyre management, and no strategic reaction to rivals' stops. Undercut and overcut emerge only from tyre pace and pit loss.
- **One reference lap time for all compounds:** no separate performance curves, and no tyre warm-up (out-lap) penalty beyond pit loss.
- **Weather:** only the explicit rain scenario. The weather forecast isn't used.
- **Lapped cars:** they keep their gap in seconds, a simplification of the timing-line rules.
- **Wet races:** pace and degradation are poorly identified, and the backtest covers mostly dry races.
