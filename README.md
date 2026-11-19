# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/patriarkma/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/patriarkma/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

`main.ts` forecasts, for each of GTA VI's 55 PS5 trophies, the percentage of Exophase-tracked players who will have earned it one week after launch, as five quantiles (5/25/50/75/95). It works in logit space, fitting a growth model to the Exophase read history for each trophy, with pooled shrinkage across trophies so noisy early reads don't throw any single forecast off.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## How it works

Each trophy's earn rate is modelled in logit space, against a time axis of `ln(1 + hours since launch)`. Per trophy, a weighted least-squares fit of `logit(rate)` on that axis uses recency weights, so the latest reads matter most. The slope used for extrapolation is the local slope at the last read, taken from a weighted quadratic fit and damped so it is never steeper than the linear slope — this captures the S-curve flattening as a trophy approaches its ceiling. That slope is then shrunk Bayesian-style toward the median slope across all 55 trophies, which keeps sparsely-read trophies from running away.

The spread is honest: a random-walk term that widens with the horizon, plus the fit residual, plus slope uncertainty times the horizon. A trophy already at 100% is forecast to stay at 100% (with a small lower tail), and rates are allowed to fall, since an influx of new players dilutes early trophies.

`main.ts` runs on Node 22 with the standard library only, fully offline, deterministic (two runs on the same case write byte-identical output), and finishes in well under 8 seconds per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9g59dt7w&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
