# NFL 2026 Super Bowl Predictor

Predicts every remaining NFL game, then simulates the rest of the season **50,000 times** to give each team's chance of making the playoffs, winning its division, winning the AFC/NFC and winning the Super Bowl.

Built on free public data (no scraping, no paid APIs). Output is a single static HTML page.

**Most important number in the repo:** the model grades itself against the Las Vegas closing spread, every week, in public.

| Benchmark (2024–25 test seasons, 570 games) | Winners called | Brier ↓ | Avg. margin error |
|---|---|---|---|
| This model | 64.9% | 0.2142 | 9.98 pts |
| Elo ratings alone | 66.1% | 0.2179 | 10.14 pts |
| **Vegas closing spread** | **68.2%** | **0.2077** | **9.69 pts** |

Vegas is still ahead. That's the point of publishing the comparison.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scikit-learn pyarrow requests
.venv/bin/python fetch_data.py      # ~2 min, downloads ~500 MB
./run_all.sh                        # ~35 s
open index.html
```

`run_all.sh` is also the weekly update: refresh data → retrain → re-simulate → rebuild the page → save a snapshot.

---

## How it works

**1. Data (nflverse, 1999–2026)** — schedules and results with Vegas spreads, play-by-play (1.28M plays → EPA per team per game), injury reports, snap counts, weekly rosters (IR/PUP), depth charts, draft picks.

**2. Ratings, built leak-free** — every game is walked in date order and features are recorded *before* kickoff: Elo with margin-of-victory and between-season regression, rolling offensive/defensive EPA, and a quarterback rating from each QB's own past starts.

**3. The model** — ridge regression on the **point margin**, converted to a win probability with a normal distribution (σ ≈ 13.4 pts):

```
margin = 0.564·Elo_diff + 0.082·off_diff + 0.129·def_diff
       + 0.176·qb_diff  + 2.240·home     + 1.516·injury_diff
```

Because it's a sum, each factor's contribution in points is known — that's where the plain-English "why" labels come from.

**4. Factors are kept only if they earn it** — each candidate is scored on validation seasons the final test never touches. Dropped so far: rest days, new head coach, rookie QB, draft capital, defensive injuries, head-to-head record, travel distance, time zones.

**5. Injuries** — snap counts measure how much a player matters; 2012–2025 history gives the odds he's still out in each future week (by status, injury type and time already missed; IR/PUP use reserve-list history). Future starting QBs come from the depth chart combined with those odds, e.g. *Lock 76% / Darnold 24%*.

**6. Simulation** — real results locked in, remaining games drawn around their predicted margin plus a per-team strength offset, then the real NFL rules: 4 division winners + 3 wild cards per conference, full tiebreakers (head-to-head, division record, common games, conference record, strength of victory, strength of schedule) and re-seeding each round.

---

## Files

| File | Purpose |
|---|---|
| `fetch_data.py` | Download nflverse data (`--current` refreshes only this season) |
| `nfl_model.py` | Ratings, features, factor selection, backtest vs Vegas |
| `injuries.py` | Player importance + injury-duration model |
| `simulate.py` | Monte Carlo season + playoffs with NFL tiebreakers |
| `tracking.py` | Pick ledger: picks frozen at kickoff, scored vs Vegas |
| `generate_html.py` | Builds `index.html` |
| `compare_models.py` | Ridge vs gradient-boosted trees |
| `versions.py` | Frozen model versions + comparison log |
| `tune.py` | Rating-parameter search (occasional, not weekly) |
| `injury_overrides.csv` | Manual injury timelines from the news |
| `run_all.sh` | The weekly update |

Data files aren't committed — `fetch_data.py` rebuilds them from source.

---

## Things that were tested and rejected

Honest results, because negative results are still results:

- **Head-to-head record** — near zero, fitted weight even slightly negative. NFL rosters turn over too fast.
- **Rest days, bye weeks, short weeks** — the NFL schedules Thursday games so *both* teams are on short rest, which cancels the effect.
- **Travel distance and time zones** — made predictions worse; "home field" already contains most of it.
- **Rookie QB, draft capital, new head coach, defensive injuries** — no reliable gain.
- **Gradient-boosted trees (XGBoost family)** — worse probabilities than the linear model; deeper trees collapsed. ~285 games a season is not much data.
- **Tuning the Elo settings** — best validation gain was noise-level and did worse on test, so the defaults stayed.

Six ideas are re-tested automatically on every run and logged to `snapshots/factor_watch.csv`. Nothing is ever added to the model automatically.

---

## Known limitations

- **Blind to off-season roster moves** until the new players actually play.
- **No access to practice reports or late lineup news**, which the betting market has.
- **Team strength is fixed within each simulated season** (no hot/cold streaks).
- The last tiebreaker steps (net points) fall back to a coin toss.

---

## Data source

All data from [nflverse](https://github.com/nflverse/nflverse-data) — an open-source NFL data project. Thanks to everyone who maintains it.
