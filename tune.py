"""
Tune rating parameters by coordinate search.
Train 2002–2017 → score on 2018–2023 (validation). The 2024–25 test seasons are never used here.
Writes data/processed/tuned_params.json, which nfl_model.py picks up automatically.
Run: .venv/bin/python tune.py      (≈ 3–5 min; only needed occasionally, not weekly)
"""
import json, time
import numpy as np
import nfl_model as nm

TRAIN = range(2002, 2018)
VALID = range(2018, 2024)
BASE_FEATURES = ["elo_diff", "off_diff", "def_diff", "qb_diff", "home"]
MIN_GAIN = 0.0005   # Brier gain needed to adopt new settings; smaller gains are noise
                    # (2026-09-19 run: best gain was 0.0003 and it did worse on 2024–25, so defaults kept)

GRID = {
    "elo_k":      [12, 16, 20, 24, 28],
    "elo_carry":  [0.5, 0.6, 2 / 3, 0.75, 0.8, 0.85, 0.9],
    "epa_alpha":  [0.05, 0.075, 0.10, 0.15, 0.20],
    "off_carry":  [0.1, 0.2, 0.3, 0.4, 0.55, 0.65, 0.75],
    "def_carry":  [0.4, 0.5, 0.6, 0.75, 0.85, 0.95],
    "qb_prior":   [-0.2, -0.15, -0.10, -0.05, 0.0],
    "qb_prior_n": [3.0, 6.0, 10.0, 15.0],
    "qb_decay":   [0.90, 0.95, 0.98, 1.0],
}


def score(games, epa, params):
    feat, _ = nm.build_features(games, epa, params, verbose=False)
    played = feat[feat["played"]]
    tr, va = played[played.season.isin(TRAIN)], played[played.season.isin(VALID)]
    m, s = nm.fit(tr, BASE_FEATURES)
    return nm.evaluate(va, m.predict(va[BASE_FEATURES]), s)["brier"]


def main():
    t0 = time.time()
    games, epa = nm.load_games(), nm.load_epa()
    best = dict(nm.PARAMS)
    best_score = score(games, epa, best)
    print(f"Start  validation Brier {best_score:.5f}")
    for rnd in range(4):
        improved = False
        for k, vals in GRID.items():
            for v in vals:
                if v == best[k]:
                    continue
                trial = dict(best, **{k: v})
                sc = score(games, epa, trial)
                if sc < best_score - 1e-5:
                    best, best_score, improved = trial, sc, True
                    print(f"  round {rnd+1}: {k} = {v:<6}  → Brier {sc:.5f}")
        if not improved:
            break
    start = score(games, epa, dict(nm.PARAMS))
    print(f"Best validation Brier {best_score:.5f}  ({time.time()-t0:.0f}s)")
    if start - best_score < MIN_GAIN:
        print(f"Gain {start - best_score:.5f} < {MIN_GAIN}: keeping current settings")
        return
    nm.TUNED_PATH.write_text(json.dumps(best, indent=1))
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
