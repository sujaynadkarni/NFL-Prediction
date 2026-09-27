"""
NFL 2026 season + playoff Monte Carlo
- Games already played are locked in; only remaining games are simulated
- Each simulation draws a hidden "true strength" offset per team (rating uncertainty)
  plus game-to-game noise, then applies NFL seeding and playoff rules
Tiebreakers: full NFL procedure (division and wild-card), except net-points steps (coin toss used instead)
Run: .venv/bin/python simulate.py
"""
import json, pickle, time
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import tracking
from nfl_model import (MODEL_PATH, CURRENT_SEASON, TEAM_DIV, TEAM_CONF, DIVISIONS,
                       LABELS, norm_cdf)

ROOT    = Path(__file__).parent
OUT     = ROOT / "data" / "processed" / "predictions.json"
N_SIM   = 50_000
TEAM_SD = 3.0      # uncertainty (pts) in each team's true strength
SEED    = 42

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
TEAMS = sorted(TEAM_DIV)
IDX   = {t: i for i, t in enumerate(TEAMS)}
NT    = len(TEAMS)


# ══════════════════════════════════════════════════════════════
# 1. GAME PREDICTIONS + "WHY"
# ══════════════════════════════════════════════════════════════

def explain(contrib, fav_sign):
    """Top reasons for the favourite, plus the biggest factor against."""
    items = sorted(contrib.items(), key=lambda kv: -kv[1] * fav_sign)
    pro = [{"label": LABELS[f], "pts": round(abs(v), 1)}
           for f, v in items if v * fav_sign >= 0.5][:3]
    con = [{"label": LABELS[f], "pts": round(abs(v), 1)}
           for f, v in reversed(items) if v * fav_sign <= -0.5][:1]
    return pro, con


def predict_row(model, feats, x, st, home, away):
    coef = dict(zip(feats, model.coef_))
    contrib = {f: coef[f] * x[f] for f in feats}
    margin = sum(contrib.values())
    total = (st.pf.get(home, 22) + st.pa.get(away, 22) + st.pf.get(away, 22) + st.pa.get(home, 22)) / 2
    return margin, contrib, total


def game_predictions(m, games_2026):
    model, sigma, feats, st = m["model"], m["sigma"], m["features"], m["state"]
    out = []
    for r in games_2026.to_dict("records"):
        x = {f: r[f] for f in feats}
        margin, contrib, total = predict_row(model, feats, x, st, r["home_team"], r["away_team"])
        p_home = norm_cdf(margin / sigma)
        fav_sign = 1 if margin >= 0 else -1
        pro, con = explain(contrib, fav_sign)
        rec = {
            "game_id": r["game_id"], "week": int(r["week"]), "gameday": str(r["gameday"].date()),
            "gametime": r["gametime"], "home": r["home_team"], "away": r["away_team"],
            "stadium": r["stadium"], "p_home": round(p_home, 4), "margin": round(margin, 1),
            "proj_home": round((total + margin) / 2), "proj_away": round((total - margin) / 2),
            "favorite": r["home_team"] if fav_sign > 0 else r["away_team"],
            "why": pro, "against": con,
            "home_qb": r.get("home_qb_name") if isinstance(r.get("home_qb_name"), str) else st.last_qb_name.get(r["home_team"]),
            "away_qb": r.get("away_qb_name") if isinstance(r.get("away_qb_name"), str) else st.last_qb_name.get(r["away_team"]),
            "vegas_home_spread": None if pd.isna(r["spread_line"]) else float(r["spread_line"]),
            "played": bool(r["played"]),
        }
        if not r["played"]:
            for side, team in [("home", r["home_team"]), ("away", r["away_team"])]:
                d = m["starters"].get((team, int(r["week"])))
                if d and d["dist"][0][1] < 0.9:
                    rec[f"{side}_qb_odds"] = [[n, round(p, 2)] for n, p in d["dist"] if p >= 0.05]
        if r["played"]:
            rec.update(home_score=int(r["home_score"]), away_score=int(r["away_score"]),
                       correct=bool((margin > 0) == (r["result"] > 0)))
        out.append(rec)
    return out


# ══════════════════════════════════════════════════════════════
# 2. STANDINGS + NFL TIEBREAKERS
#    W[t,o] = wins of t over o (ties = 0.5), G[t,o] = games between them
# ══════════════════════════════════════════════════════════════

DIV_IDX  = np.array([[TEAM_DIV[a] == TEAM_DIV[b] for b in TEAMS] for a in TEAMS])
CONF_IDX = np.array([[TEAM_CONF[a] == TEAM_CONF[b] for b in TEAMS] for a in TEAMS])


class Standings:
    def __init__(self, W, G, rng):
        self.W, self.G, self.rng = W, G, rng
        self.wins = W.sum(1)
        self.pct = self.wins / np.maximum(G.sum(1), 1)

    def rec(self, t, mask):
        g = self.G[t][mask].sum()
        return self.W[t][mask].sum() / g if g else None

    def h2h(self, t, group):
        m = np.zeros(NT, bool); m[[o for o in group if o != t]] = True
        return self.rec(t, m)

    def common(self, t, group, minimum=1):
        played_by_all = np.all(self.G[group] > 0, axis=0)
        played_by_all[group] = False
        if self.G[t][played_by_all].sum() < minimum:
            return None
        return self.rec(t, played_by_all)

    def sov(self, t):
        w = self.W[t]; return (w * self.pct).sum() / w.sum() if w.sum() else 0

    def sos(self, t):
        g = self.G[t]; return (g * self.pct).sum() / g.sum()

    def sweep(self, t, group):
        """3+ team head-to-head: 1 if t beat all others, -1 if lost to all, else 0 (NFL sweep rule)."""
        others = [o for o in group if o != t]
        if not all(self.G[t, o] > 0 for o in others):
            return 0
        if all(self.W[t, o] == self.G[t, o] for o in others):
            return 1
        if all(self.W[t, o] == 0 for o in others):
            return -1
        return 0

    def best_of(self, group, division):
        """Winner of a tie among `group` (same pct) per NFL procedure; restarts after each elimination."""
        group = list(group)
        while len(group) > 1:
            if division:
                steps = [lambda t, g: self.h2h(t, g), lambda t, g: self.rec(t, DIV_IDX[t]),
                         lambda t, g: self.common(t, g), lambda t, g: self.rec(t, CONF_IDX[t]),
                         lambda t, g: self.sov(t), lambda t, g: self.sos(t)]
            else:
                h2h = (lambda t, g: self.h2h(t, g)) if len(group) == 2 else (lambda t, g: self.sweep(t, g))
                steps = [h2h, lambda t, g: self.rec(t, CONF_IDX[t]),
                         lambda t, g: self.common(t, g, minimum=4),
                         lambda t, g: self.sov(t), lambda t, g: self.sos(t)]
            narrowed = False
            for step in steps:
                vals = {t: step(t, group) for t in group}
                if any(v is None for v in vals.values()):
                    continue
                best = max(round(v, 9) for v in vals.values())
                top = [t for t in group if round(vals[t], 9) == best]
                if len(top) < len(group):
                    group, narrowed = top, True
                    break
            if not narrowed:
                return group[int(self.rng.integers(len(group)))]       # coin toss
        return group[0]

    def order(self, teams, division):
        """Rank teams best → worst."""
        remaining, out = list(teams), []
        while remaining:
            top_pct = max(self.pct[t] for t in remaining)
            tied = [t for t in remaining if self.pct[t] == top_pct]
            if len(tied) > 1 and not division:
                # wild card: only the best-ranked club from each division enters the comparison
                reps = []
                for d in {TEAM_DIV[TEAMS[t]] for t in tied}:
                    same = [t for t in tied if TEAM_DIV[TEAMS[t]] == d]
                    reps.append(self.order(same, True)[0] if len(same) > 1 else same[0])
                tied = reps
            pick = self.best_of(tied, division) if len(tied) > 1 else tied[0]
            out.append(pick)
            remaining.remove(pick)
        return out


def seed_conference(conf, S):
    winners, others = [], []
    for d, ts in DIVISIONS.items():
        if d.startswith(conf):
            ranked = S.order([IDX[t] for t in ts], division=True)
            winners.append(ranked[0])
            others += ranked[1:]
    seeds = S.order(winners, division=False)
    wild = S.order(others, division=False)[:3]
    return seeds + wild, set(winners)


# ══════════════════════════════════════════════════════════════
# 3. MONTE CARLO
# ══════════════════════════════════════════════════════════════

def simulate(m, games_2026, n_sim=N_SIM):
    model, sigma, feats, st = m["model"], m["sigma"], m["features"], m["state"]
    rng = np.random.default_rng(SEED)
    coef = np.array(model.coef_)

    hi = games_2026["home_team"].map(IDX).values
    ai = games_2026["away_team"].map(IDX).values
    played = games_2026["played"].values
    G = np.zeros((NT, NT))
    np.add.at(G, (hi, ai), 1); np.add.at(G, (ai, hi), 1)
    mu = games_2026[feats].values @ coef
    actual = np.where(played, np.sign(games_2026["result"].fillna(0).values), 0)

    # Playoff matchups: current ratings + end-of-season projected QB and injuries (week 18)
    last_w = int(games_2026["week"].max())
    inj_proj = m["inj_proj"]
    def pair_mu(h, a, neutral):
        row = {"home_team": h, "away_team": a, "season": CURRENT_SEASON, "neutral": int(neutral),
               "home_rest": 7, "away_rest": 7}
        x = {f: 0.0 for f in feats}
        x.update({k: v for k, v in st.features(row).items() if k in x})
        sh, sa = m["starters"].get((h, last_w)), m["starters"].get((a, last_w))
        if sh and sa:
            x["qb_diff"] = (sh["ev"] - sa["ev"]) * 100
        z = {"off": 0.0, "def": 0.0}
        ih, ia = inj_proj.get((h, last_w), z), inj_proj.get((a, last_w), z)
        if "inj_off_diff" in x: x["inj_off_diff"] = ia["off"] - ih["off"]
        if "inj_def_diff" in x: x["inj_def_diff"] = ia["def"] - ih["def"]
        return float(np.dot(coef, [x[f] for f in feats]))
    mu_home = np.array([[pair_mu(h, a, False) if h != a else 0 for a in TEAMS] for h in TEAMS])
    mu_neut = np.array([[pair_mu(h, a, True) if h != a else 0 for a in TEAMS] for h in TEAMS])

    keys = ["playoffs", "division", "seed1", "div_round", "conf_game", "super_bowl", "champion"]
    seed_counts = np.zeros((NT, 7))
    tally = {k: np.zeros(NT) for k in keys}
    wins_total = np.zeros(NT)

    for _ in range(n_sim):
        strength = rng.normal(0, TEAM_SD, NT)
        margin = mu + strength[hi] - strength[ai] + rng.normal(0, sigma, len(mu))
        res = np.where(played, actual, np.sign(margin))
        hw = (res > 0).astype(float) + 0.5 * (res == 0)
        aw = 1 - hw
        W = np.zeros((NT, NT))
        np.add.at(W, (hi, ai), hw); np.add.at(W, (ai, hi), aw)
        S = Standings(W, G, rng)
        wins_total += S.wins

        def play(h, a, neutral=False):
            base = (mu_neut if neutral else mu_home)[h, a]
            return h if base + strength[h] - strength[a] + rng.normal(0, sigma) > 0 else a

        champs = []
        for conf in ["AFC", "NFC"]:
            seeds, div_winners = seed_conference(conf, S)
            for t in seeds:
                tally["playoffs"][t] += 1
            for t in div_winners:
                tally["division"][t] += 1
            tally["seed1"][seeds[0]] += 1
            for k, t in enumerate(seeds):
                seed_counts[t, k] += 1
            sd = {t: k for k, t in enumerate(seeds)}
            alive = [seeds[0]] + [play(seeds[k], seeds[7 - k]) for k in (1, 2, 3)]
            for t in alive:
                tally["div_round"][t] += 1
            alive.sort(key=sd.get)                       # re-seed
            semi = [play(alive[0], alive[3]), play(alive[1], alive[2])]
            semi.sort(key=sd.get)
            for t in semi:
                tally["conf_game"][t] += 1
            champ = play(semi[0], semi[1])
            tally["super_bowl"][champ] += 1
            champs.append(champ)
        tally["champion"][play(champs[0], champs[1], neutral=True)] += 1

    return {TEAMS[i]: {**{k: round(tally[k][i] / n_sim, 4) for k in keys},
                       "avg_wins": round(wins_total[i] / n_sim, 1),
                       "seeds": [round(x / n_sim, 4) for x in seed_counts[i]]} for i in range(NT)}


# ══════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    m = pickle.load(open(MODEL_PATH, "rb"))
    feat, st = m["feat"], m["state"]
    g26 = feat[(feat["season"] == CURRENT_SEASON) & (feat["game_type"] == "REG")].reset_index(drop=True)

    games = game_predictions(m, g26)
    print(f"Simulating the rest of 2026 ({(~g26.played).sum()} games) × {N_SIM:,} …")
    teams = simulate(m, g26)

    # current record
    for t in teams:
        hg = g26[(g26.home_team == t) & g26.played]; ag = g26[(g26.away_team == t) & g26.played]
        w = (hg.result > 0).sum() + (ag.result < 0).sum()
        l = (hg.result < 0).sum() + (ag.result > 0).sum()
        teams[t].update(name=TEAM_NAMES[t], conf=TEAM_CONF[t], div_name=TEAM_DIV[t],
                        record=f"{w}-{l}", elo=round(st.elo[t]), qb=st.last_qb_name.get(t))

    upcoming_week = min((g["week"] for g in games if not g["played"]), default=None)
    ledger, n_logged = tracking.update(games, CURRENT_SEASON, upcoming_week)
    card = tracking.scorecard(games, ledger, m["sigma"])
    for g in games:                                   # played cards show the pick that was actually on record
        if g["game_id"] in card["games"]:
            c = card["games"][g["game_id"]]
            g.update(correct=c["correct"], pick_source=c["source"], p_home_pre=c["p_home"])
    out = {
        "generated": str(date.today()), "season": CURRENT_SEASON, "n_sim": N_SIM,
        "upcoming_week": upcoming_week, "games": games, "teams": teams,
        "track_record": {"played": card["all"]["games"], "correct": card["all"]["correct"]},
        "scorecard": {k: v for k, v in card.items() if k != "games"},
        "backtest": {k: v for k, v in m["backtest"].items() if k != "ablation"},
        "factor_test": m["backtest"]["ablation"],
    }
    inj = [{k: p.get(k) for k in ["team", "name", "pos", "status", "injury", "expected_return", "source"]}
           | {"importance": round(p["imp"], 2),
              "p_out": {int(w): round(v, 2) for w, v in p["p"].items()}}
           for p in m["injuries"] if p["imp"] >= 0.3 or p["pos"] == "QB"]
    out["injuries"] = sorted(inj, key=lambda p: (p["team"], -p["importance"]))
    out["factor_selection"] = m["selection"]
    out["features"] = m["features"]
    OUT.write_text(json.dumps(out, indent=1, default=str))
    if upcoming_week:
        (tracking.SNAPS / f"{CURRENT_SEASON}_week{upcoming_week:02d}_pre.json").write_text(json.dumps(out, default=str))
    sc = card["all"]
    print(f"Pick ledger: {n_logged} Week {upcoming_week} picks recorded before kickoff | "
          f"2026 record {sc['correct']}/{sc['games']} (Vegas {sc['vegas_correct']}/{sc['vegas_games']})")

    print(f"\nDone in {time.time()-t0:.0f}s\n")
    top = sorted(teams.items(), key=lambda kv: -kv[1]["champion"])
    print(f"{'Team':<24}{'Rec':>5}{'Wins':>6}{'Playoff':>9}{'Div':>7}{'Conf':>7}{'SB':>7}")
    for t, v in top[:16]:
        print(f"{v['name']:<24}{v['record']:>5}{v['avg_wins']:>6}{v['playoffs']:>9.0%}"
              f"{v['division']:>7.0%}{v['super_bowl']:>7.0%}{v['champion']:>7.1%}")
    print(f"\nWeek {upcoming_week} picks:")
    for g in games:
        if g["week"] == upcoming_week and not g["played"]:
            p = g["p_home"] if g["favorite"] == g["home"] else 1 - g["p_home"]
            why = ", ".join(w["label"] for w in g["why"])
            print(f"  {g['away']:>3} @ {g['home']:<3}  {g['favorite']:>3} {p:.0%}  "
                  f"({g['proj_away']}-{g['proj_home']})  Vegas {g['vegas_home_spread']}  | {why}")


if __name__ == "__main__":
    main()
