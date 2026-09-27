"""
NFL 2026 Predictor — ratings, features, model, backtest
Model : Elo + rolling EPA efficiency + QB value → Ridge regression on point margin
        margin → win probability via a normal distribution
Train : 2002–2023   Test : 2024–2025 (compared against the Vegas spread)
Run   : .venv/bin/python nfl_model.py
"""
import warnings, pickle
warnings.filterwarnings("ignore")
from pathlib import Path
from math import erf, sqrt, log
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT  = Path(__file__).parent
RAW   = ROOT / "data" / "raw"
PROC  = ROOT / "data" / "processed"
MODEL_PATH = PROC / "model.pkl"

CURRENT_SEASON = 2026
TRAIN_SEASONS  = range(2002, 2024)
TEST_SEASONS   = [2024, 2025]

# ── Franchise mapping (relocations) ──────────────────────────────────────────
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

DIVISIONS = {
    "AFC East":  ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West":  ["DEN", "KC", "LV", "LAC"],
    "NFC East":  ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West":  ["ARI", "LA", "SF", "SEA"],
}
TEAM_DIV  = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAM_CONF = {t: d.split()[0] for t, d in TEAM_DIV.items()}

# ── Rating parameters (tuned by tune.py on 2018–2023; see TUNED_PARAMS) ─────
ELO_BASE, ELO_HFA = 1505, 48
PARAMS = {
    "elo_k":      20,     # how far Elo moves after one game
    "elo_carry":  2 / 3,  # share of Elo edge kept into the next season
    "epa_alpha":  0.10,   # weight of the newest game in rolling efficiency
    "off_carry":  0.65,   # offense carry-over between seasons …
    "def_carry":  0.50,   # … defense usually regresses harder
    "qb_prior":  -0.10,   # assumed EPA/dropback for a QB with no history
    "qb_prior_n": 6.0,    # games of evidence before a QB's own record dominates
    "qb_decay":   0.95,   # older QB games count less
}
TUNED_PATH = Path(__file__).parent / "data" / "processed" / "tuned_params.json"
if TUNED_PATH.exists():
    import json as _json
    PARAMS.update(_json.loads(TUNED_PATH.read_text()))

BASE_FEATURES = ["elo_diff", "off_diff", "def_diff", "qb_diff", "home"]
# Candidates are kept only if they improve the 2018–2023 validation seasons (select_features)
CANDIDATES = ["inj_off_diff", "inj_def_diff", "rookie_qb_diff", "draft_diff", "rest_diff", "new_coach_diff"]
MIN_GAIN   = 0.0005   # validation Brier gain a candidate must deliver (same noise bar as tune.py)
VALID_TRAIN, VALID_SEASONS = range(2002, 2018), range(2018, 2024)
# Watch list: re-tested every run and logged to snapshots/factor_watch.csv, but NEVER added
# automatically — we review the trend and decide together (agreed 2026-09-19)
WATCH = ["bye_diff", "extra_rest_diff", "travel", "tz_shift", "west_early", "h2h_wins_3"]
WATCH_LOG = Path(__file__).parent / "snapshots" / "factor_watch.csv"

# Home stadium (lat, lon, hours behind Eastern) — incl. pre-relocation cities
STADIUM = {
 "ARI": (33.53, -112.26, 3), "ATL": (33.76, -84.40, 0), "BAL": (39.28, -76.62, 0), "BUF": (42.77, -78.79, 0),
 "CAR": (35.23, -80.85, 0), "CHI": (41.86, -87.62, 1), "CIN": (39.10, -84.52, 0), "CLE": (41.51, -81.70, 0),
 "DAL": (32.75, -97.09, 1), "DEN": (39.74, -105.02, 2), "DET": (42.34, -83.05, 0), "GB": (44.50, -88.06, 1),
 "HOU": (29.68, -95.41, 1), "IND": (39.76, -86.16, 0), "JAX": (30.32, -81.64, 0), "KC": (39.05, -94.48, 1),
 "LA": (33.95, -118.34, 3), "LAC": (33.95, -118.34, 3), "LV": (36.09, -115.18, 3), "MIA": (25.96, -80.24, 0),
 "MIN": (44.97, -93.26, 1), "NE": (42.09, -71.26, 0), "NO": (29.95, -90.08, 1), "NYG": (40.81, -74.07, 0),
 "NYJ": (40.81, -74.07, 0), "PHI": (39.90, -75.17, 0), "PIT": (40.45, -80.02, 0), "SEA": (47.60, -122.33, 3),
 "SF": (37.40, -121.97, 3), "TB": (27.98, -82.50, 0), "TEN": (36.17, -86.77, 1), "WAS": (38.91, -76.86, 0),
 "OAK": (37.75, -122.20, 3), "SD": (32.78, -117.12, 3), "STL": (38.63, -90.19, 1),
}

LABELS = {
    "elo_diff":       "Stronger overall team",
    "off_diff":       "Better offense",
    "def_diff":       "Better defense",
    "qb_diff":        "Quarterback edge",
    "home":           "Home field",
    "inj_off_diff":   "Healthier offense",
    "inj_def_diff":   "Healthier defense",
    "rookie_qb_diff": "Veteran QB vs rookie",
    "draft_diff":     "Stronger draft class",
    "rest_diff":      "Rest advantage",
    "new_coach_diff": "Coaching stability",
    "bye_diff":        "Coming off a bye",
    "extra_rest_diff": "Extra rest",
    "travel":          "Less travel",
    "tz_shift":        "No time-zone change",
    "west_early":      "Body-clock edge",
    "h2h_wins_3":      "Head-to-head record",
}

DRAFT_TEAM = {"GNB": "GB", "KAN": "KC", "LAR": "LA", "RAM": "LA", "LVR": "LV", "OAK": "LV", "NOR": "NO",
              "NWE": "NE", "SDG": "LAC", "SFO": "SF", "STL": "LA", "TAM": "TB"}


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


# ══════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════

def load_games():
    g = pd.read_csv(RAW / "games.csv")
    g["home_raw"], g["away_raw"] = g["home_team"], g["away_team"]
    for c in ["home_team", "away_team"]:
        g[c] = g[c].replace(FRANCHISE)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    g["played"]  = g["result"].notna()
    g = g.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    return g


def load_epa():
    e = pd.read_parquet(PROC / "team_game_epa.parquet")
    for c in ["posteam", "defteam"]:
        e[c] = e[c].replace(FRANCHISE)
    return e


# ══════════════════════════════════════════════════════════════
# 2. RATINGS — walk through every game in order (leak-free:
#    each game's features use only games played before it)
# ══════════════════════════════════════════════════════════════

class RatingState:
    def __init__(self, params=None):
        self.p = dict(PARAMS, **(params or {}))
        self.elo, self.off, self.dfn = {}, {}, {}
        self.qb = {}                 # qb_id → [weighted sum, weight]
        self.last_qb = {}            # team → most recent starter id
        self.last_qb_name = {}
        self.coach_seasons = {}      # (team, season) → coach
        self.pf, self.pa = {}, {}    # rolling points scored / allowed (for projected score)
        self.season = None

    def new_season(self, season):
        for t in self.elo:
            self.elo[t] = ELO_BASE + self.p["elo_carry"] * (self.elo[t] - ELO_BASE)
        for t in self.pf:
            self.pf[t] = 22 + 0.6 * (self.pf[t] - 22)
            self.pa[t] = 22 + 0.6 * (self.pa[t] - 22)
        for t in self.off:
            self.off[t] *= self.p["off_carry"]
            self.dfn[t] *= self.p["def_carry"]
        self.season = season

    def qb_value(self, qb_id):
        s, w = self.qb.get(qb_id, (0.0, 0.0))
        return (s + self.p["qb_prior"] * self.p["qb_prior_n"]) / (w + self.p["qb_prior_n"])

    def team_qb(self, team, qb_id):
        if not isinstance(qb_id, str):
            qb_id = self.last_qb.get(team)
        return self.qb_value(qb_id) if qb_id else self.p["qb_prior"]

    def new_coach(self, team, season, coach):
        prev = self.coach_seasons.get((team, season - 1))
        return int(prev is not None and coach != prev)

    def features(self, row):
        h, a = row["home_team"], row["away_team"]
        season = row["season"]
        rest = np.clip((row.get("home_rest", 7) or 7) - (row.get("away_rest", 7) or 7), -7, 7)
        return {
            "elo_diff": (self.elo.get(h, ELO_BASE) - self.elo.get(a, ELO_BASE)) / 25,
            "off_diff": (self.off.get(h, 0) - self.off.get(a, 0)) * 100,
            "def_diff": (self.dfn.get(a, 0) - self.dfn.get(h, 0)) * 100,   # lower allowed = better
            "qb_diff":  (self.team_qb(h, row.get("home_qb_id")) - self.team_qb(a, row.get("away_qb_id"))) * 100,
            "home":     1 - row["neutral"],
            "rest_diff": rest if not np.isnan(rest) else 0,
            "new_coach_diff": (self.new_coach(h, season, row.get("home_coach"))
                               - self.new_coach(a, season, row.get("away_coach"))),
        }

    def update(self, row, epa_by_game):
        h, a, m = row["home_team"], row["away_team"], row["result"]
        # Elo with margin-of-victory multiplier (FiveThirtyEight style)
        rh, ra = self.elo.setdefault(h, ELO_BASE), self.elo.setdefault(a, ELO_BASE)
        hfa = 0 if row["neutral"] else ELO_HFA
        exp_h = 1 / (1 + 10 ** (-(rh + hfa - ra) / 400))
        s_h = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
        wdiff = (rh + hfa - ra) if m > 0 else (ra - rh - hfa)
        mult = log(abs(m) + 1) * 2.2 / (wdiff * 0.001 + 2.2) if m != 0 else 1.0
        delta = self.p["elo_k"] * mult * (s_h - exp_h)
        self.elo[h], self.elo[a] = rh + delta, ra - delta
        A = self.p["epa_alpha"]
        # Rolling points (projected score line)
        for team, scored, allowed in [(h, row["home_score"], row["away_score"]),
                                      (a, row["away_score"], row["home_score"])]:
            self.pf[team] = (1 - A) * self.pf.get(team, 22) + A * scored
            self.pa[team] = (1 - A) * self.pa.get(team, 22) + A * allowed
        # Rolling efficiency
        for team, opp in [(h, a), (a, h)]:
            rec = epa_by_game.get((row["game_id"], team))
            if rec is None:
                continue
            self.off[team] = (1 - A) * self.off.get(team, 0) + A * rec["epa_per_play"]
            opp_rec = epa_by_game.get((row["game_id"], opp))
            if opp_rec is not None:
                self.dfn[team] = (1 - A) * self.dfn.get(team, 0) + A * opp_rec["epa_per_play"]
        # QB value: starter credited with team's pass EPA in the game
        for team, side in [(h, "home"), (a, "away")]:
            qid = row.get(f"{side}_qb_id")
            rec = epa_by_game.get((row["game_id"], team))
            if isinstance(qid, str):
                self.last_qb[team] = qid
                self.last_qb_name[team] = row.get(f"{side}_qb_name")
                if rec is not None and not np.isnan(rec["pass_epa"]):
                    s, w = self.qb.get(qid, (0.0, 0.0))
                    d = self.p["qb_decay"]
                    self.qb[qid] = (s * d + rec["pass_epa"], w * d + 1)
        for team, side in [(h, "home"), (a, "away")]:
            self.coach_seasons[(team, row["season"])] = row.get(f"{side}_coach")


def build_features(games, epa, params=None, verbose=True):
    if verbose:
        print("Building ratings and features (1999 → today) …")
    epa_by_game = {(r.game_id, r.posteam): {"epa_per_play": r.epa_per_play, "pass_epa": r.pass_epa}
                   for r in epa.itertuples()}
    st = RatingState(params)
    rows = []
    for row in games.to_dict("records"):
        if row["season"] != st.season:
            st.new_season(row["season"])
        f = st.features(row)
        f.update(game_id=row["game_id"])
        rows.append(f)
        if row["played"]:
            st.update(row, epa_by_game)
    feat = pd.DataFrame(rows)
    return games.merge(feat, on="game_id"), st



# ══════════════════════════════════════════════════════════════
# 2b. EXTRA FEATURES — injuries, rookie QBs, draft capital
# ══════════════════════════════════════════════════════════════

def rookie_years():
    ry = {}
    d = pd.read_parquet(RAW / "draft_picks.parquet", columns=["season", "gsis_id"]).dropna()
    ry.update(zip(d.gsis_id, d.season))
    for f in sorted(RAW.glob("roster_20*.parquet")):
        r = pd.read_parquet(f, columns=["gsis_id", "rookie_year"]).dropna()
        for gid, y in zip(r.gsis_id, r.rookie_year):
            ry.setdefault(gid, int(y))
    return ry


def draft_values():
    d = pd.read_parquet(RAW / "draft_picks.parquet", columns=["season", "pick", "team"])
    d["team"] = d["team"].replace(DRAFT_TEAM)
    d["v"] = np.exp(-(d["pick"] - 1) / 50)          # pick 1 = 1.0, pick 50 ≈ 0.37, pick 150 ≈ 0.05
    return d.groupby(["season", "team"])["v"].sum().to_dict()


def early_weight(week, game_type):
    return max(0.0, 1 - (week - 1) / 8) if game_type == "REG" else 0.0


def add_extra_features(feat, im):
    hist = im.team_week_absences(range(2012, CURRENT_SEASON + 1))
    ry, dv = rookie_years(), draft_values()
    z = {"off": 0.0, "def": 0.0}
    rows = []
    for r in feat[["season", "week", "game_type", "home_team", "away_team", "home_qb_id", "away_qb_id"]].itertuples(index=False):
        ih, ia = hist.get((r.season, r.week, r.home_team), z), hist.get((r.season, r.week, r.away_team), z)
        rh = int(ry.get(r.home_qb_id) == r.season); ra = int(ry.get(r.away_qb_id) == r.season)
        rows.append({"inj_off_diff": ia["off"] - ih["off"], "inj_def_diff": ia["def"] - ih["def"],
                     "rookie_qb_diff": ra - rh,
                     "draft_diff": (dv.get((r.season, r.home_team), 0) - dv.get((r.season, r.away_team), 0))
                                   * early_weight(r.week, r.game_type)})
    return pd.concat([feat.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_watch_features(feat):
    """Schedule / travel factors and head-to-head — positive values favour the home team."""
    from math import radians, sin, cos, asin, sqrt
    def km(a, b):
        la1, lo1, _ = STADIUM[a]; la2, lo2, _ = STADIUM[b]
        la1, lo1, la2, lo2 = map(radians, [la1, lo1, la2, lo2])
        return 6371 * 2 * asin(sqrt(sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2))
    hr, ar = feat["home_rest"].fillna(7), feat["away_rest"].fillna(7)
    feat["bye_diff"] = (hr >= 13).astype(int) - (ar >= 13).astype(int)
    feat["extra_rest_diff"] = hr.between(10, 12).astype(int) - ar.between(10, 12).astype(int)
    at_home = feat["neutral"] == 0
    feat["travel"] = [km(a, h) / 1000 if n else 0.0 for a, h, n in zip(feat.away_raw, feat.home_raw, at_home)]
    feat["tz_shift"] = [abs(STADIUM[a][2] - STADIUM[h][2]) if n else 0 for a, h, n in zip(feat.away_raw, feat.home_raw, at_home)]
    feat["west_early"] = [int(n and STADIUM[a][2] == 3 and STADIUM[h][2] == 0 and str(t) <= "13:05")
                          for a, h, n, t in zip(feat.away_raw, feat.home_raw, at_home, feat.gametime)]
    # head-to-head record over the last 3 seasons, only games before this one, shrunk toward 0
    hist, vals = {}, []
    for r in feat.sort_values(["gameday", "game_id"]).itertuples():
        key = tuple(sorted((r.home_team, r.away_team))); sgn = 1 if key[0] == r.home_team else -1
        ws = [w for s, w in hist.get(key, []) if r.season - s <= 3]
        vals.append((r.Index, sgn * sum(ws) / (len(ws) + 2)))
        if r.played:
            hist.setdefault(key, []).append((r.season, np.sign(sgn * r.result)))
    feat["h2h_wins_3"] = pd.Series(dict(vals))
    return feat


def watch_report(played, features, upcoming_week):
    """Re-test each watch-list factor (validation + test gain) and append to the log."""
    tr, va = played[played["season"].isin(VALID_TRAIN)], played[played["season"].isin(VALID_SEASONS)]
    tr2, te = played[played["season"].isin(TRAIN_SEASONS)], played[played["season"].isin(TEST_SEASONS)]
    def brier(train, test, cols):
        m, s = fit(train, cols)
        return evaluate(test, m.predict(test[cols]), s)["brier"], m
    tr3, cur = played[played["season"].between(2002, CURRENT_SEASON - 1)], played[played["season"] == CURRENT_SEASON]
    bv, _ = brier(tr, va, features); bt, _ = brier(tr2, te, features)
    bc = brier(tr3, cur, features)[0] if len(cur) else None
    rows = []
    for c in WATCH:
        v, mv = brier(tr, va, features + [c]); t, _ = brier(tr2, te, features + [c])
        g26 = round(bc - brier(tr3, cur, features + [c])[0], 5) if bc is not None else None
        rows.append({"run_date": pd.Timestamp.today().date(), "upcoming_week": upcoming_week, "factor": c,
                     "validation_gain": round(bv - v, 5), "test_gain": round(bt - t, 5),
                     "gain_2026": g26, "games_2026": len(cur),
                     "effect_pts": round(float(mv.coef_[-1]), 2),
                     "passes_bar": (bv - v) > MIN_GAIN and (bt - t) > 0})
    log = pd.DataFrame(rows)
    WATCH_LOG.parent.mkdir(exist_ok=True)
    log.to_csv(WATCH_LOG, mode="a", header=not WATCH_LOG.exists(), index=False)
    return log


def depth_chart_qbs():
    dc = pd.read_parquet(RAW / f"depth_charts_{CURRENT_SEASON}.parquet")
    dc = dc[(dc["pos_abb"] == "QB") & (dc["dt"] == dc["dt"].max())].sort_values("pos_rank")
    dc["team"] = dc["team"].replace(FRANCHISE)
    return {t: list(zip(g.gsis_id, g.player_name)) for t, g in dc.groupby("team")}


def project_future(feat, st, im):
    """Unplayed 2026 games: expected QB (depth chart × injury odds) and projected injury absences."""
    todo = feat.index[(feat["season"] == CURRENT_SEASON) & ~feat["played"]]
    if not len(todo):
        return feat, [], {}
    upcoming = int(feat.loc[todo, "week"].min())
    players = im.project(CURRENT_SEASON, upcoming, feat)
    proj = im.team_week_projection(players)
    pout = {p["gsis_id"]: p["p"] for p in players if p["gsis_id"]}
    qbs, ry = depth_chart_qbs(), rookie_years()
    starters = {}

    def expected_qb(team, week):
        left, ev, rk, dist = 1.0, 0.0, 0.0, []
        cands = qbs.get(team, [])
        for k, (gid, name) in enumerate(cands):
            p_out = pout.get(gid, {}).get(week, 0.0) if k < len(cands) - 1 else 0.0
            p_start = left * (1 - p_out)
            ev += p_start * st.qb_value(gid)
            rk += p_start * int(ry.get(gid) == CURRENT_SEASON)
            dist.append((name, p_start))
            left *= p_out
        if not cands:
            return st.team_qb(team, None), 0.0, [(st.last_qb_name.get(team), 1.0)]
        return ev, rk, sorted(dist, key=lambda x: -x[1])

    for i in todo:
        r = feat.loc[i]
        wk, h, a = int(r["week"]), r["home_team"], r["away_team"]
        eh, rh, dh = expected_qb(h, wk)
        ea, ra, da = expected_qb(a, wk)
        feat.at[i, "qb_diff"] = (eh - ea) * 100
        feat.at[i, "rookie_qb_diff"] = ra - rh
        ih, ia = proj.get((h, wk), {"off": 0, "def": 0}), proj.get((a, wk), {"off": 0, "def": 0})
        feat.at[i, "inj_off_diff"] = ia["off"] - ih["off"]
        feat.at[i, "inj_def_diff"] = ia["def"] - ih["def"]
        feat.at[i, "home_qb_name"], feat.at[i, "away_qb_name"] = dh[0][0], da[0][0]
        starters[(h, wk)], starters[(a, wk)] = {"ev": eh, "dist": dh}, {"ev": ea, "dist": da}
    return feat, players, starters

# ══════════════════════════════════════════════════════════════
# 3. TRAIN + BACKTEST
# ══════════════════════════════════════════════════════════════

def fit(df, cols):
    m = Ridge(alpha=1.0, fit_intercept=False).fit(df[cols], df["result"])
    sigma = float(np.std(df["result"] - m.predict(df[cols])))
    return m, sigma


def evaluate(df, pred, sigma):
    y = df["result"].values
    p_home = np.array([norm_cdf(x / sigma) for x in pred])
    win = (y > 0).astype(float)
    mask = y != 0
    return {
        "games":    int(len(df)),
        "mae":      float(np.mean(np.abs(y - pred))),
        "accuracy": float(np.mean((pred[mask] > 0) == (y[mask] > 0))),
        "brier":    float(np.mean((p_home[mask] - win[mask]) ** 2)),
    }


def select_features(played):
    """Forward check of each candidate on the validation seasons (2018–2023)."""
    tr = played[played["season"].isin(VALID_TRAIN)]
    va = played[played["season"].isin(VALID_SEASONS)]
    def brier(cols):
        m, s = fit(tr, cols)
        return evaluate(va, m.predict(va[cols]), s)["brier"]
    feats, base = list(BASE_FEATURES), brier(BASE_FEATURES)
    report = {}
    for c in CANDIDATES:
        gain = base - brier(feats + [c])
        keep = gain > MIN_GAIN
        report[c] = {"validation_gain": gain, "kept": keep}
        if keep:
            feats.append(c)
            base = brier(feats)
    return feats, report


def backtest(feat, features):
    played = feat[feat["played"]]
    train  = played[played["season"].isin(TRAIN_SEASONS)]
    test   = played[played["season"].isin(TEST_SEASONS)]

    model, sigma = fit(train, features)
    res = {"model": evaluate(test, model.predict(test[features]), sigma)}

    elo_m, elo_s = fit(train, ["elo_diff", "home"])
    res["elo_only"] = evaluate(test, elo_m.predict(test[["elo_diff", "home"]]), elo_s)

    vt = test[test["spread_line"].notna()]
    res["vegas"] = evaluate(vt, vt["spread_line"].values, sigma)

    # Test-set check of every factor (kept ones removed / dropped ones added)
    ablation = {}
    for f in BASE_FEATURES + CANDIDATES:
        cols = [c for c in features if c != f] if f in features else features + [f]
        m, s = fit(train, cols)
        r = evaluate(test, m.predict(test[cols]), s)
        sign = 1 if f in features else -1
        ablation[f] = {"in_model": f in features,
                       "brier_gain": sign * (r["brier"] - res["model"]["brier"]),
                       "mae_gain": sign * (r["mae"] - res["model"]["mae"])}
    res["ablation"] = ablation

    cur = played[played["season"] == CURRENT_SEASON]
    if len(cur):
        res["season_2026_so_far"] = evaluate(cur, model.predict(cur[features]), sigma)
    return res


# ══════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    from injuries import InjuryModel
    games = load_games()
    epa   = load_epa()
    feat, state = build_features(games, epa)
    im = InjuryModel(games)
    feat = add_extra_features(feat, im)
    feat = add_watch_features(feat)
    feat, players, starters = project_future(feat, state, im)

    played = feat[feat["played"]]
    features, selection = select_features(played)
    print("\nFactor selection on 2018–2023 validation (Brier gain; kept if > %.4f):" % MIN_GAIN)
    for c, v in selection.items():
        print(f"  {c:<16} {v['validation_gain']:+.5f}  {'KEPT' if v['kept'] else 'dropped'}")

    upcoming = feat.loc[(feat["season"] == CURRENT_SEASON) & ~feat["played"], "week"].min()
    watch = watch_report(played, features, None if pd.isna(upcoming) else int(upcoming))
    print("\nWatch list — re-tested each run, never auto-added (gain > %.4f on validation and > 0 on test = worth discussing):" % MIN_GAIN)
    for r in watch.itertuples():
        print(f"  {r.factor:<16} validation {r.validation_gain:+.5f}  test {r.test_gain:+.5f}  "
              f"2026 so far {r.gain_2026:+.5f} ({r.games_2026} games)  effect {r.effect_pts:+.2f} pts"
              f"{'   ← passes the bar, review' if r.passes_bar else ''}")

    res = backtest(feat, features)
    fmt = lambda r: f"MAE {r['mae']:.2f} pts | winner {r['accuracy']:.1%} | Brier {r['brier']:.4f} | n={r['games']}"
    print("\nBacktest on 2024–2025 (trained on 2002–2023):")
    print(f"  Our model   : {fmt(res['model'])}")
    print(f"  Elo only    : {fmt(res['elo_only'])}")
    print(f"  Vegas spread: {fmt(res['vegas'])}")
    print("\n  Test-set check (positive = factor helps):")
    for f, a in res["ablation"].items():
        print(f"    {f:<16} {'in ' if a['in_model'] else 'out'}  Brier {a['brier_gain']:+.4f}  MAE {a['mae_gain']:+.3f}")
    if "season_2026_so_far" in res:
        print(f"\n  2026 so far (pre-kickoff predictions): {fmt(res['season_2026_so_far'])}")

    model, sigma = fit(played[played["season"] >= 2002], features)
    print("\nFinal coefficients (points per unit):")
    for f, c in zip(features, model.coef_):
        print(f"  {f:<16} {c:+.3f}")
    print(f"  game-to-game noise (sigma): {sigma:.2f} pts")

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": model, "sigma": sigma, "features": features, "selection": selection,
                     "state": state, "feat": feat, "backtest": res,
                     "injuries": players, "starters": starters,
                     "inj_proj": im.team_week_projection(players)}, fh)
    print(f"\nSaved {MODEL_PATH.name}")


if __name__ == "__main__":
    import nfl_model          # so the pickled RatingState belongs to module nfl_model, not __main__
    nfl_model.main()
