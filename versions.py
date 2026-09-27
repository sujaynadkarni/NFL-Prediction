"""
Model versions — v1 stays frozen, new versions are compared against it on the same seasons.

v1  current live model: raw EPA, Elo, QB factor on top of the team rating
v2  three changes:
      1. opponent-adjusted efficiency (beating a good defence counts for more)
      2. fumble luck removed (recoveries are ~coin flips, so they are valued at their average)
      3. QB separated from the team rating (so a backup starting moves the number properly)

Run: .venv/bin/python versions.py            compare every version
     .venv/bin/python versions.py --rebuild  re-aggregate play-by-play first (~1 min)
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import nfl_model as nm

ROOT = Path(__file__).parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
EPA_V2 = PROC / "team_game_epa_v2.parquet"
LOG    = ROOT / "snapshots" / "versions.csv"

PBP_COLS = ["game_id", "season", "week", "posteam", "defteam", "play_type", "epa",
            "pass", "rush", "success", "fumble", "fumble_lost"]


# ── 1. luck-adjusted play-by-play ────────────────────────────────────────────

def rebuild_epa():
    """Per game and team: raw EPA plus EPA with fumble recoveries valued at their average."""
    print("Re-aggregating play-by-play with fumble luck removed …")
    out = []
    for f in sorted(RAW.glob("pbp_*.parquet")):
        p = pd.read_parquet(f, columns=PBP_COLS)
        p = p[p["play_type"].isin(["pass", "run"]) & p["epa"].notna()].copy()
        fum = p["fumble"] == 1
        if fum.any():
            p.loc[fum, "epa_luck"] = p.loc[fum, "epa"].mean()   # recovery treated as a coin flip
        p["epa_luck"] = p["epa_luck"].fillna(p["epa"])
        g = p.groupby(["game_id", "season", "week", "posteam", "defteam"])
        agg = g.agg(plays=("epa", "size"), epa_per_play=("epa", "mean"),
                    epa_luck=("epa_luck", "mean"), success_rate=("success", "mean")).reset_index()
        pas = p[p["pass"] == 1].groupby(["game_id", "posteam"]).agg(
            pass_epa=("epa_luck", "mean"), pass_plays=("epa", "size")).reset_index()
        run = p[p["rush"] == 1].groupby(["game_id", "posteam"])["epa_luck"].mean().rename("rush_epa").reset_index()
        out.append(agg.merge(pas, on=["game_id", "posteam"], how="left").merge(run, on=["game_id", "posteam"], how="left"))
    df = pd.concat(out, ignore_index=True)
    df["posteam"] = df["posteam"].replace(nm.FRANCHISE)
    df["defteam"] = df["defteam"].replace(nm.FRANCHISE)
    df["pass_share"] = df["pass_plays"] / df["plays"]
    df.to_parquet(EPA_V2, index=False)
    print(f"  {len(df):,} team-games → {EPA_V2.name}")
    return df


# ── 2. v2 ratings ────────────────────────────────────────────────────────────

class RatingStateV2(nm.RatingState):
    """Same as v1 plus: opponent adjustment, luck-adjusted EPA, and a QB-free offence rating."""

    def __init__(self, params=None):
        super().__init__(params)
        self.off_x = {}          # offence excluding the quarterback

    def new_season(self, season):
        super().new_season(season)
        for t in self.off_x:
            self.off_x[t] *= self.p["off_carry"]

    def features(self, row):
        f = super().features(row)
        h, a = row["home_team"], row["away_team"]
        f["off_x_diff"] = (self.off_x.get(h, 0) - self.off_x.get(a, 0)) * 100
        return f

    def update(self, row, epa_by_game):
        h, a = row["home_team"], row["away_team"]
        A = self.p["epa_alpha"]
        pre_off = dict(self.off)
        pre_def = dict(self.dfn)
        super().update(row, epa_by_game)
        # redo efficiency with an opponent adjustment: credit is relative to what
        # the opponent normally allows / normally produces
        for team, opp in [(h, a), (a, h)]:
            rec = epa_by_game.get((row["game_id"], team))
            if rec is None:
                continue
            adj_off = rec["epa_luck"] - pre_def.get(opp, 0.0)
            self.off[team] = (1 - A) * pre_off.get(team, 0) + A * adj_off
            opp_rec = epa_by_game.get((row["game_id"], opp))
            if opp_rec is not None:
                adj_def = opp_rec["epa_luck"] - pre_off.get(opp, 0.0)
                self.dfn[team] = (1 - A) * pre_def.get(team, 0) + A * adj_def
            # offence minus the starting QB's own contribution
            qid = row.get("home_qb_id" if team == h else "away_qb_id")
            qb_part = self.qb_value(qid) * (rec.get("pass_share") or 0.55) if isinstance(qid, str) else 0.0
            self.off_x[team] = (1 - A) * self.off_x.get(team, 0) + A * (adj_off - qb_part)


def build_v2(games, epa):
    epa_by_game = {(r.game_id, r.posteam): {"epa_per_play": r.epa_per_play, "epa_luck": r.epa_luck,
                                            "pass_epa": r.pass_epa, "pass_share": r.pass_share}
                   for r in epa.itertuples()}
    st = RatingStateV2()
    rows = []
    for row in games.to_dict("records"):
        if row["season"] != st.season:
            st.new_season(row["season"])
        f = st.features(row)
        f["game_id"] = row["game_id"]
        rows.append(f)
        if row["played"]:
            st.update(row, epa_by_game)
    return games.merge(pd.DataFrame(rows), on="game_id"), st


# ── 3. scoring ───────────────────────────────────────────────────────────────

def score(feat, features, label):
    played = feat[feat["played"]]
    tr, va = played[played.season.isin(nm.VALID_TRAIN)], played[played.season.isin(nm.VALID_SEASONS)]
    tr2, te = played[played.season.isin(nm.TRAIN_SEASONS)], played[played.season.isin(nm.TEST_SEASONS)]
    cur = played[played.season == nm.CURRENT_SEASON]
    m1, s1 = nm.fit(tr, features)
    m2, s2 = nm.fit(tr2, features)
    v = nm.evaluate(va, m1.predict(va[features]), s1)
    t = nm.evaluate(te, m2.predict(te[features]), s2)
    c = nm.evaluate(cur, m2.predict(cur[features]), s2) if len(cur) else {}
    return {"version": label, "features": " ".join(features),
            "valid_brier": round(v["brier"], 4), "test_brier": round(t["brier"], 4),
            "test_acc": round(t["accuracy"], 4), "test_mae": round(t["mae"], 2),
            "brier_2026": round(c.get("brier", float("nan")), 4), "games_2026": c.get("games", 0)}


def main():
    t0 = time.time()
    games = nm.load_games()
    if "--rebuild" in sys.argv or not EPA_V2.exists():
        epa2 = rebuild_epa()
    else:
        epa2 = pd.read_parquet(EPA_V2)

    # v1 — exactly the live model
    import pickle
    m = pickle.load(open(nm.MODEL_PATH, "rb"))
    rows = [score(m["feat"], m["features"], "v1 (live)")]

    # v2 — three changes, built on the same extra features (injuries etc.)
    from injuries import InjuryModel
    feat2, st2 = build_v2(games, epa2)
    im = InjuryModel(games)
    feat2 = nm.add_extra_features(feat2, im)
    feat2, _, _ = nm.project_future(feat2, st2, im)
    base2 = ["elo_diff", "off_diff", "def_diff", "qb_diff", "home", "inj_off_diff"]
    rows.append(score(feat2, base2, "v2 (opp-adj + luck-adj)"))
    rows.append(score(feat2, [c if c != "off_diff" else "off_x_diff" for c in base2], "v2b (+ QB split out)"))

    df = pd.DataFrame(rows)
    df.insert(0, "run_date", pd.Timestamp.today().date())
    LOG.parent.mkdir(exist_ok=True)
    df.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print("\n" + f"{'version':<26}{'valid':>8}{'test':>8}{'test win%':>11}{'test MAE':>10}{'2026':>8}")
    for r in rows:
        print(f"{r['version']:<26}{r['valid_brier']:>8.4f}{r['test_brier']:>8.4f}{r['test_acc']:>11.1%}"
              f"{r['test_mae']:>10.2f}{r['brier_2026']:>8.4f}")
    print(f"\nVegas on the same test seasons: Brier {m['backtest']['vegas']['brier']:.4f}  "
          f"winners {m['backtest']['vegas']['accuracy']:.1%}  MAE {m['backtest']['vegas']['mae']:.2f}")
    print(f"Logged to {LOG.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
