"""
Injury layer
- Importance : each player's snap share over his last 4 games (from snap counts, 2012+)
- History    : from 2012–2025, P(player still out d weeks later | report status, injury type,
               already missing games) and, for reserve lists, P(still on IR d weeks later | IR type)
- Features   : snap-weighted expected absences per team-week (offense / defense; QBs handled separately)
- Projection : every current injury and IR placement projected week by week for the rest of 2026,
               plus manual overrides from injury_overrides.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW  = ROOT / "data" / "raw"
OVERRIDES = ROOT / "injury_overrides.csv"
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
HIST_SEASONS = range(2012, 2026)
MAX_D = 18

OFF_POS = {"RB", "FB", "WR", "TE", "T", "G", "C", "OL", "OT", "OG"}
DEF_POS = {"DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "CB", "S", "FS", "SS", "DB", "DL", "EDGE"}
CATS = ["Concussion", "Hamstring", "Knee", "Ankle", "Foot", "Shoulder", "Calf", "Groin", "Back",
        "Achilles", "Illness", "Hip", "Quadricep", "Oblique", "Pectoral"]


def side_of(pos):
    return "QB" if pos == "QB" else "off" if pos in OFF_POS else "def" if pos in DEF_POS else None


def injury_cat(txt):
    if not isinstance(txt, str):
        return "Other"
    for c in CATS:
        if c.lower() in txt.lower():
            return c
    return "Other"


# ══════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════

def id_map():
    """gsis_id (injury reports, rosters) → pfr id (snap counts)."""
    frames = []
    for f in sorted(RAW.glob("roster_20*.parquet")) + sorted(RAW.glob("rosters_weekly_20*.parquet")):
        r = pd.read_parquet(f, columns=["gsis_id", "pfr_id"])
        frames.append(r.dropna())
    m = pd.concat(frames).drop_duplicates("gsis_id")
    return dict(zip(m.gsis_id, m.pfr_id))


def load_snaps():
    frames = [pd.read_parquet(f) for f in sorted(RAW.glob("snap_counts_20*.parquet"))]
    s = pd.concat(frames, ignore_index=True)
    s = s[s["game_type"] == "REG"].copy()
    s["team"] = s["team"].replace(FRANCHISE)
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1).fillna(0)
    s["t"] = s["season"] * 100 + s["week"]
    s = s.sort_values(["pfr_player_id", "t"])
    s["imp"] = s.groupby("pfr_player_id")["share"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    return s


def load_reports(seasons):
    frames = []
    for y in seasons:
        f = RAW / f"injuries_{y}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    r = pd.concat(frames, ignore_index=True)
    r = r[(r["game_type"] == "REG") & r["report_status"].isin(["Out", "Doubtful", "Questionable"])].copy()
    r["team"] = r["team"].replace(FRANCHISE)
    r["cat"] = r["report_primary_injury"].map(injury_cat)
    r["side"] = r["position"].map(side_of)
    return r.drop_duplicates(["season", "week", "team", "gsis_id"])


def team_game_weeks(games):
    g = games[games["game_type"] == "REG"]
    return set(zip(g.season, g.week, g.home_team)) | set(zip(g.season, g.week, g.away_team))


def attach_importance(df, snaps, pfr):
    """Add each player's importance as of *before* (season, week)."""
    df = df.copy()
    df["pfr_player_id"] = df["gsis_id"].map(pfr)
    df["t"] = (df["season"] * 100 + df["week"]).astype("int64")
    left = df[df["pfr_player_id"].notna()].sort_values("t")
    right = snaps[["pfr_player_id", "t", "imp"]].rename(columns={"t": "t_last"})
    right["t_last"] = right["t_last"].astype("int64")
    right = right.sort_values("t_last")
    m = pd.merge_asof(left, right, left_on="t", right_on="t_last", by="pfr_player_id",
                      direction="backward", allow_exact_matches=False)
    m.loc[(m["t"] // 100) - (m["t_last"] // 100) > 1, "imp"] = 0      # stale (>1 season old)
    df = df.merge(m[["season", "week", "team", "gsis_id", "imp"]], on=["season", "week", "team", "gsis_id"], how="left")
    df["imp"] = df["imp"].fillna(0)
    return df


# ══════════════════════════════════════════════════════════════
# HISTORY → probability tables
# ══════════════════════════════════════════════════════════════

def report_history(games, snaps, pfr):
    """P(out d weeks after a report | status, injury type, already out)."""
    # Only regular players (≥40% of snaps): for them, no snaps really means "still injured"
    r = attach_importance(load_reports(HIST_SEASONS), snaps, pfr)
    r = r[r["pfr_player_id"].notna() & (r["imp"] >= 0.4)]
    played = set(zip(snaps.season, snaps.week, snaps.pfr_player_id))
    tg = team_game_weeks(games)
    prev_game = {}
    for (s, w, t) in sorted(tg):
        prev_game.setdefault((s, t), []).append(w)
    def prev_week(s, w, t):
        ws = [x for x in prev_game.get((s, t), []) if x < w]
        return ws[-1] if ws else None
    r["prev_w"] = [prev_week(s, w, t) for s, w, t in zip(r.season, r.week, r.team)]
    r["already_out"] = [pw is not None and (s, pw, p) not in played
                        for s, pw, p in zip(r.season, r.prev_w, r.pfr_player_id)]
    rows = []
    for rec in r[["season", "week", "team", "pfr_player_id", "report_status", "cat", "already_out"]].itertuples(index=False):
        for d in range(MAX_D):
            w = rec.week + d
            if (rec.season, w, rec.team) not in tg:
                continue
            rows.append((rec.report_status, rec.cat, rec.already_out, d,
                         (rec.season, w, rec.pfr_player_id) not in played))
    h = pd.DataFrame(rows, columns=["status", "cat", "already_out", "d", "out"])
    fine = h.groupby(["status", "cat", "already_out", "d"])["out"].agg(["mean", "size"])
    coarse = h.groupby(["status", "already_out", "d"])["out"].agg(["mean", "size"])
    return fine, coarse


def ir_history():
    """P(still on a reserve list d weeks later | reserve type)."""
    frames = []
    for y in HIST_SEASONS:
        f = RAW / f"rosters_weekly_{y}.parquet"
        if f.exists():
            r = pd.read_parquet(f, columns=["season", "week", "game_type", "gsis_id", "status", "status_description_abbr"])
            frames.append(r[r["game_type"] == "REG"])
    r = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "week", "gsis_id"])
    res = set(zip(r.loc[r.status == "RES", "season"], r.loc[r.status == "RES", "week"], r.loc[r.status == "RES", "gsis_id"]))
    lastw = r.groupby("season")["week"].max().to_dict()
    cur = r[r["status"] == "RES"].copy()
    cur["kind"] = cur["status_description_abbr"].map(reserve_kind)
    first = cur.groupby(["season", "gsis_id"])["week"].transform("min")
    cur["on_list"] = (cur["week"] - first + 1).map(on_list_bucket)
    rows = []
    for rec in cur.itertuples(index=False):
        for d in range(MAX_D):
            w = rec.week + d
            if w > lastw[rec.season]:
                break
            rows.append((rec.kind, rec.on_list, d, (rec.season, w, rec.gsis_id) in res))
    h = pd.DataFrame(rows, columns=["kind", "on_list", "d", "out"])
    return h.groupby(["kind", "on_list", "d"])["out"].mean()


def reserve_kind(code):
    return {"R48": "returning", "R04": "pup", "R01": "ir"}.get(code, "other")


def on_list_bucket(weeks):
    return "1-2" if weeks <= 2 else "3-5" if weeks <= 5 else "6+"


class InjuryModel:
    def __init__(self, games):
        print("Learning injury durations from 2012–2025 …")
        self.pfr = id_map()
        self.snaps = load_snaps()
        self.fine, self.coarse = report_history(games, self.snaps, self.pfr)
        self.ir = ir_history()

    def p_out(self, status, cat, already_out, d):
        key = (status, cat, already_out, d)
        if key in self.fine.index and self.fine.loc[key, "size"] >= 30:
            return float(self.fine.loc[key, "mean"])
        key = (status, already_out, d)
        if key in self.coarse.index:
            return float(self.coarse.loc[key, "mean"])
        return 0.0

    def p_ir(self, kind, on_list, d):
        d = min(d, MAX_D - 1)
        for key in [(kind, on_list, d), ("ir", on_list, d)]:
            if key in self.ir.index:
                return float(self.ir[key])
        return 0.5

    # ── historical feature (training) ───────────────────────────
    def team_week_absences(self, seasons):
        r = attach_importance(load_reports(seasons), self.snaps, self.pfr)
        r = r[r["side"].isin(["off", "def"])]
        r["p"] = [self.p_out(s, c, False, 0) for s, c in zip(r.report_status, r.cat)]
        r["w"] = r["imp"] * r["p"]
        t = r.pivot_table(index=["season", "week", "team"], columns="side", values="w", aggfunc="sum", fill_value=0)
        return {k: {"off": v.get("off", 0), "def": v.get("def", 0)} for k, v in t.to_dict("index").items()}

    # ── current season projection ───────────────────────────────
    def project(self, season, upcoming_week, games):
        """One row per currently injured player with P(out) for each remaining week."""
        weeks = list(range(upcoming_week, 19))
        players = []
        # 1. latest injury report
        rep_f = RAW / f"injuries_{season}.parquet"
        if rep_f.exists():
            rep = load_reports([season])
            rep = rep[rep["week"] == rep["week"].max()]
            rep_week = int(rep["week"].max()) if len(rep) else upcoming_week
            rep = attach_importance(rep, self.snaps, self.pfr)
            snaps_now = self.snaps[self.snaps.season == season]
            played = set(zip(snaps_now.week, snaps_now.pfr_player_id))
            for rec in rep.itertuples(index=False):
                prev = [w for (s, w, t) in team_game_weeks(games) if s == season and t == rec.team and w < rep_week]
                already = bool(prev) and (max(prev), rec.pfr_player_id) not in played
                probs = {w: self.p_out(rec.report_status, rec.cat, already, max(0, w - rep_week)) for w in weeks}
                players.append(dict(team=rec.team, gsis_id=rec.gsis_id, name=rec.full_name, pos=rec.position,
                                    side=rec.side, status=rec.report_status, injury=rec.report_primary_injury,
                                    imp=float(rec.imp), source="report", p=probs))
        # 2. reserve lists (injured reserve, PUP, …)
        ros_f = RAW / f"rosters_weekly_{season}.parquet"
        seen = {p["gsis_id"] for p in players}
        if ros_f.exists():
            ros = pd.read_parquet(ros_f)
            ros = ros[ros["game_type"] == "REG"]
            ros["team"] = ros["team"].replace(FRANCHISE)
            latest = ros[ros["week"] == ros["week"].max()]
            res = latest[(latest["status"] == "RES") & ~latest["gsis_id"].isin(seen)].copy()
            first_res = ros[ros.status == "RES"].groupby("gsis_id")["week"].min()
            res["season"], res["week"] = season, upcoming_week
            res = attach_importance(res, self.snaps, self.pfr)
            now_w = int(latest["week"].max())
            for rec in res.itertuples(index=False):
                kind = reserve_kind(rec.status_description_abbr)
                on_list = on_list_bucket(now_w - int(first_res.get(rec.gsis_id, now_w)) + 1)
                probs = {w: self.p_ir(kind, on_list, w - now_w) for w in weeks}
                label = {"R48": "IR – designated to return", "R01": "Injured reserve",
                         "R04": "PUP list", "R05": "Non-football injury"}.get(rec.status_description_abbr, "Reserve list")
                players.append(dict(team=rec.team, gsis_id=rec.gsis_id, name=rec.full_name, pos=rec.position,
                                    side=side_of(rec.position), status=label, injury=None,
                                    imp=float(rec.imp), source="reserve", p=probs,
                                    since_week=int(first_res.get(rec.gsis_id, now_w))))
        # 3. manual overrides
        if OVERRIDES.exists():
            ov = pd.read_csv(OVERRIDES, comment="#").dropna(how="all")
            for rec in ov.itertuples(index=False):
                match = [p for p in players if p["team"] == rec.team and p["name"] == rec.player]
                probs = {w: 1.0 if w <= int(rec.out_through_week) else 0.0 for w in weeks}
                if match:
                    for p in match:
                        p.update(p=probs, source="override", note=getattr(rec, "note", None))
                else:
                    players.append(dict(team=rec.team, gsis_id=None, name=rec.player, pos=None, side=None,
                                        status="Out (manual)", injury=getattr(rec, "note", None), imp=0.0,
                                        source="override", p=probs))
        for p in players:
            ret = [w for w in weeks if p["p"][w] < 0.5]
            p["expected_return"] = ret[0] if ret else None     # None = likely out for the season
        return players

    @staticmethod
    def team_week_projection(players):
        out = {}
        for p in players:
            if p["side"] not in ("off", "def") or p["imp"] <= 0:
                continue
            for w, pr in p["p"].items():
                d = out.setdefault((p["team"], w), {"off": 0.0, "def": 0.0})
                d[p["side"]] += p["imp"] * pr
        return out
