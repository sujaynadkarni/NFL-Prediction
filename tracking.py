"""
Pick tracking — an honest 2026 scorecard
- Ledger: the latest prediction made BEFORE kickoff for each game (frozen once the game starts)
- Scorecard: our picks vs results and vs the Vegas spread, by week
- Games with no saved pick (Week 1, Week 2 Thursday) are scored as "reconstructed":
  the model's leak-free pre-game prediction, reported separately from live picks
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT   = Path(__file__).parent
SNAPS  = ROOT / "snapshots"
LEDGER = SNAPS / "pick_ledger.json"
ET     = ZoneInfo("America/New_York")
KEEP   = ["week", "home", "away", "p_home", "margin", "favorite", "vegas_home_spread", "proj_home", "proj_away"]


def kickoff(g):
    return datetime.strptime(f"{g['gameday']} {g['gametime'] or '13:00'}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def load():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def seed_from_snapshots(ledger):
    """Add pre-kickoff picks from saved snapshot files (e.g. the Week 2 snapshot taken before this ledger existed)."""
    for f in sorted(SNAPS.glob("*_pre.json")):
        made = datetime.fromtimestamp(f.stat().st_mtime, ET)
        snap = json.loads(f.read_text())
        for g in snap["games"]:
            if (g["game_id"] not in ledger and not g["played"] and made < kickoff(g)
                    and g["week"] == snap["upcoming_week"]):
                ledger[g["game_id"]] = {k: g[k] for k in KEEP} | {"made_at": made.isoformat(timespec="minutes"),
                                                                  "source": f.name}
    return ledger


def update(games, season, upcoming_week):
    """Record/refresh picks for games that haven't kicked off yet; save this run's snapshot."""
    SNAPS.mkdir(exist_ok=True)
    ledger = seed_from_snapshots(load())
    now = datetime.now(ET)
    n = 0
    for g in games:
        if not g["played"] and now < kickoff(g) and g["week"] == upcoming_week:
            ledger[g["game_id"]] = {k: g[k] for k in KEEP} | {"made_at": now.isoformat(timespec="minutes"),
                                                              "source": "weekly run"}
            n += 1
    LEDGER.write_text(json.dumps(ledger, indent=1))
    return ledger, n


def scorecard(games, ledger, sigma):
    from nfl_model import norm_cdf
    rows = []
    for g in games:
        if not g["played"] or g["home_score"] == g["away_score"]:
            continue
        e = ledger.get(g["game_id"])
        src = "live" if e else "reconstructed"
        p = e["p_home"] if e else g["p_home"]
        home_won = g["home_score"] > g["away_score"]
        spread = (e or g).get("vegas_home_spread")
        r = {"game_id": g["game_id"], "week": g["week"], "source": src, "p_home": p,
             "correct": (p > 0.5) == home_won, "brier": (p - home_won) ** 2}
        if spread is not None and spread != 0:
            pv = norm_cdf(spread / sigma)
            r.update(vegas_correct=(spread > 0) == home_won, vegas_brier=(pv - home_won) ** 2)
        rows.append(r)

    def agg(rs):
        v = [r for r in rs if "vegas_correct" in r]
        return {"games": len(rs), "correct": sum(r["correct"] for r in rs),
                "brier": round(sum(r["brier"] for r in rs) / len(rs), 4) if rs else None,
                "vegas_games": len(v), "vegas_correct": sum(r["vegas_correct"] for r in v),
                "vegas_brier": round(sum(r["vegas_brier"] for r in v) / len(v), 4) if v else None}

    weeks = sorted({r["week"] for r in rows})
    return {
        "all":           agg(rows),
        "live":          agg([r for r in rows if r["source"] == "live"]),
        "reconstructed": agg([r for r in rows if r["source"] == "reconstructed"]),
        "by_week": [{"week": w, **agg([r for r in rows if r["week"] == w]),
                     "live": sum(r["source"] == "live" for r in rows if r["week"] == w)} for w in weeks],
        "games": {r["game_id"]: {"source": r["source"], "correct": r["correct"], "p_home": r["p_home"]} for r in rows},
    }
