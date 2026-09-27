"""
NFL data download (nflverse)
- games.csv          : every game 1999-2026 (scores, spreads, QBs, coaches, rest, stadium)
- play_by_play_YYYY  : every play with EPA, aggregated here to team-game efficiency
- injuries_YYYY      : weekly injury reports (2009+)
- draft_picks        : all draft picks
- snap_counts_YYYY   : player snap shares (2012+) — how important an injured player is
- rosters_weekly     : weekly roster status incl. injured reserve (2016+)
- depth_charts       : current season QB order

Run:  .venv/bin/python fetch_data.py            (full download)
      .venv/bin/python fetch_data.py --current  (weekly: only refresh current season)
"""
import sys, time
from pathlib import Path
import requests
import pandas as pd

ROOT     = Path(__file__).parent
RAW      = ROOT / "data" / "raw"
PROC     = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

FIRST_SEASON   = 1999
CURRENT_SEASON = 2026
REL = "https://github.com/nflverse/nflverse-data/releases/download"

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam",
            "play_type", "epa", "pass", "rush", "success", "qb_dropback"]


def download(url, dest, force=False):
    if dest.exists() and not force:
        return False
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except Exception as e:
            print(f"  retry {attempt+1} for {dest.name}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download {url}")


def team_game_epa(pbp):
    """Collapse plays into one row per (game, offense team): efficiency stats."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()]
    g = p.groupby(["game_id", "season", "week", "posteam", "defteam"])
    out = g.agg(plays=("epa", "size"),
                epa_per_play=("epa", "mean"),
                success_rate=("success", "mean")).reset_index()
    pas = p[p["pass"] == 1].groupby(["game_id", "posteam"])["epa"].mean().rename("pass_epa")
    run = p[p["rush"] == 1].groupby(["game_id", "posteam"])["epa"].mean().rename("rush_epa")
    return out.merge(pas, on=["game_id", "posteam"], how="left") \
              .merge(run, on=["game_id", "posteam"], how="left")


def main(current_only=False):
    t0 = time.time()
    seasons = [CURRENT_SEASON] if current_only else range(FIRST_SEASON, CURRENT_SEASON + 1)

    print("Games / schedule …")
    download("https://github.com/nflverse/nfldata/raw/master/data/games.csv",
             RAW / "games.csv", force=True)

    print("Play-by-play …")
    frames = []
    for s in seasons:
        f = RAW / f"pbp_{s}.parquet"
        download(f"{REL}/pbp/play_by_play_{s}.parquet", f, force=(s == CURRENT_SEASON))
        pbp = pd.read_parquet(f, columns=PBP_COLS)
        frames.append(team_game_epa(pbp))
        print(f"  {s}: {len(pbp):,} plays")
    new = pd.concat(frames, ignore_index=True)
    out = PROC / "team_game_epa.parquet"
    if current_only and out.exists():
        old = pd.read_parquet(out)
        new = pd.concat([old[old["season"] != CURRENT_SEASON], new], ignore_index=True)
    new.to_parquet(out, index=False)

    print("Injuries …")
    for s in (seasons if current_only else range(2009, CURRENT_SEASON + 1)):
        try:
            download(f"{REL}/injuries/injuries_{s}.parquet", RAW / f"injuries_{s}.parquet",
                     force=(s == CURRENT_SEASON))
        except RuntimeError as e:
            print(f"  skipped {s}: {e}")

    print("Snap counts (2012+) …")
    for s in (seasons if current_only else range(2012, CURRENT_SEASON + 1)):
        download(f"{REL}/snap_counts/snap_counts_{s}.parquet", RAW / f"snap_counts_{s}.parquet",
                 force=(s == CURRENT_SEASON))

    print("Weekly rosters / reserve lists (2016+) …")
    for s in (seasons if current_only else range(2016, CURRENT_SEASON + 1)):
        download(f"{REL}/weekly_rosters/roster_weekly_{s}.parquet", RAW / f"rosters_weekly_{s}.parquet",
                 force=(s == CURRENT_SEASON))

    print("Season rosters (player id map, 2012+) …")
    for s in (seasons if current_only else range(2012, CURRENT_SEASON + 1)):
        download(f"{REL}/rosters/roster_{s}.parquet", RAW / f"roster_{s}.parquet",
                 force=(s == CURRENT_SEASON))

    print("Depth charts (current season) …")
    download(f"{REL}/depth_charts/depth_charts_{CURRENT_SEASON}.parquet",
             RAW / f"depth_charts_{CURRENT_SEASON}.parquet", force=True)

    print("Draft picks …")
    download(f"{REL}/draft_picks/draft_picks.parquet", RAW / "draft_picks.parquet",
             force=not current_only)

    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(current_only="--current" in sys.argv)
