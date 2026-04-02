import asyncio
import aiohttp
import nest_asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from understat import Understat

nest_asyncio.apply()

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Understat Player Explorer", layout="wide")

# --------------------------------------------------
# Constants
# --------------------------------------------------
LEAGUE_OPTIONS = {
    "EPL": "epl",
    "La Liga": "la_liga",
    "Bundesliga": "bundesliga",
    "Serie A": "serie_a",
    "Ligue 1": "ligue_1",
    "RFPL": "rfpl",
}

SEASON_START_YEARS = list(range(2014, 2026))
SEASON_OPTIONS = ["All seasons"] + SEASON_START_YEARS

SHOT_NUMERIC_COLS = [
    "id", "minute", "X", "Y", "xG", "player_id", "match_id", "h_goals", "a_goals", "season"
]

MATCH_NUMERIC_COLS = [
    "id", "goals", "shots", "time", "h_goals", "a_goals", "xG", "xA", "assists",
    "key_passes", "npg", "npxG", "xGChain", "xGBuildup", "season"
]

# --------------------------------------------------
# Async helpers
# --------------------------------------------------
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def fetch_league_players_and_teams_one(league_code: str, season_start: int):
    async with aiohttp.ClientSession() as session:
        u = Understat(session)
        try:
            players = await u.get_league_players(league_code, season_start)
        except Exception:
            players = []
        try:
            teams = await u.get_teams(league_code, season_start)
        except Exception:
            teams = []
    return players, teams


async def fetch_player_bundle(player_id: int):
    async with aiohttp.ClientSession() as session:
        u = Understat(session)
        try:
            shots = await u.get_player_shots(player_id)
        except Exception:
            shots = []
        try:
            matches = await u.get_player_matches(player_id)
        except Exception:
            matches = []
        try:
            grouped = await u.get_player_grouped_stats(player_id)
        except Exception:
            grouped = None
    return shots, matches, grouped

# --------------------------------------------------
# Cleaners
# --------------------------------------------------
def clean_players_df(players_raw, league_code, season_start):
    df = pd.DataFrame(players_raw).copy()
    if df.empty:
        return df

    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    df["selected_league"] = league_code
    df["selected_season"] = season_start

    if "team_title" in df.columns:
        df["team_title"] = df["team_title"].astype(str)

    keep_first = [
        "id", "player_name", "team_title", "position", "games", "time", "goals", "shots",
        "xG", "xA", "assists", "key_passes", "npg", "npxG", "xGChain", "xGBuildup",
        "selected_league", "selected_season"
    ]
    keep_first = [c for c in keep_first if c in df.columns]
    rest = [c for c in df.columns if c not in keep_first]
    df = df[keep_first + rest]

    sort_cols = [c for c in ["team_title", "player_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def clean_shots_df(shots_raw):
    df = pd.DataFrame(shots_raw).copy()
    if df.empty:
        return df

    rename_map = {
        "id": "shot_id",
        "h_team": "home_team",
        "a_team": "away_team",
    }
    df = df.rename(columns=rename_map)

    for col in SHOT_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    if "match_id" in df.columns:
        df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    return df


def clean_matches_df(matches_raw):
    df = pd.DataFrame(matches_raw).copy()
    if df.empty:
        return df

    rename_map = {
        "id": "match_id",
        "h_team": "home_team",
        "a_team": "away_team",
    }
    df = df.rename(columns=rename_map)

    for col in MATCH_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "match_id" in df.columns:
        df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    return df

# --------------------------------------------------
# Filtering helpers
# --------------------------------------------------
def parse_player_selection(selection_text):
    if not selection_text or "| id=" not in selection_text:
        return None
    try:
        return int(selection_text.split("| id=")[-1].strip())
    except Exception:
        return None


def split_team_string(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(",") if t.strip()]


def build_team_list_from_roster(roster_df):
    teams = set()
    if roster_df is None or roster_df.empty or "team_title" not in roster_df.columns:
        return []
    for val in roster_df["team_title"].dropna():
        for t in split_team_string(val):
            teams.add(t)
    return sorted(teams)


def filter_roster_for_team(roster_df, team_name):
    if roster_df is None or roster_df.empty:
        return pd.DataFrame()
    if team_name == "All teams":
        return roster_df.copy()

    mask = roster_df["team_title"].apply(lambda x: team_name in split_team_string(x))
    return roster_df.loc[mask].copy()


def build_player_options(roster_df):
    if roster_df is None or roster_df.empty:
        return []

    out = roster_df.copy()
    out = out.dropna(subset=["id", "player_name"])
    out["id"] = out["id"].astype(int)

    label_team = out["team_title"] if "team_title" in out.columns else ""
    label_season = out["selected_season"] if "selected_season" in out.columns else ""

    out["label"] = (
        out["player_name"].astype(str)
        + " | " + label_team.astype(str)
        + " | season=" + label_season.astype(str)
        + " | id=" + out["id"].astype(str)
    )

    out = out.sort_values(["player_name", "team_title", "selected_season"])
    return out["label"].drop_duplicates().tolist()


def filter_matches_by_selection(matches_df, season_choice, team_choice):
    if matches_df is None or matches_df.empty:
        return pd.DataFrame()

    out = matches_df.copy()

    if season_choice != "All seasons" and "season" in out.columns:
        out = out[out["season"] == int(season_choice)]

    if team_choice != "All teams":
        mask = pd.Series(False, index=out.index)
        if "home_team" in out.columns:
            mask = mask | (out["home_team"] == team_choice)
        if "away_team" in out.columns:
            mask = mask | (out["away_team"] == team_choice)
        out = out[mask]

    return out.reset_index(drop=True)


def filter_shots_by_matches(shots_df, valid_match_ids, season_choice=None):
    if shots_df is None or shots_df.empty:
        return pd.DataFrame()

    out = shots_df.copy()

    if "match_id" in out.columns:
        out = out[out["match_id"].isin(valid_match_ids)]

    if season_choice != "All seasons" and "season" in out.columns:
        out = out[out["season"] == int(season_choice)]

    return out.reset_index(drop=True)


def derive_home_away_dummies(df, team_choice):
    out = df.copy()
    out["is_home"] = np.nan
    out["is_away"] = np.nan

    if team_choice != "All teams":
        if "home_team" in out.columns and "away_team" in out.columns:
            out["is_home"] = np.where(out["home_team"] == team_choice, 1, 0)
            out["is_away"] = np.where(out["away_team"] == team_choice, 1, 0)
    else:
        side_col = None
        for c in ["h_a", "side", "match_h_a", "match_side"]:
            if c in out.columns:
                side_col = c
                break
        if side_col is not None:
            out["is_home"] = np.where(out[side_col].astype(str).str.lower().isin(["h", "home"]), 1, 0)
            out["is_away"] = np.where(out[side_col].astype(str).str.lower().isin(["a", "away"]), 1, 0)

    return out


def build_shot_level_table(shots_df, matches_df, season_choice, team_choice):
    season_matches = filter_matches_by_selection(matches_df, season_choice, team_choice)

    if season_matches.empty or "match_id" not in season_matches.columns:
        return pd.DataFrame(), pd.DataFrame()

    valid_match_ids = set(season_matches["match_id"].dropna().astype(int).tolist())
    selected_shots = filter_shots_by_matches(shots_df, valid_match_ids, season_choice=season_choice)

    if selected_shots.empty:
        return pd.DataFrame(), season_matches

    match_cols = [
        c for c in [
            "match_id", "date", "season", "home_team", "away_team",
            "time", "goals", "shots", "xG", "xA", "assists", "key_passes",
            "npg", "npxG", "xGChain", "xGBuildup", "position",
            "h_goals", "a_goals", "h_a", "side"
        ] if c in season_matches.columns
    ]

    match_small = season_matches[match_cols].copy()

    rename_map = {
        "date": "match_date",
        "season": "match_season",
        "home_team": "match_home_team",
        "away_team": "match_away_team",
        "time": "match_minutes",
        "goals": "match_goals",
        "shots": "match_shots",
        "xG": "match_xG",
        "xA": "match_xA",
        "assists": "match_assists",
        "key_passes": "match_key_passes",
        "npg": "match_npg",
        "npxG": "match_npxG",
        "xGChain": "match_xGChain",
        "xGBuildup": "match_xGBuildup",
        "position": "match_position",
        "h_goals": "match_home_goals",
        "a_goals": "match_away_goals",
        "h_a": "match_h_a",
        "side": "match_side",
    }
    match_small = match_small.rename(columns=rename_map)

    selected_shots = selected_shots.merge(match_small, on="match_id", how="left")

    if "home_team" not in selected_shots.columns and "match_home_team" in selected_shots.columns:
        selected_shots["home_team"] = selected_shots["match_home_team"]
    if "away_team" not in selected_shots.columns and "match_away_team" in selected_shots.columns:
        selected_shots["away_team"] = selected_shots["match_away_team"]

    selected_shots = derive_home_away_dummies(selected_shots, team_choice)

    if "result" in selected_shots.columns:
        selected_shots["is_goal"] = np.where(selected_shots["result"].astype(str) == "Goal", 1, 0)
    else:
        selected_shots["is_goal"] = np.nan

    first_cols = [
        "player_id", "match_id", "shot_id", "match_date", "match_season",
        "home_team", "away_team", "is_home", "is_away",
        "minute", "result", "is_goal", "situation", "shotType", "X", "Y", "xG",
        "player_assisted", "lastAction",
        "match_minutes", "match_goals", "match_shots", "match_xG", "match_xA",
        "match_assists", "match_key_passes", "match_npg", "match_npxG",
        "match_xGChain", "match_xGBuildup", "match_position",
        "match_home_goals", "match_away_goals"
    ]
    first_cols = [c for c in first_cols if c in selected_shots.columns]
    remaining = [c for c in selected_shots.columns if c not in first_cols]
    selected_shots = selected_shots[first_cols + remaining]

    return selected_shots, season_matches


def build_match_level_table(matches_df, season_choice, team_choice):
    out = filter_matches_by_selection(matches_df, season_choice, team_choice)

    if out.empty:
        return out

    out = derive_home_away_dummies(out, team_choice)

    rename_map = {
        "date": "match_date",
        "season": "match_season",
        "time": "match_minutes",
        "goals": "match_goals",
        "shots": "match_shots",
        "xG": "match_xG",
        "xA": "match_xA",
        "assists": "match_assists",
        "key_passes": "match_key_passes",
        "npg": "match_npg",
        "npxG": "match_npxG",
        "xGChain": "match_xGChain",
        "xGBuildup": "match_xGBuildup",
        "position": "match_position",
        "h_goals": "match_home_goals",
        "a_goals": "match_away_goals",
    }
    out = out.rename(columns=rename_map)

    first_cols = [
        "match_id", "match_date", "match_season", "home_team", "away_team",
        "is_home", "is_away", "match_minutes", "match_goals", "match_shots",
        "match_xG", "match_xA", "match_assists", "match_key_passes",
        "match_npg", "match_npxG", "match_xGChain", "match_xGBuildup",
        "match_position", "match_home_goals", "match_away_goals"
    ]
    first_cols = [c for c in first_cols if c in out.columns]
    remaining = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + remaining]

    return out.reset_index(drop=True)

# --------------------------------------------------
# Summary helpers
# --------------------------------------------------
def safe_numeric(series):
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def shot_summary_df(shots_df):
    if shots_df is None or shots_df.empty:
        return pd.DataFrame({"Metric": [], "Value": []})

    xg = safe_numeric(shots_df["xG"]) if "xG" in shots_df.columns else pd.Series(dtype=float)
    n_shots = len(shots_df)
    n_matches = shots_df["match_id"].nunique() if "match_id" in shots_df.columns else np.nan
    n_goals = int(shots_df["is_goal"].sum()) if "is_goal" in shots_df.columns else np.nan

    xg_nonmissing = xg.dropna()
    xg_sd = xg_nonmissing.std(ddof=1) if len(xg_nonmissing) > 1 else 0.0
    xg_var = xg_nonmissing.var(ddof=1) if len(xg_nonmissing) > 1 else 0.0

    metrics = {
        "Total shots": n_shots,
        "Total goals": n_goals,
        "Total xG": xg.sum(),
        "Mean shot xG": xg.mean(),
        "Median shot xG": xg.median(),
        "SD shot xG": xg_sd,
        "Variance shot xG": xg_var,
        "Shots per match": (n_shots / n_matches) if pd.notna(n_matches) and n_matches > 0 else np.nan,
        "Goal rate": (n_goals / n_shots) if n_shots > 0 and pd.notna(n_goals) else np.nan,
        "Home-shot share": shots_df["is_home"].mean() if "is_home" in shots_df.columns else np.nan,
        "Away-shot share": shots_df["is_away"].mean() if "is_away" in shots_df.columns else np.nan,
        "Mean shot minute": safe_numeric(shots_df["minute"]).mean() if "minute" in shots_df.columns else np.nan,
    }
    return pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})


def match_summary_df(match_df):
    if match_df is None or match_df.empty:
        return pd.DataFrame({"Metric": [], "Value": []})

    xg = safe_numeric(match_df["match_xG"]) if "match_xG" in match_df.columns else pd.Series(dtype=float)
    xa = safe_numeric(match_df["match_xA"]) if "match_xA" in match_df.columns else pd.Series(dtype=float)
    mins = safe_numeric(match_df["match_minutes"]) if "match_minutes" in match_df.columns else pd.Series(dtype=float)

    xg_nonmissing = xg.dropna()
    xg_sd = xg_nonmissing.std(ddof=1) if len(xg_nonmissing) > 1 else 0.0
    xg_var = xg_nonmissing.var(ddof=1) if len(xg_nonmissing) > 1 else 0.0

    metrics = {
        "Matches": len(match_df),
        "Total match xG": xg.sum(),
        "Mean match xG": xg.mean(),
        "Median match xG": xg.median(),
        "SD match xG": xg_sd,
        "Variance match xG": xg_var,
        "Mean match xA": xa.mean(),
        "Mean minutes": mins.mean(),
    }
    return pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})

# --------------------------------------------------
# Cached loaders
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_roster_cached(league_code, season_choice):
    if season_choice == "All seasons":
        roster_parts = []
        for yr in SEASON_START_YEARS:
            try:
                players_raw, _ = run_async(fetch_league_players_and_teams_one(league_code, int(yr)))
            except Exception:
                players_raw = []
            if not players_raw:
                continue
            players_df = clean_players_df(players_raw, league_code, int(yr))
            if not players_df.empty:
                roster_parts.append(players_df)
        if roster_parts:
            return pd.concat(roster_parts, ignore_index=True).drop_duplicates()
        return pd.DataFrame()

    try:
        players_raw, _ = run_async(fetch_league_players_and_teams_one(league_code, int(season_choice)))
    except Exception:
        players_raw = []

    if not players_raw:
        return pd.DataFrame()

    return clean_players_df(players_raw, league_code, int(season_choice))


@st.cache_data(show_spinner=False)
def load_player_data_cached(player_id):
    try:
        shots_raw, matches_raw, _ = run_async(fetch_player_bundle(int(player_id)))
    except Exception:
        shots_raw, matches_raw = [], []
    shots_df = clean_shots_df(shots_raw)
    matches_df = clean_matches_df(matches_raw)
    return shots_df, matches_df

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Understat Player Explorer")
st.caption(
    "Choose a data type, league, season, team, and player. "
    "Shot mode returns shot-level rows with match context merged in."
)

col1, col2, col3 = st.columns([1.4, 1, 1])
with col1:
    mode = st.radio("Data type", ["Player shot data", "Player match data"], horizontal=True)
with col2:
    league_label = st.selectbox("League", list(LEAGUE_OPTIONS.keys()), index=0)
with col3:
    season_choice = st.selectbox("Season", SEASON_OPTIONS, index=SEASON_OPTIONS.index(2024))

league_code = LEAGUE_OPTIONS[league_label]

with st.spinner("Loading roster..."):
    roster_df = load_roster_cached(league_code, season_choice)

if roster_df.empty:
    st.error(
        "No roster data could be loaded for this selection. "
        "This usually means the Understat site structure changed or that the package could not parse the page. "
        "Try another league/season, or redeploy later."
    )
    st.stop()

team_options = ["All teams"] + build_team_list_from_roster(roster_df)
team_choice = st.selectbox("Team", team_options, index=0)

roster_filtered = filter_roster_for_team(roster_df, team_choice)
player_options = build_player_options(roster_filtered)

if not player_options:
    st.warning("No players found for this team/season selection.")
    st.stop()

player_label = st.selectbox("Player", player_options)
player_id = parse_player_selection(player_label)
player_name = player_label.split("|")[0].strip() if player_label else "Player"

load_clicked = st.button("Load player data", type="primary")

if load_clicked:
    if player_id is None:
        st.error("Could not parse the selected player ID.")
        st.stop()

    with st.spinner(f"Loading player data for {player_name}..."):
        shots_df, matches_df = load_player_data_cached(player_id)

    if mode == "Player shot data":
        current_table, filtered_matches = build_shot_level_table(
            shots_df=shots_df,
            matches_df=matches_df,
            season_choice=season_choice,
            team_choice=team_choice,
        )

        if current_table.empty:
            st.warning("No shot rows found for this selection.")
            st.stop()

        st.success("Shot-level data loaded.")

        st.subheader("Shot-level summary")
        summary_df = shot_summary_df(current_table)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            if "xG" in current_table.columns:
                vals = safe_numeric(current_table["xG"]).dropna().values
                if len(vals) > 0:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.hist(vals, bins=min(20, max(5, len(vals))))
                    ax.set_title("Shot xG distribution")
                    ax.set_xlabel("Shot xG")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)

        with plot_col2:
            if "result" in current_table.columns:
                counts = current_table["result"].astype(str).value_counts()
                if len(counts) > 0:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    counts.plot(kind="bar", ax=ax)
                    ax.set_title("Shot results")
                    ax.set_xlabel("Result")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)

        if "situation" in current_table.columns:
            counts = current_table["situation"].astype(str).value_counts()
            if len(counts) > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                counts.plot(kind="bar", ax=ax)
                ax.set_title("Shot situations")
                ax.set_xlabel("Situation")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

        st.subheader("Shot-level table")
        st.dataframe(current_table, use_container_width=True, hide_index=True)

        csv_data = current_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download shot-level CSV",
            data=csv_data,
            file_name=f"{player_name.replace(' ', '_')}_{league_code}_{season_choice}_shot.csv",
            mime="text/csv",
        )

    else:
        current_table = build_match_level_table(
            matches_df=matches_df,
            season_choice=season_choice,
            team_choice=team_choice,
        )

        if current_table.empty:
            st.warning("No match rows found for this selection.")
            st.stop()

        st.success("Match-level data loaded.")

        st.subheader("Match-level summary")
        summary_df = match_summary_df(current_table)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if "match_xG" in current_table.columns:
            vals = safe_numeric(current_table["match_xG"]).dropna().values
            if len(vals) > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(vals, bins=min(15, max(5, len(vals))))
                ax.set_title("Match xG distribution")
                ax.set_xlabel("Match xG")
                ax.set_ylabel("Count")
                st.pyplot(fig)

        st.subheader("Match-level table")
        st.dataframe(current_table, use_container_width=True, hide_index=True)

        csv_data = current_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download match-level CSV",
            data=csv_data,
            file_name=f"{player_name.replace(' ', '_')}_{league_code}_{season_choice}_match.csv",
            mime="text/csv",
        )
