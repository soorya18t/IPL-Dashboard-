"""
app.py – Unified IPL Analytics Dashboard, Feature Engineering & ML Pipeline
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- PREMIUM PAGE CONFIG ---
st.set_page_config(page_title="Pro Cricket Analytics", page_icon="🏏", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR A COLORFUL, ATTRACTIVE UI ---
def inject_custom_css():
    st.markdown("""
    <style>
    /* Stylish Metric Cards with Neon Left Borders */
    [data-testid="stMetric"] {
        background-color: #1E2130;
        border-left: 5px solid #00F2FE;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-left: 5px solid #FF0844;
    }
    /* Gradient Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11141E 0%, #0B0C10 100%);
    }
    /* Colorful Headers */
    h1, h2 {
        background: -webkit-linear-gradient(45deg, #FF9A9E 0%, #FECFEF 99%, #FECFEF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ═══════════════════════════════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR            = Path(".")
DELIVERIES_PATH     = DATA_DIR / "deliveries_yearwise.csv"
MATCHES_PATH        = DATA_DIR / "matches_yearwise.csv"
OUTPUT_PLAYER_STATS = DATA_DIR / "player_stats.csv"
OUTPUT_MATCH_STATS  = DATA_DIR / "player_match_stats.csv"
OUTPUT_TRAINING     = DATA_DIR / "training_data.csv"
MODEL_PATH          = DATA_DIR / "ipl_model.joblib"
SQUADS_PATH         = DATA_DIR / "current_squads.csv"
FIXES_PATH          = DATA_DIR / "player_fixes.csv"

BATTING_DEFAULTS = {"bat_sr":115.0,"bat_avg":20.0,"pp_sr":110.0,"death_sr":120.0,"boundary_pct":30.0}
BOWLING_DEFAULTS = {"bowl_economy":8.5,"bowl_sr":25.0,"pp_economy":8.0,"death_economy":10.0}

PLAYER_ROLE_COLORS   = {"Batter": "#00F2FE", "Bowler": "#FF0844", "All-rounder": "#F5AF19", "WK-Batter": "#00C9FF", "Wicket-keeper": "#00C9FF", "Unknown": "#888888"}

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: DATA CLEANING & LOADING (SILENT UI)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_raw_data():
    deliveries = pd.read_csv(DELIVERIES_PATH, low_memory=False)
    matches = pd.read_csv(MATCHES_PATH)
    
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    season_mapping = {"2007/08": "2008", "2009/10": "2010", "2020/21": "2020"}
    matches["season"] = matches["season"].astype(str).replace(season_mapping)
    matches["season"] = matches["season"].str.extract(r'(\d{4})')[0]
    
    team_mapping = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Rising Pune Supergiant": "Rising Pune Supergiants",
        "Rising Pune Supergiants": "Rising Pune Supergiants",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Pune Warriors": "Pune Warriors" 
    }
    matches["team1"] = matches["team1"].replace(team_mapping)
    matches["team2"] = matches["team2"].replace(team_mapping)
    matches["toss_winner"] = matches["toss_winner"].replace(team_mapping)
    matches["winner"] = matches["winner"].replace(team_mapping)
    deliveries["batting_team"] = deliveries["batting_team"].replace(team_mapping)
    deliveries["bowling_team"] = deliveries["bowling_team"].replace(team_mapping)

    venue_mapping = {
        "M Chinnaswamy Stadium, Bengaluru": "M. Chinnaswamy Stadium",
        "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
        "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
        "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
        "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
        "Feroz Shah Kotla": "Arun Jaitley Stadium",
        "Feroz Shah Kotla Ground": "Arun Jaitley Stadium",
        "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
        "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
        "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
        "Wankhede Stadium, Mumbai": "Wankhede Stadium",
        "Eden Gardens, Kolkata": "Eden Gardens",
        "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
        "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
        "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
        "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
        "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
        "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
        "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
        "Brabourne Stadium, Mumbai": "Brabourne Stadium",
        "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
        "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
        "Sharjah Cricket Stadium": "Sharjah Cricket Stadium",
        "Dubai International Cricket Stadium": "Dubai International Cricket Stadium",
        "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium"
    }
    matches["venue"] = matches["venue"].replace(venue_mapping)

    if FIXES_PATH.exists():
        fixes_df = pd.read_csv(FIXES_PATH)
        player_mapping = dict(zip(fixes_df["Wrong_Name"], fixes_df["Correct_Name"]))
    else:
        player_mapping = {}

    matches["player_of_match"] = matches["player_of_match"].replace(player_mapping)
    deliveries["batter"] = deliveries["batter"].replace(player_mapping)
    deliveries["bowler"] = deliveries["bowler"].replace(player_mapping)
    deliveries["non_striker"] = deliveries["non_striker"].replace(player_mapping)
    if "player_dismissed" in deliveries.columns:
        deliveries["player_dismissed"] = deliveries["player_dismissed"].replace(player_mapping)

    return deliveries, matches

@st.cache_data
def load_processed_data():
    player_stats = pd.read_csv("player_stats.csv")
    if "player" in player_stats.columns:
        player_stats.set_index("player", inplace=True)
    player_stats = player_stats.fillna(0)
    
    match_perf = pd.read_csv("player_match_stats.csv").fillna(0)
    return player_stats, match_perf

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: FEATURE ENGINEERING & ML LOGIC (Backend)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_batting_stats(deliveries):
    df = deliveries.copy()
    overall = (
        df.groupby("batter")
        .agg(bat_runs=("batsman_runs","sum"), bat_balls=("batsman_runs","count"),
             bat_dismissals=("is_wicket","sum"), bat_matches=("match_id","nunique"),
             bat_fours=("batsman_runs", lambda x:(x==4).sum()),
             bat_sixes=("batsman_runs", lambda x:(x==6).sum()))
        .reset_index()
    )
    overall["boundary_runs"] = overall["bat_fours"]*4 + overall["bat_sixes"]*6
    overall["bat_sr"]  = (overall["bat_runs"]/overall["bat_balls"]*100).round(2)
    overall["bat_avg"] = np.where(overall["bat_dismissals"]>0, overall["bat_runs"]/overall["bat_dismissals"], overall["bat_runs"]).round(2)
    overall["boundary_pct"] = np.where(overall["bat_runs"]>0, overall["boundary_runs"]/overall["bat_runs"]*100, 0.0).round(2)
    pp = df[df["over"]<6].groupby("batter").agg(pp_runs=("batsman_runs","sum"), pp_balls=("batsman_runs","count")).reset_index()
    pp["pp_sr"] = (pp["pp_runs"]/pp_balls["pp_balls"]*100).round(2) if 'pp_balls' in locals() else (pp["pp_runs"]/pp["pp_balls"]*100).round(2)
    death = df[df["over"]>=15].groupby("batter").agg(death_runs=("batsman_runs","sum"), death_balls=("batsman_runs","count")).reset_index()
    death["death_sr"] = (death["death_runs"]/death["death_balls"]*100).round(2)
    bat = overall.merge(pp[["batter","pp_sr"]], on="batter", how="left").merge(death[["batter","death_sr"]], on="batter", how="left")
    bat.rename(columns={"batter":"player"}, inplace=True)
    return bat

def compute_bowling_stats(deliveries):
    df = deliveries.copy()
    # NEW FOOLPROOF FILTER: Only keep normal balls, legbyes, byes, and penalties.
    legal_mask = df["extras_type"].isna() | df["extras_type"].isin(["legbyes", "byes", "penalty", "legbys"])
    legal = df[legal_mask]
    # ... rest of the function remains the same
    bowl = (
        df.groupby("bowler")["total_runs"].sum().reset_index(name="bowl_runs")
        .merge(legal.groupby("bowler")["total_runs"].count().reset_index(name="bowl_balls"), on="bowler", how="left")
        .merge(df[df["is_wicket"]==1].groupby("bowler")["is_wicket"].sum().reset_index(name="bowl_wickets"), on="bowler", how="left")
        .merge(df.groupby("bowler")["match_id"].nunique().reset_index(name="bowl_matches"), on="bowler", how="left")
        .fillna({"bowl_wickets":0})
    )
    bowl["bowl_wickets"] = bowl["bowl_wickets"].astype(int)
    bowl["bowl_economy"] = np.where(bowl["bowl_balls"]>0, bowl["bowl_runs"]/bowl["bowl_balls"]*6, np.nan).round(2)
    bowl["bowl_sr"]      = np.where(bowl["bowl_wickets"]>0, bowl["bowl_balls"]/bowl["bowl_wickets"], np.nan).round(2)
    pp_r = df[df["over"]<6].groupby("bowler")["total_runs"].sum().reset_index(name="pp_bowl_runs")
    pp_b = legal[legal["over"]<6].groupby("bowler")["total_runs"].count().reset_index(name="pp_bowl_balls")
    pp_bowl = pp_r.merge(pp_b, on="bowler", how="left")
    pp_bowl["pp_economy"] = np.where(pp_bowl["pp_bowl_balls"]>0, pp_bowl["pp_bowl_runs"]/pp_bowl["pp_bowl_balls"]*6, np.nan).round(2)
    d_r = df[df["over"]>=15].groupby("bowler")["total_runs"].sum().reset_index(name="d_bowl_runs")
    d_b = legal[legal["over"]>=15].groupby("bowler")["total_runs"].count().reset_index(name="d_bowl_balls")
    death_bowl = d_r.merge(d_b, on="bowler", how="left")
    death_bowl["death_economy"] = np.where(death_bowl["d_bowl_balls"]>0, death_bowl["d_bowl_runs"]/death_bowl["d_bowl_balls"]*6, np.nan).round(2)
    bowl = bowl.merge(pp_bowl[["bowler","pp_economy"]], on="bowler", how="left").merge(death_bowl[["bowler","death_economy"]], on="bowler", how="left")
    bowl.rename(columns={"bowler":"player"}, inplace=True)
    return bowl

def classify_roles(bat_stats, bowl_stats, deliveries):
    df = bat_stats[["player","bat_balls","bat_sr","bat_avg"]].merge(bowl_stats[["player","bowl_balls","bowl_wickets","bowl_economy"]], on="player", how="outer").fillna(0)
    wk_players = set(deliveries[deliveries["dismissal_kind"]=="stumped"]["fielder"].dropna().unique())
    
    def assign_role(row):
        if row["player"] in wk_players and row["bat_balls"] > 0: return "WK-Batter"
        total_activity = row["bat_balls"] + row["bowl_balls"]
        if total_activity == 0: return "Unknown"
        bat_pct = row["bat_balls"] / total_activity
        bowl_pct = row["bowl_balls"] / total_activity
        
        if row["bat_balls"] >= 120 and row["bowl_balls"] >= 120 and bat_pct >= 0.20 and bowl_pct >= 0.20:
            return "All-rounder"
        if bat_pct >= bowl_pct:
            return "Batter"
        else:
            return "Bowler"
        
    df["role"] = df.apply(assign_role, axis=1)
    return df[["player","role"]]

def build_player_stats(deliveries):
    bat = compute_batting_stats(deliveries)
    bowl = compute_bowling_stats(deliveries)
    roles = classify_roles(bat, bowl, deliveries)
    ps = bat.merge(bowl, on="player", how="outer").merge(roles, on="player", how="left")
    ps["role"] = ps["role"].fillna("Unknown")
    ps.set_index("player", inplace=True)
    return ps

def compute_match_performance(deliveries, matches):
    bat = (
        deliveries.groupby(["match_id","batter","batting_team"])
        .agg(runs=("batsman_runs","sum"), balls_faced=("batsman_runs","count"), fours=("batsman_runs",lambda x:(x==4).sum()), sixes=("batsman_runs",lambda x:(x==6).sum()))
        .reset_index().rename(columns={"batter":"player","batting_team":"team"})
    )
    bat["bat_sr"] = (bat["runs"]/bat["balls_faced"]*100).round(1)
    dismissed = deliveries[deliveries["is_wicket"]==1].groupby(["match_id","player_dismissed"]).size().reset_index(name="_d").rename(columns={"player_dismissed":"player"})
    dismissed["dismissed"] = 1
    bat = bat.merge(dismissed[["match_id","player","dismissed"]], on=["match_id","player"], how="left")
    bat["dismissed"] = bat["dismissed"].fillna(0).astype(int)

    legal_mask = deliveries["extras_type"].isna() | deliveries["extras_type"].isin(["legbyes", "byes", "penalty", "legbys"])
    legal = deliveries[legal_mask]
    bowl = (
        legal.groupby(["match_id","bowler","bowling_team"]).size().reset_index(name="legal_balls").rename(columns={"bowler":"player","bowling_team":"team"})
        .merge(deliveries.groupby(["match_id","bowler"])["total_runs"].sum().reset_index(name="runs_conceded").rename(columns={"bowler":"player"}), on=["match_id","player"], how="left")
        .merge(deliveries[deliveries["is_wicket"]==1].groupby(["match_id","bowler"])["is_wicket"].sum().reset_index(name="wickets").rename(columns={"bowler":"player"}), on=["match_id","player"], how="left")
        .fillna({"wickets":0,"runs_conceded":0})
    )
    bowl["wickets"] = bowl["wickets"].astype(int)
    # FIX: Correct cricket overs decimal calculation (e.g., 26 balls -> 4.2)
    # NEW LINE (Shows only completed whole overs):
    bowl["overs_bowl"] = bowl["legal_balls"] // 6
    bowl["economy"] = np.where(bowl["legal_balls"]>0, bowl["runs_conceded"]/(bowl["legal_balls"]/6), np.nan).round(2)

    perf = bat[["match_id","player","team","runs","balls_faced","fours","sixes","bat_sr","dismissed"]].merge(
        bowl[["match_id","player","wickets","runs_conceded","overs_bowl","economy"]], on=["match_id","player"], how="outer")
    bowl_team = bowl[["match_id","player","team"]].rename(columns={"team":"_bt"})
    perf = perf.merge(bowl_team, on=["match_id","player"], how="left")
    perf["team"] = perf["team"].fillna(perf["_bt"]); perf.drop(columns=["_bt"], inplace=True)
    for c in ["runs","balls_faced","fours","sixes","dismissed"]: perf[c] = perf[c].fillna(0).astype(int)
    perf["wickets"] = perf["wickets"].fillna(0).astype(int)

    meta = matches[["id","date","season","venue","team1","team2","winner"]].rename(columns={"id":"match_id"})
    perf = perf.merge(meta, on="match_id", how="left")
    perf["date"] = pd.to_datetime(perf["date"])
    
    def get_opponent(row):
        if pd.isna(row["team"]) or pd.isna(row["team1"]): return "Unknown"
        return row["team2"] if row["team"] == row["team1"] else row["team1"]
    perf["opponent"] = perf.apply(get_opponent, axis=1)
    
    return perf.sort_values(["player","date"], ascending=[True,False])

def compute_team_strength(xi, player_stats):
    # IMPACT PLAYER FIX: Allow 11 or 12 players
    if len(xi) not in [11, 12]: return {"team_batting_score":0, "team_bowling_score":0}
    
    def _fill_player(player, stats, kind):
        defaults = BATTING_DEFAULTS if kind=="batting" else BOWLING_DEFAULTS
        if player not in stats.index: return defaults.copy()
        row = stats.loc[player]
        return {col:(default if (pd.isna(row.get(col,np.nan)) or row.get(col,0)==0) else row.get(col)) for col,default in defaults.items()}

    bat_records = []
    for p in xi:
        d = _fill_player(p, player_stats, "batting")
        d["_q"] = d["bat_sr"] * d["bat_avg"]
        bat_records.append(d)
        
    # Take top 8 batters if 12 players (impact sub), else 7
    num_bat = 8 if len(xi) == 12 else 7
    top_bat = pd.DataFrame(bat_records).sort_values("_q", ascending=False).head(num_bat)
    w_bat = np.arange(num_bat, 0, -1, dtype=float); w_bat /= w_bat.sum()
    batting_score = (w_bat[0]*(top_bat["bat_sr"].values*w_bat).sum()/100 + w_bat[1]*(top_bat["bat_avg"].values*w_bat).sum()/50 + w_bat[2]*(top_bat["pp_sr"].values*w_bat).sum()/100 + w_bat[3]*(top_bat["death_sr"].values*w_bat).sum()/100 + w_bat[4]*(top_bat["boundary_pct"].values*w_bat).sum()/100) * 100
    
    bowl_records = []
    for p in xi:
        d = _fill_player(p, player_stats, "bowling")
        d["_q"] = -(d["bowl_economy"] * (d["bowl_sr"] if not pd.isna(d.get("bowl_sr")) else 30))
        bowl_records.append(d)
        
    # Take top 6 bowlers if 12 players (impact sub), else 5
    num_bowl = 6 if len(xi) == 12 else 5
    top_bowl = pd.DataFrame(bowl_records).sort_values("_q", ascending=False).head(num_bowl)
    w_bowl = np.arange(num_bowl, 0, -1, dtype=float); w_bowl /= w_bowl.sum()
    M = 15.0
    bowling_score = (w_bowl[0]*((M-top_bowl["bowl_economy"].values)*w_bowl).sum()/M + w_bowl[1]*((M-top_bowl["pp_economy"].values)*w_bowl).sum()/M + w_bowl[2]*((M-top_bowl["death_economy"].values)*w_bowl).sum()/M)*100
    
    return {"team_batting_score":round(float(batting_score),4), "team_bowling_score":round(float(bowling_score),4)}

def build_training_data(matches, deliveries, player_stats):
    rows = []
    del_by_match = deliveries.groupby("match_id")
    matches = matches.sort_values("date") # Ensure chronological order for H2H
    
    for idx, match in matches.iterrows():
        mid = match["id"]
        if mid not in del_by_match.groups: continue
        md = del_by_match.get_group(mid)
        t1, t2, winner = match["team1"], match["team2"], match["winner"]
        if pd.isna(winner) or winner not in [t1, t2]: continue
        
        # Determine rosters (Allow up to 12 if IP was active, otherwise 11)
        t1_xi = list(set(md[md["batting_team"]==t1]["batter"].unique())|set(md[md["bowling_team"]==t1]["bowler"].unique()))[:12]
        t2_xi = list(set(md[md["batting_team"]==t2]["batter"].unique())|set(md[md["bowling_team"]==t2]["bowler"].unique()))[:12]
        
        if len(t1_xi)<5 or len(t2_xi)<5: continue
        while len(t1_xi)<11: t1_xi.append("__unknown__")
        while len(t2_xi)<11: t2_xi.append("__unknown__")
        
        # H2H Feature: Last 2 games before this date
        past_matches = matches[(matches["date"] < match["date"]) & 
                               (((matches["team1"] == t1) & (matches["team2"] == t2)) | 
                                ((matches["team1"] == t2) & (matches["team2"] == t1)))]
        last_2 = past_matches.tail(2)
        t1_h2h_wins = len(last_2[last_2["winner"] == t1])
        
        try:
            s1 = compute_team_strength(t1_xi, player_stats)
            s2 = compute_team_strength(t2_xi, player_stats)
            rows.append({
                "match_id": mid, "venue": match["venue"] if not pd.isna(match["venue"]) else "Unknown",
                "toss_winner_is_t1": int(match["toss_winner"] == t1), 
                "toss_decision": match["toss_decision"] if not pd.isna(match["toss_decision"]) else "field",
                "t1_batting_score": s1["team_batting_score"], "t1_bowling_score": s1["team_bowling_score"],
                "t2_batting_score": s2["team_batting_score"], "t2_bowling_score": s2["team_bowling_score"],
                "t1_h2h_wins": t1_h2h_wins, # NEW H2H FEATURE
                "winner": int(winner==t1)
            })
        except: continue
    return pd.DataFrame(rows)

def train_and_save_model(df: pd.DataFrame):
    CATEGORICAL_FEATURES = ["venue", "toss_decision"]
    NUMERIC_FEATURES = ["toss_winner_is_t1", "t1_batting_score", "t1_bowling_score", "t2_batting_score", "t2_bowling_score", "t1_h2h_wins"]
    df = df.dropna(subset=CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["winner"])
    X, y = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES], df["winner"]
    
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    classifier = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3: FRONTEND UI & DASHBOARDS
# ═══════════════════════════════════════════════════════════════════════════════
def role_badge(label, color):
    return f'<span style="background:{color};color:#fff;padding:4px 12px;border-radius:20px;font-size:0.9rem;font-weight:700;letter-spacing:0.5px;box-shadow:0 0 8px {color}80;">{label}</span>'

def section_overview(matches, deliveries):
    st.markdown("##  Tournament Overview")
    st.markdown("Dive into the historical data of the world's premier T20 league.")
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches", f"{len(matches):,}")
    c2.metric("Seasons Played", matches["season"].nunique())
    
    total_runs = int(deliveries["total_runs"].sum())
    total_sixes = deliveries[deliveries["batsman_runs"] == 6].shape[0]
    c3.metric("Total Runs Scored", f"{total_runs:,}")
    c4.metric("Total Sixes Hit", f"{total_sixes:,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Matches per Season")
    
    season_counts = matches["season"].value_counts().sort_index().reset_index()
    season_counts.columns = ["SEASON", "MATCHES"]
    
    fig = px.bar(
        season_counts, 
        x="SEASON", 
        y="MATCHES", 
        text="MATCHES",
        color="MATCHES", 
        color_continuous_scale="Plasma"
    )
    fig.update_xaxes(type='category', title_text="IPL Season") 
    fig.update_yaxes(title_text="Total Matches")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

def section_team_analytics(matches, deliveries):
    st.markdown("##  Team Analytics")
    st.markdown("Select a franchise to view their all-time win rates and legendary performers.")
    st.markdown("---")
    
    teams = sorted(set(matches["team1"].dropna().unique()) | set(matches["team2"].dropna().unique()))
    team = st.selectbox("Select Franchise", teams)

    team_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)]
    wins = len(team_matches[team_matches["winner"] == team])
    total = len(team_matches)
    win_pct = (wins / total * 100) if total > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Matches Played", f"{total:,}")
    c2.metric("Total Wins", f"{wins:,}")
    c3.metric("Win Percentage", f"{win_pct:.1f}%")

    st.markdown("---")
    st.subheader(f"Top Performers for {team}")
    
    team_del = deliveries[deliveries["batting_team"] == team]
    top_batters = team_del.groupby("batter")["batsman_runs"].sum().sort_values(ascending=False).head(5).reset_index()
    top_batters.columns = ["PLAYER", "TOTAL RUNS"]
    
    team_bowl = deliveries[deliveries["bowling_team"] == team]
    top_bowlers = team_bowl[team_bowl["is_wicket"] == 1].groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(5).reset_index()
    top_bowlers.columns = ["PLAYER", "TOTAL WICKETS"]

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Highest Run Scorers**")
        fig_bat = px.bar(top_batters, x="TOTAL RUNS", y="PLAYER", orientation='h', color="TOTAL RUNS", color_continuous_scale="agsunset", text="TOTAL RUNS")
        fig_bat.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_bat, use_container_width=True)
        st.dataframe(top_batters, use_container_width=True, hide_index=True)
        
    with colB:
        st.markdown("**Highest Wicket Takers**")
        fig_bowl = px.bar(top_bowlers, x="TOTAL WICKETS", y="PLAYER", orientation='h', color="TOTAL WICKETS", color_continuous_scale="oryel", text="TOTAL WICKETS")
        fig_bowl.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_bowl, use_container_width=True)
        st.dataframe(top_bowlers, use_container_width=True, hide_index=True)

def section_player_analytics(deliveries, player_stats, match_perf):
    st.markdown("##  Player Analytics")
    st.markdown("Analyze career statistics, match-by-match breakdowns, and recent form for any player.")
    st.markdown("---")
    
    if player_stats.empty or match_perf.empty:
        st.info("No data available to display.")
        return
        
    all_known_players = deliveries["batter"].dropna().tolist() + deliveries["bowler"].dropna().tolist()
    if SQUADS_PATH.exists():
        sq_df = pd.read_csv(SQUADS_PATH)
        if "Player" in sq_df.columns:
            all_known_players += sq_df["Player"].dropna().tolist()
            
    player = st.selectbox("Search for a Player", sorted(set(all_known_players)))
    role = player_stats.loc[player, "role"] if player in player_stats.index else "Unknown"
    
    st.markdown(f"**Primary Role:** {role_badge(role, PLAYER_ROLE_COLORS.get(role, '#888'))}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([" Career Stats", " Match-by-Match", " Recent Form Analyzer"])
    with tab1:
        c1, c2 = st.columns(2)
        r = player_stats.loc[player] if player in player_stats.index else {}
        with c1:
            st.subheader(" Batting Metrics")
            st.metric("Total Runs", f"{int(r.get('bat_runs', 0)):,}")
            st.metric("Strike Rate", f"{r.get('bat_sr', 0):.1f}")
            st.metric("Average", f"{r.get('bat_avg', 0):.1f}")
        with c2:
            st.subheader(" Bowling Metrics")
            st.metric("Wickets Taken", f"{int(r.get('bowl_wickets', 0)):,}" if not pd.isna(r.get("bowl_wickets", np.nan)) else 0)
            st.metric("Career Economy", f"{r.get('bowl_economy', 0):.2f}" if not pd.isna(r.get("bowl_economy", np.nan)) else "N/A")

    with tab2:
        pm = match_perf[match_perf["player"] == player].copy()
        if pm.empty:
            st.info("No historical match data available for this player (likely a new debutant).")
        else:
            st.markdown("####  Batting Appearances")
            bat_df = pm[pm["balls_faced"] > 0][["date","opponent","venue","runs","balls_faced","fours","sixes","bat_sr"]].sort_values("date", ascending=False)
            bat_df["date"] = pd.to_datetime(bat_df["date"]).dt.strftime("%d %b %Y")
            bat_df.columns = [col.replace("_", " ").upper() for col in bat_df.columns]
            st.dataframe(bat_df, use_container_width=True, hide_index=True)
            
            st.markdown("####  Bowling Appearances")
            bowl_df = pm[pm["overs_bowl"] > 0][["date","opponent","venue","overs_bowl","runs_conceded","wickets","economy"]].sort_values("date", ascending=False)
            bowl_df["date"] = pd.to_datetime(bowl_df["date"]).dt.strftime("%d %b %Y")
            bowl_df.columns = [col.replace("_", " ").upper() for col in bowl_df.columns]
            st.dataframe(bowl_df, use_container_width=True, hide_index=True)

    with tab3:
        pm = match_perf[match_perf["player"] == player].sort_values("date", ascending=False)
        if pm.empty:
            st.info("No recent form data available.")
        else:
            max_val = max(5, len(pm))
            num_matches = st.slider("Select number of recent matches to analyze:", min_value=1, max_value=max_val, value=min(10, max_val))
            
            recent = pm.head(num_matches)
            st.markdown("<br>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Runs (Last {num_matches})", int(recent['runs'].sum()))
            c2.metric("Avg SR", f"{recent['bat_sr'].mean():.1f}")
            c3.metric(f"Wickets (Last {num_matches})", int(recent['wickets'].sum()))
            c4.metric("Avg Economy", f"{recent[recent['overs_bowl']>0]['economy'].mean():.2f}" if (recent['overs_bowl']>0).any() else "N/A")
            
            st.markdown("<br>", unsafe_allow_html=True)
            recent_df = recent[["date", "opponent", "runs", "bat_sr", "wickets", "economy"]].copy()
            recent_df["date"] = pd.to_datetime(recent_df["date"]).dt.strftime("%d %b %Y")
            recent_df.columns = [col.replace("_", " ").upper() for col in recent_df.columns]
            st.dataframe(recent_df, use_container_width=True, hide_index=True)

def section_match_role_analysis(matches, deliveries, match_perf):
    st.markdown("##  Match Role Analysis")
    st.markdown("Analyze how specific players performed in a given historical match.")
    st.markdown("---")
    
    if match_perf.empty:
        st.info("No data available to display.")
        return
        
    c1, c2 = st.columns(2)
    with c1: season = st.selectbox("Select Season", sorted(matches["season"].unique(), reverse=True))
    season_matches = matches[matches["season"] == season].sort_values("date")
    match_labels = season_matches.apply(lambda r: f"{r['date'].strftime('%d %b')} | {r['team1']} vs {r['team2']}", axis=1).tolist()
    with c2: chosen_idx = st.selectbox("Select Match", range(len(match_labels)), format_func=lambda i: match_labels[i])
    
    selected_match = season_matches.iloc[chosen_idx]
    mid = selected_match["id"]
    
    # Calculate Team Totals
    match_dels = deliveries[deliveries["match_id"] == mid]
    team_scores = match_dels.groupby("batting_team").agg(
        runs=("total_runs", "sum"), 
        wickets=("is_wicket", "sum")
    ).to_dict('index')
    
    st.markdown(f"### 🏆 Match Winner: **{selected_match['winner']}**")
    score_cols = st.columns(len(team_scores))
    for i, (team, stats) in enumerate(team_scores.items()):
        score_cols[i].metric(f"{team}", f"{stats['runs']}/{stats['wickets']}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    pm = match_perf[match_perf["match_id"] == mid].copy()
    
    st.markdown("####  Batting Scorecard")
    bat_display = pm[pm["balls_faced"] > 0][["player","team","runs","balls_faced","fours","sixes","bat_sr"]]
    bat_display.columns = ["PLAYER", "TEAM", "RUNS", "BALLS", "4S", "6S", "STRIKE RATE"]
    st.dataframe(bat_display.sort_values(["TEAM", "RUNS"], ascending=[True, False]).reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("####  Bowling Scorecard")
    bowl_display = pm[pm["overs_bowl"] > 0][["player","team","overs_bowl","runs_conceded","wickets","economy"]]
    bowl_display.columns = ["PLAYER", "TEAM", "OVERS", "RUNS", "WICKETS", "ECONOMY"]
    st.dataframe(bowl_display.sort_values(["TEAM", "WICKETS"], ascending=[True, False]).reset_index(drop=True), use_container_width=True, hide_index=True)

def section_match_predictor(matches, deliveries, player_stats, model):
    st.markdown("##  Advanced Match Predictor")
    st.markdown("Use our Machine Learning model to simulate a match outcome based on customized Playing XIs and venue conditions.")
    st.markdown("---")
    
    if model is None:
        st.info("Model initializing. Please ensure backend setup is complete.")
        return

    squads_df = pd.read_csv(SQUADS_PATH) if SQUADS_PATH.exists() else pd.DataFrame()
    teams = sorted(set(matches["team1"].dropna().unique()) | set(matches["team2"].dropna().unique()))
    
    # Generate the list of seasons available in the dataset (Removed "All-Time 11")
    available_seasons = sorted(matches["season"].dropna().unique().tolist(), reverse=True)
    season_options = ["2026", "2025"] + available_seasons
    season_options = sorted(list(set(season_options)), reverse=True)

    def get_team_squad_and_top_n(team_name, season_choice, max_players):
        if season_choice in ["2025", "2026"]:
            if not squads_df.empty and "Team" in squads_df.columns and team_name in squads_df["Team"].values:
                squad = squads_df[squads_df["Team"] == team_name]["Player"].dropna().unique().tolist()
                return squad, squad[:max_players]
            return [], []
            
        season_matches = matches[matches["season"] == season_choice]["id"]
        season_dels = deliveries[deliveries["match_id"].isin(season_matches)]
            
        batters = season_dels[season_dels["batting_team"] == team_name]["batter"]
        bowlers = season_dels[season_dels["bowling_team"] == team_name]["bowler"]
        all_team_players = pd.concat([batters, bowlers]).dropna()
        
        squad = sorted(all_team_players.unique().tolist())
        top_n = all_team_players.value_counts().head(max_players).index.tolist() if not all_team_players.empty else []
        return squad, top_n

    with st.expander("⚙️ 1. Match Settings & Teams", expanded=True):
        match_season = st.selectbox("📅 Match Season", season_options)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1: franchise_a = st.selectbox("🔵 Select Team A", teams, index=0 if len(teams)>0 else 0)
        with col_t2: franchise_b = st.selectbox("🔴 Select Team B", teams, index=1 if len(teams)>1 else 0)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1: venue = st.selectbox("Venue", sorted(matches["venue"].dropna().unique()))
        with c2: toss_winner = st.radio("Toss Winner", ["Team A", "Team B"], horizontal=True)
        with c3: toss_decision = st.radio("Toss Decision", ["Bat First", "Bowl First"], horizontal=True)

    # If season is 2023 or newer, allow 12 players (Impact Player rule)
    allow_impact_player = match_season >= "2023"
    max_p = 12 if allow_impact_player else 11

    squad_a, default_a = get_team_squad_and_top_n(franchise_a, match_season, max_p)
    squad_b, default_b = get_team_squad_and_top_n(franchise_b, match_season, max_p)

    st.markdown(f"### 2. Confirm Playing  (Max {max_p} players)")
    if allow_impact_player:
        st.info("💡 Impact Player rule is active for this season. You may select up to 12 players per team.")
    
    if not squad_a:
        st.warning(f"⚠️ No squad data found for {franchise_a} in {match_season}.")
    if not squad_b:
        st.warning(f"⚠️ No squad data found for {franchise_b} in {match_season}.")

    ca, cb = st.columns(2)
    with ca: xi_a = st.multiselect(f"🔵 {franchise_a}", squad_a, default=default_a, max_selections=max_p)
    with cb: xi_b = st.multiselect(f"🔴 {franchise_b}", squad_b, default=default_b, max_selections=max_p)

    st.markdown("<br>", unsafe_allow_html=True)
    if len(xi_a) in [11, 12] and len(xi_b) in [11, 12] and not set(xi_a) & set(xi_b):
        if st.button("⚡ Run ML Simulation", type="primary", use_container_width=True):
            
            str_a = compute_team_strength(xi_a, player_stats)
            str_b = compute_team_strength(xi_b, player_stats)
            
            toss_format = "bat" if toss_decision == "Bat First" else "field"
            
            # Fetch Last 2 H2H matches for the ML Pipeline
            past_matches = matches[(((matches["team1"] == franchise_a) & (matches["team2"] == franchise_b)) |
                                    ((matches["team1"] == franchise_b) & (matches["team2"] == franchise_a)))]
            last_2 = past_matches.sort_values("date").tail(2)
            t1_h2h_wins = len(last_2[last_2["winner"] == franchise_a])
            
            inp = pd.DataFrame([{
                "venue": venue, 
                "toss_decision": toss_format, 
                "toss_winner_is_t1": int(toss_winner == "Team A"),
                "t1_batting_score": str_a["team_batting_score"], 
                "t1_bowling_score": str_a["team_bowling_score"],
                "t2_batting_score": str_b["team_batting_score"], 
                "t2_bowling_score": str_b["team_bowling_score"],
                "t1_h2h_wins": t1_h2h_wins # Included H2H
            }])
            
            probs = model.predict_proba(inp)[0]
            
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>🏆 Simulation Results</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 0.2, 1])
            with col1:
                st.info(f"**🔵 {franchise_a}**\n\nWin Probability: **{probs[1]*100:.1f}%**")
            with col3:
                st.error(f"**🔴 {franchise_b}**\n\nWin Probability: **{probs[0]*100:.1f}%**")
            
            
            
    elif set(xi_a) & set(xi_b):
        st.warning("⚠️ A player cannot play for both teams at the same time! Please remove duplicate players.")
    else:
        st.info(f"ℹ️ Please ensure both teams have either 11 or 12 players selected to run the prediction.")

def section_system_setup(matches, deliveries):
    st.header("⚙️ System Setup & Backend")
    st.markdown("Use this section to regenerate data features or retrain the Machine Learning model if you have added new data to your CSV files.")
    
    st.subheader("Step 1: Feature Engineering")
    st.markdown("Calculates historical player stats, roles, and match performances.")
    if st.button("🛠️ Run Data Processing"):
        with st.spinner("Processing data... this might take a minute."):
            st.cache_data.clear()
            ps = build_player_stats(deliveries)
            ps.to_csv(OUTPUT_PLAYER_STATS)
            mp = compute_match_performance(deliveries, matches)
            mp.to_csv(OUTPUT_MATCH_STATS, index=False)
            td = build_training_data(matches, deliveries, ps)
            td.to_csv(OUTPUT_TRAINING, index=False)
            st.success("✅ Feature Engineering Complete! Player Stats and Training Data updated.")

    st.subheader("Step 2: Train Machine Learning Model")
    st.markdown("Trains the Random Forest model on the processed data.")
    if st.button("🧠 Train Model"):
        if not OUTPUT_TRAINING.exists():
            st.error("Training data not found. Please run Step 1 first.")
        else:
            with st.spinner("Training Random Forest model..."):
                td = pd.read_csv(OUTPUT_TRAINING)
                train_and_save_model(td)
                st.cache_resource.clear() 
                st.success("✅ Model Trained and Saved successfully!")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    deliveries, matches = load_raw_data()
    player_stats, match_perf = load_processed_data()
    model = load_model()

    st.sidebar.markdown("<h2>🏏 IPL Dashboard</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    section = st.sidebar.radio("", [
        "📊 Overview", 
        "📈 Team Analytics", 
        "🏏 Player Analytics", 
        "🎭 Match Role Analysis", 
        "🔮 Match Predictor"
         ,#"⚙️ System Setup"
    ])
    
    st.sidebar.markdown("---")
    
    if section == "📊 Overview": section_overview(matches, deliveries)
    elif section == "📈 Team Analytics": section_team_analytics(matches, deliveries)
    elif section == "🏏 Player Analytics": section_player_analytics(deliveries, player_stats, match_perf)
    elif section == "🎭 Match Role Analysis": section_match_role_analysis(matches, deliveries, match_perf)
    elif section == "🔮 Match Predictor": section_match_predictor(matches, deliveries, player_stats, model)
    elif section == "⚙️ System Setup": section_system_setup(matches, deliveries) 

if __name__ == "__main__":
    main()