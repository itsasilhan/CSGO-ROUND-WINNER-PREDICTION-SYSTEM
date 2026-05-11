
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.ensemble        import RandomForestClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

st.set_page_config(page_title="CS:GO Round Predictor", layout="wide")

st.markdown("""
<style>
.sec  { font-size:13px; font-weight:700; color:#888;
        text-transform:uppercase; letter-spacing:1px;
        margin-top:18px; margin-bottom:4px; }
.pill { display:flex; justify-content:space-between; align-items:center;
        padding:10px 16px; border-radius:10px; margin:5px 0;
        font-size:16px; font-weight:600; }
.row  { display:flex; justify-content:space-between; font-size:14px;
        padding:5px 0; border-bottom:1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ── Locate CSV ────────────────────────────────────────────────────────────────
CSV_FILE = "csgo_round_snapshots.csv"

def find_csv():
    checks = [
        CSV_FILE,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILE),
        os.path.join(os.path.expanduser("~"), CSV_FILE),
        os.path.join(os.path.expanduser("~"), "Downloads", CSV_FILE),
        os.path.join(os.path.expanduser("~"), "csgo_data", CSV_FILE),
    ]
    for p in checks:
        if os.path.exists(p):
            return p
    return None

# ── Train model once, keep in memory ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)

    df["ct_health"] = df["ct_health"].clip(0, 500)
    df["t_health"]  = df["t_health"].clip(0, 500)
    df["ct_money"]  = df["ct_money"].clip(0, 16_000)
    df["t_money"]   = df["t_money"].clip(0, 16_000)
    df["ct_players_alive"] = df["ct_players_alive"].clip(0, 5)
    df["t_players_alive"]  = df["t_players_alive"].clip(0, 5)

    nc = {}
    nc["health_advantage"]  = df["ct_health"]  - df["t_health"]
    nc["money_advantage"]   = df["ct_money"]   - df["t_money"]
    nc["players_advantage"] = df["ct_players_alive"] - df["t_players_alive"]
    nc["armor_advantage"]   = df["ct_armor"]   - df["t_armor"]
    nc["awp_advantage"]     = df["ct_weapon_awp"] - df["t_weapon_awp"]

    ct_r = ["ct_weapon_ak47","ct_weapon_m4a4","ct_weapon_m4a1s",
            "ct_weapon_aug","ct_weapon_famas","ct_weapon_galilar","ct_weapon_sg553"]
    t_r  = ["t_weapon_ak47","t_weapon_m4a4","t_weapon_m4a1s",
            "t_weapon_aug","t_weapon_famas","t_weapon_galilar","t_weapon_sg553"]
    ct_g = [c for c in df.columns if c.startswith("ct_grenade")]
    t_g  = [c for c in df.columns if c.startswith("t_grenade")]

    nc["ct_total_rifles"]   = df[ct_r].sum(axis=1)
    nc["t_total_rifles"]    = df[t_r].sum(axis=1)
    nc["rifle_advantage"]   = nc["ct_total_rifles"]   - nc["t_total_rifles"]
    nc["ct_total_grenades"] = df[ct_g].sum(axis=1)
    nc["t_total_grenades"]  = df[t_g].sum(axis=1)
    nc["grenade_advantage"] = nc["ct_total_grenades"] - nc["t_total_grenades"]
    nc["late_round"]        = (df["time_left"] < 45).astype(int)

    le_map = LabelEncoder()
    nc["map_encoded"]  = le_map.fit_transform(df["map"])
    nc["bomb_encoded"] = df["bomb_planted"].astype(int)

    df = pd.concat([df, pd.DataFrame(nc, index=df.index)], axis=1)
    df["winner_encoded"] = (df["round_winner"] == "T").astype(int)

    drop = ["map", "bomb_planted", "round_winner", "winner_encoded"]
    feature_cols = [c for c in df.columns if c not in drop]

    X = df[feature_cols]
    y = df["winner_encoded"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=3, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    return model, feature_cols, le_map


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, feature_cols, le_map, p):
    row = pd.Series(0.0, index=feature_cols)

    row["time_left"]        = float(p["time_left"])
    row["ct_score"]         = float(p["ct_score"])
    row["t_score"]          = float(p["t_score"])
    row["ct_health"]        = float(p["ct_health"])
    row["t_health"]         = float(p["t_health"])
    row["ct_armor"]         = float(p["ct_armor"])
    row["t_armor"]          = float(p["t_armor"])
    row["ct_money"]         = float(p["ct_money"])
    row["t_money"]          = float(p["t_money"])
    row["ct_helmets"]       = float(p["ct_helmets"])
    row["t_helmets"]        = float(p["t_helmets"])
    row["ct_defuse_kits"]   = float(p["ct_kits"])
    row["ct_players_alive"] = float(p["ct_players"])
    row["t_players_alive"]  = float(p["t_players"])
    row["ct_weapon_awp"]    = float(p["ct_awp"])
    row["t_weapon_awp"]     = float(p["t_awp"])

    if p["map_name"] in le_map.classes_:
        row["map_encoded"] = float(le_map.transform([p["map_name"]])[0])
    row["bomb_encoded"] = 1.0 if p["bomb_planted"] else 0.0

    row["health_advantage"]  = row["ct_health"]        - row["t_health"]
    row["money_advantage"]   = row["ct_money"]         - row["t_money"]
    row["players_advantage"] = row["ct_players_alive"] - row["t_players_alive"]
    row["armor_advantage"]   = row["ct_armor"]         - row["t_armor"]
    row["awp_advantage"]     = row["ct_weapon_awp"]    - row["t_weapon_awp"]
    row["rifle_advantage"]   = float(p["ct_rifles"]    - p["t_rifles"])
    row["ct_total_rifles"]   = float(p["ct_rifles"])
    row["t_total_rifles"]    = float(p["t_rifles"])
    row["grenade_advantage"] = float(p["ct_grens"]     - p["t_grens"])
    row["ct_total_grenades"] = float(p["ct_grens"])
    row["t_total_grenades"]  = float(p["t_grens"])
    row["late_round"]        = float(p["time_left"] < 45)

    X_in = pd.DataFrame([row[feature_cols]])
    prob = model.predict_proba(X_in)[0]
    pred = model.predict(X_in)[0]
    return ("T" if pred == 1 else "CT"), round(prob[0]*100, 1), round(prob[1]*100, 1)


# ── Load model ────────────────────────────────────────────────────────────────
st.title("CS:GO Round Winner Predictor")
st.markdown("Enter the current round state and click **Predict Winner**.")

csv_path = find_csv()
if csv_path is None:
    st.error(
        f"`{CSV_FILE}` not found. "
        "Put csgo_round_snapshots.csv in the same folder as this script, then reload."
    )
    st.stop()

with st.spinner("Training model on first launch - about 30 seconds..."):
    model, feature_cols, le_map = load_model(csv_path)

st.success("Model ready!")
st.markdown("---")

# ── Inputs ───────────────────────────────────────────────
st.markdown('<p class="sec">General</p>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
map_name     = c1.selectbox("Map", ["de_dust2","de_mirage","de_inferno",
                                     "de_nuke","de_overpass","de_train",
                                     "de_vertigo","de_cache"])
time_left    = c2.number_input("Time Left (s)",  0, 175,  90, 1)
bomb_planted = c3.selectbox("Bomb Planted?", ["No", "Yes"])
ct_score     = c4.number_input("CT Score",        0,  30,   8, 1)
t_score      = c5.number_input("T Score",         0,  30,   7, 1)

st.markdown("---")
st.markdown('<p class="sec">Players and Health</p>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)
ct_players = c1.number_input("CT Players Alive", 0,   5,   4, 1)
t_players  = c2.number_input("T Players Alive",  0,   5,   4, 1)
ct_health  = c3.number_input("CT Total Health",  0, 500, 350, 5)
t_health   = c4.number_input("T Total Health",   0, 500, 300, 5)
ct_armor   = c5.number_input("CT Total Armor",   0, 500, 300, 5)
t_armor    = c6.number_input("T Total Armor",    0, 500, 250, 5)

st.markdown("---")
st.markdown('<p class="sec">Economy</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
ct_money   = c1.number_input("CT Money ($)",  0, 16000, 8000, 50)
t_money    = c2.number_input("T Money ($)",   0, 16000, 5000, 50)
ct_helmets = c3.number_input("CT Helmets",    0,     5,    3,  1)
t_helmets  = c4.number_input("T Helmets",     0,     5,    2,  1)

st.markdown("---")
st.markdown('<p class="sec">Weapons and Utilities</p>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
ct_awp   = c1.number_input("CT AWPs",       0,  5, 1, 1)
t_awp    = c2.number_input("T AWPs",        0,  5, 0, 1)
ct_rifles= c3.number_input("CT Rifles",      0,  5, 3, 1)
t_rifles = c4.number_input("T Rifles",       0,  5, 2, 1)
ct_kits  = c5.number_input("CT Defuse Kits", 0,  5, 2, 1)
ct_grens = c6.number_input("CT Grenades",    0, 20, 3, 1)
t_grens  = c7.number_input("T Grenades",     0, 20, 2, 1)

st.markdown("---")
_, btn_col, _ = st.columns([2, 1, 2])
predict_clicked = btn_col.button(
    "Predict Winner",
    use_container_width=True,
    type="primary",
)

# ── Validation + Output ───────────────────────────────────────────────────────
if predict_clicked:

    errors = []

    if bomb_planted == "Yes" and time_left > 40:
        errors.append(
            f"Bomb is planted — timer cannot exceed 40s. You entered {time_left}s."
        )

    if ct_players == 0 and (ct_health > 0 or ct_armor > 0):
        errors.append(
            "CT has 0 players alive — health and armor must be 0."
        )

    if t_players == 0 and (t_health > 0 or t_armor > 0):
        errors.append(
            "T has 0 players alive — health and armor must be 0."
        )

    if ct_players > 0 and ct_health > ct_players * 100:
        errors.append(
            f"CT health cannot exceed {ct_players * 100} "
            f"({ct_players} players x 100 HP max). You entered {int(ct_health)}."
        )

    if t_players > 0 and t_health > t_players * 100:
        errors.append(
            f"T health cannot exceed {t_players * 100} "
            f"({t_players} players x 100 HP max). You entered {int(t_health)}."
        )

    if ct_players > 0 and ct_armor > ct_players * 100:
        errors.append(
            f"CT armor cannot exceed {ct_players * 100} "
            f"({ct_players} players x 100 armor max). You entered {int(ct_armor)}."
        )

    if t_players > 0 and t_armor > t_players * 100:
        errors.append(
            f"T armor cannot exceed {t_players * 100} "
            f"({t_players} players x 100 armor max). You entered {int(t_armor)}."
        )

    if ct_helmets > ct_players:
        errors.append(
            f"CT helmets cannot exceed CT players alive ({ct_players}). "
            f"You entered {ct_helmets}."
        )

    if t_helmets > t_players:
        errors.append(
            f"T helmets cannot exceed T players alive ({t_players}). "
            f"You entered {t_helmets}."
        )

    if ct_kits > ct_players:
        errors.append(
            f"CT defuse kits cannot exceed CT players alive ({ct_players}). "
            f"You entered {ct_kits}."
        )

    if ct_awp + ct_rifles > ct_players:
        errors.append(
            f"CT weapons (AWPs + rifles = {ct_awp + ct_rifles}) cannot exceed "
            f"CT players alive ({ct_players})."
        )

    if t_awp + t_rifles > t_players:
        errors.append(
            f"T weapons (AWPs + rifles = {t_awp + t_rifles}) cannot exceed "
            f"T players alive ({t_players})."
        )

    if errors:
        st.markdown("---")
        st.markdown("### Input errors")
        for msg in errors:
            st.error(msg)

    else:
        params = dict(
            map_name=map_name,
            time_left=time_left,
            bomb_planted=(bomb_planted == "Yes"),
            ct_score=ct_score,     t_score=t_score,
            ct_players=ct_players, t_players=t_players,
            ct_health=ct_health,   t_health=t_health,
            ct_armor=ct_armor,     t_armor=t_armor,
            ct_money=ct_money,     t_money=t_money,
            ct_helmets=ct_helmets, t_helmets=t_helmets,
            ct_kits=ct_kits,
            ct_awp=ct_awp,         t_awp=t_awp,
            ct_rifles=ct_rifles,   t_rifles=t_rifles,
            ct_grens=ct_grens,     t_grens=t_grens,
        )

        winner, ct_pct, t_pct = predict(model, feature_cols, le_map, params)

        CT  = "#1d6fa5"
        T   = "#c45c1a"
        WIN = CT if winner == "CT" else T

        st.markdown("---")
        st.markdown("## Result")

        left, mid, right = st.columns([1.2, 1, 1.2])

        with left:
            st.markdown(f"""
            <div style="border-radius:14px; padding:30px 20px; text-align:center;
                        background:{WIN}18; border:2px solid {WIN}">
                <div style="font-size:56px; font-weight:900;
                            color:{WIN}; letter-spacing:3px">{winner}</div>
                <div style="font-size:14px; color:#888; margin-top:6px">
                    Predicted Round Winner
                </div>
                <div style="font-size:13px; color:#666; margin-top:10px">
                    <b>{map_name}</b> &nbsp;·&nbsp;
                    <b>{time_left}s left</b> &nbsp;·&nbsp;
                    Bomb: <b>{"Yes" if bomb_planted == "Yes" else "No"}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with mid:
            st.markdown("**Win Probabilities**")
            st.markdown(
                f'<div class="pill" style="background:{CT}18">'
                f'<span>🔵 CT</span><span style="color:{CT}">{ct_pct}%</span></div>',
                unsafe_allow_html=True)
            st.progress(ct_pct / 100)
            st.markdown(
                f'<div class="pill" style="background:{T}18">'
                f'<span>🟠 T</span><span style="color:{T}">{t_pct}%</span></div>',
                unsafe_allow_html=True)
            st.progress(t_pct / 100)

            conf = max(ct_pct, t_pct)
            if conf >= 80:
                label, clr = "High confidence", "#2e7d32"
            elif conf >= 65:
                label, clr = "Moderate confidence", "#f57f17"
            else:
                label, clr = "Low confidence (close round)", "#c62828"

            st.markdown(
                f"<div style='margin-top:8px;font-size:13px;font-weight:600;"
                f"color:{clr}'>● {label}</div>",
                unsafe_allow_html=True)

        with right:
            st.markdown("**Advantage Breakdown (CT - T)**")
            items = [
                ("Players alive", ct_players - t_players),
                ("Health",        ct_health  - t_health),
                ("Money ($)",     ct_money   - t_money),
                ("Armor",         ct_armor   - t_armor),
                ("AWPs",          ct_awp     - t_awp),
                ("Rifles",        ct_rifles  - t_rifles),
                ("Grenades",      ct_grens   - t_grens),
            ]
            for lbl, val in items:
                clr = CT if val > 0 else T if val < 0 else "#888"
                txt = f"▲ {abs(val)}" if val > 0 else f"▼ {abs(val)}" if val < 0 else "Even"
                st.markdown(
                    f'<div class="row"><span style="color:#bbb">{lbl}</span>'
                    f'<span style="color:{clr};font-weight:700">{txt}</span></div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        chart_col, summary_col = st.columns([2, 1])

        with chart_col:
            st.markdown("**Visual Advantage Breakdown**")
            labels = ["Players", "Health", "Money\n(div100)", "Armor", "AWPs", "Rifles", "Grenades"]
            vals   = [
                ct_players - t_players,
                ct_health  - t_health,
                (ct_money  - t_money) / 100,
                ct_armor   - t_armor,
                ct_awp     - t_awp,
                ct_rifles  - t_rifles,
                ct_grens   - t_grens,
            ]
            colors = [CT if v > 0 else T if v < 0 else "#555" for v in vals]

            LABEL_CLR = "#e0e0e0"
            TICK_CLR  = "#aaaaaa"
            GRID_CLR  = "#444444"

            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.55)
            ax.axhline(0, color=GRID_CLR, lw=0.8)
            ax.set_ylabel("CT - T", color=LABEL_CLR, fontsize=10)
            ax.set_title("Blue = CT ahead   |   Orange = T ahead",
                         fontsize=10, fontweight="bold", color=LABEL_CLR)
            ax.tick_params(axis="x", colors=TICK_CLR, labelsize=9)
            ax.tick_params(axis="y", colors=TICK_CLR, labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            span = max(abs(v) for v in vals) if any(vals) else 1
            for bar, v in zip(bars, vals):
                if v != 0:
                    y = bar.get_height() + span * (0.04 if v > 0 else -0.12)
                    ax.text(bar.get_x() + bar.get_width() / 2, y,
                            f"{v:.0f}", ha="center", fontsize=9,
                            fontweight="600", color=LABEL_CLR)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with summary_col:
            st.markdown("**Round Summary**")
            rows = {
                "Score":      f"CT {ct_score} - {t_score} T",
                "Time left":  f"{time_left}s {'- Late round' if time_left < 45 else ''}",
                "Bomb":       "Planted" if bomb_planted == "Yes" else "Not planted",
                "CT players": f"{ct_players} / 5",
                "T players":  f"{t_players} / 5",
                "CT money":   f"${ct_money:,}",
                "T money":    f"${t_money:,}",
            }
            for k, v in rows.items():
                st.markdown(
                    f'<div class="row"><span style="color:#888">{k}</span>'
                    f'<b style="color:#e0e0e0">{v}</b></div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "Model: Random Forest · n_estimators=200 · min_samples_leaf=3 · "
            "Test Accuracy 84.95% · AUC-ROC 0.928 · "
            "Dataset: CS:GO Round Snapshots (117,448 rows)"
        )