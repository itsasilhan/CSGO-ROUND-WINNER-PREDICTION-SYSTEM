"""
CS:GO Round Winner Predictor — Streamlit App
CSS 324 - Introduction to Machine Learning - Final Project

HOW TO RUN:
  1. Put csgo_app.py and csgo_round_snapshots.csv in the same folder
  2. pip install streamlit scikit-learn pandas numpy matplotlib Pillow
  3. streamlit run csgo_app.py
"""

import os
import io
import json
import base64
import warnings
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.ensemble        import RandomForestClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = "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"
CSV_FILE = "csgo_round_snapshots.csv"
MAPS     = ["de_dust2","de_mirage","de_inferno","de_nuke",
            "de_overpass","de_train","de_vertigo","de_cache"]
RIFLES   = {"ak47","m4a4","m4a1s","aug","famas","galilar","sg553"}
CT_CLR   = "#1d6fa5"
T_CLR    = "#c45c1a"

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
.palive { background:#1a2a1a; border-left:3px solid #4caf50;
          padding:4px 8px; margin:2px 0; border-radius:4px; font-size:13px; }
.pdead  { background:#1a1a1a; border-left:3px solid #555;
          padding:4px 8px; margin:2px 0; border-radius:4px;
          font-size:13px; color:#555; }
</style>
""", unsafe_allow_html=True)

# ── CSV search ────────────────────────────────────────────────────────────────
def find_csv():
    for p in [CSV_FILE,
              os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILE),
              os.path.join(os.path.expanduser("~"), CSV_FILE),
              os.path.join(os.path.expanduser("~"), "Downloads", CSV_FILE),
              os.path.join(os.path.expanduser("~"), "csgo_data", CSV_FILE)]:
        if os.path.exists(p):
            return p
    return None

# ── Train model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(csv_path):
    df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    for c in ["ct_health","t_health"]:     df[c] = df[c].clip(0, 500)
    for c in ["ct_money","t_money"]:       df[c] = df[c].clip(0, 16000)
    for c in ["ct_players_alive","t_players_alive"]: df[c] = df[c].clip(0, 5)

    nc = {}
    nc["health_advantage"]  = df["ct_health"] - df["t_health"]
    nc["money_advantage"]   = df["ct_money"]  - df["t_money"]
    nc["players_advantage"] = df["ct_players_alive"] - df["t_players_alive"]
    nc["armor_advantage"]   = df["ct_armor"]  - df["t_armor"]
    nc["awp_advantage"]     = df["ct_weapon_awp"] - df["t_weapon_awp"]

    ct_r = ["ct_weapon_ak47","ct_weapon_m4a4","ct_weapon_m4a1s","ct_weapon_aug",
            "ct_weapon_famas","ct_weapon_galilar","ct_weapon_sg553"]
    t_r  = ["t_weapon_ak47","t_weapon_m4a4","t_weapon_m4a1s","t_weapon_aug",
            "t_weapon_famas","t_weapon_galilar","t_weapon_sg553"]
    ct_g = [c for c in df.columns if c.startswith("ct_grenade")]
    t_g  = [c for c in df.columns if c.startswith("t_grenade")]

    nc["ct_total_rifles"]   = df[ct_r].sum(1)
    nc["t_total_rifles"]    = df[t_r].sum(1)
    nc["rifle_advantage"]   = nc["ct_total_rifles"] - nc["t_total_rifles"]
    nc["ct_total_grenades"] = df[ct_g].sum(1)
    nc["t_total_grenades"]  = df[t_g].sum(1)
    nc["grenade_advantage"] = nc["ct_total_grenades"] - nc["t_total_grenades"]
    nc["late_round"]        = (df["time_left"] < 45).astype(int)

    le = LabelEncoder()
    nc["map_encoded"]  = le.fit_transform(df["map"])
    nc["bomb_encoded"] = df["bomb_planted"].astype(int)

    df = pd.concat([df, pd.DataFrame(nc, index=df.index)], axis=1)
    df["winner_encoded"] = (df["round_winner"] == "T").astype(int)

    drop = ["map","bomb_planted","round_winner","winner_encoded"]
    fcols = [c for c in df.columns if c not in drop]
    X, y  = df[fcols], df["winner_encoded"]
    Xt, _, yt, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                n_jobs=-1, random_state=42)
    m.fit(Xt, yt)
    return m, fcols, le

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, fcols, le, p):
    row = pd.Series(0.0, index=fcols)
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
    if p["map_name"] in le.classes_:
        row["map_encoded"] = float(le.transform([p["map_name"]])[0])
    row["bomb_encoded"]      = 1.0 if p["bomb_planted"] else 0.0
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
    prob = model.predict_proba(pd.DataFrame([row[fcols]]))[0]
    pred = model.predict(pd.DataFrame([row[fcols]]))[0]
    return ("T" if pred == 1 else "CT"), round(prob[0]*100,1), round(prob[1]*100,1)

# ── Validate ──────────────────────────────────────────────────────────────────
def validate(p):
    e = []
    if p["bomb_planted"] and p["time_left"] > 40:
        e.append(f"Bomb planted — timer cannot exceed 40s. Got {p['time_left']}s.")
    if p["ct_players"] == 0 and (p["ct_health"] > 0 or p["ct_armor"] > 0):
        e.append("CT has 0 players alive — health and armor must be 0.")
    if p["t_players"] == 0 and (p["t_health"] > 0 or p["t_armor"] > 0):
        e.append("T has 0 players alive — health and armor must be 0.")
    if p["ct_players"] > 0 and p["ct_health"] > p["ct_players"]*100:
        e.append(f"CT health {int(p['ct_health'])} exceeds max {p['ct_players']*100}.")
    if p["t_players"] > 0 and p["t_health"] > p["t_players"]*100:
        e.append(f"T health {int(p['t_health'])} exceeds max {p['t_players']*100}.")
    if p["ct_players"] > 0 and p["ct_armor"] > p["ct_players"]*100:
        e.append(f"CT armor {int(p['ct_armor'])} exceeds max {p['ct_players']*100}.")
    if p["t_players"] > 0 and p["t_armor"] > p["t_players"]*100:
        e.append(f"T armor {int(p['t_armor'])} exceeds max {p['t_players']*100}.")
    if p["ct_helmets"] > p["ct_players"]:
        e.append(f"CT helmets ({p['ct_helmets']}) > CT players ({p['ct_players']}).")
    if p["t_helmets"] > p["t_players"]:
        e.append(f"T helmets ({p['t_helmets']}) > T players ({p['t_players']}).")
    if p["ct_kits"] > p["ct_players"]:
        e.append(f"CT defuse kits ({p['ct_kits']}) > CT players ({p['ct_players']}).")
    if p["ct_awp"] + p["ct_rifles"] > p["ct_players"]:
        e.append(f"CT AWPs+rifles ({p['ct_awp']+p['ct_rifles']}) > CT players ({p['ct_players']}).")
    if p["t_awp"] + p["t_rifles"] > p["t_players"]:
        e.append(f"T AWPs+rifles ({p['t_awp']+p['t_rifles']}) > T players ({p['t_players']}).")
    return e

# ── Show result ───────────────────────────────────────────────────────────────
def show_result(winner, ct_pct, t_pct, p):
    WIN = CT_CLR if winner == "CT" else T_CLR
    st.markdown("---")
    st.markdown("## Result")

    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    with c1:
        st.markdown(f"""
        <div style="border-radius:14px;padding:30px 20px;text-align:center;
                    background:{WIN}18;border:2px solid {WIN}">
            <div style="font-size:56px;font-weight:900;color:{WIN};letter-spacing:3px">{winner}</div>
            <div style="font-size:14px;color:#888;margin-top:6px">Predicted Round Winner</div>
            <div style="font-size:13px;color:#666;margin-top:10px">
                <b>{p['map_name']}</b> &nbsp;·&nbsp; <b>{p['time_left']}s left</b> &nbsp;·&nbsp;
                Bomb: <b>{"Yes" if p["bomb_planted"] else "No"}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("**Win Probabilities**")
        st.markdown(f'<div class="pill" style="background:{CT_CLR}18"><span>CT</span>'
                    f'<span style="color:{CT_CLR}">{ct_pct}%</span></div>', unsafe_allow_html=True)
        st.progress(ct_pct/100)
        st.markdown(f'<div class="pill" style="background:{T_CLR}18"><span>T</span>'
                    f'<span style="color:{T_CLR}">{t_pct}%</span></div>', unsafe_allow_html=True)
        st.progress(t_pct/100)
        conf = max(ct_pct, t_pct)
        lbl, clr = (("High confidence","#2e7d32") if conf>=80
                    else ("Moderate confidence","#f57f17") if conf>=65
                    else ("Low confidence (close round)","#c62828"))
        st.markdown(f"<div style='margin-top:8px;font-size:13px;font-weight:600;color:{clr}'>● {lbl}</div>",
                    unsafe_allow_html=True)

    with c3:
        st.markdown("**Advantage Breakdown (CT - T)**")
        for lbl, val in [("Players alive", p["ct_players"]-p["t_players"]),
                         ("Health",        p["ct_health"] -p["t_health"]),
                         ("Money ($)",     p["ct_money"]  -p["t_money"]),
                         ("Armor",         p["ct_armor"]  -p["t_armor"]),
                         ("AWPs",          p["ct_awp"]    -p["t_awp"]),
                         ("Rifles",        p["ct_rifles"] -p["t_rifles"]),
                         ("Grenades",      p["ct_grens"]  -p["t_grens"])]:
            clr = CT_CLR if val>0 else T_CLR if val<0 else "#888"
            txt = f"▲ {abs(val)}" if val>0 else f"▼ {abs(val)}" if val<0 else "Even"
            st.markdown(f'<div class="row"><span style="color:#bbb">{lbl}</span>'
                        f'<span style="color:{clr};font-weight:700">{txt}</span></div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    ch, sc = st.columns([2,1])
    with ch:
        st.markdown("**Visual Advantage Breakdown**")
        lbls = ["Players","Health","Money\n(div100)","Armor","AWPs","Rifles","Grenades"]
        vals = [p["ct_players"]-p["t_players"], p["ct_health"]-p["t_health"],
                (p["ct_money"]-p["t_money"])/100, p["ct_armor"]-p["t_armor"],
                p["ct_awp"]-p["t_awp"], p["ct_rifles"]-p["t_rifles"],
                p["ct_grens"]-p["t_grens"]]
        cols = [CT_CLR if v>0 else T_CLR if v<0 else "#555" for v in vals]
        L,T,G = "#e0e0e0","#aaaaaa","#444444"
        fig, ax = plt.subplots(figsize=(8,3))
        bars = ax.bar(lbls, vals, color=cols, edgecolor="white", width=0.55)
        ax.axhline(0, color=G, lw=0.8)
        ax.set_ylabel("CT - T", color=L, fontsize=10)
        ax.set_title("Blue = CT ahead   |   Orange = T ahead", fontsize=10,
                     fontweight="bold", color=L)
        ax.tick_params(axis="x", colors=T, labelsize=9)
        ax.tick_params(axis="y", colors=T, labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor(G)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        span = max(abs(v) for v in vals) if any(vals) else 1
        for bar, v in zip(bars, vals):
            if v != 0:
                y = bar.get_height() + span*(0.04 if v>0 else -0.12)
                ax.text(bar.get_x()+bar.get_width()/2, y, f"{v:.0f}",
                        ha="center", fontsize=9, fontweight="600", color=L)
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with sc:
        st.markdown("**Round Summary**")
        for k, v in {"Score": f"CT {p['ct_score']} - {p['t_score']} T",
                     "Time left": f"{p['time_left']}s {'- Late' if p['time_left']<45 else ''}",
                     "Bomb": "Planted" if p["bomb_planted"] else "Not planted",
                     "CT players": f"{p['ct_players']} / 5",
                     "T players":  f"{p['t_players']} / 5",
                     "CT money":   f"${p['ct_money']:,}",
                     "T money":    f"${p['t_money']:,}"}.items():
            st.markdown(f'<div class="row"><span style="color:#888">{k}</span>'
                        f'<b style="color:#e0e0e0">{v}</b></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Model: Random Forest · n_estimators=200 · min_samples_leaf=3 · "
               "Test Accuracy 84.95% · AUC-ROC 0.928 · Dataset: CS:GO Round Snapshots (117,448 rows)")

# ── Vision API ────────────────────────────────────────────────────────────────
def parse_screenshot(image_bytes: bytes):
    # Step 1: compress image to JPEG ≤1280px wide to avoid timeout
    img = Image.open(io.BytesIO(image_bytes))
    if img.width > 1280:
        ratio = 1280 / img.width
        img = img.resize((1280, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "Analyze this CS:GO broadcast screenshot. "
        "Return ONLY valid JSON (no markdown, no extra text) with this exact structure:\n"
        '{"time_left":<seconds as int>,"ct_score":<int>,"t_score":<int>,"bomb_planted":<bool>,'
        '"map_name":"<de_dust2|de_mirage|de_inferno|de_nuke|de_overpass|de_train|de_vertigo|de_cache>",'
        '"ct_players":[{"name":"<str>","alive":<bool>,"hp":<0-100>,"money":<int>,'
        '"weapon":"<ak47|m4a4|m4a1s|awp|mp9|famas|aug|galilar|sg553|pistol|none>",'
        '"has_helmet":<bool>,"has_armor":<bool>,"grenades":<int>,"has_defuse_kit":<bool>}],'
        '"t_players":[{"name":"<str>","alive":<bool>,"hp":<0-100>,"money":<int>,'
        '"weapon":"<ak47|m4a4|m4a1s|awp|mp9|famas|aug|galilar|sg553|pistol|none>",'
        '"has_helmet":<bool>,"has_armor":<bool>,"grenades":<int>}]}\n\n'
        "Rules: CT side = panel with blue background. T side = panel with gold (yellow) background. "
        "Dead players have grey row and skull icon, hp=0. "
        "HP = white number on right of alive row. Money = green $ under name. "
        "Timer M:SS convert to seconds. bomb_planted=true only if bomb timer visible."
    )

    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg",
                                         "data": img_b64}},
            {"type": "text", "text": prompt}
        ]}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": ANTHROPIC_API_KEY,
        }
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            raw  = data["content"][0]["text"].strip()
            # Strip markdown fences if model added them
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rstrip("`").strip()
            return json.loads(raw)

        except urllib.error.HTTPError as e:
            msg = e.read().decode()
            st.error(f"API error {e.code}: {msg[:300]}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON from API: {e}")
            return None
        except Exception as e:
            if attempt < 2:
                continue   # retry on connection errors
            st.error(f"Vision API error after 3 attempts: {e}")
            return None

    return None

def gs_to_params(gs: dict) -> dict:
    ct = gs.get("ct_players", [])
    t  = gs.get("t_players",  [])

    def alive(pl):      return [p for p in pl if p.get("alive", False)]
    def sumk(pl, k):    return sum(int(p.get(k, 0) or 0) for p in alive(pl))

    ca = alive(ct); ta = alive(t)
    ct_awp = sum(1 for p in ca if p.get("weapon","") == "awp")
    t_awp  = sum(1 for p in ta if p.get("weapon","") == "awp")
    ct_rfl = sum(1 for p in ca if p.get("weapon","") in RIFLES)
    t_rfl  = sum(1 for p in ta if p.get("weapon","") in RIFLES)

    return dict(
        map_name    = gs.get("map_name","de_dust2"),
        time_left   = int(gs.get("time_left", 90)),
        bomb_planted= bool(gs.get("bomb_planted", False)),
        ct_score    = int(gs.get("ct_score", 0)),
        t_score     = int(gs.get("t_score", 0)),
        ct_players  = len(ca),
        t_players   = len(ta),
        ct_health   = min(sumk(ct,"hp"), len(ca)*100),
        t_health    = min(sumk(t, "hp"), len(ta)*100),
        ct_armor    = min(sum(100 for p in ca if p.get("has_armor",False)), len(ca)*100),
        t_armor     = min(sum(100 for p in ta if p.get("has_armor",False)), len(ta)*100),
        ct_money    = min(sumk(ct,"money"), 16000),
        t_money     = min(sumk(t, "money"), 16000),
        ct_helmets  = min(sum(1 for p in ca if p.get("has_helmet",False)), len(ca)),
        t_helmets   = min(sum(1 for p in ta if p.get("has_helmet",False)), len(ta)),
        ct_kits     = min(sum(1 for p in ca if p.get("has_defuse_kit",False)), len(ca)),
        ct_awp      = ct_awp,
        t_awp       = t_awp,
        ct_rifles   = min(ct_rfl, max(0, len(ca)-ct_awp)),
        t_rifles    = min(t_rfl,  max(0, len(ta)-t_awp)),
        ct_grens    = sumk(ct,"grenades"),
        t_grens     = sumk(t, "grenades"),
    )

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
st.title("CS:GO Round Winner Predictor")

csv_path = find_csv()
if csv_path is None:
    st.error(f"`{CSV_FILE}` not found. Put csgo_round_snapshots.csv in the same folder.")
    st.stop()

with st.spinner("Training model on first launch — about 30 seconds..."):
    model, fcols, le = load_model(csv_path)
st.success("Model ready!")
st.markdown("---")

tab1, tab2 = st.tabs(["Manual Input", "Screenshot (Auto-fill)"])

# ── TAB 1: Manual ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="sec">General</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns([2,1,1,1,1])
    map_name     = c1.selectbox("Map", MAPS, key="m_map")
    time_left    = c2.number_input("Time Left (s)", 0,175,90,1, key="m_ti")
    bomb_planted = c3.selectbox("Bomb Planted?", ["No","Yes"], key="m_bo")
    ct_score     = c4.number_input("CT Score", 0,30,8,1, key="m_cs")
    t_score      = c5.number_input("T Score",  0,30,7,1, key="m_ts")

    st.markdown("---")
    st.markdown('<p class="sec">Players and Health</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    ct_players = c1.number_input("CT Players Alive",0,5,4,1,key="m_cp")
    t_players  = c2.number_input("T Players Alive", 0,5,4,1,key="m_tp")
    ct_health  = c3.number_input("CT Total Health", 0,500,350,5,key="m_ch")
    t_health   = c4.number_input("T Total Health",  0,500,300,5,key="m_th")
    ct_armor   = c5.number_input("CT Total Armor",  0,500,300,5,key="m_ca")
    t_armor    = c6.number_input("T Total Armor",   0,500,250,5,key="m_ta")

    st.markdown("---")
    st.markdown('<p class="sec">Economy</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    ct_money   = c1.number_input("CT Money ($)", 0,16000,8000,50,key="m_cm")
    t_money    = c2.number_input("T Money ($)",  0,16000,5000,50,key="m_tm")
    ct_helmets = c3.number_input("CT Helmets",   0,5,3,1,key="m_che")
    t_helmets  = c4.number_input("T Helmets",    0,5,2,1,key="m_the")

    st.markdown("---")
    st.markdown('<p class="sec">Weapons and Utilities</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    ct_awp   = c1.number_input("CT AWPs",       0,5,1,1,key="m_caw")
    t_awp    = c2.number_input("T AWPs",        0,5,0,1,key="m_taw")
    ct_rifles= c3.number_input("CT Rifles",      0,5,3,1,key="m_cri")
    t_rifles = c4.number_input("T Rifles",       0,5,2,1,key="m_tri")
    ct_kits  = c5.number_input("CT Defuse Kits", 0,5,2,1,key="m_cki")
    ct_grens = c6.number_input("CT Grenades",    0,20,3,1,key="m_cgr")
    t_grens  = c7.number_input("T Grenades",     0,20,2,1,key="m_tgr")

    st.markdown("---")
    _,btn,_ = st.columns([2,1,2])
    if btn.button("Predict Winner", use_container_width=True, type="primary", key="m_btn"):
        params = dict(
            map_name=map_name, time_left=time_left,
            bomb_planted=(bomb_planted=="Yes"),
            ct_score=ct_score, t_score=t_score,
            ct_players=ct_players, t_players=t_players,
            ct_health=ct_health, t_health=t_health,
            ct_armor=ct_armor, t_armor=t_armor,
            ct_money=ct_money, t_money=t_money,
            ct_helmets=ct_helmets, t_helmets=t_helmets,
            ct_kits=ct_kits,
            ct_awp=ct_awp, t_awp=t_awp,
            ct_rifles=ct_rifles, t_rifles=t_rifles,
            ct_grens=ct_grens, t_grens=t_grens,
        )
        errs = validate(params)
        if errs:
            st.markdown("---")
            st.markdown("### Input errors")
            for msg in errs: st.error(msg)
        else:
            show_result(*predict(model, fcols, le, params), params)

# ── TAB 2: Screenshot ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("Upload a broadcast screenshot — Claude Vision reads all stats automatically.")
    st.markdown("---")

    uploaded = st.file_uploader("Drop screenshot here", type=["png","jpg","jpeg"],
                                key="uploader")

    if uploaded:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption="Uploaded screenshot", use_container_width=True)

        _,pbtn,_ = st.columns([2,1,2])
        if pbtn.button("Parse Screenshot + Predict", use_container_width=True,
                       type="primary", key="s_btn"):

            with st.spinner("Reading screenshot with Claude Vision..."):
                gs = parse_screenshot(img_bytes)

            if gs:
                params = gs_to_params(gs)

                st.markdown("---")
                st.markdown("### Parsed Game State")
                i1, i2, i3 = st.columns(3)

                with i1:
                    st.markdown("**Match Info**")
                    for k,v in {"Map": gs.get("map_name","?"),
                                "Timer": f"{gs.get('time_left','?')}s",
                                "Score": f"CT {gs.get('ct_score','?')} - {gs.get('t_score','?')} T",
                                "Bomb": "Planted" if gs.get("bomb_planted") else "Not planted",
                                "CT alive": f"{params['ct_players']} / 5",
                                "T alive":  f"{params['t_players']} / 5"}.items():
                        st.markdown(f'<div class="row"><span style="color:#888">{k}</span>'
                                    f'<b style="color:#e0e0e0">{v}</b></div>',
                                    unsafe_allow_html=True)

                with i2:
                    st.markdown(f'<b style="color:{CT_CLR}">CT Players</b>', unsafe_allow_html=True)
                    for p in gs.get("ct_players",[]):
                        cls = "palive" if p.get("alive") else "pdead"
                        hp  = f'HP:{p.get("hp",0)}' if p.get("alive") else "DEAD"
                        st.markdown(
                            f'<div class="{cls}"><b>{p.get("name","?")}</b> &nbsp; '
                            f'{hp} &nbsp; {p.get("weapon","") if p.get("alive") else ""} &nbsp; '
                            f'${p.get("money",0):,}</div>', unsafe_allow_html=True)

                with i3:
                    st.markdown(f'<b style="color:{T_CLR}">T Players</b>', unsafe_allow_html=True)
                    for p in gs.get("t_players",[]):
                        cls = "palive" if p.get("alive") else "pdead"
                        hp  = f'HP:{p.get("hp",0)}' if p.get("alive") else "DEAD"
                        st.markdown(
                            f'<div class="{cls}"><b>{p.get("name","?")}</b> &nbsp; '
                            f'{hp} &nbsp; {p.get("weapon","") if p.get("alive") else ""} &nbsp; '
                            f'${p.get("money",0):,}</div>', unsafe_allow_html=True)

                # Auto-clamp any out-of-range values silently
                params["ct_health"]  = min(params["ct_health"],  params["ct_players"]*100)
                params["t_health"]   = min(params["t_health"],   params["t_players"]*100)
                params["ct_armor"]   = min(params["ct_armor"],   params["ct_players"]*100)
                params["t_armor"]    = min(params["t_armor"],    params["t_players"]*100)
                params["ct_helmets"] = min(params["ct_helmets"], max(0, params["ct_players"]))
                params["t_helmets"]  = min(params["t_helmets"],  max(0, params["t_players"]))
                params["ct_kits"]    = min(params["ct_kits"],    max(0, params["ct_players"]))
                params["ct_awp"]     = min(params["ct_awp"],     params["ct_players"])
                params["t_awp"]      = min(params["t_awp"],      params["t_players"])
                params["ct_rifles"]  = min(params["ct_rifles"],  max(0, params["ct_players"]-params["ct_awp"]))
                params["t_rifles"]   = min(params["t_rifles"],   max(0, params["t_players"]-params["t_awp"]))

                show_result(*predict(model, fcols, le, params), params)