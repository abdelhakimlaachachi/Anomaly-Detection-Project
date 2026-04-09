import streamlit as st
import cv2
import numpy as np
import time
import collections
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import base64
import tempfile
import pandas as pd


st.set_page_config(
    page_title="SENTINEL AI — PFE",
    page_icon="🔺",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* ── BASE ──────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #020b12 !important;
    color: #a0d8ef !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #020b12 !important;
}

/* Grille de fond animée */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    animation: gridScroll 20s linear infinite;
}
@keyframes gridScroll { from{background-position:0 0} to{background-position:40px 40px} }

/* Ligne de scan animée */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,255,255,0.15), rgba(0,255,255,0.4), rgba(0,255,255,0.15), transparent);
    animation: scanLine 6s linear infinite;
    pointer-events: none;
    z-index: 1;
}
@keyframes scanLine { from{top:-2px} to{top:100vh} }

* { font-family: 'Share Tech Mono', monospace !important; }

/* ── SIDEBAR ───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #000d16 !important;
    border-right: 1px solid rgba(0,255,255,0.15) !important;
}
[data-testid="stSidebar"] * { color: #7ecfed !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: #041520 !important;
    border: 1px solid rgba(0,255,255,0.25) !important;
    color: #00ffff !important;
    border-radius: 2px !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #00ffff !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(0,255,255,0.15) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(0,255,255,0.4) !important;
    color: #00ffff !important;
    border-radius: 2px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,255,255,0.1) !important;
    box-shadow: 0 0 12px rgba(0,255,255,0.3) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    border-color: rgba(0,255,255,0.7) !important;
    box-shadow: 0 0 8px rgba(0,255,255,0.2), inset 0 0 8px rgba(0,255,255,0.05) !important;
}

/* ── TITRES ─────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Orbitron', monospace !important;
    color: #00ffff !important;
    letter-spacing: 0.08em !important;
    text-shadow: 0 0 20px rgba(0,255,255,0.4) !important;
}
h1 { font-size: 22px !important; font-weight: 900 !important; }
h4 { font-size: 12px !important; font-weight: 400 !important;
     color: rgba(0,255,255,0.6) !important; text-shadow: none !important; }
hr {
    border: none !important;
    border-top: 1px solid rgba(0,255,255,0.15) !important;
    margin: 1rem 0 !important;
}

/* ── KPI CARDS ──────────────────────────────────────── */
.kpi-card {
    background: #020f1c;
    border: 1px solid rgba(0,255,255,0.2);
    border-top: 2px solid rgba(0,255,255,0.5);
    padding: 14px 16px;
    text-align: center;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,255,0.6), transparent);
}
.kpi-label {
    font-size: 10px;
    color: rgba(0,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Orbitron', monospace !important;
    font-size: 28px;
    font-weight: 700;
    margin: 4px 0 2px;
    line-height: 1;
}
.kpi-sub { font-size: 10px; color: rgba(0,255,255,0.35); margin-top: 4px; }
.kpi-green  { color: #00ff9f; text-shadow: 0 0 12px rgba(0,255,159,0.6); }
.kpi-red    { color: #ff2a2a; text-shadow: 0 0 12px rgba(255,42,42,0.7); }
.kpi-orange { color: #ffaa00; text-shadow: 0 0 12px rgba(255,170,0,0.6); }
.kpi-cyan   { color: #00ffff; text-shadow: 0 0 12px rgba(0,255,255,0.6); }

/* ── STATUS BANNER ──────────────────────────────────── */
.status-banner {
    padding: 10px 20px;
    text-align: center;
    font-family: 'Orbitron', monospace !important;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 8px;
    border-left: 3px solid;
    border-right: 3px solid;
    clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);
}
.status-normal {
    background: rgba(0,255,159,0.06);
    color: #00ff9f;
    border-color: rgba(0,255,159,0.5);
    text-shadow: 0 0 10px rgba(0,255,159,0.5);
}
.status-alert {
    background: rgba(255,42,42,0.1);
    color: #ff2a2a;
    border-color: rgba(255,42,42,0.7);
    text-shadow: 0 0 10px rgba(255,42,42,0.8);
    animation: alertPulse 0.8s ease-in-out infinite;
}
.status-cooldown {
    background: rgba(255,170,0,0.07);
    color: #ffaa00;
    border-color: rgba(255,170,0,0.5);
    text-shadow: 0 0 10px rgba(255,170,0,0.5);
}
@keyframes alertPulse {
    0%,100% { opacity:1; box-shadow: 0 0 15px rgba(255,42,42,0.3); }
    50%      { opacity:0.75; box-shadow: 0 0 30px rgba(255,42,42,0.6); }
}

/* ── SCORE BAR ──────────────────────────────────────── */
.score-bar-wrap { margin-top: 8px; padding: 8px 0; }
.score-bar-labels {
    display: flex; justify-content: space-between;
    font-size: 10px; color: rgba(0,255,255,0.4);
    margin-bottom: 5px; letter-spacing: 0.08em;
}
.score-bar-track {
    background: rgba(0,255,255,0.06);
    border: 1px solid rgba(0,255,255,0.15);
    height: 8px; position: relative;
}
.score-bar-fill { height: 100%; transition: width 0.3s ease; }
.score-bar-fill.green  { background: linear-gradient(90deg, #003d25, #00ff9f); box-shadow: 2px 0 8px #00ff9f; }
.score-bar-fill.orange { background: linear-gradient(90deg, #3d2800, #ffaa00); box-shadow: 2px 0 8px #ffaa00; }
.score-bar-fill.red    { background: linear-gradient(90deg, #3d0000, #ff2a2a); box-shadow: 2px 0 8px #ff2a2a; }
.score-bar-threshold {
    position: absolute; top: -4px;
    width: 1px; height: 16px;
    background: rgba(255,255,255,0.5);
}
.score-bar-sub {
    font-size: 9px; color: rgba(0,255,255,0.3);
    margin-top: 3px; text-align: right; letter-spacing: 0.08em;
}

/* ── ALERT ROWS ─────────────────────────────────────── */
.alert-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(0,255,255,0.08);
    font-size: 12px;
}
.alert-row:last-child { border-bottom: none; }
.alert-dot-red {
    width:8px; height:8px; border-radius:50%;
    background:#ff2a2a; box-shadow:0 0 6px #ff2a2a;
    flex-shrink:0; margin-top:3px;
    animation: dotBlink 1.5s ease-in-out infinite;
}
.alert-dot-orange {
    width:8px; height:8px; border-radius:50%;
    background:#ffaa00; box-shadow:0 0 6px #ffaa00;
    flex-shrink:0; margin-top:3px;
}
@keyframes dotBlink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.alert-label { color: #a0d8ef; font-size: 12px; }
.alert-meta  { color: rgba(0,255,255,0.35); font-size: 10px; margin-top: 2px; letter-spacing: 0.05em; }
.alerts-empty { font-size: 11px; color: rgba(0,255,255,0.25); letter-spacing: 0.1em; padding: 8px 0; }

/* ── PANEL TITLES ───────────────────────────────────── */
.panel-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 10px !important;
    color: rgba(0,255,255,0.45) !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,255,255,0.12);
}

/* ── STREAMLIT OVERRIDES ────────────────────────────── */
[data-testid="stMarkdownContainer"] p { color: #7ecfed; }
[data-testid="stImage"] img { border: 1px solid rgba(0,255,255,0.2) !important; }
[data-testid="stAlert"] {
    background: rgba(0,255,255,0.05) !important;
    border: 1px solid rgba(0,255,255,0.2) !important;
    color: #7ecfed !important;
    border-radius: 0 !important;
}
[data-testid="stProgressBar"] > div > div {
    background-color: rgba(0,255,255,0.1) !important;
}
[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #003d4d, #00ffff) !important;
    box-shadow: 2px 0 6px rgba(0,255,255,0.5) !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(0,255,159,0.06) !important;
    border: 1px solid rgba(0,255,159,0.2) !important;
    color: #00ff9f !important;
}
</style>
""", unsafe_allow_html=True)


SEQUENCE_LENGTH = 30
FRAME_SKIP      = 1
FPS_INFERENCE   = 20
BACKUP_DIR      = "local_saved_clips"

if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)


for k, v in [
    ("running", False), ("alerts", []), ("score_history", []),
    ("frame_count", 0), ("alert_count", 0), ("persons_count", 0),
    ("current_score", 0.0),
]:
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource
def load_smart_model(path_model):
    try:
        lstm_model = load_model(path_model, compile=False)
        yolo_model = YOLO('yolov8s-pose.pt')
        return lstm_model, yolo_model
    except Exception as e:
        st.error(f"ERREUR CHARGEMENT MODÈLE : {e}")
        return None, None


def extract_keypoints(results):
    if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
        return results[0].keypoints.xyn[0].cpu().numpy().flatten()
    return np.zeros(34)


def process_sequence(seq):
    seq      = seq.reshape(seq.shape[0], -1, 2)
    seq      = seq - seq[:, :1, :]
    seq      = seq.reshape(seq.shape[0], -1)
    velocity = seq[1:] - seq[:-1]
    velocity = np.vstack([velocity, np.zeros((1, seq.shape[1]))])
    return np.concatenate([seq, velocity], axis=1)


def send_video_clip_robust(video_path, score, webhook_url):
    time.sleep(1)
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return False
    try:
        import requests
        with open(video_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode('utf-8')
        r = requests.post(
            webhook_url,
            json={'camera': 'SENTINEL_CAM', 'score': str(score), 'video_base64': video_b64},
            timeout=60,
            headers={'ngrok-skip-browser-warning': 'true', 'Connection': 'close'},
            verify=False
        )
        if r.status_code == 200:
            try: os.remove(video_path)
            except: pass
            return True
    except Exception as e:
        print(f"Webhook error: {e}")
    return False


mean = np.load("models/mean.npy")
std  = np.load("models/std.npy")


with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 16px;">
        <div style="font-family:'Orbitron',monospace; font-size:16px; font-weight:900;
                    color:#00ffff; letter-spacing:0.25em; text-shadow:0 0 20px rgba(0,255,255,0.6);">
            SENTINEL
        </div>
        <div style="font-size:9px; color:rgba(0,255,255,0.35); letter-spacing:0.3em; margin-top:2px;">
            AI SURVEILLANCE SYSTEM v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    model_files = [f for f in os.listdir('models') if f.endswith('.keras')]
    video_files = [f for f in os.listdir('testing_videos') if f.endswith(('.mp4', '.avi'))]

    selected_model_path = st.selectbox("MODÈLE IA", model_files, index=0 if model_files else None)
    source_option       = st.selectbox("SOURCE VIDÉO", ["Webcam (0)"] + video_files)
    video_source        = 0 if source_option == "Webcam (0)" else source_option
    threshold           = st.slider("SEUIL ANOMALIE", 0.50, 0.99, 0.80, format="%.2f")
    webhook_url         = st.text_input("WEBHOOK URL", "")
    cooldown            = st.number_input("ANTI-SPAM (sec)", 10, 300, 30)

    st.markdown("---")
    col_s, col_x = st.columns(2)
    with col_s: start_btn = st.button("▶ START", type="primary", use_container_width=True)
    with col_x: stop_btn  = st.button("■ STOP",  use_container_width=True)
    st.markdown("---")

    if st.session_state.running:
        st.success("● SYSTÈME ACTIF")
    else:
        st.info("○ SYSTÈME ARRÊTÉ")

    st.markdown('<div style="font-size:10px;color:rgba(0,255,255,0.4);letter-spacing:0.1em;margin-top:10px;margin-bottom:4px;">ANTI-SPAM COOLDOWN</div>', unsafe_allow_html=True)
    cooldown_ph_sidebar = st.empty()

if start_btn:
    st.session_state.running       = True
    st.session_state.frame_count   = 0
    st.session_state.alert_count   = 0
    st.session_state.score_history = []
    st.session_state.alerts        = []

if stop_btn:
    st.session_state.running = False


st.markdown("""
<div style="display:flex; align-items:baseline; gap:16px; margin-bottom:4px;">
    <div style="font-family:'Orbitron',monospace; font-size:22px; font-weight:900;
                color:#00ffff; letter-spacing:0.15em;
                text-shadow:0 0 25px rgba(0,255,255,0.5);">
        ◈ SENTINEL AI
    </div>
    <div style="font-size:11px; color:rgba(0,255,255,0.35); letter-spacing:0.2em; padding-bottom:2px;">
        SYSTÈME DE SURVEILLANCE INTELLIGENTE — PFE
    </div>
</div>
<div style="height:1px; background:linear-gradient(90deg,rgba(0,255,255,0.6),transparent); margin-bottom:16px;"></div>
""", unsafe_allow_html=True)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi_score_ph   = kpi1.empty()
kpi_alerts_ph  = kpi2.empty()
kpi_frames_ph  = kpi3.empty()
kpi_persons_ph = kpi4.empty()

def render_kpis(score, alerts, frames, persons, thr):
    score_cls = "kpi-red" if score > thr else ("kpi-orange" if score > thr * 0.75 else "kpi-green")
    kpi_score_ph.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">// SCORE ANOMALIE</div>
        <div class="kpi-value {score_cls}">{score:.2f}</div>
        <div class="kpi-sub">SEUIL : {thr:.2f}</div>
    </div>""", unsafe_allow_html=True)

    kpi_alerts_ph.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">// ALERTES SESSION</div>
        <div class="kpi-value {'kpi-red' if alerts > 0 else 'kpi-green'}">{alerts}</div>
        <div class="kpi-sub">{'⚠ INCIDENTS DÉTECTÉS' if alerts > 0 else 'NOMINAL'}</div>
    </div>""", unsafe_allow_html=True)

    kpi_frames_ph.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">// FRAMES TRAITÉES</div>
        <div class="kpi-value kpi-cyan">{frames:,}</div>
        <div class="kpi-sub">{FPS_INFERENCE} FPS INFÉRENCE</div>
    </div>""", unsafe_allow_html=True)

    kpi_persons_ph.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">// CIBLES DÉTECTÉES</div>
        <div class="kpi-value kpi-orange">{persons}</div>
        <div class="kpi-sub">YOLO-POSE TRACKER</div>
    </div>""", unsafe_allow_html=True)

render_kpis(
    st.session_state.current_score, st.session_state.alert_count,
    st.session_state.frame_count,   st.session_state.persons_count, threshold
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


col_video, col_panel = st.columns([3, 1])

with col_video:
    status_banner_ph  = st.empty()
    video_placeholder = st.empty()
    score_bar_ph      = st.empty()

with col_panel:
    st.markdown('<div class="panel-title">// COURBE DE MENACE</div>', unsafe_allow_html=True)
    chart_ph = st.empty()
    st.markdown('<div class="panel-title" style="margin-top:14px;">// JOURNAL DES INCIDENTS</div>', unsafe_allow_html=True)
    alerts_ph = st.empty()


def render_status_banner(text, color):
    css = {"green": "status-normal", "red": "status-alert", "orange": "status-cooldown"}.get(color, "status-normal")
    status_banner_ph.markdown(f'<div class="status-banner {css}">{text}</div>', unsafe_allow_html=True)

def render_score_bar(score, thr):
    pct = int(score * 100)
    cls = "red" if score > thr else ("orange" if score > thr * 0.75 else "green")
    score_bar_ph.markdown(f"""
    <div class="score-bar-wrap">
        <div class="score-bar-labels">
            <span>NIVEAU DE MENACE</span><span>{score:.2f}</span>
        </div>
        <div class="score-bar-track">
            <div class="score-bar-fill {cls}" style="width:{pct}%"></div>
            <div class="score-bar-threshold" style="left:{int(thr*100)}%"></div>
        </div>
        <div class="score-bar-sub">SEUIL DE DÉCLENCHEMENT : {thr:.2f}</div>
    </div>""", unsafe_allow_html=True)

def render_score_chart(history):
    if not history:
        chart_ph.empty()
        return
    df = pd.DataFrame({"Menace": history[-60:]})
    chart_ph.line_chart(df, height=155, use_container_width=True)

def render_alerts(alerts_list):
    if not alerts_list:
        alerts_ph.markdown('<div class="alerts-empty">// AUCUN INCIDENT ENREGISTRÉ</div>', unsafe_allow_html=True)
        return
    html = ""
    for a in reversed(alerts_list[-5:]):
        dot = "alert-dot-red" if a["score"] >= 0.9 else "alert-dot-orange"
        html += f"""
        <div class="alert-row">
            <div class="{dot}"></div>
            <div>
                <div class="alert-label">{a['label'].upper()}</div>
                <div class="alert-meta">{a['time']} &nbsp;·&nbsp; SCORE {a['score']:.2f}</div>
            </div>
        </div>"""
    alerts_ph.markdown(html, unsafe_allow_html=True)

render_status_banner("// EN ATTENTE — CONFIGUREZ ET DÉMARREZ", "green")
render_score_chart(st.session_state.score_history)
render_alerts(st.session_state.alerts)


if st.session_state.running:

    if not selected_model_path:
        st.error("ERREUR : AUCUN MODÈLE SÉLECTIONNÉ")
        st.stop()

    lstm_model, yolo_model = load_smart_model(os.path.join("models", selected_model_path))

    if isinstance(video_source, str) and os.path.exists(os.path.join("testing_videos", video_source)):
        cap = cv2.VideoCapture(os.path.join("testing_videos", video_source))
    elif isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
    else:
        uploaded_file = st.file_uploader("UPLOAD VIDÉO", type=["mp4", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.info("// UPLOADER UNE VIDÉO OU SÉLECTIONNER WEBCAM")
            st.stop()

    if not cap.isOpened():
        st.error("ERREUR : SOURCE VIDÉO INACCESSIBLE")
        st.stop()

    buffer_maxlen           = SEQUENCE_LENGTH * FRAME_SKIP
    frame_buffer_raw        = collections.deque(maxlen=buffer_maxlen)
    frame_buffer_annotated  = collections.deque(maxlen=buffer_maxlen)
    keypoints_buffer        = collections.deque(maxlen=buffer_maxlen)
    pred_buffer             = collections.deque(maxlen=5)

    is_recording                = False
    recording_counter           = 0
    video_writer_raw            = None
    video_writer_annotated      = None
    current_clip_path_raw       = ""
    current_clip_path_annotated = ""
    detection_score             = 0.0
    last_alert_time             = 0
    frame_count_global          = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("// FIN DU FLUX VIDÉO")
            st.session_state.running = False
            break

        frame           = cv2.resize(frame, (640, 480))
        results         = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        keypoints       = extract_keypoints(results)

        n_persons = len(results[0].boxes) if results[0].boxes is not None else 0
        st.session_state.persons_count = n_persons

        frame_buffer_raw.append(frame.copy())
        frame_buffer_annotated.append(annotated_frame.copy())
        keypoints_buffer.append(keypoints)

        current_time        = time.time()
        frame_count_global += 1
        st.session_state.frame_count = frame_count_global

        current_loss = 0.0
        status_text  = "// FLUX NOMINAL — AUCUNE ANOMALIE"
        status_color = "green"

        if len(frame_buffer_raw) >= buffer_maxlen and not is_recording:
            if frame_count_global % FRAME_SKIP == 0:
                time_since_alert = current_time - last_alert_time

                if time_since_alert > cooldown:
                    cooldown_ph_sidebar.progress(1.0)

                    kp_subset  = list(keypoints_buffer)[::FRAME_SKIP]
                    seq_input  = np.array(kp_subset)
                    seq_input  = process_sequence(seq_input)
                    seq_input  = seq_input.reshape(1, SEQUENCE_LENGTH, 68)
                    seq_input  = (seq_input - mean) / std

                    prediction     = lstm_model.predict(seq_input, verbose=0)
                    current_loss   = prediction[0][0]

                    pred_buffer.append(current_loss)
                    smoothed_score = np.mean(pred_buffer)

                    st.session_state.score_history.append(float(current_loss))
                    st.session_state.current_score = float(current_loss)

                    if smoothed_score > threshold and sum(p > threshold for p in pred_buffer) >= 3:
                        is_recording      = True
                        detection_score   = smoothed_score
                        recording_counter = FPS_INFERENCE * 5

                        timestamp = datetime.now().strftime('%H%M%S')
                        current_clip_path_raw        = os.path.join(BACKUP_DIR, f"preuve_webhook_{timestamp}.mp4")
                        current_clip_path_annotated  = os.path.join(BACKUP_DIR, f"preuve_local_{timestamp}.mp4")

                        h, w, _ = frame.shape
                        video_writer_raw       = cv2.VideoWriter(current_clip_path_raw,        cv2.VideoWriter_fourcc(*'mp4v'), FPS_INFERENCE, (w, h))
                        video_writer_annotated = cv2.VideoWriter(current_clip_path_annotated,  cv2.VideoWriter_fourcc(*'mp4v'), FPS_INFERENCE, (w, h))

                        for old_raw, old_ann in zip(frame_buffer_raw, frame_buffer_annotated):
                            video_writer_raw.write(old_raw)
                            video_writer_annotated.write(old_ann)

                        st.session_state.alert_count += 1
                        st.session_state.alerts.append({
                            "label": "Anomalie détectée",
                            "score": float(smoothed_score),
                            "time":  datetime.now().strftime('%H:%M:%S')
                        })

                        status_text  = "⚠ ALERTE — COMPORTEMENT ANORMAL DÉTECTÉ"
                        status_color = "red"

                else:
                    remaining    = cooldown - time_since_alert
                    progress_val = max(0.0, min(1.0, remaining / cooldown))
                    cooldown_ph_sidebar.progress(progress_val, text=f"COOLDOWN : {int(remaining)}s")
                    status_text  = f"// PAUSE ANTI-SPAM — {int(remaining)}s"
                    status_color = "orange"

        if is_recording:
            video_writer_raw.write(frame)
            video_writer_annotated.write(annotated_frame)
            recording_counter -= 1
            status_text  = f"⚠ ENREGISTREMENT EN COURS — SCORE : {detection_score:.2f}"
            status_color = "red"

            if recording_counter <= 0:
                is_recording = False
                video_writer_raw.release()
                video_writer_annotated.release()
                send_video_clip_robust(current_clip_path_raw, detection_score, webhook_url)
                last_alert_time = time.time()
                frame_buffer_raw.clear()
                frame_buffer_annotated.clear()
                keypoints_buffer.clear()
                pred_buffer.clear()

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        render_status_banner(status_text, status_color)
        render_score_bar(current_loss, threshold)
        render_kpis(current_loss, st.session_state.alert_count, frame_count_global, n_persons, threshold)

        if frame_count_global % 10 == 0:
            render_score_chart(st.session_state.score_history)
            render_alerts(st.session_state.alerts)

    cap.release()
    st.info("// SURVEILLANCE TERMINÉE — SYSTÈME EN VEILLE")

else:
    render_status_banner("// EN ATTENTE — CONFIGUREZ ET DÉMARREZ", "green")
    video_placeholder.empty()
    render_score_chart(st.session_state.score_history)
    render_alerts(st.session_state.alerts)