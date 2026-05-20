"""
Sign Language to Text — Real-Time Streamlit Dashboard
======================================================
Captures webcam via OpenCV, detects hand landmarks with MediaPipe Tasks API,
and classifies signs using the trained model in real-time.
"""

import streamlit as st
import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from collections import deque, Counter
from datetime import datetime

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="HoloSign: Universal Gesture Communication System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.05rem;
        font-weight: 300;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }

    .prediction-box {
        text-align: center;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        margin: 1rem 0;
    }
    .prediction-sign {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    .prediction-label {
        color: #8892b0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    .confidence-container { margin: 1rem 0; }
    .confidence-bar-bg {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .confidence-text {
        color: #a0aec0;
        font-size: 0.8rem;
        margin-top: 0.3rem;
        display: flex;
        justify-content: space-between;
    }

    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-active {
        background: rgba(72, 187, 120, 0.15);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .status-inactive {
        background: rgba(245, 101, 101, 0.15);
        color: #f56565;
        border: 1px solid rgba(245, 101, 101, 0.3);
    }

    .history-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        margin-bottom: 0.4rem;
        border-left: 3px solid #667eea;
    }
    .history-sign { color: #e2e8f0; font-weight: 600; font-size: 0.95rem; }
    .history-time { color: #718096; font-size: 0.75rem; }

    .section-title {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .stat-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #667eea; }
    .stat-label {
        font-size: 0.7rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(102, 126, 234, 0.3); border-radius: 3px; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "sign_language_model.pkl")
LABEL_MAP_PATH = os.path.join("model", "label_map.pkl")
HAND_MODEL_PATH = os.path.join("model", "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3


# ── Load Resources ─────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)
    return model, label_map


@st.cache_resource
def load_hand_landmarker():
    if not os.path.exists(HAND_MODEL_PATH):
        return None
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


model, label_map = load_ml_model()
landmarker = load_hand_landmarker()


def get_confidence_color(conf):
    if conf >= 0.8:
        return "#48bb78"
    elif conf >= 0.5:
        return "#ecc94b"
    else:
        return "#f56565"


def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks


def extract_two_hand_features(result):
    """Build 126-feature vector (left hand + right hand), zeros for missing hand."""
    empty = [0.0] * (NUM_LANDMARKS * FEATURES_PER_LANDMARK)
    left_coords = empty[:]
    right_coords = empty[:]
    for i, hand_lms in enumerate(result.hand_landmarks):
        label = result.handedness[i][0].category_name.lower()
        coords = extract_landmarks(hand_lms)
        if label == "left":
            left_coords = coords
        else:
            right_coords = coords
    return left_coords + right_coords


def draw_hand_on_frame(frame, hand_landmarks):
    """Draw hand landmarks on OpenCV frame."""
    h, w = frame.shape[:2]
    for start_idx, end_idx in HAND_CONNECTIONS:
        s = hand_landmarks[start_idx]
        e = hand_landmarks[end_idx]
        cv2.line(frame, (int(s.x * w), int(s.y * h)),
                 (int(e.x * w), int(e.y * h)), (0, 255, 128), 2, cv2.LINE_AA)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 128), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)


# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤟 HoloSign: Universal Gesture Communication System</h1>
    <p>Real-time hand gesture recognition powered by MediaPipe + Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ── Pre-flight Checks ─────────────────────────────────────────
if landmarker is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <h2 style="color: #f56565;">⚠️ Hand Landmarker Model Not Found</h2>
        <p style="color: #a0aec0;">Download <code>hand_landmarker.task</code> into the <code>model/</code> folder.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if model is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <h2 style="color: #f56565;">⚠️ ML Model Not Found</h2>
        <p style="color: #a0aec0; font-size: 1.1rem;">
            Run the data collection and training scripts first:
        </p>
        <br>
        <code style="color: #667eea; font-size: 1rem;">
            1. python collect_data.py<br>
            2. python train_model.py
        </code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Session State ──────────────────────────────────────────────
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=20)
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0
if "running" not in st.session_state:
    st.session_state.running = False

# ── Layout ─────────────────────────────────────────────────────
col_video, col_panel = st.columns([3, 2], gap="large")

with col_video:
    st.markdown('<div class="section-title">📹 Live Camera Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶ Start Camera", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏹ Stop Camera", use_container_width=True)

with col_panel:
    st.markdown('<div class="section-title">🎯 Prediction</div>', unsafe_allow_html=True)
    prediction_placeholder = st.empty()
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">📊 Statistics</div>',
                unsafe_allow_html=True)
    stats_placeholder = st.empty()
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">📜 History</div>',
                unsafe_allow_html=True)
    history_placeholder = st.empty()


def render_prediction(sign, confidence):
    color = get_confidence_color(confidence)
    conf_pct = confidence * 100
    prediction_placeholder.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Detected Sign</div>
        <div class="prediction-sign">{sign}</div>
        <div class="confidence-container">
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: {conf_pct}%; background: {color};"></div>
            </div>
            <div class="confidence-text">
                <span>Confidence</span>
                <span style="color: {color}; font-weight: 600;">{conf_pct:.1f}%</span>
            </div>
        </div>
        <div class="status-indicator status-active">● Hand Detected</div>
    </div>
    """, unsafe_allow_html=True)


def render_no_hand():
    prediction_placeholder.markdown("""
    <div class="prediction-box" style="border-color: rgba(245, 101, 101, 0.2); background: rgba(245, 101, 101, 0.05);">
        <div class="prediction-label">Waiting for Input</div>
        <div class="prediction-sign" style="font-size: 2.5rem; -webkit-text-fill-color: #718096;">—</div>
        <p style="color: #718096; font-size: 0.9rem;">Show a hand gesture to the camera</p>
        <div class="status-indicator status-inactive">● No Hand Detected</div>
    </div>
    """, unsafe_allow_html=True)


def render_stats(total, sign_counts):
    most_common = max(sign_counts, key=sign_counts.get) if sign_counts else "—"
    stats_placeholder.markdown(f"""
    <div class="glass-card">
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Predictions</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{most_common}</div>
                <div class="stat-label">Most Frequent</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(sign_counts)}</div>
                <div class="stat-label">Signs Detected</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(label_map)}</div>
                <div class="stat-label">Signs Available</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_history(history):
    if not history:
        history_placeholder.markdown("""
        <div class="glass-card" style="text-align: center;">
            <p style="color: #718096;">No predictions yet</p>
        </div>
        """, unsafe_allow_html=True)
        return
    import pandas as pd
    rows = [{"Sign": sign, "Time": timestamp, "Accuracy": f"{conf*100:.1f}%" if conf is not None else "N/A"}
            for sign, timestamp, *rest in reversed(list(history))
            for conf in [rest[0] if rest else None]]
    df = pd.DataFrame(rows)
    history_placeholder.dataframe(df, use_container_width=True, hide_index=True)


# ── Camera Loop ────────────────────────────────────────────────
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    prediction_buffer = deque(maxlen=10)
    sign_counts = {}
    last_stable_sign = None

    render_no_hand()
    render_stats(st.session_state.total_predictions, sign_counts)
    render_history(st.session_state.prediction_history)

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands using Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    draw_hand_on_frame(frame, hand_lms)

                # Predict using 2-hand feature vector
                features = np.array(extract_two_hand_features(result)).reshape(1, -1)
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                sign_name = label_map.get(prediction, "Unknown")
                prediction_buffer.append(sign_name)

                # Majority vote
                if len(prediction_buffer) >= 5:
                    vote = Counter(prediction_buffer).most_common(1)[0][0]
                    if vote != last_stable_sign and confidence > 0.5:
                        last_stable_sign = vote
                        st.session_state.total_predictions += 1
                        sign_counts[vote] = sign_counts.get(vote, 0) + 1
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.prediction_history.append((vote, timestamp, confidence))
                    render_prediction(vote, confidence)
                else:
                    render_prediction(sign_name, confidence)

                render_stats(st.session_state.total_predictions, sign_counts)
                render_history(st.session_state.prediction_history)

                # Green border
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1] - 1, frame.shape[0] - 1),
                              (0, 255, 0), 3)
            else:
                prediction_buffer.clear()
                last_stable_sign = None
                render_no_hand()
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1] - 1, frame.shape[0] - 1),
                              (0, 0, 255), 2)

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame, channels="RGB", use_container_width=True)

    except Exception as e:
        st.error(f"Camera error: {str(e)}")
    finally:
        cap.release()

else:
    video_placeholder.markdown("""
    <div class="glass-card" style="text-align: center; padding: 4rem 2rem; min-height: 350px;
         display: flex; flex-direction: column; justify-content: center; align-items: center;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📹</div>
        <h3 style="color: #e2e8f0; font-weight: 600;">Camera Inactive</h3>
        <p style="color: #718096; font-size: 0.95rem;">
            Click <strong>Start Camera</strong> to begin real-time sign language translation
        </p>
    </div>
    """, unsafe_allow_html=True)
    render_no_hand()
    render_stats(st.session_state.total_predictions, {})
    render_history(st.session_state.prediction_history)

# ── Text to Speech + Speech to Text (fully self-contained) ──────
st.components.v1.html("""
<style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: 'Inter', sans-serif; background: transparent; }
    .section-title {
        color: #e2e8f0; font-size: 1.1rem; font-weight: 600;
        margin: 0 0 1rem 0; display: flex; align-items: center; gap: 8px;
    }
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    textarea {
        width: 100%; min-height: 90px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 0.8rem;
        color: #e2e8f0; font-size: 0.95rem;
        resize: vertical; font-family: inherit;
        margin-bottom: 1rem;
    }
    textarea::placeholder { color: #718096; }
    .btn-row { display: flex; gap: 0.8rem; flex-wrap: wrap; }
    button {
        padding: 0.55rem 1.4rem; border: none;
        border-radius: 10px; font-weight: 600;
        font-size: 0.9rem; cursor: pointer; transition: all 0.2s;
    }
    .btn-speak  { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .btn-start  { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .btn-stop   { background: rgba(245,101,101,0.2); color: #f56565; border: 1px solid rgba(245,101,101,0.3); }
    .btn-clear  { background: rgba(255,255,255,0.06); color: #a0aec0; border: 1px solid rgba(255,255,255,0.1); }
    .status { font-size: 0.8rem; margin-top: 0.8rem; color: #718096; }
    .listening { color: #48bb78; }
</style>

<!-- TTS -->
<div class="section-title">🔊 Text to Speech</div>
<div class="glass-card">
    <textarea id="tts-input" placeholder="Enter text here..."></textarea>
    <div class="btn-row">
        <button class="btn-speak" onclick="speakText()">🔊 Speak</button>
        <button class="btn-clear" onclick="document.getElementById('tts-input').value=''">🗑 Clear</button>
    </div>
</div>

<!-- STT -->
<div class="section-title">🎤 Speech to Text</div>
<div class="glass-card">
    <textarea id="stt-output" placeholder="Your speech will appear here..."></textarea>
    <div class="btn-row">
        <button class="btn-start" onclick="startListening()">🎤 Start Recording</button>
        <button class="btn-stop"  onclick="stopListening()">⏹ Stop</button>
        <button class="btn-clear" onclick="clearSTT()">🗑 Clear</button>
    </div>
    <div id="stt-status" class="status">Click Start Recording and speak...</div>
</div>

<script>
    // ── TTS ──
    function speakText() {
        var text = document.getElementById('tts-input').value.trim();
        if (!text) { alert('Please enter some text first.'); return; }
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance(text);
        u.rate = 0.9; u.pitch = 1.0;
        window.speechSynthesis.speak(u);
    }

    // ── STT ──
    var recognition = null;

    function clearSTT() {
        if (recognition) { recognition.abort(); recognition = null; }
        document.getElementById('stt-output').value = '';
        document.getElementById('stt-status').className = 'status';
        document.getElementById('stt-status').innerText = 'Click Start Recording and speak...';
    }

    function startListening() {
        var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
            document.getElementById('stt-status').innerText = '❌ Not supported. Use Chrome.';
            return;
        }
        if (recognition) { recognition.abort(); }
        recognition = new SR();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            var s = document.getElementById('stt-status');
            s.className = 'status listening';
            s.innerText = '● Listening...';
        };
        recognition.onresult = function(e) {
            var final = '', interim = '';
            for (var i = e.resultIndex; i < e.results.length; i++) {
                if (e.results[i].isFinal) final += e.results[i][0].transcript + ' ';
                else interim += e.results[i][0].transcript;
            }
            if (final) document.getElementById('stt-output').value += final;
            var s = document.getElementById('stt-status');
            s.innerText = interim ? '● Listening: ' + interim : '● Listening...';
        };
        recognition.onerror = function(e) {
            var s = document.getElementById('stt-status');
            s.className = 'status';
            s.innerText = '❌ Error: ' + e.error;
        };
        recognition.onend = function() {
            var s = document.getElementById('stt-status');
            s.className = 'status';
            s.innerText = 'Recording stopped.';
        };
        recognition.start();
    }

    function stopListening() {
        if (recognition) { recognition.stop(); recognition = null; }
    }
</script>
""", height=620)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem; color: #4a5568; font-size: 0.8rem;">
    <p>Built with ❤️ using MediaPipe + Scikit-learn + Streamlit</p>
    <p>Signs: Hello • Thank You • Yes • No • Help • Alright • Happy • Sorry • Friends • How Are You</p>
</div>
""", unsafe_allow_html=True)
