# 🤟 HoloSign: Universal Gesture Communication System

> *Bridging the silence — one hand gesture at a time.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-00897B?style=flat-square)](https://mediapipe.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🌟 What is HoloSign?

HoloSign is a **real-time multimodal AI communication platform** that translates hand gestures into text and speech — and vice versa — using your webcam. No special hardware. No gloves. No delays.

It was built to break down communication barriers for the deaf and hard-of-hearing community, using the power of computer vision, machine learning, and browser-native speech APIs — all packaged into a sleek, interactive Streamlit dashboard.

🏆 **Winner — Project Expo-2k26**, Andhra Loyola Institute of Engineering & Technology

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖐️ **Real-Time Gesture Recognition** | Detects hand signs live from your webcam using MediaPipe Hand Landmarker |
| 🧠 **ML-Powered Classification** | Random Forest model trained on 126-feature hand landmark vectors (2 hands × 21 landmarks × 3 axes) |
| 🗳️ **Majority Vote Smoothing** | 10-frame prediction buffer with majority voting for rock-solid, flicker-free results |
| 🔊 **Text-to-Speech (TTS)** | Type any text and hear it spoken aloud via the Web Speech API |
| 🎤 **Speech-to-Text (STT)** | Speak naturally and watch your words transcribe in real time |
| 📊 **Live Statistics Dashboard** | Track total predictions, most frequent signs, and unique signs detected |
| 📜 **Prediction History** | Time-stamped log of every sign detected in your session |
| 🎨 **Glassmorphism UI** | Dark-mode dashboard with gradient accents and smooth animations |

---

## 🎯 Supported Signs

```
Hello  •  Thank You  •  Yes  •  No  •  Help
Alright  •  Happy  •  Sorry  •  Friends  •  How Are You
```

---

## 🧱 System Architecture

```
📷 Webcam Input
      │
      ▼
┌─────────────────────────┐
│   MediaPipe Hand        │  ← Detects up to 2 hands
│   Landmarker (.task)    │  ← 21 landmarks per hand (x, y, z)
└───────────┬─────────────┘
            │  126 features (2 hands × 21 × 3)
            ▼
┌─────────────────────────┐
│  Random Forest          │  ← Trained on your own collected data
│  Classifier (.pkl)      │  ← 100 estimators, 80/20 train-test split
└───────────┬─────────────┘
            │  Prediction + Confidence Score
            ▼
┌─────────────────────────┐
│  Majority Vote Buffer   │  ← 10-frame sliding window
│  (deque, maxlen=10)     │  ← Filters noise and jitter
└───────────┬─────────────┘
            │
            ▼
    Streamlit Dashboard
   ┌────────┴────────┐
   │                 │
 Sign Text       Confidence
 History         Statistics
   │
   ▼
TTS / STT (Web Speech API)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A working webcam
- Chrome browser (for Speech-to-Text)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Holosign_keerthi_project.git
cd Holosign_keerthi_project
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe Hand Landmarker Model

Download `hand_landmarker.task` from the [MediaPipe Models page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it inside the `model/` folder:

```
model/
└── hand_landmarker.task   ← place here
```

### 5. Collect Training Data

```bash
python collect_data.py
```

Follow the on-screen prompts. For each sign, a 5-second countdown gives you time to get ready, then **200 samples** are captured automatically. Data is saved to `data/sign_language_data.csv`.

### 6. Train the Model

```bash
python train_model.py
```

This trains a Random Forest classifier and saves:
- `model/sign_language_model.pkl` — the trained model
- `model/label_map.pkl` — the label encoding map

### 7. Launch the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, click **▶ Start Camera**, and start signing!

---

## 📁 Project Structure

```
Holosign_keerthi_project/
│
├── app.py                  # Main Streamlit dashboard
├── collect_data.py         # Webcam-based data collection script
├── train_model.py          # Model training script (Random Forest)
├── requirements.txt        # Python dependencies
│
├── data/
│   └── sign_language_data.csv    # Collected landmark data (generated)
│
└── model/
    ├── hand_landmarker.task      # MediaPipe model (download separately)
    ├── sign_language_model.pkl   # Trained classifier (generated)
    └── label_map.pkl             # Label encoder map (generated)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit, Custom CSS (Glassmorphism) |
| **Computer Vision** | OpenCV, MediaPipe Tasks API |
| **Machine Learning** | scikit-learn (Random Forest Classifier) |
| **Speech** | Web Speech API (TTS + STT via browser) |
| **Data** | NumPy, Pandas |
| **Language** | Python 3.10+ |

---

## 🧠 How the ML Pipeline Works

1. **Data Collection** — `collect_data.py` uses MediaPipe to capture 21 hand landmarks (x, y, z) per hand. With support for 2 hands, each sample is a **126-dimensional feature vector**. Missing hands are zero-padded for consistency.

2. **Training** — `train_model.py` loads the CSV, encodes labels, splits data 80/20, and trains a `RandomForestClassifier` with 100 estimators. A classification report and confusion matrix are printed at the end.

3. **Inference** — `app.py` captures frames in real time, extracts the same 126-feature vector, and runs `model.predict()` + `model.predict_proba()` to get both the predicted sign and a confidence score.

4. **Smoothing** — A 10-frame sliding window buffer applies majority voting, so only stable, confirmed gestures are logged — eliminating flickering from transitional frames.

---

## 📸 Demo

> *Start camera → show a hand gesture → watch HoloSign translate it instantly.*

The dashboard features:
- **Left panel** — Live camera feed with landmark overlay (green skeleton on your hand)
- **Right panel** — Detected sign, confidence bar, session statistics, and prediction history
- **Bottom section** — Text-to-Speech and Speech-to-Text panels

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas to extend HoloSign:

- [ ] Add more signs from the ASL alphabet
- [ ] Integrate a full sentence builder from consecutive signs
- [ ] Export session history as a text/PDF transcript
- [ ] Add multilingual TTS support
- [ ] Train with a deep learning model (CNN/LSTM) for higher accuracy

```bash
# Fork the repo, create a branch, and send a PR
git checkout -b feature/your-feature-name
```

---

## 👩‍💻 Author

**Singamsetti Keerthi**  
B.Tech CSE | AI/ML Enthusiast | Project Expo-2k26 Winner  
📧 keerthisingamsetty093@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/singamsetti-keerthi) • [GitHub](https://github.com/keerthi123345)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

<div align="center">
  <i>Built with ❤️ using MediaPipe + scikit-learn + Streamlit</i><br>
  <i>"Technology is most powerful when it gives voice to those who need it most."</i>
</div>
