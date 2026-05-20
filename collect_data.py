"""
Sign Language Data Collection Script
=====================================
Captures hand landmarks using MediaPipe Tasks API and saves to CSV
for training a sign language recognition model.

Supports up to 2 hands. Feature vector = 126 values (21 landmarks x 3 x 2 hands).
Missing hand is padded with zeros for consistent feature size.

Signs: Hello, Thank You, Yes, No, Help, Alright, Happy, Sorry, Friends, How Are You
"""

import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# ── Configuration ──────────────────────────────────────────────
SIGNS = ["Hello", "Thank You", "Yes", "No", "Help", "Alright", "Happy", "Sorry", "Friends", "How Are You"]
SAMPLES_PER_SIGN = 200
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "sign_language_data.csv")
COUNTDOWN_SECONDS = 5
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3  # x, y, z
NUM_HANDS = 2
TOTAL_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK * NUM_HANDS  # 126
MODEL_ASSET = os.path.join("model", "hand_landmarker.task")

# Signs that require both hands
TWO_HAND_SIGNS = {"Alright", "Happy", "How Are You", "Friends"}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def create_header():
    """Create CSV header for 2-hand landmarks."""
    header = []
    for hand in ["left", "right"]:
        for i in range(NUM_LANDMARKS):
            header.extend([f"{hand}_x_{i}", f"{hand}_y_{i}", f"{hand}_z_{i}"])
    header.append("label")
    return header


def extract_landmarks(hand_landmarks):
    """Extract flattened x,y,z from a single hand's landmarks."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def extract_two_hand_features(result):
    """
    Build a fixed 126-feature vector from up to 2 detected hands.
    Hands are sorted by handedness: left hand first, right hand second.
    Missing hand slots are filled with zeros.
    """
    empty = [0.0] * (NUM_LANDMARKS * FEATURES_PER_LANDMARK)
    left_coords = empty[:]
    right_coords = empty[:]

    for i, hand_lms in enumerate(result.hand_landmarks):
        # handedness label for this hand
        label = result.handedness[i][0].category_name.lower()  # "left" or "right"
        coords = extract_landmarks(hand_lms)
        if label == "left":
            left_coords = coords
        else:
            right_coords = coords

    return left_coords + right_coords


def draw_hand_landmarks(frame, hand_landmarks, color=(0, 255, 128)):
    """Draw hand landmarks and connections on the frame."""
    h, w = frame.shape[:2]
    for start_idx, end_idx in HAND_CONNECTIONS:
        s = hand_landmarks[start_idx]
        e = hand_landmarks[end_idx]
        cv2.line(frame, (int(s.x * w), int(s.y * h)),
                 (int(e.x * w), int(e.y * h)), color, 2, cv2.LINE_AA)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 128), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)


def draw_info_overlay(frame, text, sub_text="", color=(0, 255, 0)):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)
    if sub_text:
        cv2.putText(frame, sub_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)


def draw_progress_bar(frame, current, total, label=""):
    h, w = frame.shape[:2]
    bar_height = 30
    bar_y = h - bar_height - 10
    bar_x = 20
    bar_width = w - 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    fill_width = int((current / total) * bar_width)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                  (0, 200, 100), -1)
    cv2.putText(frame, f"{label} {current}/{total}", (bar_x + 10, bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hand_status(frame, num_hands_detected, need_two):
    h, w = frame.shape[:2]
    if need_two:
        color = (0, 255, 0) if num_hands_detected >= 2 else (0, 165, 255)
    else:
        color = (0, 255, 0) if num_hands_detected >= 1 else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)


def draw_countdown(frame, seconds_left):
    h, w = frame.shape[:2]
    text = str(seconds_left)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 5.0, 8
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    tx = (w - text_size[0]) // 2
    ty = (h + text_size[1]) // 2
    cv2.putText(frame, text, (tx + 3, ty + 3), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, scale, (0, 200, 255), thickness, cv2.LINE_AA)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(MODEL_ASSET):
        print(f"ERROR: Hand landmarker model not found at {MODEL_ASSET}")
        return

    # Initialize CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        csv.writer(f).writerow(create_header())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # num_hands=2 to support two-hand signs
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_ASSET),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    print("\n" + "=" * 55)
    print("  SIGN LANGUAGE DATA COLLECTOR  (2-hand support)")
    print("=" * 55)
    print(f"\n  Signs: {', '.join(SIGNS)}")
    print(f"  Samples per sign: {SAMPLES_PER_SIGN}")
    print(f"  Feature vector: {TOTAL_FEATURES} (2 hands x 21 landmarks x 3)")
    print(f"  Two-hand signs: {', '.join(TWO_HAND_SIGNS)}")
    print("\n  Press 'Q' at any time to quit")
    print("=" * 55 + "\n")

    total_signs = len(SIGNS)

    for sign_idx, sign in enumerate(SIGNS):
        need_two = sign in TWO_HAND_SIGNS
        hand_hint = "USE BOTH HANDS" if need_two else "one hand"
        print(f"\n>> '{sign}' ({sign_idx + 1}/{total_signs}) — {hand_hint}")

        # ── Countdown ──
        countdown_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            elapsed = time.time() - countdown_start
            seconds_left = COUNTDOWN_SECONDS - int(elapsed)
            if seconds_left <= 0:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            for lms in result.hand_landmarks:
                draw_hand_landmarks(frame, lms)

            draw_hand_status(frame, len(result.hand_landmarks), need_two)
            hint = f"[{hand_hint}] Get ready! Starting in {seconds_left}s..."
            draw_info_overlay(frame, f"Next: '{sign}'", hint, (0, 200, 255))
            draw_countdown(frame, seconds_left)

            cv2.imshow("Sign Language Data Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); landmarker.close()
                print("\n>> Cancelled."); return

        # ── Collection ──
        collected = 0
        samples = []
        print(f"   Collecting '{sign}'...")

        while collected < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            num_detected = len(result.hand_landmarks)
            # For two-hand signs require at least 2 hands; single-hand signs need at least 1
            ready = (num_detected >= 2) if need_two else (num_detected >= 1)

            for lms in result.hand_landmarks:
                draw_hand_landmarks(frame, lms)

            if ready:
                features = extract_two_hand_features(result)
                features.append(sign)
                samples.append(features)
                collected += 1

            draw_hand_status(frame, num_detected, need_two)
            if need_two:
                status = "RECORDING (2 hands)" if ready else f"NEED BOTH HANDS ({num_detected}/2)"
            else:
                status = "RECORDING" if ready else "NO HAND DETECTED"
            color = (0, 255, 0) if ready else (0, 0, 255)
            draw_info_overlay(frame, f"Sign: '{sign}' - {status}",
                              f"Sign {sign_idx + 1}/{total_signs}", color)
            draw_progress_bar(frame, collected, SAMPLES_PER_SIGN, "Samples:")

            cv2.imshow("Sign Language Data Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); landmarker.close()
                print("\n>> Cancelled."); return

        with open(OUTPUT_FILE, "a", newline="") as f:
            csv.writer(f).writerows(samples)
        print(f"   Done: {collected} samples for '{sign}'")

    print("\n" + "=" * 55)
    print(f"  Collection complete! {total_signs * SAMPLES_PER_SIGN} total samples")
    print(f"  Saved to: {OUTPUT_FILE}")
    print("=" * 55)

    for _ in range(90):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        draw_info_overlay(frame, "Collection Complete! Press Q to exit.", color=(0, 255, 0))
        cv2.imshow("Sign Language Data Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
