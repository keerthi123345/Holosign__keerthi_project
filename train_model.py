"""
Sign Language Model Training Script
====================================
Loads collected landmark data and trains a RandomForest classifier.
Saves the trained model and label mapping for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ── Configuration ──────────────────────────────────────────────
DATA_FILE = os.path.join("data", "sign_language_data.csv")
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "sign_language_model.pkl")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100


def main():
    print("\n" + "=" * 50)
    print("  SIGN LANGUAGE MODEL TRAINER")
    print("=" * 50)
    
    # ── Load Data ─────────────────────────────────────────────
    if not os.path.exists(DATA_FILE):
        print(f"\n  ❌ Data file not found: {DATA_FILE}")
        print("  Run collect_data.py first!")
        return
    
    print(f"\n  Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    print(f"  Total samples: {len(df)}")
    print(f"\n  Samples per sign:")
    for sign, count in df["label"].value_counts().items():
        print(f"    • {sign}: {count}")
    
    # ── Prepare Features & Labels ──────────────────────────────
    X = df.drop("label", axis=1).values
    y_raw = df["label"].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    label_map = dict(zip(
        label_encoder.transform(label_encoder.classes_),
        label_encoder.classes_
    ))
    print(f"\n  Label mapping: {label_map}")
    
    # ── Train/Test Split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n  Training set: {len(X_train)} samples")
    print(f"  Test set:     {len(X_test)} samples")
    
    # ── Train Model ────────────────────────────────────────────
    print(f"\n  Training RandomForest (n_estimators={N_ESTIMATORS})...")
    
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # ── Evaluate ───────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  ✅ Training complete!")
    print(f"\n  {'=' * 40}")
    print(f"  ACCURACY: {accuracy * 100:.2f}%")
    print(f"  {'=' * 40}")
    
    print(f"\n  Classification Report:")
    print("  " + "-" * 40)
    
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    report = classification_report(y_test, y_pred, target_names=target_names)
    for line in report.split("\n"):
        print(f"  {line}")
    
    # ── Confusion Matrix ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print("  " + "-" * 40)
    
    # Header
    header = "        " + "  ".join(f"{name[:5]:>5}" for name in target_names)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>5}" for val in row)
        print(f"  {target_names[i][:5]:>5}   {row_str}")
    
    # ── Save Model ─────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Model saved to: {MODEL_FILE}")
    
    with open(LABEL_MAP_FILE, "wb") as f:
        pickle.dump(label_map, f)
    print(f"  Label map saved to: {LABEL_MAP_FILE}")
    
    # ── Feature Importance ─────────────────────────────────────
    importances = model.feature_importances_
    top_n = 10
    top_indices = np.argsort(importances)[-top_n:][::-1]
    
    print(f"\n  Top {top_n} Important Features:")
    print("  " + "-" * 40)
    for idx in top_indices:
        landmark_num = idx // 3
        coord = ["x", "y", "z"][idx % 3]
        print(f"    Landmark {landmark_num} ({coord}): {importances[idx]:.4f}")
    
    print("\n" + "=" * 50)
    print(f"  ✅ All done! Run: streamlit run app.py")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
