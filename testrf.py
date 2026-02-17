# ===============================================
# Full RandomForest Pipeline — Train + Predict + Kaggle Submission
# ===============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# ===============================================
# 1️⃣ Shared preprocessing function
# ===============================================
def preprocess(df, is_train=True):
    df = df.copy()

    # 🧹 Train: remove rows with missing values / Test: keep missing values
    if is_train:
        df = df.dropna(subset=["Arrival Delay in Minutes"])

    df.drop(columns=["id"], inplace=True, errors="ignore")

    # ===== Delay Severity (NaN allowed)
    if "Departure Delay in Minutes" in df.columns:
        df["Delay_Severity"] = (
            df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
        ) / 2

    # ===== Service Quality Score (weighted for Online boarding)
    service_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
    ]
    df["Service_Quality_Score"] = (
        df[service_cols].mean(axis=1) * 0.8 +
        df["Online boarding"] * 0.2
    )

    # ===== Label Encoding
    label_cols = ["Gender", "Customer Type", "Type of Travel"]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # ===== One-hot Encoding
    df = pd.get_dummies(df, columns=["Class"], drop_first=True)

    # ===== Target Encoding (train only)
    if is_train and "satisfaction" in df.columns:
        df["satisfaction"] = df["satisfaction"].map({
            "neutral or dissatisfied": 0,
            "satisfied": 1
        })

    return df


# ===============================================
# 2️⃣ Training + evaluation
# ===============================================
def train_and_evaluate():
    train = pd.read_csv("data/train_subset.csv")
    train = preprocess(train, is_train=True)

    X = train.drop("satisfaction", axis=1)
    y = train["satisfaction"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ RandomForest (uses the same preprocessing as XGBoost)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[🔍 RandomForest Performance]")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler, X.columns


# ===============================================
# 3️⃣ Generate prediction file for Kaggle submission
# ===============================================
def generate_submission(model, scaler, feature_cols):
    test_raw = pd.read_csv("data/test_kaggle_features.csv")
    unknown = preprocess(test_raw.copy(), is_train=False)

    # Match feature columns
    missing_cols = set(feature_cols) - set(unknown.columns)
    for col in missing_cols:
        unknown[col] = 0
    unknown = unknown[feature_cols]

    # Temporarily convert NaN to 0 (for scaler input)
    X_unknown_scaled = scaler.transform(unknown.fillna(0))
    y_pred = model.predict(X_unknown_scaled)

    y_pred_label = ["satisfied" if p == 1 else "neutral or dissatisfied" for p in y_pred]

    submission = pd.DataFrame({
        "id": test_raw["id"],
        "satisfaction": y_pred_label
    })

    submission.to_csv("submission.csv", index=False)
    print("\n✅ submission.csv created successfully! (Row count preserved)")
    print(f"Total rows: {len(submission)}")
    print(submission.head())


# ===============================================
# 4️⃣ Run
# ===============================================
if __name__ == "__main__":
    model, scaler, feature_cols = train_and_evaluate()
    generate_submission(model, scaler, feature_cols)
