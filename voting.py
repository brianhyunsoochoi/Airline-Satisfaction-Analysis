# ===============================================
# Voting Ensemble — Decision Tree + KNN + SVM
# (Same Preprocessing and Evaluation Flow)
# ===============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import time

# ===============================================
# 1️⃣ Preprocessing function (same as before)
# ===============================================
def preprocess(df, is_train=True):
    df = df.copy()
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)
    df.drop(columns=["id"], inplace=True, errors="ignore")

    service_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Seat comfort", "Online boarding",
        "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
    ]
    df["Service_Quality_Score"] = (
        df[service_cols].mean(axis=1) * 0.8 + df["Online boarding"] * 0.2
    )

    if is_train and "satisfaction" in df.columns:
        df["satisfaction"] = df["satisfaction"].map({
            "neutral or dissatisfied": 0,
            "satisfied": 1
        })

    label_cols = ["Gender", "Customer Type", "Type of Travel"]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=["Class"], drop_first=True)
    return df


# ===============================================
# 2️⃣ Define Voting model
# ===============================================
def get_voting_model():
    dt = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=7)
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)

    # Soft Voting (probability-based combination)
    voting_clf = VotingClassifier(
        estimators=[("dt", dt), ("knn", knn), ("svm", svm)],
        voting="soft"
    )
    return voting_clf


# ===============================================
# 3️⃣ Training & evaluation
# ===============================================
def run_voting():
    print("[1️⃣] Loading and preprocessing data...")
    start = time.time()
    train = pd.read_csv("data/train_subset.csv")
    train = preprocess(train, is_train=True)

    X = train.drop("satisfaction", axis=1)
    y = train["satisfaction"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✅ Preprocessing complete. Shape: {X_scaled.shape}, Time: {time.time() - start:.2f}s")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[2️⃣] Training Voting Ensemble (DecisionTree + KNN + SVM)...")
    model = get_voting_model()
    model.fit(X_train, y_train)
    print("✅ Training complete.")

    print("\n[3️⃣] Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n📊 Results — Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ===============================================
# 4️⃣ Run
# ===============================================
if __name__ == "__main__":
    run_voting()
