# ===============================================
# Full Stacking Pipeline — Consistent with XGBoost Preprocessing
# ===============================================

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ===============================================
# 1️⃣ Shared preprocessing function (same as XGBoost version)
# ===============================================
def preprocess(df, is_train=True):
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)
    df.drop(columns=["id"], inplace=True, errors="ignore")

    # Delay Severity
    if "Departure Delay in Minutes" in df.columns:
        df["Delay_Severity"] = (df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]) / 2

  # Service Quality Score (includes 8:2 weight for Online boarding)
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

    # Label Encoding
    label_cols = ["Gender", "Customer Type", "Type of Travel"]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot Encoding
    df = pd.get_dummies(df, columns=["Class"], drop_first=True)

    # Target Encoding (train only)
    if is_train and "satisfaction" in df.columns:
        df["satisfaction"] = df["satisfaction"].map({
            "neutral or dissatisfied": 0,
            "satisfied": 1
        })

    return df


# ===============================================
# 2️⃣ Data loading and preprocessing
# ===============================================
def load_data():
    print("[1️⃣] Loading and preprocessing data...")
    start = time.time()

    train = pd.read_csv("data/train_subset.csv")
    train = preprocess(train, is_train=True)

    X = train.drop("satisfaction", axis=1)
    y = train["satisfaction"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"✅ Preprocessing complete. Shape: {X_scaled.shape}, Time: {time.time() - start:.2f}s\n")
    return X_scaled, y


# ===============================================
# 3️⃣ Run stacking
# ===============================================
def run_stacking():
    print("[2️⃣] Splitting train/test data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data split done. Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    # Base models
    print("[3️⃣] Initializing base models...")
    base_models = [
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)),
        ("gb", GradientBoostingClassifier(
            n_estimators=150, random_state=42)),
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ]
    print("✅ Base models ready.\n")

    # Meta model
    print("[4️⃣] Initializing meta model (Logistic Regression)...")
    meta_model = LogisticRegression(max_iter=500, random_state=42)

    # Stacking model
    print("[5️⃣] Building stacking classifier (5-fold CV)...")
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    print("[6️⃣] Training stacking model...")
    start_train = time.time()
    stack_model.fit(X_train, y_train)
    print(f"✅ Training complete. Time: {time.time() - start_train:.2f}s\n")

    print("[7️⃣] Evaluating model...")
    y_pred = stack_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n✅ Stacking Model Evaluation")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\n🏁 All steps completed successfully!")


# ===============================================
# 4️⃣ Run
# ===============================================
if __name__ == "__main__":
    run_stacking()
