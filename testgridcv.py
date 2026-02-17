# ===============================================
# Test RandomForest with GridSearchCV (Accuracy & F1)
# ===============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ===============================================
# 1️⃣ Data loading & preprocessing (train_only)
# ===============================================
def load_data():
    train = pd.read_csv("data/train_subset.csv")

    # Handle missing values
    train["Arrival Delay in Minutes"] = train["Arrival Delay in Minutes"].fillna(0)
    train.drop(columns=["id"], inplace=True, errors='ignore')

    # ===== Delay Severity Index =====
    train["Delay_Severity"] = (train["Departure Delay in Minutes"] + train["Arrival Delay in Minutes"]) / 2

    # Remove original delay columns
    train.drop(["Departure Delay in Minutes", "Arrival Delay in Minutes", "Delay_Severity", "Delay_Severity_log"],
               axis=1, inplace=True, errors='ignore')

    # ===== Service Quality Score =====
    service_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
    ]
    valid_cols = [col for col in service_cols if col in train.columns]
    train["Service_Quality_Score"] = train[valid_cols].mean(axis=1)

    # Target encoding
    train["satisfaction"] = train["satisfaction"].map({
        "neutral or dissatisfied": 0,
        "satisfied": 1
    })

    # Label Encoding
    label_cols = ["Gender", "Customer Type", "Type of Travel"]
    le = LabelEncoder()
    for col in label_cols:
        train[col] = le.fit_transform(train[col])

    # One-hot Encoding
    train = pd.get_dummies(train, columns=["Class"], drop_first=True)

    # Split
    X = train.drop("satisfaction", axis=1)
    y = train["satisfaction"]

    # Standard scaling (all features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# ===============================================
# 2️⃣ RandomForest + GridSearchCV
# ===============================================
def run_grid_search():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base model
    rf = RandomForestClassifier(random_state=42)

    # GridSearch parameters (twice as dense as before)
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300, 400],
        'max_depth': [6, 8, 10, 12, 15, 20, None],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3, 4]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\n✅ Best Parameters Found:")
    print(grid_search.best_params_)
    print(f"✅ Best F1 Score (CV avg): {grid_search.best_score_:.4f}")

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n=== Final Evaluation on Test Set ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ===============================================
# 3️⃣ Run
# ===============================================
if __name__ == "__main__":
    run_grid_search()
