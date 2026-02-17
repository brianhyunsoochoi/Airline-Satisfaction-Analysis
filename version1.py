# ===============================================
# K-Fold Evaluation (Accuracy, F1, AUC) + Confusion Matrix + ROC Image Save + Kaggle Submission
# ===============================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_curve, auc,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ===============================================
# 1. Preprocessing
# ===============================================
def preprocess(df, is_train=True):
    df = df.copy()

    # Drop missing rows for training only
    if is_train:
        df = df.dropna(subset=["Arrival Delay in Minutes"])

    df.drop(columns=["id"], inplace=True, errors="ignore")

    # Compute weighted Service Quality Score
    service_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Seat comfort", "Online boarding",
        "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
    ]
    df["Service_Quality_Score"] = (
        df[service_cols].mean(axis=1) * 0.8 + df["Online boarding"] * 0.2
    )

    # Target encoding
    if is_train and "satisfaction" in df.columns:
        df["satisfaction"] = df["satisfaction"].map({
            "neutral or dissatisfied": 0,
            "satisfied": 1
        })

    # Label Encoding
    label_cols = ["Gender", "Customer Type", "Type of Travel"]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot Encoding
    df = pd.get_dummies(df, columns=["Class"], drop_first=True)
    return df


# ===============================================
# 2. Model Definitions
# ===============================================
def get_models():
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=5, random_state=42
        ),
        "GradientBoost": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.05, random_state=42
        ),
        # "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=330,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    }


# ===============================================
# 3. K-Fold Evaluation (Accuracy, F1, AUC)
# =============================================== 
def evaluate_models(X, y, k=1):
    models = get_models()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Create folder to save images
    os.makedirs("results", exist_ok=True)

    results = []
    print(f"\nStarting {k}-Fold evaluation for {len(models)} models...\n")

    for model_idx, (name, model) in enumerate(models.items(), start=1):
        print(f"\n==============================")
        print(f"[{model_idx}/{len(models)}] Evaluating model: {name}")
        print("==============================")

        aucs, accs, f1s = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)

            # Prediction and probability
            ##Does the model has the method predict_proba()
            if hasattr(model, "predict_proba"):
                y_pred_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_prob = model.decision_function(X_test)
                #Min–Max Normalization
                y_pred_prob = (y_pred_prob - y_pred_prob.min()) / (y_pred_prob.max() - y_pred_prob.min())

            y_pred = model.predict(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)

            aucs.append(roc_auc)
            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))

            # ROC Curve Plot
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title(f"{name} - ROC Curve (Fold {fold_idx})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/{name}_ROC_Fold{fold_idx}.png", dpi=300)
            plt.show()

            # Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title(f"{name} - Confusion Matrix (Fold {fold_idx})")
            plt.tight_layout()
            plt.savefig(f"results/{name}_Confusion_Fold{fold_idx}.png", dpi=300)
            plt.show()

        mean_auc = np.mean(aucs)
        mean_acc = np.mean(accs)
        mean_f1 = np.mean(f1s)
        print(f"Done — Accuracy={mean_acc:.4f}, F1={mean_f1:.4f}, AUC={mean_auc:.4f}")

        results.append([name, mean_acc, mean_f1, mean_auc])

    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "AUC"]).sort_values(by="Accuracy", ascending=False)
    print("\n=== Model Performance ===")
    print(df_results)
    return df_results


# ===============================================
# 4. Kaggle Submission
# ===============================================
def generate_submission(best_model_name, scaler, feature_cols, X_scaled, y):
    test_raw = pd.read_csv("data/test_kaggle_features.csv")
    unknown = preprocess(test_raw, is_train=False)

    # Align columns
    missing_cols = set(feature_cols) - set(unknown.columns)
    for col in missing_cols:
        unknown[col] = 0
    unknown = unknown[feature_cols]

    X_unknown_scaled = scaler.transform(unknown)
    model = get_models()[best_model_name]
    model.fit(X_scaled, y)
    y_pred = model.predict(X_unknown_scaled)

    submission = pd.DataFrame({
        "id": test_raw["id"],
        "satisfaction": ["satisfied" if p == 1 else "neutral or dissatisfied" for p in y_pred]
    })
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission file created successfully — using {best_model_name}")

# ===============================================
# 5. Run
# ===============================================
if __name__ == "__main__":
    train = pd.read_csv("data/train_subset.csv")
    train = preprocess(train, is_train=True)

    X = train.drop("satisfaction", axis=1)
    y = train["satisfaction"].values  # Convert to NumPy array

    #fit: checks unique value
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = get_models()
    print("\nModels to evaluate:")
    for i, name in enumerate(models.keys(), start=1):
        print(f"{i}. {name}")
    print("=====================================\n")

    # Run K-Fold evaluation
    results = evaluate_models(X_scaled, y, k=2)

    best_model_name = results.iloc[0]["Model"]
    print(f"\nBest Model by Accuracy: {best_model_name}")

    generate_submission(best_model_name, scaler, X.columns, X_scaled, y)
