# ===============================================
# A3 Full ML Pipeline: preprocessing → training → evaluation → tuning → prediction
# ===============================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# =============================
# 1️⃣ Data loading & preprocessing
# =============================
print("[1️⃣] Loading and preprocessing data...")

train = pd.read_csv("data/train_subset.csv")
unknown = pd.read_csv("data/test_kaggle_features.csv")

# Handle missing values
train["Arrival Delay in Minutes"] = train["Arrival Delay in Minutes"].fillna(0)
unknown["Arrival Delay in Minutes"] = unknown["Arrival Delay in Minutes"].fillna(0)

# Remove ID
train.drop(columns=["id"], inplace=True)
unknown.drop(columns=["id"], inplace=True, errors='ignore')

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
    if col in unknown.columns:
        unknown[col] = le.transform(unknown[col])

# One-hot Encoding
train = pd.get_dummies(train, columns=["Class"], drop_first=True)
unknown = pd.get_dummies(unknown, columns=["Class"], drop_first=True)

# Align to the same columns
missing_cols = set(train.columns) - set(unknown.columns)
missing_cols.discard("satisfaction")
for col in missing_cols:
    unknown[col] = 0
unknown = unknown[train.drop("satisfaction", axis=1).columns]

# Scaling
scaler = StandardScaler()
X = train.drop("satisfaction", axis=1)
y = train["satisfaction"]
X_scaled = scaler.fit_transform(X)
X_unknown_scaled = scaler.transform(unknown)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Preprocessing done. Train size:", X_train.shape)

# =============================
# 2️⃣ Train & evaluate multiple models
# =============================
print("\n[2️⃣] Training multiple classifiers...")

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoost": GradientBoostingClassifier(random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results.append([name, acc, f1])
    print(f"{name:15s} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"]).sort_values(by="F1", ascending=False)
print("\n=== Model Performance ===")
print(results_df)

