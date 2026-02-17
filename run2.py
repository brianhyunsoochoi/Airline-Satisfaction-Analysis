# =============================
# 3️⃣ Hyperparameter tuning (example: RandomForest)
# =============================
print("\n[3️⃣] Hyperparameter tuning (RandomForest example)...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Best RF params:", grid_rf.best_params_)
print("Best RF F1 score:", grid_rf.best_score_)

best_model = grid_rf.best_estimator_

# =============================
# 4️⃣ Final evaluation
# =============================
y_pred_best = best_model.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average='weighted')

print("\n[4️⃣] Final Best Model Evaluation")
print(f"Accuracy: {acc_best:.4f}, F1: {f1_best:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# =============================
# 5️⃣ Feature importance
# =============================
print("\n[5️⃣] Top Feature Importances:")
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))

# Save
importances.head(20).to_csv("results/feature_importance.csv")

# =============================
# 6️⃣ Predict on unknown dataset (Kaggle submission)
# =============================
print("\n[6️⃣] Predicting on unknown dataset...")

unknown_pred = best_model.predict(X_unknown_scaled)
submission = pd.DataFrame({
    "ID": np.arange(len(unknown_pred)),
    "label": unknown_pred
})
submission.to_csv("results/kaggle_submission.csv", index=False)
print("✅ Predictions saved to results/kaggle_submission.csv")

# =============================
# 7️⃣ Save summary
# =============================
results_df.to_csv("results/model_performance.csv", index=False)
joblib.dump(best_model, "results/best_model.pkl")
print("✅ All results saved. Pipeline complete!")
