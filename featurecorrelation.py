import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================================
# Correlation analysis
# ===============================================
def show_feature_correlation():
    train = pd.read_csv("data/train_subset.csv")
    train["Arrival Delay in Minutes"] = train["Arrival Delay in Minutes"].fillna(0)
    train.drop(columns=["id"], inplace=True, errors='ignore')

    # Create new variables
    train["Delay_Severity"] = (train["Departure Delay in Minutes"] + train["Arrival Delay in Minutes"]) / 2
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

    # Select numeric columns only
    numeric_cols = train.select_dtypes(include=[np.number])

    # Compute correlation coefficients
    corr = numeric_cols.corr()

    # Print TOP 10 features most correlated with satisfaction
    top_corr = corr["satisfaction"].sort_values(ascending=False).head(10)
    print("\n📈 TOP 10 correlations with satisfaction\n")
    print(top_corr)

    # Visualization (heatmap)
    plt.figure(figsize=(30, 6))
    sns.heatmap(corr[["satisfaction"]].sort_values(by="satisfaction", ascending=False), 
                annot=True, cmap="RdYlGn", center=0)
    plt.title("Feature Correlation with Satisfaction")
    plt.show()


# Run
if __name__ == "__main__":
    show_feature_correlation()
