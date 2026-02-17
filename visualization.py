# viz_1A.py
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Settings ----------
EXCEL_PATH = "32130_AT2_25716438.xlsx"   # Place this in the same folder
CSV_PATH   = "32130_AT2_25716438.csv"    # Ignored if not present
SHEET_NAME = 0                            # Use 0 for the first sheet instead of "Sheet1"
OUT_DIR    = "figs_1A"

# ---------- Utilities ----------
def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\s\-]", "_", str(s))
    return s.replace(" ", "_")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------- Data loading ----------
def load_dataset():
    if os.path.exists(EXCEL_PATH):
        try:
            return pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
        except Exception:
            pass
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError("Dataset not found. Put Excel/CSV next to this script.")

df = load_dataset()
ensure_dir(OUT_DIR)

# ---------- Column groups ----------
nominal_cols = ["Gender", "Customer Type", "Type of Travel"]
ordinal_service_cols = [
    "Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking",
    "Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment",
    "On-board service","Leg room service","Baggage handling","Checkin service","Inflight service","Cleanliness"
]
ratio_cols = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

satisfaction_col = next((c for c in ["satisfaction","Satisfaction"] if c in df.columns), None)
class_col = "Class" if "Class" in df.columns else None

saved = []

# ---------- Nominal: bar charts ----------
for col in nominal_cols:
    if col not in df.columns: continue
    counts = df[col].value_counts(dropna=False)
    fig = plt.figure()
    counts.plot(kind="bar")
    plt.title(f"Frequency of {col}"); plt.xlabel(col); plt.ylabel("Count")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"nominal_{safe_name(col)}_bar.png")
    plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

# ---------- Ordinal: Class ----------
if class_col:
    counts = df[class_col].value_counts(dropna=False)
    fig = plt.figure()
    counts.plot(kind="bar")
    plt.title(f"Frequency of {class_col}"); plt.xlabel(class_col); plt.ylabel("Count")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"ordinal_{safe_name(class_col)}_bar.png")
    plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

# ---------- Ordinal: Satisfaction ----------
if satisfaction_col:
    counts = df[satisfaction_col].value_counts(dropna=False)
    fig = plt.figure()
    counts.plot(kind="bar")
    plt.title(f"Frequency of {satisfaction_col}"); plt.xlabel(satisfaction_col); plt.ylabel("Count")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"ordinal_{safe_name(satisfaction_col)}_bar.png")
    plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

# ---------- Ordinal: service Likert scale (bar + box) ----------
for col in ordinal_service_cols:
    if col not in df.columns: continue

    # Frequency bar chart
    counts = df[col].value_counts(dropna=False).sort_index()
    fig = plt.figure()
    counts.plot(kind="bar")
    plt.title(f"Frequency of {col}"); plt.xlabel(col); plt.ylabel("Count")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"ordinal_{safe_name(col)}_bar.png")
    plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # Boxplot (for Wi-Fi, exclude 0=N/A)
    series = df[col]
    if col == "Inflight wifi service":
        series = series.replace(0, np.nan)
    series = series.dropna()
    if series.shape[0] > 0 and pd.api.types.is_numeric_dtype(series):
        fig = plt.figure()
        plt.boxplot(series, vert=True, showfliers=True)
        plt.title(f"Boxplot of {col}"); plt.ylabel(col)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, f"ordinal_{safe_name(col)}_box.png")
        plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

# ---------- Ratio: histogram + box ----------
for col in ratio_cols:
    if col not in df.columns: continue
    s = df[col].dropna()
    if s.shape[0] == 0: continue

    fig = plt.figure()
    plt.hist(s, bins=30)
    plt.title(f"Histogram of {col}"); plt.xlabel(col); plt.ylabel("Frequency")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"ratio_{safe_name(col)}_hist.png")
    plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    fig = plt.figure()
    plt.boxplot(s, vert=True, showfliers=True)
    plt.title(f"Boxplot of {col}"); plt.ylabel(col)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"ratio_{safe_name(col)}_box.png")
    plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

# ---------- Multivariate: scatter & grouped box ----------
# Departure vs Arrival Delay
if ("Departure Delay in Minutes" in df.columns) and ("Arrival Delay in Minutes" in df.columns):
    s1, s2 = df["Departure Delay in Minutes"], df["Arrival Delay in Minutes"]
    m = s1.notna() & s2.notna()
    if m.sum() > 0:
        fig = plt.figure()
        plt.scatter(s1[m], s2[m], s=5)
        plt.title("Scatter: Departure vs Arrival Delay")
        plt.xlabel("Departure Delay in Minutes"); plt.ylabel("Arrival Delay in Minutes")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "scatter_departure_vs_arrival_delay.png")
        plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

# Age vs Flight Distance
if ("Age" in df.columns) and ("Flight Distance" in df.columns):
    s1, s2 = df["Age"], df["Flight Distance"]
    m = s1.notna() & s2.notna()
    if m.sum() > 0:
        fig = plt.figure()
        plt.scatter(s1[m], s2[m], s=5)
        plt.title("Scatter: Age vs Flight Distance")
        plt.xlabel("Age"); plt.ylabel("Flight Distance")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "scatter_age_vs_flight_distance.png")
        plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

# Departure Delay by Satisfaction (grouped box)
if satisfaction_col and ("Departure Delay in Minutes" in df.columns):
    order_map = {"neutral or dissatisfied": 0, "satisfied": 1}
    s_map = df[satisfaction_col].map(order_map)
    g0 = df.loc[s_map == 0, "Departure Delay in Minutes"].dropna()
    g1 = df.loc[s_map == 1, "Departure Delay in Minutes"].dropna()
    if len(g0) > 0 and len(g1) > 0:
        fig = plt.figure()
        plt.boxplot([g0, g1], labels=["Neutral/Dissatisfied", "Satisfied"], showfliers=True)
        plt.title("Departure Delay by Satisfaction"); plt.ylabel("Departure Delay in Minutes")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "box_departure_delay_by_satisfaction.png")
        plt.savefig(p, dpi=150); plt.close(fig); saved.append(p)

# ---------- Print saved file list ----------
print(f"Saved {len(saved)} figures to: {OUT_DIR}")
for p in saved:
    print(p)
