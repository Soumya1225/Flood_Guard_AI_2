from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "Datasets" / "processed_data"
MODEL_DIR = BASE / "models"

INPUT = PROCESSED / "ml_training_data.csv"
MODEL_OUTPUT = MODEL_DIR / "flood_guard_random_forest.pkl"

print("=" * 70)
print("       FLOOD GUARD AI - RANDOM FOREST TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading ML dataset...")

df = pd.read_csv(INPUT)

print("   Shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES
# ============================================================

print("\n[2] Preparing features...")

feature_columns = [
    "water_level_mean_m",
    "water_level_max_m",
    "water_level_min_m",
    "water_level_change_m",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "water_level_3day_mean_m",
    "water_level_7day_mean_m",
    "month",
    "day_of_year"
]

target_column = "high_water_event"

X = df[feature_columns]
y = df[target_column]

print("   Features:", len(feature_columns))
print("   Target:", target_column)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\n[3] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("   Training rows:", len(X_train))
print("   Testing rows:", len(X_test))


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

print("\n[4] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("   Training complete.")


# ============================================================
# 5. PREDICTIONS
# ============================================================

print("\n[5] Evaluating model...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

try:
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )
except ValueError:
    roc_auc = 0.0


print("\n" + "-" * 70)
print("MODEL PERFORMANCE")
print("-" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "High Water"
        ],
        zero_division=0
    )
)


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance:")

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

for _, row in importance.iterrows():
    print(
        f"   {row['feature']:<30} "
        f"{row['importance']:.4f}"
    )


# ============================================================
# 9. SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_OUTPUT
)

print("\nModel saved:")
print(MODEL_OUTPUT)

print("\n" + "=" * 70)
print("RANDOM FOREST TRAINING COMPLETE")
print("=" * 70)