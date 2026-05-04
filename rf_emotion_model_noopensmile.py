import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_analysis.csv")

features = [
    "pitch_delta",
    "energy_delta",
    "speech_rate_delta",
    "spectral_centroid_delta",
    "thickness_score_delta",
    "pitch_std_delta",
    "pitch_velocity_delta",
    "pitch_std",
    "pitch_velocity",
    "low_band_ratio",
    "low_mid_ratio",
    "high_band_ratio",
    "thickness_score",
    "energy",
    "speech_rate",
    "spectral_centroid",
    "zcr",
]

X = df[features]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

# cross validation
scores = cross_val_score(model, X, y_encoded, cv=5)
print("Cross Validation Scores:", scores)
print("Mean Accuracy:", round(scores.mean(), 4))
print("Std:", round(scores.std(), 4))

# train
model.fit(X, y_encoded)

# feature importance
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== Feature Importance ===")
print(importance_df.round(4).to_string(index=False))
importance_df.to_csv("rf_librosa_feature_importance.csv", index=False)

# classification report (train reference)
y_pred = model.predict(X)
print("\nClassification Report (train reference)")
print(classification_report(y_encoded, y_pred, target_names=le.classes_))

# confusion matrix 그래프
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_encoded, y_pred,
    display_labels=le.classes_,
    ax=ax,
    colorbar=True
)
ax.set_title("Librosa Feature RandomForest Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_librosa_confusion_matrix.png", dpi=150)
plt.show()
print("Saved → rf_librosa_confusion_matrix.png")