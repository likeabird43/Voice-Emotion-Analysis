import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("opensmile_features.csv")

##frequnecy_dynamics_experiment added

freq_df = pd.read_csv("frequency_dynamics_experiment.csv")

df = df.merge(
    freq_df[
        [
            "environment",
            "file",
            "main_freq_velocity_mean",
            "main_freq_jump_ratio",
            "main_freq_smoothness",
        ]
    ],
    on=["environment", "file"],
    how="left"
)

# label
y = df["label"]

# feature 제거
X = df.drop(columns=["label","file","environment"])

# label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

# cross validation
scores = cross_val_score(model, X, y_encoded, cv=5)

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# train
model.fit(X, y_encoded)

## feature importance added

import numpy as np

importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values("importance", ascending=False)

print("\n=== Top 20 Important Features ===")
print(importance_df.head(20))

importance_df.to_csv("rf_feature_importance.csv", index=False)


# prediction
y_pred = model.predict(X)

print("\nClassification Report")
print(classification_report(y_encoded, y_pred, target_names=le.classes_))