import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

DATA_CSV = "dataset_analysis.csv"
OUT_DIR = "plots"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_CSV)

    # 사용할 feature들
    feature_cols = [
        "pitch_delta",
        "energy_delta",
        "speech_rate_delta",
        "spectral_centroid_delta",
        "thickness_score_delta",
        "pitch_std_delta",
        "pitch_velocity_delta",
        "pitch_std",
        "pitch_velocity",
        "spectral_centroid",
        "spectral_rolloff",
        "spectral_bandwidth",
        "zcr",
        "low_band_ratio",
        "low_mid_ratio",
        "high_band_ratio",
        "thickness_score",
    ]

    # 혹시 빠진 컬럼이 있으면 자동 제외
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["label"].copy()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
    )

    # 교차검증 accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="accuracy")

    print("=== Cross-Validation Accuracy ===")
    print("Fold scores:", [round(s, 4) for s in scores])
    print("Mean accuracy:", round(scores.mean(), 4))
    print("Std:", round(scores.std(), 4))

    # 전체 데이터로 학습 후 중요도 추출
    clf.fit(X, y_encoded)
    importances = clf.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n=== Feature Importance Ranking ===")
    print(importance_df.to_string(index=False))

    importance_df.to_csv("feature_importance_ranking.csv", index=False)
    print("\nSaved → feature_importance_ranking.csv")

    # 그래프 저장
    plt.figure(figsize=(8, 6))
    top_n = min(12, len(importance_df))
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=200)
    plt.close()

    print(f"Saved → {os.path.join(OUT_DIR, 'feature_importance.png')}")


if __name__ == "__main__":
    main()