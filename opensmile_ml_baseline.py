import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

INPUT_CSV = "opensmile_features.csv"
PLOT_DIR = "plots"


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # metadata 제외
    meta_cols = {"environment", "file", "label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy()
    y = df["label"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # CV accuracy
    scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")

    print("=== Cross-Validation Accuracy ===")
    print("Fold scores:", [round(float(s), 4) for s in scores])
    print("Mean accuracy:", round(float(scores.mean()), 4))
    print("Std:", round(float(scores.std()), 4))

    # CV predictions for confusion matrix
    y_pred = cross_val_predict(clf, X, y_enc, cv=cv)

    print("\n=== Classification Report ===")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")
    plt.title("openSMILE RandomForest Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(PLOT_DIR, "opensmile_rf_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"Saved → {cm_path}")

    # fit full model for feature importance
    clf.fit(X, y_enc)
    importances = clf.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    out_csv = "opensmile_feature_importance.csv"
    importance_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    print("\n=== Top 20 Features ===")
    print(importance_df.head(20).to_string(index=False))

    # plot top 15
    top_n = min(15, len(importance_df))
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.title("Top openSMILE Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(PLOT_DIR, "opensmile_rf_feature_importance.png")
    plt.savefig(fi_path, dpi=200)
    plt.close()

    print(f"Saved → {fi_path}")


if __name__ == "__main__":
    main()