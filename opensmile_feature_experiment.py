import os
import pandas as pd
import opensmile

SAMPLES_DIR = "audio/samples"
OUTPUT_CSV = "opensmile_features.csv"


def parse_state(filename: str) -> str:
    filename = filename.lower()
    if "happy" in filename:
        return "happy"
    if "stressed" in filename:
        return "stressed"
    if "tired" in filename:
        return "tired"
    return "unknown"


def main():
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    rows = []

    for env in os.listdir(SAMPLES_DIR):
        env_path = os.path.join(SAMPLES_DIR, env)
        if not os.path.isdir(env_path):
            continue

        print(f"Processing: {env}")

        for file in sorted(os.listdir(env_path)):
            if not file.endswith(".wav"):
                continue

            label = parse_state(file)
            if label == "unknown":
                continue

            file_path = os.path.join(env_path, file)

            feats = smile.process_file(file_path)
            feat_dict = feats.iloc[0].to_dict()

            row = {
                "environment": env,
                "file": file,
                "label": label,
                **feat_dict,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved → {OUTPUT_CSV}")
    print("\n=== Preview ===")
    print(df.head())
    print("\nShape:", df.shape)

    print("\n=== Mean by label (first 10 numeric columns) ===")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    preview_cols = numeric_cols[:10]
    print(df.groupby("label")[preview_cols].mean().round(3))


if __name__ == "__main__":
    main()