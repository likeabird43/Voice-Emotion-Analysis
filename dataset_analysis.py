import os
import re
import numpy as np
import pandas as pd

from voice_analysis import extract_features, compute_feature_delta, infer_state
from frequency_dynamics_experiment import extract_main_frequency_track

BASELINE_DIR = "audio/baselines"
SAMPLES_DIR = "audio/samples"


def compute_baseline(environment):
    path = os.path.join(BASELINE_DIR, environment)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline folder not found: {path}")

    wav_files = [f for f in os.listdir(path) if f.endswith(".wav") and "_cleaned" not in f]
    if not wav_files:
        raise ValueError(f"No WAV files found in: {path}")

    baseline_features = []
    baseline_freq_features = []

    for file in wav_files:
        file_path = os.path.join(path, file)

        features = extract_features(file_path)
        freq_features = extract_main_frequency_track(file_path)

        baseline_features.append(features)
        baseline_freq_features.append(freq_features)

    return {
        "pitch": float(np.median([f["pitch"] for f in baseline_features])),
        "pitch_std": float(np.mean([f["pitch_std"] for f in baseline_features])),
        "pitch_velocity": float(np.mean([f["pitch_velocity"] for f in baseline_features])),
        "energy": float(np.mean([f["energy"] for f in baseline_features])),
        "speech_rate": float(np.mean([f["speech_rate"] for f in baseline_features])),
        "spectral_centroid": float(np.mean([f["spectral_centroid"] for f in baseline_features])),
        "duration": float(np.mean([f["duration"] for f in baseline_features])),
        "spectral_rolloff": float(np.mean([f["spectral_rolloff"] for f in baseline_features])),
        "spectral_bandwidth": float(np.mean([f["spectral_bandwidth"] for f in baseline_features])),
        "zcr": float(np.mean([f["zcr"] for f in baseline_features])),
        "low_band_ratio": float(np.mean([f["low_band_ratio"] for f in baseline_features])),
        "low_mid_ratio": float(np.mean([f["low_mid_ratio"] for f in baseline_features])),
        "high_band_ratio": float(np.mean([f["high_band_ratio"] for f in baseline_features])),
        "thickness_score": float(np.mean([f["thickness_score"] for f in baseline_features])),

        # Frequency dynamics baseline
        "main_freq_velocity_mean": float(np.mean([f["main_freq_velocity_mean"] for f in baseline_freq_features])),
        "main_freq_jump_ratio": float(np.mean([f["main_freq_jump_ratio"] for f in baseline_freq_features])),
        "main_freq_smoothness": float(np.mean([f["main_freq_smoothness"] for f in baseline_freq_features])),
    }


def parse_state(filename):
    match = re.search(r"(happy|tired|stressed)", filename.lower())
    return match.group(1) if match else "unknown"


def analyze_dataset():
    rows = []

    environments = [
        env for env in os.listdir(SAMPLES_DIR)
        if os.path.isdir(os.path.join(SAMPLES_DIR, env))
    ]

    if not environments:
        raise FileNotFoundError(f"No environment folders found in: {SAMPLES_DIR}")

    for env in environments:
        print(f"Processing: {env}")
        baseline = compute_baseline(env)
        sample_path = os.path.join(SAMPLES_DIR, env)

        for file in sorted(os.listdir(sample_path)):
            if not file.endswith(".wav"):
                continue
            if "_cleaned" in file:
                continue

            label = parse_state(file)
            if label == "unknown":
                print(f"  Warning: skipping '{file}'")
                continue

            file_path = os.path.join(sample_path, file)

            features = extract_features(file_path)
            freq_features = extract_main_frequency_track(file_path)

            # frequency dynamics를 features에 합치기
            features.update({
                "main_freq_velocity_mean": freq_features["main_freq_velocity_mean"],
                "main_freq_jump_ratio": freq_features["main_freq_jump_ratio"],
                "main_freq_smoothness": freq_features["main_freq_smoothness"],
            })

            deltas = compute_feature_delta(features, baseline)
            inferred_state, confidence, _ = infer_state(features, baseline=baseline)

            row = {
                "environment": env,
                "file": file,
                "label": label,
                "inferred_state": inferred_state,
                "confidence": confidence,
                "correct": label == inferred_state,

                "pitch": features["pitch"],
                "pitch_std": features["pitch_std"],
                "pitch_velocity": features["pitch_velocity"],
                "energy": features["energy"],
                "speech_rate": features["speech_rate"],
                "spectral_centroid": features["spectral_centroid"],
                "duration": features["duration"],

                "spectral_rolloff": features["spectral_rolloff"],
                "spectral_bandwidth": features["spectral_bandwidth"],
                "zcr": features["zcr"],
                "low_band_ratio": features["low_band_ratio"],
                "low_mid_ratio": features["low_mid_ratio"],
                "high_band_ratio": features["high_band_ratio"],
                "thickness_score": features["thickness_score"],

                # Frequency dynamics raw features
                "main_freq_velocity_mean": features["main_freq_velocity_mean"],
                "main_freq_jump_ratio": features["main_freq_jump_ratio"],
                "main_freq_smoothness": features["main_freq_smoothness"],

                **deltas,
            }
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = analyze_dataset()

    print("\n=== Overall Accuracy ===")
    print(round(df["correct"].mean(), 4))

    print("\n=== Accuracy by State ===")
    print(df.groupby("label")["correct"].mean().round(2))

    print("\n=== Accuracy by Environment ===")
    print(df.groupby("environment")["correct"].mean().round(2))

    print("\n=== Pitch Velocity by State ===")
    print(df.groupby("label")["pitch_velocity"].mean().round(2))

    print("\n=== Mean Features by State ===")
    print(
        df.groupby("label")[[
            "pitch_delta",
            "energy_delta",
            "speech_rate_delta",
            "spectral_centroid_delta",
            "thickness_score_delta",
            "pitch_std_delta",
            "pitch_velocity_delta",
            "main_freq_velocity_mean_delta",
            "main_freq_jump_ratio_delta",
            "main_freq_smoothness_delta",
        ]].mean().round(4)
    )

    print("\n=== Mean Acoustic Features by State ===")
    print(
        df.groupby("label")[[
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
            "main_freq_velocity_mean",
            "main_freq_jump_ratio",
            "main_freq_smoothness",
        ]].mean().round(4)
    )

    df.to_csv("dataset_analysis.csv", index=False)
    print("\nSaved → dataset_analysis.csv")