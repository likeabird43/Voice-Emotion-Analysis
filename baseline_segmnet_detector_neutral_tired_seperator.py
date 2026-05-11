import os
import numpy as np
import pandas as pd
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

MIXED_FILE = "audio/mixed/outside_mixed_v1.wav"
OUTPUT_DIR = "analysis_outputs"


# Janet님의 mixed speech timeline
# (start_sec, end_sec, label)
TIMELINE = [
    (0, 28, "neutral"),
    (29, 46, "happy"),
    (47, 67, "neutral"),
    (68, 87, "stressed"),
    (88, 107, "neutral"),
    (108, 128, "tired"),
    (129, 149, "neutral"),
    (150, 174, "stressed"),
]


def get_label_for_window(start_sec: float, end_sec: float) -> str:
    """
    window의 midpoint를 기준으로 라벨 부여
    """
    mid = (start_sec + end_sec) / 2.0
    for s, e, label in TIMELINE:
        if s <= mid <= e:
            return label
    return "unknown"


def extract_window_features(
    file_path: str,
    window_sec: float = 2.5,
    hop_sec: float = 1.0,
) -> pd.DataFrame:
    y, sr = librosa.load(file_path, sr=None)

    window_size = int(window_sec * sr)
    hop_size = int(hop_sec * sr)

    rows = []

    for start in range(0, len(y) - window_size + 1, hop_size):
        end = start + window_size
        segment = y[start:end]

        start_sec = round(start / sr, 2)
        end_sec = round(end / sr, 2)

        pitch = librosa.yin(segment, fmin=50, fmax=300)
        valid_pitch = pitch[np.isfinite(pitch)]

        mean_pitch = float(np.mean(valid_pitch)) if len(valid_pitch) > 0 else 0.0
        pitch_std = float(np.std(valid_pitch)) if len(valid_pitch) > 0 else 0.0
        pitch_velocity = float(np.mean(np.abs(np.diff(valid_pitch)))) if len(valid_pitch) > 1 else 0.0

        rms = librosa.feature.rms(y=segment)[0]
        mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0

        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
        mean_centroid = float(np.mean(spectral_centroid)) if len(spectral_centroid) > 0 else 0.0

        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
        mean_rolloff = float(np.mean(spectral_rolloff)) if len(spectral_rolloff) > 0 else 0.0

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
        mean_bandwidth = float(np.mean(spectral_bandwidth)) if len(spectral_bandwidth) > 0 else 0.0

        zcr = librosa.feature.zero_crossing_rate(segment)[0]
        mean_zcr = float(np.mean(zcr)) if len(zcr) > 0 else 0.0

        onsets = librosa.onset.onset_detect(y=segment, sr=sr)
        duration = librosa.get_duration(y=segment, sr=sr)
        speech_rate = float(len(onsets) / duration) if duration > 0 else 0.0

        S = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        total_energy = np.sum(S)

        def band_ratio(low_hz: float, high_hz: float) -> float:
            if total_energy <= 0:
                return 0.0
            mask = (freqs >= low_hz) & (freqs < high_hz)
            return float(np.sum(S[mask, :]) / total_energy)

        low_band_ratio = band_ratio(80, 500)
        low_mid_ratio = band_ratio(500, 1500)
        high_band_ratio = band_ratio(1500, 8000)

        thickness_score = (
            (2.0 * low_band_ratio)
            + (1.2 * low_mid_ratio)
            - (1.0 * high_band_ratio)
            - (mean_centroid / 5000.0)
            - (mean_rolloff / 8000.0)
            - (mean_zcr * 5.0)
        )

        # frequency dynamics
        freq_mask = (freqs >= 60.0) & (freqs <= 2000.0)
        S_valid = S[freq_mask, :]
        freqs_valid = freqs[freq_mask]

        main_freq_track = []
        for t in range(S_valid.shape[1]):
            frame_mag = S_valid[:, t]
            peak_idx = int(np.argmax(frame_mag))
            main_freq_track.append(freqs_valid[peak_idx])

        main_freq_track = np.array(main_freq_track, dtype=float)
        diffs = np.diff(main_freq_track)
        abs_diffs = np.abs(diffs)

        main_freq_velocity_mean = float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else 0.0
        main_freq_std = float(np.std(main_freq_track)) if len(main_freq_track) > 0 else 0.0
        main_freq_range = float(np.max(main_freq_track) - np.min(main_freq_track)) if len(main_freq_track) > 0 else 0.0

        jump_threshold = 80.0
        jump_count = int(np.sum(abs_diffs > jump_threshold))
        main_freq_jump_ratio = float(jump_count / len(abs_diffs)) if len(abs_diffs) > 0 else 0.0

        second_diffs = np.diff(main_freq_track, n=2)
        main_freq_smoothness = (
            float(1.0 / (1.0 + np.mean(np.abs(second_diffs))))
            if len(second_diffs) > 0 else 0.0
        )

        label = get_label_for_window(start_sec, end_sec)

        rows.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label,
            "pitch": mean_pitch,
            "pitch_std": pitch_std,
            "pitch_velocity": pitch_velocity,
            "energy": mean_rms,
            "speech_rate": speech_rate,
            "spectral_centroid": mean_centroid,
            "spectral_rolloff": mean_rolloff,
            "spectral_bandwidth": mean_bandwidth,
            "zcr": mean_zcr,
            "low_band_ratio": low_band_ratio,
            "low_mid_ratio": low_mid_ratio,
            "high_band_ratio": high_band_ratio,
            "thickness_score": float(thickness_score),
            "main_freq_velocity_mean": main_freq_velocity_mean,
            "main_freq_jump_ratio": main_freq_jump_ratio,
            "main_freq_smoothness": main_freq_smoothness,
            "main_freq_std": main_freq_std,
            "main_freq_range": main_freq_range,
        })

    return pd.DataFrame(rows)


def run_neutral_tired_analysis():
    if not os.path.exists(MIXED_FILE):
        raise FileNotFoundError(f"Mixed file not found: {MIXED_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Neutral vs Tired Separation Analysis ===")
    print(f"Mixed file: {MIXED_FILE}")

    df = extract_window_features(MIXED_FILE, window_sec=2.5, hop_sec=1.0)
    df.to_csv(os.path.join(OUTPUT_DIR, "neutral_tired_all_windows.csv"), index=False)

    # neutral / tired만 보기
    nt_df = df[df["label"].isin(["neutral", "tired"])].copy()
    nt_df.to_csv(os.path.join(OUTPUT_DIR, "neutral_tired_windows_only.csv"), index=False)

    print("\n=== Window Counts ===")
    print(nt_df["label"].value_counts())

    feature_cols = [
        "pitch",
        "pitch_std",
        "pitch_velocity",
        "energy",
        "speech_rate",
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
        "main_freq_std",
        "main_freq_range",
    ]

    print("\n=== Mean Feature Comparison (Neutral vs Tired) ===")
    mean_df = nt_df.groupby("label")[feature_cols].mean().round(4)
    print(mean_df.to_string())

    mean_df.to_csv(os.path.join(OUTPUT_DIR, "neutral_tired_mean_comparison.csv"))

    # RF classification: neutral vs tired
    X = nt_df[feature_cols].fillna(0.0)
    y = nt_df["label"].map({"neutral": 0, "tired": 1})

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )

    n_min = nt_df["label"].value_counts().min()
    n_splits = min(5, n_min) if n_min >= 2 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    print("\n=== RF Neutral vs Tired CV Accuracy ===")
    print("Fold scores:", [round(s, 4) for s in scores])
    print("Mean accuracy:", round(scores.mean(), 4))
    print("Std:", round(scores.std(), 4))

    clf.fit(X, y)
    y_pred = clf.predict(X)

    print("\n=== Classification Report (train set reference) ===")
    print(classification_report(y, y_pred, target_names=["neutral", "tired"]))

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(os.path.join(OUTPUT_DIR, "neutral_tired_feature_importance.csv"), index=False)

    print("\n=== Top 10 Important Features (Neutral vs Tired) ===")
    print(importance_df.head(10).round(4).to_string(index=False))

    print("\nSaved:")
    print(f"- {os.path.join(OUTPUT_DIR, 'neutral_tired_all_windows.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'neutral_tired_windows_only.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'neutral_tired_mean_comparison.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'neutral_tired_feature_importance.csv')}")


if __name__ == "__main__":
    run_neutral_tired_analysis()