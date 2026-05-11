import os
import numpy as np
import pandas as pd
import librosa

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


def zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - mean) / std


def get_label_for_window(start_sec: float, end_sec: float) -> str:
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

        # basic acoustic
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

        # band ratios
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

        rows.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": get_label_for_window(start_sec, end_sec),
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


def build_neutral_target_profile(df: pd.DataFrame) -> dict:
    neutral_df = df[df["label"] == "neutral"].copy()
    if neutral_df.empty:
        raise ValueError("No neutral windows found from timeline labels.")

    target = {}
    for col in [
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
    ]:
        target[col] = float(neutral_df[col].mean())

    return target


def add_neutral_profile_score(df: pd.DataFrame, target: dict) -> pd.DataFrame:
    out = df.copy()

    # neutral target로부터의 거리 계산
    feature_weights = {
        "pitch": 0.4,
        "pitch_std": 0.7,
        "pitch_velocity": 0.9,
        "energy": 0.8,
        "speech_rate": 0.7,
        "spectral_centroid": 0.7,
        "spectral_rolloff": 0.4,
        "spectral_bandwidth": 0.4,
        "zcr": 0.3,
        "low_band_ratio": 0.6,
        "low_mid_ratio": 0.6,
        "high_band_ratio": 0.6,
        "thickness_score": 0.6,
        "main_freq_velocity_mean": 1.0,
        "main_freq_jump_ratio": 0.8,
        "main_freq_smoothness": 1.0,
        "main_freq_std": 1.0,
        "main_freq_range": 0.9,
    }

    distance_terms = []

    for feature, weight in feature_weights.items():
        values = out[feature].values
        target_val = target[feature]

        # 전체 window 기준 z-scale
        std = np.std(values)
        if std < 1e-8:
            dist = np.zeros_like(values)
        else:
            dist = np.abs(values - target_val) / std

        out[f"{feature}_dist"] = dist
        distance_terms.append(weight * dist)

    total_distance = np.sum(distance_terms, axis=0)
    out["neutral_profile_distance"] = total_distance

    # 점수는 distance의 음수 형태
    out["baseline_like_score_v2"] = -total_distance

    return out


def detect_baseline_segments_v2(
    file_path: str,
    window_sec: float = 2.5,
    hop_sec: float = 1.0,
    top_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = extract_window_features(file_path, window_sec=window_sec, hop_sec=hop_sec)
    target = build_neutral_target_profile(df)
    df = add_neutral_profile_score(df, target)

    n_select = max(3, int(len(df) * top_ratio))
    selected = df.sort_values("baseline_like_score_v2", ascending=False).head(n_select).copy()

    return df, selected, target


def build_estimated_baseline_profile(selected_df: pd.DataFrame, window_sec: float = 2.5) -> dict:
    return {
        "pitch": float(np.median(selected_df["pitch"])),
        "pitch_std": float(np.mean(selected_df["pitch_std"])),
        "pitch_velocity": float(np.mean(selected_df["pitch_velocity"])),
        "energy": float(np.mean(selected_df["energy"])),
        "speech_rate": float(np.mean(selected_df["speech_rate"])),
        "spectral_centroid": float(np.mean(selected_df["spectral_centroid"])),
        "duration": float(window_sec),
        "spectral_rolloff": float(np.mean(selected_df["spectral_rolloff"])),
        "spectral_bandwidth": float(np.mean(selected_df["spectral_bandwidth"])),
        "zcr": float(np.mean(selected_df["zcr"])),
        "low_band_ratio": float(np.mean(selected_df["low_band_ratio"])),
        "low_mid_ratio": float(np.mean(selected_df["low_mid_ratio"])),
        "high_band_ratio": float(np.mean(selected_df["high_band_ratio"])),
        "thickness_score": float(np.mean(selected_df["thickness_score"])),
        "main_freq_velocity_mean": float(np.mean(selected_df["main_freq_velocity_mean"])),
        "main_freq_jump_ratio": float(np.mean(selected_df["main_freq_jump_ratio"])),
        "main_freq_smoothness": float(np.mean(selected_df["main_freq_smoothness"])),
        "main_freq_std": float(np.mean(selected_df["main_freq_std"])),
        "main_freq_range": float(np.mean(selected_df["main_freq_range"])),
    }


def evaluate_candidate_labels(selected_df: pd.DataFrame):
    print("\n=== Label Distribution of Top Baseline-like Segments ===")
    print(selected_df["label"].value_counts())


def main():
    if not os.path.exists(MIXED_FILE):
        raise FileNotFoundError(f"Mixed file not found: {MIXED_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Baseline Segment Detection V2 (Neutral Target Profile) ===")
    print(f"Mixed file: {MIXED_FILE}")

    all_windows_df, selected_df, neutral_target = detect_baseline_segments_v2(
        MIXED_FILE,
        window_sec=2.5,
        hop_sec=1.0,
        top_ratio=0.20,
    )

    estimated_baseline = build_estimated_baseline_profile(selected_df, window_sec=2.5)

    all_path = os.path.join(OUTPUT_DIR, "baseline_detector_v2_all_windows.csv")
    selected_path = os.path.join(OUTPUT_DIR, "baseline_detector_v2_candidates.csv")
    target_path = os.path.join(OUTPUT_DIR, "baseline_detector_v2_neutral_target.csv")
    profile_path = os.path.join(OUTPUT_DIR, "baseline_detector_v2_profile.csv")

    all_windows_df.to_csv(all_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    pd.DataFrame([neutral_target]).to_csv(target_path, index=False)
    pd.DataFrame([estimated_baseline]).to_csv(profile_path, index=False)

    print("\nSaved:")
    print(f"- {all_path}")
    print(f"- {selected_path}")
    print(f"- {target_path}")
    print(f"- {profile_path}")

    evaluate_candidate_labels(selected_df)

    preview_cols = [
        "start_sec",
        "end_sec",
        "label",
        "baseline_like_score_v2",
        "pitch_velocity",
        "energy",
        "speech_rate",
        "low_band_ratio",
        "low_mid_ratio",
        "high_band_ratio",
        "main_freq_velocity_mean",
        "main_freq_jump_ratio",
        "main_freq_smoothness",
        "main_freq_std",
        "main_freq_range",
    ]

    print("\n=== Top Baseline-like Segments (V2) ===")
    print(
        selected_df[preview_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== Neutral Target Profile ===")
    for k, v in neutral_target.items():
        print(f"{k}: {round(v, 4)}")

    print("\n=== Estimated Baseline Profile from Selected Segments ===")
    for k, v in estimated_baseline.items():
        print(f"{k}: {round(v, 4)}")


if __name__ == "__main__":
    main()