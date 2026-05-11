import re
import os
import numpy as np
import pandas as pd
import librosa

from voice_analysis import extract_features, infer_state
from frequency_dynamics_experiment import extract_main_frequency_track

BASELINE_DIR = "audio/baselines"
SAMPLES_DIR = "audio/samples"
MIXED_FILE = "audio/mixed/outside_mixed_v1.wav"


def compute_folder_baseline(environment: str) -> dict:
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
        "main_freq_velocity_mean": float(np.mean([f["main_freq_velocity_mean"] for f in baseline_freq_features])),
        "main_freq_jump_ratio": float(np.mean([f["main_freq_jump_ratio"] for f in baseline_freq_features])),
        "main_freq_smoothness": float(np.mean([f["main_freq_smoothness"] for f in baseline_freq_features])),
        "main_freq_std": float(np.mean([f.get("main_freq_std", 0.0) for f in baseline_freq_features])),
        "main_freq_range": float(np.mean([f.get("main_freq_range", 0.0) for f in baseline_freq_features])),
    }


def zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - mean) / std


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

        pitch = librosa.yin(segment, fmin=50, fmax=300)
        valid_pitch = pitch[np.isfinite(pitch)]
        mean_pitch = float(np.mean(valid_pitch)) if len(valid_pitch) > 0 else 0.0
        pitch_std = float(np.std(valid_pitch)) if len(valid_pitch) > 0 else 0.0
        pitch_velocity = float(np.mean(np.abs(np.diff(valid_pitch)))) if len(valid_pitch) > 1 else 0.0

        rms = librosa.feature.rms(y=segment)[0]
        mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0

        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
        mean_centroid = float(np.mean(spectral_centroid)) if len(spectral_centroid) > 0 else 0.0

        onsets = librosa.onset.onset_detect(y=segment, sr=sr)
        duration = librosa.get_duration(y=segment, sr=sr)
        speech_rate = float(len(onsets) / duration) if duration > 0 else 0.0

        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
        mean_rolloff = float(np.mean(spectral_rolloff)) if len(spectral_rolloff) > 0 else 0.0

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
        mean_bandwidth = float(np.mean(spectral_bandwidth)) if len(spectral_bandwidth) > 0 else 0.0

        zcr = librosa.feature.zero_crossing_rate(segment)[0]
        mean_zcr = float(np.mean(zcr)) if len(zcr) > 0 else 0.0

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

        jump_threshold = 80.0
        jump_count = int(np.sum(abs_diffs > jump_threshold))
        jump_ratio = float(jump_count / len(abs_diffs)) if len(abs_diffs) > 0 else 0.0

        second_diffs = np.diff(main_freq_track, n=2)
        smoothness = float(1.0 / (1.0 + np.mean(np.abs(second_diffs)))) if len(second_diffs) > 0 else 0.0

        rows.append({
            "start_sec": round(start / sr, 2),
            "end_sec": round(end / sr, 2),
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
            "main_freq_velocity_mean": float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else 0.0,
            "main_freq_jump_ratio": jump_ratio,
            "main_freq_smoothness": smoothness,
            "main_freq_std": float(np.std(main_freq_track)) if len(main_freq_track) > 0 else 0.0,
            "main_freq_range": float(np.max(main_freq_track) - np.min(main_freq_track)) if len(main_freq_track) > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def add_neutral_like_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["z_pitch_velocity"] = zscore(out["pitch_velocity"].values)
    out["z_main_freq_velocity"] = zscore(out["main_freq_velocity_mean"].values)
    out["z_jump_ratio"] = zscore(out["main_freq_jump_ratio"].values)
    out["z_smoothness"] = zscore(out["main_freq_smoothness"].values)
    out["z_main_freq_std"] = zscore(out["main_freq_std"].values)
    out["z_main_freq_range"] = zscore(out["main_freq_range"].values)

    out["z_energy"] = zscore(out["energy"].values)
    out["z_speech_rate"] = zscore(out["speech_rate"].values)

    energy_center_penalty = np.abs(out["z_energy"].values)
    rate_center_penalty = np.abs(out["z_speech_rate"].values)

    out["z_low_band"] = zscore(out["low_band_ratio"].values)
    out["z_low_mid_band"] = zscore(out["low_mid_ratio"].values)
    out["z_high_band"] = zscore(out["high_band_ratio"].values)

    low_band_penalty = np.abs(out["z_low_band"].values)
    low_mid_penalty = np.abs(out["z_low_mid_band"].values)
    high_band_penalty = np.abs(out["z_high_band"].values)

    out["neutral_like_score"] = (
        (-0.6 * out["z_pitch_velocity"])
        + (-0.6 * out["z_main_freq_velocity"])
        + (-0.5 * out["z_jump_ratio"])
        + ( 0.9 * out["z_smoothness"])
        + (-0.6 * out["z_main_freq_std"])
        + (-0.5 * out["z_main_freq_range"])
        + (-1.0 * energy_center_penalty)
        + (-1.0 * rate_center_penalty)
        + (-0.6 * low_band_penalty)
        + (-0.6 * low_mid_penalty)
        + (-0.6 * high_band_penalty)
    )

    return out


def estimate_baseline_from_mixed(
    mixed_file: str,
    window_sec: float = 2.5,
    hop_sec: float = 1.0,
    top_ratio: float = 0.20,
) -> tuple[dict, pd.DataFrame]:
    df = extract_window_features(mixed_file, window_sec=window_sec, hop_sec=hop_sec)
    df = add_neutral_like_score(df)

    n_select = max(3, int(len(df) * top_ratio))
    selected = df.sort_values("neutral_like_score", ascending=False).head(n_select).copy()

    baseline = {
        "pitch": float(np.median(selected["pitch"])),
        "pitch_std": float(np.mean(selected["pitch_std"])),
        "pitch_velocity": float(np.mean(selected["pitch_velocity"])),
        "energy": float(np.mean(selected["energy"])),
        "speech_rate": float(np.mean(selected["speech_rate"])),
        "spectral_centroid": float(np.mean(selected["spectral_centroid"])),
        "duration": float(window_sec),
        "spectral_rolloff": float(np.mean(selected["spectral_rolloff"])),
        "spectral_bandwidth": float(np.mean(selected["spectral_bandwidth"])),
        "zcr": float(np.mean(selected["zcr"])),
        "low_band_ratio": float(np.mean(selected["low_band_ratio"])),
        "low_mid_ratio": float(np.mean(selected["low_mid_ratio"])),
        "high_band_ratio": float(np.mean(selected["high_band_ratio"])),
        "thickness_score": float(np.mean(selected["thickness_score"])),
        "main_freq_velocity_mean": float(np.mean(selected["main_freq_velocity_mean"])),
        "main_freq_jump_ratio": float(np.mean(selected["main_freq_jump_ratio"])),
        "main_freq_smoothness": float(np.mean(selected["main_freq_smoothness"])),
        "main_freq_std": float(np.mean(selected["main_freq_std"])),
        "main_freq_range": float(np.mean(selected["main_freq_range"])),
    }

    return baseline, df


def parse_state(filename: str) -> str:
    match = re.search(r"(happy|tired|stressed)", filename.lower())
    return match.group(1) if match else "unknown"


def evaluate_baseline_on_samples(environment: str, baseline: dict) -> pd.DataFrame:
    sample_path = os.path.join(SAMPLES_DIR, environment)
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample folder not found: {sample_path}")

    rows = []

    for file in sorted(os.listdir(sample_path)):
        if not file.endswith(".wav"):
            continue
        if "_cleaned" in file:
            continue

        label = parse_state(file)
        if label == "unknown":
            continue

        file_path = os.path.join(sample_path, file)

        features = extract_features(file_path)
        freq_features = extract_main_frequency_track(file_path)
        features.update({
            "main_freq_velocity_mean": freq_features["main_freq_velocity_mean"],
            "main_freq_jump_ratio": freq_features["main_freq_jump_ratio"],
            "main_freq_smoothness": freq_features["main_freq_smoothness"],
            "main_freq_std": freq_features.get("main_freq_std", 0.0),
            "main_freq_range": freq_features.get("main_freq_range", 0.0),
        })

        inferred_state, confidence, _ = infer_state(features, baseline=baseline)

        rows.append({
            "file": file,
            "label": label,
            "inferred_state": inferred_state,
            "confidence": confidence,
            "correct": label == inferred_state,
        })

    return pd.DataFrame(rows)


def compare_baselines(environment: str = "outside"):
    print(f"=== Mixed Speech Baseline Estimation ({environment}) ===")

    if not os.path.exists(MIXED_FILE):
        raise FileNotFoundError(f"Mixed file not found: {MIXED_FILE}")

    folder_baseline = compute_folder_baseline(environment)
    mixed_baseline, window_df = estimate_baseline_from_mixed(MIXED_FILE)

    os.makedirs("analysis_outputs", exist_ok=True)
    window_df.to_csv("analysis_outputs/mixed_window_features.csv", index=False)

    pd.DataFrame([folder_baseline]).to_csv(
        f"analysis_outputs/{environment}_folder_baseline.csv", index=False
    )
    pd.DataFrame([mixed_baseline]).to_csv(
        f"analysis_outputs/{environment}_mixed_estimated_baseline.csv", index=False
    )

    print("\nSaved:")
    print("- analysis_outputs/mixed_window_features.csv")
    print(f"- analysis_outputs/{environment}_folder_baseline.csv")
    print(f"- analysis_outputs/{environment}_mixed_estimated_baseline.csv")

    folder_eval = evaluate_baseline_on_samples(environment, folder_baseline)
    mixed_eval = evaluate_baseline_on_samples(environment, mixed_baseline)

    folder_eval.to_csv(
        f"analysis_outputs/{environment}_folder_baseline_eval.csv", index=False
    )
    mixed_eval.to_csv(
        f"analysis_outputs/{environment}_mixed_baseline_eval.csv", index=False
    )

    print("\n=== Accuracy Comparison ===")
    print(f"Folder baseline accuracy: {folder_eval['correct'].mean():.4f}")
    print(f"Mixed estimated baseline accuracy: {mixed_eval['correct'].mean():.4f}")

    print("\n=== Accuracy by State (Folder Baseline) ===")
    print(folder_eval.groupby("label")["correct"].mean().round(2))

    print("\n=== Accuracy by State (Mixed Estimated Baseline) ===")
    print(mixed_eval.groupby("label")["correct"].mean().round(2))

    print("\n=== Top Neutral-like Windows ===")
    preview_cols = [
        "start_sec",
        "end_sec",
        "neutral_like_score",
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
    print(
        window_df.sort_values("neutral_like_score", ascending=False)
        .head(10)[preview_cols]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    compare_baselines(environment="outside")