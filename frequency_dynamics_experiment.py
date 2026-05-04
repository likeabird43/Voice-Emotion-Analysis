import os
import numpy as np
import pandas as pd
import librosa

SAMPLES_DIR = "audio/samples"


def _safe_mean(arr) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def _safe_std(arr) -> float:
    return float(np.std(arr)) if len(arr) > 0 else 0.0


def parse_state(filename: str) -> str:
    filename = filename.lower()
    if "happy" in filename:
        return "happy"
    if "stressed" in filename:
        return "stressed"
    if "tired" in filename:
        return "tired"
    return "unknown"


def extract_main_frequency_track(
    file_path: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 60.0,
    fmax: float = 2000.0,
) -> dict:
    y, sr = librosa.load(file_path, sr=None)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 관심 주파수 대역만 사용
    valid_mask = (freqs >= fmin) & (freqs <= fmax)
    S_valid = S[valid_mask, :]
    freqs_valid = freqs[valid_mask]

    main_freq_track = []
    peak_energy_track = []

    for t in range(S_valid.shape[1]):
        frame_mag = S_valid[:, t]
        peak_idx = int(np.argmax(frame_mag))
        main_freq_track.append(freqs_valid[peak_idx])
        peak_energy_track.append(frame_mag[peak_idx])

    main_freq_track = np.array(main_freq_track, dtype=float)
    peak_energy_track = np.array(peak_energy_track, dtype=float)

    # 프레임 간 변화량
    diffs = np.diff(main_freq_track)
    abs_diffs = np.abs(diffs)

    # jump threshold: 경험적으로 80Hz 이상이면 큰 점프로 간주
    jump_threshold = 80.0
    jump_count = int(np.sum(abs_diffs > jump_threshold))
    jump_ratio = float(jump_count / len(abs_diffs)) if len(abs_diffs) > 0 else 0.0

    # smoothness: second-order diff가 작을수록 더 부드러움
    second_diffs = np.diff(main_freq_track, n=2)
    smoothness = float(1.0 / (1.0 + np.mean(np.abs(second_diffs)))) if len(second_diffs) > 0 else 0.0

    return {
        "main_freq_mean": _safe_mean(main_freq_track),
        "main_freq_std": _safe_std(main_freq_track),
        "main_freq_velocity_mean": _safe_mean(abs_diffs),
        "main_freq_velocity_std": _safe_std(abs_diffs),
        "main_freq_range": float(np.max(main_freq_track) - np.min(main_freq_track)) if len(main_freq_track) > 0 else 0.0,
        "main_freq_jump_count": jump_count,
        "main_freq_jump_ratio": jump_ratio,
        "main_freq_smoothness": smoothness,
        "peak_energy_mean": _safe_mean(peak_energy_track),
        "num_frames": int(len(main_freq_track)),
    }


def run_experiment():
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
            feats = extract_main_frequency_track(file_path)

            row = {
                "environment": env,
                "file": file,
                "label": label,
                **feats,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    print("\n=== Mean Frequency Dynamics by State ===")
    print(
        df.groupby("label")[[
            "main_freq_mean",
            "main_freq_std",
            "main_freq_velocity_mean",
            "main_freq_velocity_std",
            "main_freq_range",
            "main_freq_jump_count",
            "main_freq_jump_ratio",
            "main_freq_smoothness",
        ]].mean().round(2)
    )

    print("\n=== Mean Frequency Dynamics by State x Environment ===")
    print(
        df.groupby(["label", "environment"])[[
            "main_freq_velocity_mean",
            "main_freq_jump_ratio",
            "main_freq_smoothness",
        ]].mean().round(3)
    )

    df.to_csv("frequency_dynamics_experiment.csv", index=False)
    print("\nSaved → frequency_dynamics_experiment.csv")


if __name__ == "__main__":
    run_experiment()
