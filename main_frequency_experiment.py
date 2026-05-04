import os
import numpy as np
import pandas as pd
import librosa

SAMPLES_DIR = "audio/samples"


def extract_main_frequency_features(file_path):
    y, sr = librosa.load(file_path)

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    main_freq_track = []

    for t in range(S.shape[1]):
        peak_idx = np.argmax(S[:, t])
        main_freq_track.append(freqs[peak_idx])

    main_freq_track = np.array(main_freq_track)

    main_freq_mean = np.mean(main_freq_track)
    main_freq_std = np.std(main_freq_track)
    main_freq_velocity = np.mean(np.abs(np.diff(main_freq_track)))
    main_freq_range = np.max(main_freq_track) - np.min(main_freq_track)

    return {
        "main_freq_mean": main_freq_mean,
        "main_freq_std": main_freq_std,
        "main_freq_velocity": main_freq_velocity,
        "main_freq_range": main_freq_range,
    }


def parse_state(filename):
    filename = filename.lower()
    if "happy" in filename:
        return "happy"
    if "stressed" in filename:
        return "stressed"
    if "tired" in filename:
        return "tired"
    return "unknown"


def run_experiment():
    rows = []

    for env in os.listdir(SAMPLES_DIR):
        env_path = os.path.join(SAMPLES_DIR, env)

        if not os.path.isdir(env_path):
            continue

        print("Processing:", env)

        for file in os.listdir(env_path):
            if not file.endswith(".wav"):
                continue

            label = parse_state(file)
            if label == "unknown":
                continue

            path = os.path.join(env_path, file)

            features = extract_main_frequency_features(path)

            row = {
                "environment": env,
                "file": file,
                "label": label,
                **features
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    print("\n=== Mean Main Frequency Features by State ===")
    print(
        df.groupby("label")[[
            "main_freq_mean",
            "main_freq_std",
            "main_freq_velocity",
            "main_freq_range"
        ]].mean().round(2)
    )

    df.to_csv("main_frequency_experiment.csv", index=False)
    print("\nSaved → main_frequency_experiment.csv")


if __name__ == "__main__":
    run_experiment()