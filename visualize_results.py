import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

DATASET_CSV = "dataset_analysis.csv"
MAIN_FREQ_CSV = "main_frequency_experiment.csv"
SAMPLES_DIR = "audio/samples"
OUT_DIR = "plots"


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def plot_bar(df, x_col, y_col, title, filename):
    plt.figure(figsize=(7, 4))
    grouped = df.groupby(x_col)[y_col].mean()
    plt.bar(grouped.index, grouped.values)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


def plot_environment_accuracy(df):
    plt.figure(figsize=(6, 4))
    grouped = df.groupby("environment")["correct"].mean()
    plt.bar(grouped.index, grouped.values)
    plt.title("Accuracy by Environment")
    plt.xlabel("Environment")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_by_environment.png"), dpi=200)
    plt.close()


def extract_pitch_contour(file_path):
    y, sr = librosa.load(file_path)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch = np.where(np.isfinite(pitch), pitch, np.nan)
    times = librosa.times_like(pitch, sr=sr)
    return times, pitch


def extract_main_freq_track(file_path):
    y, sr = librosa.load(file_path)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    track = []
    for t in range(S.shape[1]):
        peak_idx = np.argmax(S[:, t])
        track.append(freqs[peak_idx])

    track = np.array(track, dtype=float)
    times = librosa.times_like(track, sr=sr)
    return times, track


def pick_example_file(label, environment="cafe"):
    env_path = os.path.join(SAMPLES_DIR, environment)
    if not os.path.isdir(env_path):
        return None

    candidates = sorted(
        f for f in os.listdir(env_path)
        if f.endswith(".wav") and label in f.lower()
    )
    if not candidates:
        return None
    return os.path.join(env_path, candidates[0])


def plot_contours_for_label(label, environment="cafe"):
    file_path = pick_example_file(label, environment)
    if file_path is None:
        print(f"No file found for {label} in {environment}")
        return

    pitch_t, pitch = extract_pitch_contour(file_path)
    mf_t, main_freq = extract_main_freq_track(file_path)

    plt.figure(figsize=(8, 4))
    plt.plot(pitch_t, pitch)
    plt.title(f"Pitch Contour - {label} ({environment})")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pitch_contour_{label}_{environment}.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(mf_t, main_freq)
    plt.title(f"Dominant Frequency Track - {label} ({environment})")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"main_freq_contour_{label}_{environment}.png"), dpi=200)
    plt.close()


def main():
    ensure_out_dir()

    dataset_df = pd.read_csv(DATASET_CSV)
    mf_df = pd.read_csv(MAIN_FREQ_CSV)

    plot_environment_accuracy(dataset_df)
    plot_bar(dataset_df, "label", "pitch_velocity", "Pitch Velocity by Emotion", "pitch_velocity_by_emotion.png")
    plot_bar(dataset_df, "label", "thickness_score", "Thickness Score by Emotion", "thickness_score_by_emotion.png")
    plot_bar(mf_df, "label", "main_freq_velocity", "Dominant Frequency Velocity by Emotion", "main_freq_velocity_by_emotion.png")
    plot_bar(mf_df, "label", "main_freq_std", "Dominant Frequency STD by Emotion", "main_freq_std_by_emotion.png")

    for label in ["happy", "stressed", "tired"]:
        plot_contours_for_label(label, environment="cafe")

    print(f"Saved plots to ./{OUT_DIR}")


if __name__ == "__main__":
    main()