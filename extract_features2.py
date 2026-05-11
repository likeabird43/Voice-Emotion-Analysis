import librosa
import numpy as np
import pandas as pd
from pathlib import Path

AUDIO_DIR = Path("/Users/j/Desktop/AI_Projects/voice-mood-device/audio")
TARGET_SR = 16000

def extract_bands(y, sr):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = np.sum(S)
    def band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(S[mask]) / total) if total > 0 else 0.0
    return {
        "b1_80_500":   band(80,   500),
        "b2_500_1k":   band(500,  1000),
        "b3_1k_2k":    band(1000, 2000),
        "b4_2k_4k":    band(2000, 4000),
        "b5_4k_8k":    band(4000, 8000),
    }

def extract_features(wav_path):
    # sr=16000, mono로 강제 통일
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    pitch = librosa.yin(y, fmin=80, fmax=400)
    valid_pitch = pitch[(pitch > 0) & np.isfinite(pitch)]

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    feats = {
        "file":               wav_path.name,
        "environment":        wav_path.parent.name,
        "sr_original":        None,  # soundfile로 따로 확인
        "duration":           round(duration, 3),
        "pitch_mean":         float(np.mean(valid_pitch)) if len(valid_pitch) > 0 else 0.0,
        "pitch_std":          float(np.std(valid_pitch))  if len(valid_pitch) > 0 else 0.0,
        "energy_rms":         float(np.mean(rms)),
        "speech_rate":        float(len(onsets) / duration) if duration > 0 else 0.0,
        "spectral_centroid":  float(np.mean(centroid)),
        "spectral_rolloff":   float(np.mean(rolloff)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "zcr":                float(np.mean(zcr)),
    }

    # 5밴드 추가
    feats.update(extract_bands(y, sr))

    # MFCC 13개 추가
    for i, val in enumerate(np.mean(mfcc, axis=1)):
        feats[f"mfcc_{i+1}"] = float(val)

    return feats

rows = []
for wav in sorted(AUDIO_DIR.rglob("*.wav")):
    # 레이블 추출 (파일명에서)
    name = wav.stem  # 예: stressed_1, happy_3
    label = name.rsplit("_", 1)[0]  # stressed, happy, tired, baseline

    try:
        feats = extract_features(wav)
        feats["label"] = label
        rows.append(feats)
        print(f"✓ {wav.parent.name}/{wav.name}")
    except Exception as e:
        print(f"✗ {wav.name}: {e}")

df = pd.DataFrame(rows)
out_path = Path("/Users/j/Desktop/AI_Projects/voice-mood-device/features_standardized.csv")
df.to_csv(out_path, index=False)
print(f"\n완료! {len(df)}개 파일 → {out_path}")
print(df["label"].value_counts())