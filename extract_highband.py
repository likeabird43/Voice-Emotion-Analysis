import librosa
import numpy as np
import pandas as pd
from pathlib import Path

AUDIO_DIR = Path("/Users/j/Desktop/AI_Projects/voice-mood-device/audio")

def extract_high_bands(path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    total = np.sum(S)

    def band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(S[mask]) / total) if total > 0 else 0.0

    return {
        'b_4k_6k':   band(4000,  6000),
        'b_6k_8k':   band(6000,  8000),
        'b_8k_11k':  band(8000,  11000),
        'b_11k_16k': band(11000, 16000),
        'b_16k_up':  band(16000, sr//2),
        'sr': sr,
    }

rows = []
for wav in sorted(AUDIO_DIR.rglob("*.wav")):
    name  = wav.stem
    label = name.rsplit("_", 1)[0]
    if label not in ['happy','tired','stressed','stressed_intense']:
        continue
    try:
        r = extract_high_bands(wav)
        r['file']        = wav.name
        r['environment'] = wav.parent.name
        r['label']       = label
        rows.append(r)
        print(f"✓ {wav.parent.name}/{wav.name}")
    except Exception as e:
        print(f"✗ {wav.name}: {e}")

df = pd.DataFrame(rows)
df = df.sort_values('sr', ascending=False)
df = df.drop_duplicates(subset=['file', 'environment'], keep='first')
df = df.reset_index(drop=True)

out = Path("/Users/j/Desktop/AI_Projects/voice-mood-device/highband_features.csv")
df.to_csv(out, index=False)
print(f"\n완료! {len(df)}개 → {out}")
print(df['label'].value_counts())