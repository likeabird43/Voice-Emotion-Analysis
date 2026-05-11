import webrtcvad
import numpy as np
import librosa


def extract_voice_only(input_path: str, aggressiveness: int = 3) -> np.ndarray:
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    y_int16 = (y * 32767).astype(np.int16)

    vad = webrtcvad.Vad(aggressiveness)
    frame_duration = 30
    frame_size = int(16000 * frame_duration / 1000)

    voiced_frames = []
    n_frames = len(y_int16) // frame_size

    for i in range(n_frames):
        start = i * frame_size
        end = start + frame_size
        frame = y_int16[start:end]
        if vad.is_speech(frame.tobytes(), sample_rate=16000):
            voiced_frames.append(frame.astype(np.float32) / 32767.0)

    if not voiced_frames:
        yt, _ = librosa.effects.trim(y, top_db=25)
        return yt

    return np.concatenate(voiced_frames)