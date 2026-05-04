import noisereduce as nr
import numpy as np
import librosa


def reduce_noise(input_path: str) -> tuple[np.ndarray, int]:
    """
    배경소음 제거 후 numpy array와 sr 반환
    VAD와 달리 음성 특성 보존하면서 노이즈만 제거
    """
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # 처음 0.5초를 노이즈 샘플로 사용 (배경소음 구간)
    noise_sample = y[:int(sr * 0.5)]
    
    # 노이즈 제거
    y_denoised = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=0.8  # 노이즈 감소 강도 (0~1)
    )
    
    return y_denoised, sr