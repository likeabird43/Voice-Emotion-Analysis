import librosa
import numpy as np


def _safe_mean(arr) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def _band_energy_ratio(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    if len(y) == 0:
        return 0.0

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    band_mask = (freqs >= low_hz) & (freqs < high_hz)
    total_energy = np.sum(S)

    if total_energy <= 0:
        return 0.0

    band_energy = np.sum(S[band_mask, :])
    return float(band_energy / total_energy)


def extract_features(file_path: str) -> dict:
    y, sr = librosa.load(file_path)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    valid_pitch = pitch[np.isfinite(pitch)]
    mean_pitch = _safe_mean(valid_pitch)
    pitch_std = float(np.std(valid_pitch)) if len(valid_pitch) > 0 else 0.0
    pitch_velocity = float(np.mean(np.abs(np.diff(valid_pitch)))) if len(valid_pitch) > 1 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    mean_rms = _safe_mean(rms)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = _safe_mean(spectral_centroid)

    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    speech_rate = float(len(onsets) / duration) if duration > 0 else 0.0

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mean_rolloff = _safe_mean(spectral_rolloff)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mean_bandwidth = _safe_mean(spectral_bandwidth)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mean_zcr = _safe_mean(zcr)

    low_band_ratio = _band_energy_ratio(y, sr, 80, 500)
    low_mid_ratio = _band_energy_ratio(y, sr, 500, 1500)
    high_band_ratio = _band_energy_ratio(y, sr, 1500, 8000)

    thickness_score = (
        (2.0 * low_band_ratio)
        + (1.2 * low_mid_ratio)
        - (1.0 * high_band_ratio)
        - (mean_centroid / 5000.0)
        - (mean_rolloff / 8000.0)
        - (mean_zcr * 5.0)
    )

    return {
        "pitch": mean_pitch,
        "pitch_std": pitch_std,
        "pitch_velocity": pitch_velocity,
        "energy": mean_rms,
        "spectral_centroid": mean_centroid,
        "speech_rate": speech_rate,
        "duration": duration,
        "spectral_rolloff": mean_rolloff,
        "spectral_bandwidth": mean_bandwidth,
        "zcr": mean_zcr,
        "low_band_ratio": low_band_ratio,
        "low_mid_ratio": low_mid_ratio,
        "high_band_ratio": high_band_ratio,
        "thickness_score": float(thickness_score),
    }


def compute_feature_delta(features: dict, baseline: dict | None) -> dict:
    if baseline is None:
        return {
            "pitch_delta": 0.0,
            "energy_delta": 0.0,
            "speech_rate_delta": 0.0,
            "spectral_centroid_delta": 0.0,
            "thickness_score_delta": 0.0,
            "pitch_std_delta": 0.0,
            "pitch_velocity_delta": 0.0,
            "main_freq_velocity_mean_delta": 0.0,
            "main_freq_jump_ratio_delta": 0.0,
            "main_freq_smoothness_delta": 0.0,
        }

    return {
        "pitch_delta": features["pitch"] - baseline["pitch"],
        "energy_delta": features["energy"] - baseline["energy"],
        "speech_rate_delta": features["speech_rate"] - baseline["speech_rate"],
        "spectral_centroid_delta": features["spectral_centroid"] - baseline["spectral_centroid"],
        "thickness_score_delta": features["thickness_score"] - baseline["thickness_score"],
        "pitch_std_delta": features["pitch_std"] - baseline["pitch_std"],
        "pitch_velocity_delta": features["pitch_velocity"] - baseline["pitch_velocity"],

        # frequency dynamics delta
        "main_freq_velocity_mean_delta": features.get("main_freq_velocity_mean", 0.0) - baseline.get("main_freq_velocity_mean", 0.0),
        "main_freq_jump_ratio_delta": features.get("main_freq_jump_ratio", 0.0) - baseline.get("main_freq_jump_ratio", 0.0),
        "main_freq_smoothness_delta": features.get("main_freq_smoothness", 0.0) - baseline.get("main_freq_smoothness", 0.0),
    }


def infer_state(features: dict, text_hint: str = "", baseline: dict | None = None) -> tuple[str, float, dict]:
    deltas = compute_feature_delta(features, baseline)

    pitch_delta = deltas["pitch_delta"]
    energy_delta = deltas["energy_delta"]
    rate_delta = deltas["speech_rate_delta"]
    thickness_delta = deltas["thickness_score_delta"]
    pitch_std_delta = deltas["pitch_std_delta"]
    pitch_velocity_delta = deltas["pitch_velocity_delta"]

    main_freq_velocity_mean_delta = deltas["main_freq_velocity_mean_delta"]
    main_freq_jump_ratio_delta = deltas["main_freq_jump_ratio_delta"]
    main_freq_smoothness_delta = deltas["main_freq_smoothness_delta"]

    low_band = features.get("low_band_ratio", 0.0)
    low_mid = features.get("low_mid_ratio", 0.0)
    high_band = features.get("high_band_ratio", 0.0)

    text_hint = text_hint.lower().strip()

    scores = {
        "tired": 0.0,
        "stressed": 0.0,
        "happy": 0.0,
        "neutral": 0.10,
    }

    if baseline is not None:
        # -------------------------
        # TIRED
        # -------------------------
        if energy_delta < -0.003:
            scores["tired"] += 0.30
        if rate_delta < -0.20:
            scores["tired"] += 0.25
        if pitch_velocity_delta < -0.5:
            scores["tired"] += 0.15
        if low_band > 0.50:
            scores["tired"] += 0.15
        if low_mid < 0.26:
            scores["tired"] += 0.10
        if energy_delta < 0 and rate_delta < 0:
            scores["tired"] += 0.10
        if pitch_delta < 8 and pitch_velocity_delta < 0:
            scores["tired"] += 0.10

        # frequency dynamics tired 보조 신호
        if main_freq_velocity_mean_delta < -3.0:
            scores["tired"] += 0.10
        if main_freq_smoothness_delta > 0.002:
            scores["tired"] += 0.05

        # -------------------------
        # HAPPY vs STRESSED shared high-arousal gate
        # -------------------------
        if pitch_delta > 10 and energy_delta > 0.010:
            # Janet data pattern:
            # happy  -> rate_delta higher
            # stressed -> rate_delta lower
            if rate_delta > 0.30:
                scores["happy"] += 0.40
            else:
                scores["stressed"] += 0.40

        # -------------------------
        # HAPPY
        # -------------------------
        if rate_delta > 0.20:
            scores["happy"] += 0.20
        if pitch_velocity_delta > 1.0:
            scores["happy"] += 0.20
        if high_band > 0.195 and low_band < 0.49:
            scores["happy"] += 0.15
        if low_mid < 0.32 and pitch_delta > 5:
            scores["happy"] += 0.10
        if thickness_delta < -0.08:
            scores["happy"] += 0.05

        # frequency dynamics happy 보조 신호
        if main_freq_velocity_mean_delta > 2.0 and main_freq_jump_ratio_delta > 0.0:
            scores["happy"] += 0.05

        # -------------------------
        # STRESSED
        # -------------------------
        if pitch_delta > 18 and energy_delta > 0.015:
            scores["stressed"] += 0.20
        if 0.00 <= rate_delta <= 0.35:
            scores["stressed"] += 0.15
        if low_mid > 0.32 and pitch_delta > 10:
            scores["stressed"] += 0.15
        if low_band < 0.46 and pitch_delta > 15:
            scores["stressed"] += 0.10
        if 0.0 <= pitch_velocity_delta <= 2.5 and pitch_delta > 10:
            scores["stressed"] += 0.10
        if pitch_std_delta > 8 and pitch_delta > 12:
            scores["stressed"] += 0.10

        # frequency dynamics stressed 보조 신호
        if main_freq_velocity_mean_delta > 5.4:
            scores["stressed"] += 0.15
        if main_freq_jump_ratio_delta > 0.014:
            scores["stressed"] += 0.10
        if main_freq_smoothness_delta < -0.002:
            scores["stressed"] += 0.05

        # -------------------------
        # Anti-confusion adjustments
        # -------------------------
        if rate_delta > 0.50 and pitch_velocity_delta > 2.0:
            scores["happy"] += 0.10
            scores["stressed"] -= 0.10

        if energy_delta < 0 and rate_delta < 0:
            scores["happy"] -= 0.10
            scores["stressed"] -= 0.05

        if low_band > 0.55 and energy_delta < 0:
            scores["happy"] -= 0.10

    else:
        # fallback
        energy = features["energy"]
        pitch = features["pitch"]
        speech_rate = features["speech_rate"]

        if energy < 0.05 and speech_rate < 3.5:
            scores["tired"] += 0.50
        if pitch > 180 and energy > 0.10:
            scores["stressed"] += 0.40
        if 130 < pitch < 180 and energy > 0.08:
            scores["happy"] += 0.40

    # -------------------------
    # Text hint boost
    # -------------------------
    if "피곤" in text_hint or "tired" in text_hint:
        scores["tired"] += 0.35
    if "스트레스" in text_hint or "stress" in text_hint or "바쁘" in text_hint or "busy" in text_hint:
        scores["stressed"] += 0.35
    if "신나" in text_hint or "좋아" in text_hint or "happy" in text_hint:
        scores["happy"] += 0.30

    # clamp negatives
    for k in scores:
        scores[k] = max(0.0, scores[k])

    state = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    top, second = sorted_scores[0], sorted_scores[1]
    confidence = min(0.95, max(0.35, 0.5 + (top - second)))

    return state, round(confidence, 2), deltas