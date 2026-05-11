# pip install sounddevice
import sounddevice as sd
import numpy as np
import tempfile, soundfile as sf
from datetime import datetime
from voice_analysis import extract_features, infer_state
from vad_preprocessor import extract_voice_only

SAMPLE_RATE = 16000
WINDOW_SEC = 5      # 5초마다 분석
STRIDE_SEC = 2      # 2초 간격으로 슬라이딩

def run_realtime(baseline: dict | None = None):
    buffer = []
    results = []

    print("🎙️  실시간 감정 모니터링 시작 (Ctrl+C로 종료)\n")

    def callback(indata, frames, time, status):
        buffer.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype='float32', callback=callback):
        try:
            while True:
                sd.sleep(STRIDE_SEC * 1000)  # STRIDE_SEC초 대기

                if len(buffer) == 0:
                    continue

                audio = np.concatenate(buffer)

                # 최근 WINDOW_SEC초만 사용
                window = audio[-(SAMPLE_RATE * WINDOW_SEC):]

                if len(window) < SAMPLE_RATE * 2:  # 최소 2초
                    continue

                # 임시 파일로 저장 후 분석
                with tempfile.NamedTemporaryFile(suffix=".wav",
                                                 delete=False) as f:
                    sf.write(f.name, window, SAMPLE_RATE)
                    cleaned = f.name.replace(".wav", "_c.wav")
                    extract_voice_only(f.name, cleaned)

                    features = extract_features(cleaned)
                    state, confidence, deltas = infer_state(
                        features, baseline=baseline
                    )

                timestamp = datetime.now().strftime("%H:%M:%S")
                results.append({
                    "time": timestamp,
                    "state": state,
                    "confidence": confidence
                })

                # 터미널 출력
                emoji = {"happy": "😊", "tired": "😴",
                         "stressed": "😤"}.get(state, "😐")
                bar = "█" * int(confidence * 20)
                print(f"[{timestamp}] {emoji} {state:8s} | "
                      f"확신도: {bar} {confidence:.0%}")

        except KeyboardInterrupt:
            print("\n\n모니터링 종료")
            return results