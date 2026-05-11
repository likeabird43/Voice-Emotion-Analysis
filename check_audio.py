import soundfile as sf
import os
from pathlib import Path
from collections import Counter

audio_dir = Path("/Users/j/Desktop/AI_Projects/voice-mood-device/audio")  # 실제 오디오 폴더 경로로 바꿔주세요

results = []
for wav in sorted(audio_dir.rglob("*.wav")):
    info = sf.info(str(wav))
    results.append({
        "file": wav.name,
        "label": wav.parent.name,
        "sr": info.samplerate,
        "channels": info.channels,
        "duration": round(info.duration, 2),
    })

# 샘플레이트 분포
srs = Counter(r["sr"] for r in results)
chs = Counter(r["channels"] for r in results)
print(f"총 파일 수: {len(results)}")
print(f"샘플레이트 분포: {dict(srs)}")
print(f"채널 분포: {dict(chs)}")

# 문제 파일만 출력
print("\n=== sr이 섞인 파일들 ===")
dominant_sr = srs.most_common(1)[0][0]
for r in results:
    if r["sr"] != dominant_sr:
        print(f"  {r['label']}/{r['file']} → sr={r['sr']}, ch={r['channels']}")