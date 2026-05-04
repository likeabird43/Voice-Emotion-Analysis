from voice_analysis import extract_features, infer_state
from suggestion_engine import get_suggestions


def main():
    test_files = {
        "/Users/j/Desktop/voice-mood-device/audio/tired.wav": "오늘 너무 피곤하다",
        "/Users/j/Desktop/voice-mood-device/audio/stressed.wav": "아 오늘 진짜 바쁘다",
        "/Users/j/Desktop/voice-mood-device/audio/happy.wav": "오늘 기분 진짜 좋다",
    }

    for file_path, text_hint in test_files.items():
        print("\n===================================")
        print("Analyzing:", file_path)

        features = extract_features(file_path)
        baseline_path = "/Users/j/Desktop/voice-mood-device/audio/baseline.wav"
        baseline_features = extract_features(baseline_path)
        state, confidence, deltas = infer_state(
            features,
            text_hint=text_hint,
            baseline=baseline_features
        )
        suggestions = get_suggestions(state, confidence)

        
        print("\n[Baseline-aware Deltas]")
        print(f"Pitch Delta: {deltas['pitch_delta']:.2f}")
        print(f"Energy Delta: {deltas['energy_delta']:.4f}")
        print(f"Speech Rate Delta: {deltas['speech_rate_delta']:.2f}")

        print("\n[Acoustic Features]")
        print(f"Pitch: {features['pitch']:.2f}")
        print(f"Energy: {features['energy']:.4f}")
        print(f"Spectral Centroid: {features['spectral_centroid']:.2f}")
        print(f"Speech Rate: {features['speech_rate']:.2f}")
        print(f"Duration: {features['duration']:.2f}s")

        print("\n[Inference Result]")
        print(f"Text Hint: {text_hint}")
        print(f"Detected State: {state}")
        print(f"Confidence Score: {confidence:.2f}")

        print("\n[Suggested Smartphone Actions]")
        for i, suggestion in enumerate(suggestions, start=1):
            print(f"{i}. {suggestion}")


if __name__ == "__main__":
    main()