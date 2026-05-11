# Voice Mood Device
### Baseline-Relative Acoustic State Inference for Personalized Smartphone UX

> *"Instead of classifying emotion, we detect deviation from your own normal."*

---

## The Problem

Current voice assistants respond only to explicit commands.  
But users constantly express their state indirectly through natural speech:

- *"I'm so tired today."*
- *"This is stressing me out."*
- *"I'm actually in a good mood."*

Existing Speech Emotion Recognition (SER) systems try to solve this by training on large multi-speaker datasets and predicting absolute emotion labels. This approach has a fundamental flaw: **emotional expression is deeply personal**. The acoustic signature of "stressed" varies enormously between individuals.

---

## The Insight

**A smartphone is almost always used by the same person.**

This single observation changes everything. Instead of asking *"does this voice sound stressed?"*, we can ask *"does this voice sound different from how this person normally sounds — and in which direction?"*

This shift — from absolute classification to **personal deviation detection** — is the core idea of this project.

---

## What Makes This Different

| Conventional SER | This Project |
|---|---|
| Multi-speaker datasets (acted) | Single-speaker, real-world recordings |
| Absolute emotion labels | Deviation from personal baseline |
| Clean studio audio | Noisy environments (cafe, outdoor) |
| Hard classification | Confidence-aware inference |
| No UX output | Adaptive smartphone suggestions |
| Manual neutral enrollment | Automatic baseline segment detection |
| Standard 3-band spectral features | Custom 5-band + interpretable acoustic indices |

---

## Key Acoustic Findings

These findings emerged from **perceptual observation** of the recordings, then were **statistically verified** on the full dataset.

### Finding 1 — Sample Rate Mismatch Corrupts Features
Initial extraction mixed 44100Hz stereo and 48000Hz mono files.  
This caused a **230Hz gap in spectral centroid** on the stressed class — making features incomparable across classes.  
All audio was re-standardized to **16kHz mono** before analysis.

### Finding 2 — Tired Is Cleanly Separable by Energy
Fatigue consistently reduces vocal energy.

```
tired energy mean:   0.0647
others energy mean:  0.0980
t-test p-value:      0.000000  ★
```

### Finding 3 — Stressed vs Happy: Frequency Band Signature
By listening to the recordings, a perceptual hypothesis emerged:
- Stressed speech feels "heavier" in the **1–2kHz range**
- Happy speech feels more resonant in the **500Hz–1kHz** range

Verified statistically on 114 utterances:

```
b3 (1k–2kHz):   stressed > happy              p=0.046  ★
b2 (500–1kHz):  stressed_intense higher        p=0.025  ★
6k–8kHz band:   happy > stressed               p=0.003  ★
```

Splitting the standard 3-band feature into **5 narrower bands** captured these signals directly.

### Finding 4 — Two Interpretable Acoustic Indices

Inspired by Spotify's audio features (valence, energy), two custom indices were designed from acoustic observations:

**VTS (Vocal Tension Score)** — how tense the voice sounds
```
VTS = 0.35 × pitch_std + 0.25 × b3(1-2kHz) + 0.25 × energy - 0.15 × b2(500-1kHz)

tired:     VTS = 0.250  (low tension)
happy:     VTS = 0.415
stressed:  VTS = 0.436  (high tension)
```

**VOI (Vocal Openness Index)** — how open/bright the voice sounds
```
VOI = 0.40 × b2(500-1kHz) + 0.35 × b_6k8k - 0.25 × b3(1-2kHz)

stressed:  VOI = 0.205  (closed, heavy)
tired:     VOI = 0.177  (low energy)
happy:     VOI = 0.275  (open, bright)  ★ happy vs stressed p=0.004
```

These indices are **human-interpretable** — not just model inputs, but meaningful acoustic descriptors.

---

## Technical Pipeline

```
Raw Audio (any sr, any channel)
        │
        ▼
┌─────────────────────┐
│   Standardization   │  sr=16kHz, mono
│   VAD / Noise Red.  │
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│       Feature Extraction        │
│                                 │
│  • pitch, energy, speech rate   │
│  • 5-band spectral ratios       │
│  • MFCC (13 coefficients)       │
│  • VTS / VOI (custom indices)   │
│  • MFCC energy, var, slope      │
└─────────────────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌───────────────┐       ┌──────────────────────┐
│   Baseline    │       │   State Inference    │
│   Detection   │       │   Random Forest      │
│               │       │                      │
│ Auto-detects  │       │  F1=0.816            │
│ neutral segs  │       │  Accuracy=0.82       │
│ → profile     │       │  (StratifiedKFold-5) │
└───────────────┘       └──────────────────────┘
        │                          │
        └──────────┬───────────────┘
                   ▼
        ┌─────────────────────┐
        │   Delta Features    │  current − baseline
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Suggestion Engine  │
        │                     │
        │  tired    → reduce  │
        │             notifs  │
        │  stressed → focus   │
        │             mode    │
        │  happy    → keep    │
        │             current │
        └─────────────────────┘
```

---

## Results

| Metric | Value |
|---|---|
| **CV F1-macro** (StratifiedKFold-5) | **0.816 ± 0.084** |
| **Accuracy** | **0.82** |
| Happy F1 | 0.78 |
| Stressed F1 | 0.78 |
| Tired F1 | 0.89 |

> Achieved on real-world noisy recordings (cafe + outdoor), single speaker, 114 utterances.  
> Competitive with noisy-condition SER literature (typical range: 0.60–0.75).

---

## Dataset

- **114 utterances** recorded by a single speaker (Samsung Galaxy, same device)
- **Environments:** cafe (noisy), outdoor (noisy) — real-world conditions
- **Labels:** happy (38), tired (35), stressed (41, includes stressed_intense)
- **Baseline:** 32 neutral utterances for personal calibration
- **Duration:** 4–8 seconds per utterance
- **Processing:** Logic Pro export → Waves NS1 noise reduction → 16kHz mono standardization

No acted or studio-recorded data. All recordings reflect natural speech in noisy real-world conditions.

---

## Project Structure

```
voice-mood-device/
│
├── analysis.ipynb                    # Full acoustic analysis & findings
├── rf_emotion_model.py               # Final classifier (F1=0.816)
├── extract_features.py               # Feature extraction pipeline
├── extract_highband.py               # High-frequency band extraction
├── check_audio.py                    # Sample rate verification
│
├── baseline_segment_detector.py      # Auto-detects neutral speech segments
├── voice_analysis.py                 # Core feature extraction
├── suggestion_engine.py              # Confidence-aware UX suggestions
├── realtime_monitor.py               # Live voice monitoring
├── noise_reducer.py                  # Noise reduction preprocessing
├── vad_preprocessor.py               # Voice activity detection
├── app.py                            # Gradio demo app
│
├── features_standardized.csv         # Extracted features (sr=16kHz mono)
├── highband_features.csv             # High-frequency band features
├── plots/                            # Analysis visualizations
└── audio/                            # Raw recordings (not included)
    ├── cafe/
    └── outside/
```

---

## Setup

```bash
git clone https://github.com/likeabird43/Voice-Emotion-Analysis
cd Voice-Emotion-Analysis
pip install -r requirements.txt
```

**Run the classifier:**
```bash
python rf_emotion_model.py
```

**Run the analysis notebook:**
```bash
jupyter notebook analysis.ipynb
```

**Re-extract features from audio:**
```bash
python extract_features.py
python extract_highband.py
```

---

## Limitations & Next Steps

**Current limitations:**
- Single speaker — calibration session required for new users
- 114 utterances — small dataset by ML standards
- Happy/stressed confusion remains (both show elevated arousal)

**Planned improvements:**
- Personal baseline calibration app (onboarding session)
- wav2vec2 / HuBERT fine-tuning for multi-speaker generalization
- Continuous stress intensity estimation
- Mobile app with background audio collection (user consent required)

---

## Related Work

| | This Project | Existing Approaches |
|---|---|---|
| Data | Real-world noisy, single speaker | Acted, studio (RAVDESS, IEMOCAP) |
| Features | Custom acoustic indices (VTS/VOI) | Standard MFCC, pitch |
| Baseline | Auto-detected from speech | Manual enrollment (Patent US10943604) |
| Output | UX suggestion + confidence | Emotion label |

The key contribution is **automatic neutral segment detection** from natural speech — unlike Patent US10943604 which requires manual neutral enrollment.

---

## Tools & Stack

`Python` `librosa` `scikit-learn` `Gradio` `Logic Pro` `Waves NS1` `NumPy` `pandas` `scipy` `matplotlib`

---

## Citation

If you use this work, please cite:
```
@misc{voicemooddevice2025,
  author = {Janet},
  title  = {Voice Mood Device},
  year   = {2025},
  url    = {https://github.com/likeabird43/Voice-Emotion-Analysis}
}
```
