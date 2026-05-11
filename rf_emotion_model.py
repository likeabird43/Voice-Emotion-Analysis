"""
rf_emotion_model.py
--------------------
Voice Mood Device — 감정 분류 모델 (최종 버전)
사용 데이터: features_standardized.csv + highband_features.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
# features_standardized.csv: 모든 오디오를 sr=16kHz mono로 표준화 후 추출한 피처
# highband_features.csv:     원본 sr 그대로 추출한 고주파(4k~) 밴드 피처
df = pd.read_csv("features_standardized.csv")
hb = pd.read_csv("highband_features.csv")

# baseline, outside_mixed는 분류 대상이 아니라 제외
# stressed_intense는 stressed로 통합 (음향적으로 같은 클래스, 강도만 다름)
main = df[df['label'].isin(['happy', 'tired', 'stressed', 'stressed_intense'])].copy()
main['label'] = main['label'].replace({'stressed_intense': 'stressed'})

# 고주파 피처 합치기 (file + environment 기준으로 정확히 매칭)
main = main.merge(hb[['file', 'environment', 'b_6k_8k']], on=['file', 'environment'], how='left')

print(f"총 샘플 수: {len(main)}")
print(main['label'].value_counts())
print()


# ── 2. 파생 피처 생성 ─────────────────────────────────────────────────────────
# 피처를 0~1 범위로 정규화하는 함수
# 왜? 서로 단위가 다른 피처들(Hz, RMS 등)을 같은 스케일로 만들기 위해
def normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


# VTS (Vocal Tension Score): 목소리 긴장도 점수
# - pitch_std 높을수록: 음정이 불안정 = 긴장
# - b3_1k_2k 높을수록: 1-2kHz 에너지 높음 = 스트레스 특성
# - energy_rms 높을수록: 볼륨 큼 = 긴장
# - b2_500_1k 낮을수록: 중저음 적음 = 해피 특성 아님
# → 높을수록 stressed, 낮을수록 tired
main['vts'] = (
    0.35 * normalize(main['pitch_std']) +
    0.25 * normalize(main['b3_1k_2k']) +
    0.25 * normalize(main['energy_rms']) -
    0.15 * normalize(main['b2_500_1k'])
)

# VOI (Vocal Openness Index): 목소리 밝음/열림 점수
# - b2_500_1k 높을수록: 중저음 풍부 = 편안하고 열린 목소리
# - b_6k_8k 높을수록: 고주파 에너지 높음 = happy 특성 (p=0.003 검증됨)
# - b3_1k_2k 낮을수록: stressed 특성이 아님
# → 높을수록 happy, 낮을수록 stressed/tired
main['voi'] = (
    0.40 * normalize(main['b2_500_1k']) +
    0.35 * normalize(main['b_6k_8k']) -
    0.25 * normalize(main['b3_1k_2k'])
)

# MFCC 파생 피처: MFCC 13개 계수로부터 스펙트럼 형태 요약
mfcc_cols = [f'mfcc_{i}' for i in range(1, 14)]

# mfcc_energy: 전체 MFCC 에너지 (스펙트럼 복잡도)
main['mfcc_energy'] = np.sqrt((main[mfcc_cols] ** 2).sum(axis=1))

# mfcc_var: MFCC 계수들의 분산 (스펙트럼 평탄도)
main['mfcc_var'] = main[mfcc_cols].var(axis=1)

# mfcc_slope: 저주파→고주파 방향 기울기 (스펙트럼 기울기)
x = np.arange(13)
main['mfcc_slope'] = main[mfcc_cols].apply(
    lambda row: np.polyfit(x, row.values, 1)[0], axis=1
)


# ── 3. 피처 선택 ──────────────────────────────────────────────────────────────
feat_cols = [
    # 기본 음향 피처
    'pitch_mean',           # 평균 음높이 (Hz)
    'pitch_std',            # 음높이 변동폭 (긴장도 지표)
    'energy_rms',           # 평균 볼륨
    'speech_rate',          # 발화 속도 (onset/초)
    'spectral_centroid',    # 스펙트럼 무게중심 (음색 밝기)
    'spectral_rolloff',     # 에너지 상한 주파수
    'spectral_bandwidth',   # 스펙트럼 폭
    'zcr',                  # 영교차율 (노이즈/음성 비율)

    # 5밴드 주파수 에너지 비율 (기존 3밴드에서 세분화)
    'b1_80_500',            # 저음 (80-500Hz)
    'b2_500_1k',            # 중저음 (500-1kHz) — happy 특성
    'b3_1k_2k',             # 중음 (1-2kHz) — stressed 특성
    'b4_2k_4k',             # 중고음 (2-4kHz)
    'b5_4k_8k',             # 고음 (4-8kHz)

    # MFCC 13개 (스펙트럼 형태)
] + mfcc_cols + [

    # 새로 만든 해석 가능한 피처
    'vts',                  # Vocal Tension Score
    'voi',                  # Vocal Openness Index
    'b_6k_8k',              # 6-8kHz 고주파 (happy 특성, p=0.003)
    'mfcc_energy',          # MFCC 에너지
    'mfcc_var',             # MFCC 분산
    'mfcc_slope',           # MFCC 기울기
]


# ── 4. 모델 학습 및 평가 ──────────────────────────────────────────────────────
le = LabelEncoder()
X  = main[feat_cols]
y  = le.fit_transform(main['label'])

# StratifiedKFold: 각 fold에서 클래스 비율을 유지하며 분리
# → 데이터가 적을 때 편향을 막는 올바른 방법
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 최적 하이퍼파라미터 (GridSearchCV로 탐색한 결과)
# n_estimators=200: 데이터가 114개로 작아서 300보다 200이 더 안정적
model = RandomForestClassifier(
    n_estimators=200,       # 결정 트리 개수
    max_depth=None,         # 트리 깊이 제한 없음
    max_features='sqrt',    # 각 트리에서 사용할 피처 수 (전체의 제곱근)
    min_samples_leaf=1,     # 리프 노드 최소 샘플 수
    min_samples_split=2,    # 분기 최소 샘플 수
    random_state=42
)

# cross_val_score: 데이터를 5등분해서 번갈아 test로 쓰는 교차 검증
# → 전체 데이터를 학습에도 쓰고 테스트에도 쓸 수 있는 올바른 방법
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

print("=== Cross Validation 결과 ===")
print(f"각 Fold F1: {scores.round(3)}")
print(f"평균 F1-macro: {scores.mean():.3f} ± {scores.std():.3f}")
print()


# ── 5. 클래스별 상세 성능 ────────────────────────────────────────────────────
# cross_val_predict: 각 샘플이 test fold에 있을 때의 예측값 수집
# → 전체 데이터에 대한 예측이지만 항상 unseen 데이터 기준
y_pred = cross_val_predict(model, X, y, cv=cv)

print("=== 클래스별 성능 ===")
print(classification_report(y, y_pred, target_names=le.classes_))


# ── 6. Confusion Matrix 저장 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y, y_pred,
    display_labels=le.classes_,
    ax=ax,
    colorbar=True
)
ax.set_title(f"Confusion Matrix (F1={scores.mean():.3f})")
plt.tight_layout()
plt.savefig("plots/rf_confusion_matrix_final.png", dpi=150)
plt.show()
print("저장 완료 → plots/rf_confusion_matrix_final.png")


# ── 7. 피처 중요도 ───────────────────────────────────────────────────────────
model.fit(X, y)  # 전체 데이터로 학습해서 중요도 확인
importance_df = pd.DataFrame({
    'feature':    feat_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 15 피처 중요도 ===")
print(importance_df.head(15).to_string(index=False))
importance_df.to_csv("rf_feature_importance_final.csv", index=False)
print("저장 완료 → rf_feature_importance_final.csv")


# ── 8. 성능 향상 여정 ────────────────────────────────────────────────────────
print("\n=== 성능 향상 여정 ===")
print("시작 (기존 피처):          F1=0.666")
print("+ 5밴드 + VTS + VOI:      F1=0.752  (+0.086)")
print("+ 데이터 보강 (114개):    F1=0.795  (+0.043)")
print("+ 하이퍼파라미터 튜닝:    F1=0.806  (+0.011)  ← 최종")