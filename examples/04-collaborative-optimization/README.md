# 협업 최적화 (Collaborative Optimization) 예제

여러 최적화 기법(Pruning, Clustering, Quantization)을 조합하여 모델을 최대한 압축하는 고급 최적화 기법입니다.

## 📚 개념

### 협업 최적화란?

기존 최적화 기법들의 약점을 보완하기 위해 여러 기법을 **순차적으로 적용**하면서 이전 단계의 효과를 **보존**하는 방식입니다.

### 문제: 기법 충돌

```
순진한 조합 (효과 소실):
모델 → 프루닝 (스파시티 도입) → 클러스터링 (고유 값 감소)
       ✅ 50% 스파시티        ❌ 스파시티 소실!

협업 최적화 (효과 보존):
모델 → 프루닝 → 스파시티 보존 클러스터링 → 양자화
      ✅             ✅                       ✅
```

### 3가지 협업 최적화 경로

#### 1️⃣ CQAT (Cluster-preserving QAT)

```
기본 모델 → 클러스터링 → CQAT → Int8 양자화
           (16개)      (보존)

효과:
- 고유 가중치: 1000 → 16 (15배 감소)
- 양자화 인식 훈련으로 추가 정확도 유지
- 압축율: 30-35%
```

**사용 사례:**

- 가중치 분포가 양극단적인 모델 (예: 이미지 분류)
- 정확도가 매우 중요한 경우

#### 2️⃣ PQAT (Sparsity-preserving QAT)

```
기본 모델 → 프루닝 → PQAT → Int8 양자화
          (50%)   (보존)

효과:
- 스파시티: 50%의 가중치가 0
- 양자화 인식 훈련으로 남은 가중치 최적화
- 압축율: 35-40%
- gzip 압축 시 추가 효율 (반복되는 0)
```

**사용 사례:**

- 가중치들이 다양한 크기를 가진 모델
- 하드웨어가 희소성 활용 가능한 경우

#### 3️⃣ PCQAT (Sparsity-Cluster-preserving QAT) ⭐

```
기본 모델 → 프루닝 → 클러스터링 → PCQAT → Int8 양자화
          (50%)    (스파시티+클러스터링 보존)

효과:
- 스파시티: 50% (프루닝)
- 고유 값: 16개 (클러스터링)
- 양자화 인식 훈련 (모든 최적화 보존)
- 압축율: 45-50% (최고!)
```

**사용 사례:**

- 최대 압축이 필요한 엣지 디바이스
- 메모리가 극도로 제한된 환경

### 협업 최적화 트리

```
                    기본 모델 (Float32)
                          |
        ________________________|________________________
        |                       |                       |
    프루닝 ← 기울임               클러스터링              QAT
        |                       |                       |
    + QAT                   + QAT                    (단순)
    (PQAT)                 (CQAT)                     |
        |                       |                    배포
        |_______________________|
                   |
            프루닝 + 클러스터링
                   |
                + QAT
              (PCQAT)
                   |
                배포 ✅
```

## 🚀 사용 방법

### 1. PC에서 모델 생성

#### 필수 설치

```bash
pip install -r requirements.txt
```

#### 예제 1: CQAT (01-cqat.py)

```bash
python 01-cqat.py
```

**학습 내용:**

- 클러스터링 + 양자화 인식 훈련의 조합
- 클러스터링 효과를 보존하는 방법

**출력:**

```
기본 모델:         98.50%
클러스터링:        98.30%
CQAT (Int8):       98.20%

모델 크기:
기본:              27.89 KB (100%)
클러스터링:        21.34 KB (76%)
CQAT:              7.23 KB (26%)
```

#### 예제 2: PQAT (02-pqat.py)

```bash
python 02-pqat.py
```

**학습 내용:**

- 프루닝 + 양자화 인식 훈련의 조합
- 스파시티 효과를 보존하는 방법

**출력:**

```
기본 모델:         98.50%
프루닝 (50%):      98.10%
PQAT (Int8):       98.05%

모델 크기:
기본:              27.89 KB (100%)
프루닝:            23.45 KB (84%)
PQAT:              6.78 KB (24%)
```

#### 예제 3: PCQAT (03-pcqat.py) ⭐

```bash
python 03-pcqat.py
```

**학습 내용:**

- 3가지 최적화 기법의 완전 조합
- 모든 효과를 보존하는 파이프라인

**출력:**

```
기본 모델:                98.50%
프루닝 (50%):             98.10%
프루닝+클러스터링:        98.00%
PCQAT (Int8 - 최종):      97.95%

모델 크기:
기본 (Float32):           27.89 KB (100%)
프루닝:                   23.45 KB (84%)
프루닝+클러스터링:        18.56 KB (67%)
PCQAT (Int8):             5.12 KB (18%) ⭐
```

#### 배치 모델 생성 (create_models.py)

```bash
python create_models.py
```

**생성 모델:**

- `mnist_model_baseline.tflite` - 기본 모델
- `mnist_model_cqat.tflite` - CQAT 모델
- `mnist_model_pqat.tflite` - PQAT 모델
- `mnist_model_pcqat.tflite` - PCQAT 모델 (최종)
- `mnist_test_images.npy` - 테스트 이미지
- `mnist_test_labels.npy` - 테스트 레이블

### 2. Raspberry Pi 4에서 벤치마킹

#### 설정

```bash
# 옵션 1: tflite-runtime (가볍고 빠름)
pip install -r requirements-rpi.txt

# 옵션 2: TensorFlow
pip install tensorflow==2.13.1 numpy
```

#### 벤치마크 실행

```bash
python benchmark_rpi4.py
```

**결과 예시:**

```
┌─────────────────────────────────────────────────────────┐
│  모델           크기      정확도    평균 추론   FPS     │
├─────────────────────────────────────────────────────────┤
│  baseline       27.89     98.50%   250.45 ms   4.0    │
│  cqat           7.23      98.20%   45.34 ms    22.0   │
│  pqat           6.78      98.05%   43.12 ms    23.2   │
│  pcqat          5.12      97.95%   38.21 ms    26.2   │
└─────────────────────────────────────────────────────────┘
```

## 📊 기대 성능

### 압축 비교 (MB 단위)

| 모델           | 크기     | 압축율 |
| -------------- | -------- | ------ |
| 기본 (Float32) | 27.89 KB | 100%   |
| CQAT (Int8)    | 7.23 KB  | 26%    |
| PQAT (Int8)    | 6.78 KB  | 24%    |
| PCQAT (Int8)   | 5.12 KB  | 18%    |

### 정확도 유지

| 모델  | 정확도 | 손실  |
| ----- | ------ | ----- |
| 기본  | 98.50% | -     |
| CQAT  | 98.20% | 0.3%  |
| PQAT  | 98.05% | 0.45% |
| PCQAT | 97.95% | 0.55% |

### 추론 속도 (Raspberry Pi 4)

| 모델           | 시간   | FPS  |
| -------------- | ------ | ---- |
| 기본 (Float32) | 250 ms | 4.0  |
| CQAT (Int8)    | 45 ms  | 22.0 |
| PQAT (Int8)    | 43 ms  | 23.2 |
| PCQAT (Int8)   | 38 ms  | 26.2 |

## 🔧 파라미터 튜닝

### 클러스터 수 선택

```python
clustering_params = {
    'number_of_clusters': 16,  # 튜닝 가능
    'cluster_centroids_init': tfmot.clustering.keras.CentroidsInitializer.LINEAR,
}
```

| 클러스터 | 압축율   | 정확도 | 추천         |
| -------- | -------- | ------ | ------------ |
| 16       | 중간     | 높음   | ✅ DEFAULT   |
| 32       | 높음     | 중간   | 더 분석 필요 |
| 8        | 낮음     | 높음   | PCQAT에만    |
| 64+      | 매우낮음 | 낮음   | 비추천       |

### 프루닝 스파시티 선택

```python
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,  # 튜닝 가능 (30-75%)
        begin_step=0,
        end_step=total_steps,
    )
}
```

| 스파시티 | 압축     | 정확도 | 추천    |
| -------- | -------- | ------ | ------- |
| 30%      | 낮음     | 높음   | 보수적  |
| 50%      | 중간     | 높음   | ✅ 권장 |
| 75%      | 높음     | 중간   | 공격적  |
| 90%      | 매우높음 | 낮음   | 위험    |

## 🎯 최적화 선택 가이드

### CQAT 선택 시기

```
✅ 장점:
- 구현이 가장 간단
- 정확도 손실 최소
- 추론 속도 개선

❌ 단점:
- 압축율이 가장 낮음
- 희소성 없음
```

### PQAT 선택 시기

```
✅ 장점:
- 희소성 활용 가능
- 매체 크기 감소 (gzip)
- 중간 수준의 구현 복잡도

❌ 단점:
- PCQAT보다 압축율 낮음
- 희소성 지원 하드웨어 필요
```

### PCQAT 선택 시기 (권장)

```
✅ 장점:
- 최고의 압축율 (50% 이상)
- 모든 최적화 효과 포함
- 정확도도 충분히 유지

❌ 단점:
- 구현이 가장 복잡
- 훈련 시간 증가
- 정확도 손실 약간 증가
```

## ⚠️ 주의사항

### 정확도 손실 최소화

```python
# ✅ 좋은 방법: 단계적으로 진행
# 1. 프루닝 최적화
# 2. 클러스터링 추가
# 3. 양자화 최종

# ❌ 피할 방법: 파라미터 과다
final_sparsity=0.9  # 너무 높음
'number_of_clusters': 4  # 너무 적음
```

### 훈련 시간 관리

- PCQAT는 3단계 훈련 (프루닝 → 클러스터링 → QAT)
- 각 단계마다 2-3 epoch 권장
- 전체 훈련 시간: 단순 모델의 6배

### 메모리 문제

```bash
# 메모리 부족 시 배치 크기 감소
model.fit(
    train_images,
    train_labels,
    batch_size=32,  # 128에서 32로
    epochs=2,
)
```

## 📈 성능 측정 예시

### 레이어별 분석

```python
for layer in model.layers:
    if len(layer.weights) > 0:
        weight = layer.weights[0].numpy()
        unique = len(np.unique(weight))
        print(f"{layer.name}: {unique} 고유값")

# 출력:
# conv2d: 16 고유값 (클러스터링)
# dense: 16 고유값 (클러스터링)
```

### 스파시티와 클러스터링 시각화

```python
# 프루닝 전후 가중치 분포
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 원본
axes[0].hist(original_weights.flatten(), bins=50)
axes[0].set_title('Original Weights')

# 프루닝
axes[1].hist(pruned_weights.flatten(), bins=50)
axes[1].set_title('After Pruning (50% sparsity)')

# PCQAT
axes[2].hist(pcqat_weights.flatten(), bins=50)
axes[2].set_title('After PCQAT')

plt.tight_layout()
plt.show()
```

## 🐛 문제 해결

### 문제: 클러스터링이 스파시티를 소실

```python
# ❌ 잘못된 방법
pruned_model → 저장 → 다시 로드 → 클러스터링

# ✅ 올바른 방법
pruned_model → 프루닝 래퍼 제거 → 클러스터링 적용
```

### 문제: PCQAT 정확도가 너무 떨어짐

```
추천 순서:
1. 스파시티 75% → 50%로 감소
2. 클러스터 수 8 → 16으로 증가
3. QAT 에폭 2 → 3으로 증가
```

### 문제: 라즈베리파이에서 추론 느림

```bash
# 1. Int8 양자화 확인
if interpreter.get_input_details()[0]['dtype'] != np.int8:
    print("⚠️ Int8 양자화가 아닙니다")

# 2. tflite-runtime 사용 (TensorFlow 대신)
pip install tflite-runtime

# 3. 배치 처리 고려
```

## 📚 참고 자료

- **TensorFlow 협업 최적화**: https://www.tensorflow.org/model_optimization/guide/combine/collaborative_optimization
- **CQAT 예제**: https://www.tensorflow.org/model_optimization/guide/combine/cqat_example
- **PQAT 예제**: https://www.tensorflow.org/model_optimization/guide/combine/pqat_example
- **PCQAT 예제**: https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example
- **Deep Compression (논문)**: https://arxiv.org/abs/1510.00149

## 📝 라이선스

이 예제는 MIT 라이선스를 따릅니다.
