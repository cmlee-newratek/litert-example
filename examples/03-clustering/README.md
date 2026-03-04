# 클러스터링 (Model Clustering) 예제

TensorFlow와 TensorFlow Model Optimization Toolkit을 사용하여 **가중치 클러스터링**으로 모델을 압축하는 예제입니다.

## 📚 개념

### 가중치 클러스터링이란?

클러스터링은 모델의 가중치들을 **N개의 그룹(클러스터)으로 묶고**, 각 그룹을 하나의 **중심값(centroid)**으로 대체하는 기법입니다.

**예시:**

```
원본 가중치:   [0.1, 0.12, 0.11, 0.89, 0.91, 0.88]
클러스터 2개:  그룹 1: [0.1, 0.12, 0.11] → 중심값 0.11로 대체
              그룹 2: [0.89, 0.91, 0.88] → 중심값 0.89로 대체
결과 가중치:   [0.11, 0.11, 0.11, 0.89, 0.89, 0.89]
```

### 압축 효과

- **gzip 압축 시 효과적**: 반복되는 값들이 많아져 gzip이 더 잘 압축
- **고유 값 감소**: 예를 들어, 1,000개의 가중치가 16개의 고유 값으로 축소
- **정확도 유지**: 적절한 클러스터 수 사용 시 정확도 손실 최소화
- **추론 속도**: 원본과 동일 (압축은 저장 단계에서만 효과)

### 클러스터 초기화 방법

| 방법         | 설명                          | 사용 경우             |
| ------------ | ----------------------------- | --------------------- |
| **LINEAR**   | 가중치 범위를 균등하게 분할   | 기본값, 대부분의 경우 |
| **KMEANS++** | K-means++ 알고리즘으로 초기화 | 더 나은 압축 필요 시  |
| **DENSITY ** | 가중치 분포에 따라 초기화     | 특정 레이어 최적화    |

### 다른 최적화 기법과의 조합

```
Pruning (희소성):  0으로 설정하여 불필요한 연결 제거
Quantization:      가중치 비트 수 감소 (예: Float32 → Int8)
Clustering:        고유 가중치 개수 감소 (값 반복)

클러스터링 + 양자화:
  → 클러스터링으로 고유 값 감소 (예: 1000 → 16)
  → 양자화로 각 값을 더 작은 비트로 인코딩
  → 최대 압축 효과 (10-15배)
```

## 🚀 사용 방법

### 1. PC에서 모델 생성

#### 필수 설치

```bash
pip install -r requirements.txt
```

#### 예제 1: 기본 클러스터링 (01-basic-clustering.py)

```bash
python 01-basic-clustering.py
```

**출력:**

- `mnist_clustered_models/mnist_model_baseline.tflite` - 원본 Float32 모델
- `mnist_clustered_models/mnist_model_clustered.tflite` - 클러스터링된 모델 (16개 클러스터)
- 정확도, 모델 크기, 고유 값 수 출력

**학습 내용:**

- 클러스터링의 기본 작동 원리
- 고유 값 개수 분석
- gzip 압축 효과

#### 예제 2: 클러스터링 + 양자화 (02-clustering-with-quantization.py)

```bash
python 02-clustering-with-quantization.py
```

**생성 모델 (5가지):**

1. `mnist_model_baseline.tflite` - 원본 Float32
2. `mnist_model_clustered.tflite` - 클러스터링만
3. `mnist_model_baseline_quant.tflite` - 양자화만
4. `mnist_model_clustered_quant.tflite` - 클러스터링 + 양자화
5. `mnist_model_clustered_int8.tflite` - 클러스터링 + Int8

**출력:**

- 각 모델의 정확도
- TFLite 모델 크기 (압축 전)
- gzip 압축 후 크기
- 레이어별 고유 값 수

#### 배치 모델 생성 (create_models.py)

```bash
python create_models.py
```

**생성 모델:**

- `mnist_model_baseline.tflite`
- `mnist_model_clustered_8.tflite` (8개 클러스터)
- `mnist_model_clustered_16.tflite` (16개 클러스터)
- `mnist_model_clustered_32.tflite` (32개 클러스터)
- `mnist_model_clustered_16_quant.tflite` (16클러스터 + 양자화)
- `mnist_model_clustered_16_int8.tflite` (16클러스터 + Int8)
- `mnist_test_images.npy` - 테스트 이미지
- `mnist_test_labels.npy` - 테스트 레이블

**장점:**

- 모든 모델을 한 번에 생성
- 라즈베리파이로 배포하기 쉬운 구조
- 테스트 데이터도 함께 저장

### 2. Raspberry Pi 4에서 벤치마킹

#### 설정 1: tflite-runtime 사용 (권장)

```bash
# 가벼운 설치 (~50MB)
pip install -r requirements-rpi.txt  # tflite-runtime 선택
```

#### 설정 2: TensorFlow 사용

```bash
# 전체 TensorFlow 설치 (~500MB)
pip install tensorflow==2.13.1 numpy
```

#### 벤치마크 실행

```bash
# 모든 모델 벤치마킹
python benchmark_rpi4.py
```

**측정 항목:**

- **정확도**: 테스트 데이터셋의 정확도
- **추론 시간**: 평균, 중앙값, 최소, 최대 (ms)
- **FPS**: 초당 처리 이미지 수
- **모델 크기**: KB 단위

**출력 예시:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 벤치마크 결과 요약:
모델                   크기         정확도         평균 추론      FPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
baseline               27.89        98.50          250.45         4.0
clustered_8           23.45        98.10          251.23         4.0
clustered_16          21.34        98.30          249.87         4.0
clustered_32          19.56        97.80          250.12         4.0
clustered_16_quant     7.23        98.20          45.34          22.0
clustered_16_int8      6.12        98.10          38.21          26.2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 📊 기대 성능 (Raspberry Pi 4)

### 모델 크기 비교

| 모델                 | 크기     | 원본 대비 |
| -------------------- | -------- | --------- |
| 원본 Float32         | 27.89 KB | 100%      |
| 8 클러스터           | 23.45 KB | 84%       |
| 16 클러스터          | 21.34 KB | 76%       |
| 32 클러스터          | 19.56 KB | 70%       |
| 16 클러스터 + 양자화 | 7.23 KB  | 26%       |
| 16 클러스터 + Int8   | 6.12 KB  | 22%       |

### gzip 압축 후 (배포 시)

```
원본 Float32:        14.5 KB
16 클러스터:         11.2 KB (77%)
16 클러스터 + 양자화: 4.2 KB (29%)
```

### 추론 속도 (Raspberry Pi 4B, tflite-runtime)

```
Float32:           250ms/이미지 (4 FPS)
Int8 양자화:        45ms/이미지 (22 FPS)
클러스터링 + Int8:  38ms/이미지 (26 FPS)
```

**주요 특징:**

- ✅ 클러스터링만으로 20-30% 크기 감소
- ✅ 클러스터링 + 양자화로 최대 70% 감소
- ✅ 정확도는 <1% 손실 (충분히 작은 손실)
- ✅ Int8 양자화가 가장 빠른 추론 속도 제공

## 🔧 파라미터 튜닝

### 클러스터 수 선택

```python
clustering_params = {
    'number_of_clusters': 16,  # 조정 가능한 파라미터
    'cluster_centroids_init': tfmot.clustering.keras.CentroidsInitializer.LINEAR,
}
```

**클러스터 수별 효과:**
| 클러스터 수 | 압축율 | 정확도 | 추천 |
|----------|------|-----|-----|
| 8 | 낮음 | 높음 | 기본 |
| 16 | 중간 | 높음 | ✅ 추천 |
| 32 | 높음 | 중간 | 더 분석 필요 |
| 64+ | 매우 낮음 | 낮음 | 비추천 |

**선택 기준:**

- **작은 모델** (< 10MB): 16-32 클러스터
- **중간 모델** (10-100MB): 8-16 클러스터
- **큰 모델** (> 100MB): 4-8 클러스터

### 중심값 초기화 방법 비교

```python
# LINEAR (기본값, 빠름)
'cluster_centroids_init': tfmot.clustering.keras.CentroidsInitializer.LINEAR

# KMEANS++ (더 나은 압축, 느림)
'cluster_centroids_init': tfmot.clustering.keras.CentroidsInitializer.KMEANS_PLUS_PLUS

# DENSITY (특수 경우)
'cluster_centroids_init': tfmot.clustering.keras.CentroidsInitializer.DENSITY_BASED
```

## 📈 성능 측정

### 정확도 평가

```python
def evaluate_accuracy(model, test_data, test_labels):
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == test_labels)
    return accuracy
```

### 추론 속도 측정

```python
import time

# 워밍업
interpreter.invoke()

# 측정
times = []
for _ in range(10):
    start = time.perf_counter()
    interpreter.invoke()
    end = time.perf_counter()
    times.append((end - start) * 1000)  # ms

avg_time = np.mean(times)
fps = 1000 / avg_time
```

## ⚠️ 주의사항

### 클러스터링이 효과적이지 않은 경우

1. **배치 정규화 앞의 층**
   - Conv2D/Dense + BatchNormalization 구조에서 배치 정규화 전 층은 효과 감소
   - 이유: 배치 정규화가 이미 가중치를 정규화함

2. **매우 빈약한 모델**
   - 가중치 수가 적으면 클러스터링 효과 미미
   - 예: 작은 Dense 레이어

3. **복잡한 분포**
   - 가중치가 광범위한 값을 가질 때 클러스터링 효과 감소

### 정확도 손실 최소화

```python
# ✅ 좋은 방법: 작은 클러스터 수로 시작
for num_clusters in [256, 128, 64, 32, 16, 8]:
    # 테스트하고 정확도 차이 확인
    accuracy = evaluate_tflite(model)
    if accuracy > 0.98:  # 목표 정확도
        break

# ❌ 피할 방법: 무조건 작은 수로 설정
'number_of_clusters': 4  # 너무 공격적
```

## 🐛 문제 해결

### 문제: TFLite 변환 실패

```
ValueError: Failed to convert TFLiteConverter
```

**해결:**

```python
# 클러스터링 모델 정리 확인
model = tfmot.clustering.keras.strip_clustering(clustered_model)

# 모델이 실제로 정리되었는지 확인
print(model.summary())
```

### 문제: 라즈베리파이에서 tflite-runtime이 안 됨

```bash
# tflite-runtime 설치 실패
pip install tflite-runtime
```

**해결:**

```bash
# 방법 1: 미리 컴파일된 바이너리 사용
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.9.0-cp39-cp39-linux_armv7l.whl

# 방법 2: TensorFlow 사용
pip install tensorflow==2.13.1

# 방법 3: 소스에서 빌드
git clone https://github.com/tensorflow/runtime
# 컴파일 후 설치
```

### 문제: 메모리 부족 (Out of Memory)

```
MemoryError: Unable to allocate ... bytes
```

**해결:**

```python
# 배치 크기 줄이기
model.fit(
    train_images,
    train_labels,
    batch_size=32,  # 128에서 32로 줄이기
    epochs=2,
)

# 또는 호출 메모리 정리
import gc
gc.collect()
```

### 문제: 정확도가 너무 많이 떨어짐

```
원본: 98.5%, 클러스터링: 85.3% (13.2% 손실)
```

**해결:**

```python
# 1. 클러스터 수 증가
'number_of_clusters': 32  # 16에서 32로 증가

# 2. 훈련 시간 증가
epochs=5  # 2에서 5로 증가

# 3. 더 나은 초기화 사용
'cluster_centroids_init': tfmot.clustering.keras.CentroidsInitializer.KMEANS_PLUS_PLUS
```

## 📚 참고 자료

- **TensorFlow 클러스터링 가이드**: https://www.tensorflow.org/model_optimization/guide/clustering
- **TensorFlow Model Optimization Toolkit**: https://github.com/tensorflow/model-optimization
- **Deep Compression** (논문): https://arxiv.org/abs/1510.00149

## 📝 라이선스

이 예제는 MIT 라이선스를 따릅니다.
