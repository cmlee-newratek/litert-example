# 프루닝 예제 (Pruning Examples)

MNIST 데이터셋을 기반으로 TensorFlow 모델에 프루닝(Pruning)을 적용하여 희소(sparse) 모델을 생성하는 예제입니다.

## 📋 개요

프루닝(Pruning)은 훈련 중에 중요하지 않은 가중치를 점진적으로 0으로 만들어 희소(sparse) 모델을 생성하는 기법입니다. 희소 모델은 압축이 용이하며, 0인 가중치를 건너뛰어 추론 속도를 개선할 수 있습니다.

프루닝과 양자화를 결합하면 최대 압축 효과 (10-15x)를 얻을 수 있습니다.

## 🎯 목표

- 프루닝의 개념과 작동 방식 이해
- Magnitude-based pruning 적용 방법 학습
- 희소성(sparsity)과 압축률의 관계 이해
- 프루닝과 양자화 결합 방법 학습
- 정확도와 모델 크기의 트레이드오프 학습
- Raspberry Pi 4에서 실전 성능 측정

## 📂 예제 구조

```
02-pruning/
├── [PC에서 실행] ──────────────────────────────
│   ├── 01-basic-pruning.py               # 기본 프루닝 예제
│   ├── 02-pruning-with-quantization.py  # 프루닝 + 양자화 결합 예제
│   └── create_models.py                  # [PC] 모든 모델 생성
│
├── [Raspberry Pi에서 실행] ───────────────────────
│   └── benchmark_rpi4.py                 # [Pi] 성능 벤치마크
│
├── mnist_pruned_models/               # (자동생성) 모델 저장 디렉토리
│   ├── mnist_model_baseline.tflite
│   ├── mnist_model_pruned.tflite
│   ├── mnist_model_baseline_quant.tflite
│   ├── mnist_model_pruned_quant.tflite
│   ├── mnist_model_pruned_int8.tflite
│   ├── mnist_test_images.npy
│   ├── mnist_test_labels.npy
│   └── benchmark_results_rpi4.json    # (Pi) 벤치마크 결과
│
└── README.md                          # 이 파일
```

## 🔍 프루닝이란?

### Magnitude-based Weight Pruning

Magnitude-based pruning은 가중치의 크기(magnitude)를 기준으로 중요도가 낮은 가중치를 점진적으로 0으로 만드는 기법입니다.

**작동 방식**:

1. 훈련 중에 각 가중치의 크기를 평가
2. 크기가 작은 가중치부터 점진적으로 0으로 설정
3. 남은 가중치는 계속 훈련하여 정확도 유지
4. 최종적으로 설정된 희소성(sparsity) 비율 달성

### 주요 특징

**장점**:

- 모델 압축률 향상 (gzip 압축 시 최대 6x)
- 정확도 손실 최소화
- 양자화와 결합 가능 (PQAT)
- 추론 속도 개선 가능 (프레임워크 지원 시)

**단점**:

- 재훈련(fine-tuning) 필요
- 훈련 시간 증가
- 프레임워크가 희소 연산을 지원해야 추론 속도 개선

## 🚀 시작하기

### 필수 요구사항

**pip를 사용한 설치**:

```bash
pip install -r requirements.txt
```

**개별 패키지 설치**:

```bash
pip install tensorflow tensorflow-model-optimization numpy
```

### 실행

**개별 예제 실행**:

```bash
# 기본 프루닝
python 01-basic-pruning.py

# 프루닝 + 양자화
python 02-pruning-with-quantization.py
```

**모든 모델 한 번에 생성**:

```bash
python create_models.py
```

## � 실행 가이드

### 1단계: PC에서 모델 생성

**방법 A: 모든 모델 한 번에 생성 (권장)**

```bash
python create_models.py
```

이 명령은:

- MNIST 데이터셋 다운로드 및 훈련
- 5가지 모델 변형 생성
- 모든 모델을 `mnist_pruned_models/` 디렉토리에 저장
- 테스트 데이터 numpy 파일로 저장

**방법 B: 개별 예제 실행**

```bash
# 기본 프루닍 예제
python 01-basic-pruning.py

# 프루닍 + 양자화 예제
python 02-pruning-with-quantization.py
```

### 2단계: Raspberry Pi 4로 전달

생성된 `mnist_pruned_models/` 디렉토리를 Raspberry Pi 4로 복사합니다.

**방법: 전체 저장소 복제**

```bash
# Pi에서 전체 저장소 복제
git clone https://github.com/newracom/litert-example.git
cd litert-example/examples/01-pruning

# PC에서 모델 생성 후 복사
```

### 3단계: Raspberry Pi 4에서 벤치마크 실행

Raspberry Pi 4에서 다음을 실행합니다.

```bash
cd examples/01-pruning

# 벤치마크 수행 (미리 생성된 모델들 사용)
python benchmark_rpi4.py
```

이 명령은:

- 시스템 정보 표시 (ARM 아키텍처, CPU, 메모리)
- 각 모델의 정확도 평가
- 각 모델의 추론 속도 측정 (50회 반복)
- 결과를 `benchmark_results_rpi4.json`에 저장

## 📊 PC vs Raspberry Pi 실행 환경

### 각 스크립트의 역할

| 스크립트                          | 실행 환경        | 목적      | 설명                                                               |
| --------------------------------- | ---------------- | --------- | ------------------------------------------------------------------ |
| `01-basic-pruning.py`             | PC               | 학습      | 기본 프루닝 개념 학습<br>모델 생성부터 평가까지 완전한 과정        |
| `02-pruning-with-quantization.py` | PC               | 학습      | 프루닝+양자화 결합 학습<br>최대 압축 목표                          |
| `create_models.py`                | **PC**           | 모델 생성 | 모든 모델을 한 번에 생성<br>Raspberry Pi용 모델 파일 생성          |
| `benchmark_rpi4.py`               | **Raspberry Pi** | 성능 측정 | 실제 Pi 환경에서 모델 성능 측정<br>ARM CPU의 실제 성능 데이터 수집 |

### 리소스 요구사항

```
PC (create_models.py):
  - 메모리: 8 GB 이상 추천
  - 저장소: 500 MB (모델 + 데이터)
  - 실행 시간: 5-10분 (GPU 있으면 더 빠름)

Raspberry Pi 4 (benchmark_rpi4.py):
  - 메모리: 2 GB 이상 (충분함)
  - 저장소: 200 MB (모델들만)
  - 실행 시간: 30초-1분
```

## 📊 기대 성능 (Raspberry Pi 4)

### create_models.py

모든 프루닝 모델을 한 번에 생성하는 스크립트입니다.

**생성되는 모델**:

1. Baseline Float32
2. Pruned Float32
3. Baseline + Quantization
4. Pruned + Quantization
5. Pruned + Int8 Quantization

**사용 시기**:

- PC에서 모든 모델을 한 번에 생성
- Raspberry Pi로 전달하여 벤치마크
- 개별 예제 실행 시간 절약

### benchmark_rpi4.py

Raspberry Pi 4에서 모델 성능을 측정하는 벤치마크 스크립트입니다.

**측정 항목**:

- 정확도 (Accuracy)
- 모델 크기 (KB)
- 추론 시간 (ms)
- FPS (Frames Per Second)
- 시스템 정보 (CPU, 메모리 등)

**최적화 항목**:

- TensorFlow 또는 tflite-runtime 지원
- numpy 파일 또는 HTTP 다운로드로 MNIST 데이터 로드
- 결과 JSON 파일로 저장

**실행 방법**:

```bash
# Raspberry Pi 4에서
cd examples/02-pruning
python benchmark_rpi4.py
```

## 💡 핵심 개념

### 1. 희소성 (Sparsity)

희소성은 모델의 가중치 중 0인 비율을 나타냅니다.

```
희소성 = (0인 가중치 수) / (전체 가중치 수) × 100%
```

- **낮은 희소성 (10-30%)**: 정확도 손실 거의 없음, 압축률 낮음
- **중간 희소성 (50-70%)**: 균형잡힌 트레이드오프 (권장)
- **높은 희소성 (80-90%)**: 높은 압축률, 정확도 손실 가능

### 2. Polynomial Decay 스케줄

프루닝 비율을 점진적으로 증가시키는 스케줄입니다.

```
sparsity(t) = final_sparsity + (initial_sparsity - final_sparsity)
              × (1 - (t - begin_step) / (end_step - begin_step))^3
```

이는 초기에는 천천히, 후반에는 빠르게 프루닝을 적용하여 모델이 적응할 시간을 줍니다.

### 3. 압축과 프루닝

프루닝된 모델은 그 자체로는 크기가 줄어들지 않습니다 (0도 저장 공간 필요). 하지만 압축 알고리즘(gzip, zip 등)과 결합하면:

- 0이 많을수록 압축률 향상
- 일반적으로 2-6x 압축률 달성
- 네트워크 전송이나 저장 시 유리

### 4. 프루닝 + 양자화 (PQAT)

프루닝과 양자화를 결합하면 최대 압축 효과를 얻을 수 있습니다:

1. **프루닝**: 가중치를 희소하게 만듦
2. **양자화**: Float32 → Int8 변환
3. **압축**: gzip 등으로 압축

이 조합으로 10-15배 크기 감소 가능합니다.

## 🎛️ 프루닝 파라미터 조정

### initial_sparsity & final_sparsity

```python
# 보수적 프루닝 (정확도 우선)
initial_sparsity=0.0
final_sparsity=0.3

# 표준 프루닝 (균형)
initial_sparsity=0.0
final_sparsity=0.5

# 공격적 프루닝 (압축 우선)
initial_sparsity=0.0
final_sparsity=0.8
```

### begin_step & end_step

```python
# 짧은 프루닝 (빠른 수렴)
epochs = 2
end_step = 훈련스텝 × epochs

# 긴 프루닝 (안정적)
epochs = 5
end_step = 훈련스텝 × epochs
```

### Pruning Schedule 종류

```python
# 1. PolynomialDecay (권장)
tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,
    begin_step=0,
    end_step=end_step,
    power=3  # 곡선 정도 (1=선형, 3=급격)
)

# 2. ConstantSparsity (단순)
tfmot.sparsity.keras.ConstantSparsity(
    target_sparsity=0.5,
    begin_step=0,
    end_step=end_step
)
```

## 📈 결과 분석

### 기대 효과

**MNIST 모델 기준** (50% 희소성):

| 항목             | 원본   | 프루닝 후 | 변화 |
| ---------------- | ------ | --------- | ---- |
| 정확도           | ~97%   | ~96%      | -1%p |
| 모델 크기        | ~84 KB | ~84 KB    | 0%   |
| 압축 크기 (gzip) | ~25 KB | ~18 KB    | -28% |
| 희소성           | 0%     | 50%       | -    |

### 레이어별 희소성

일반적으로:

- **Conv 레이어**: 30-50% 희소성 (중요한 특징 보존)
- **Dense 레이어**: 50-80% 희소성 (과적합 방지)

## 🔧 문제 해결

### "정확도 손실이 너무 큼"

**해결 방법**:

1. 희소성 비율 낮추기 (0.5 → 0.3)
2. 프루닝 epochs 늘리기 (2 → 5)
3. 학습률 낮추기
4. 더 긴 baseline 훈련

### "압축 효과가 적음"

**확인 사항**:

1. 희소성이 실제로 달성되었는지 확인
2. gzip 압축을 사용하고 있는지 확인
3. 더 높은 희소성 시도 (0.5 → 0.7)

### "훈련 시간이 너무 오래 걸림"

**해결 방법**:

1. epochs 줄이기 (단, 정확도 손실 가능)
2. batch_size 늘리기
3. GPU 사용 (tensorflow-gpu)

## 🔄 다음 단계

프루닝을 익힌 후:

1. **양자화 학습**: `examples/01-quantization/` 참고
2. **PQAT**: 프루닝 + 양자화 결합 (02-pruning-with-quantization.py)
3. **실제 모델 적용**: 자신의 모델에 프루닝 적용
4. **하드웨어 최적화**: Raspberry Pi 4, EdgeTPU, ARM NEON 등

## 📚 참고 자료

- [TensorFlow Model Optimization - Pruning](https://www.tensorflow.org/model_optimization/guide/pruning)
- [Pruning with Keras](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras)
- [Pruning Comprehensive Guide](https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide)
- [Research Paper: To prune, or not to prune](https://arxiv.org/pdf/1710.01878.pdf)
