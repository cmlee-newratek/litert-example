# 양자화 예제 (Quantization Examples)

MNIST 데이터셋을 기반으로 TensorFlow 모델을 다양한 방식으로 양자화하여 LiteRT로 변환하는 예제입니다.

## 📋 개요

양자화는 모델의 파라미터와 활성화값의 정밀도를 줄여 모델 크기와 추론 속도를 개선하는 기법입니다.

## 🎯 목표

- 각 양자화 방식의 특징 이해
- MNIST 모델을 통한 실습
- 모델 크기와 추론 속도의 트레이드오프 학습
- 상황에 맞는 최적의 양자화 방식 선택

## 📂 예제 구조

```
02-quantization/
├── [PC에서 실행] ──────────────────────────────
│   ├── 01-float16-quantization.py           # Float16 양자화 개별 예제
│   ├── 02-dynamic-range-quantization.py    # 동적 범위 양자화 개별 예제
│   ├── 03-integer-quantization.py          # 정수(Int8) 양자화 개별 예제
│   ├── 04-quantization-aware-training.py   # QAT 개별 예제
│   ├── 05-int16-int8-quantization.py       # 16x8 양자화 개별 예제
│   └── create_models.py                     # [PC] 모든 모델 생성
│
├── [Raspberry Pi에서 실행] ────────────────────
│   └── benchmark_rpi4.py                    # [Pi] 성능 벤치마크
│
├── mnist_tflite_models/                     # (자동생성) 모델 저장 디렉토리
│   ├── mnist_model_float32.tflite
│   ├── mnist_model_quant_f16.tflite
│   ├── mnist_model_quant_dynamic.tflite
│   ├── mnist_model_quant_int8.tflite
│   ├── mnist_model_quant_qat.tflite
│   ├── mnist_model_quant_16x8.tflite
│   └── benchmark_results_rpi4.json          # (Pi) 벤치마크 결과
│
├── requirements.txt                         # Python 패키지 의존성
└── README.md                                # 이 파일
```

## 🔍 양자화 방식 비교

### 1. Float16 양자화

**파일**: `01-float16-quantization.py`

**특징**:

- 추가 데이터 필요 없음
- 모델 크기 ~50% 감소 (2x 축소)
- 정확도 손실 최소화
- GPU 연산 최적화

**사용 시기**:

- GPU가 있는 기기
- 정확도가 매우 중요한 경우
- 빠른 변환이 필요한 경우

**실행**:

```bash
python 01-float16-quantization.py
```

**공식 문서**:
https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_float16_quant

---

### 2. 동적 범위 양자화 (Dynamic Range)

**파일**: `02-dynamic-range-quantization.py`

**특징**:

- 추가 데이터 필요 없음
- 모델 크기 ~75% 감소 (4x 축소)
- Float16보다 더 큰 압축
- CPU/GPU 지원

**사용 시기**:

- 대부분의 일반적인 경우
- 특별한 데이터가 없는 경우
- 빠른 배포가 필요한 경우

**실행**:

```bash
python 02-dynamic-range-quantization.py
```

**공식 문서**:
https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quant

---

### 3. 정수 양자화 (Post-Training Integer)

**파일**: `03-integer-quantization.py`

**특징**:

- 대표 데이터셋 필요
- 모델 크기 ~75% 감소 (4x 축소)
- 가장 빠른 추론 (CPU)
- EdgeTPU 완벽 지원

**사용 시기**:

- EdgeTPU를 사용하려는 경우
- CPU 성능이 중요한 경우
- 임베디드 기기용

**실행**:

```bash
python 03-integer-quantization.py
```

**공식 문서**:
https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant

---

### 4. 양자화 인식 훈련 (QAT)

**파일**: `04-quantization-aware-training.py`

**특징**:

- 훈련 데이터 필요
- 정수 양자화보다 높은 정확도
- 훈련 시간 소요
- EdgeTPU 완벽 지원

**사용 시기**:

- 정확도가 매우 중요한 경우
- 충분한 훈련 데이터가 있는 경우
- 최고의 성능을 원하는 경우

**실행**:

```bash
python 04-quantization-aware-training.py
```

**공식 문서**:
https://www.tensorflow.org/model_optimization/guide/quantization/training_example

---

### 5. Int16 활성화 + Int8 가중치

**파일**: `05-int16-int8-quantization.py`

**특징**:

- 대표 데이터셋 필요
- 정수 양자화보다 높은 정확도
- 활성화에 민감한 모델에 적합
- 약 3-4x 모델 크기 감소

**사용 시기**:

- 활성화에 민감한 모델
- 더 높은 정확도가 필요한 경우
- 특수 하드웨어 지원시

**실행**:

```bash
python 05-int16-int8-quantization.py
```

**공식 문서**:
https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant_16x8

---

## 📊 성능 비교 (MNIST 기준)

### 모델 크기 비교

```
Float32 (원본)      ████████████████████ 100%
Float16 (2x)        ██████████ 50%
Dynamic Range (4x)  █████ 25%
Integer (4x)        █████ 25%
16x8 (3-4x)         ███████ 30%
```

### 참고: Raspberry Pi 4에서의 예상 성능

| 양자화 방식    | 모델 크기 | 추론 시간 | 정확도 | FPS  |
| -------------- | --------- | --------- | ------ | ---- |
| Float32        | ~100 KB   | ~250 ms   | 96-97% | ~4   |
| Float16        | ~50 KB    | ~200 ms   | 96-97% | ~5   |
| Dynamic Range  | ~25 KB    | ~180 ms   | 95-96% | ~5.5 |
| Integer (Int8) | ~25 KB    | ~100 ms   | 95-96% | ~10  |
| QAT            | ~25 KB    | ~120 ms   | 96-97% | ~8   |
| 16x8           | ~30 KB    | ~150 ms   | 96-97% | ~6.5 |

_실제 성능은 Raspberry Pi 모델과 시스템 상태에 따라 달라질 수 있습니다._

---

## 🚀 시작하기

### 필수 요구사항

**방법 1: requirements.txt 사용 (권장)**

```bash
pip install -r requirements.txt
```

**방법 2: 개별 설치**

```bash
pip install tensorflow tensorflow-model-optimization numpy
```

**환경별 설치 옵션:**

```bash
# PC - CPU만 사용
pip install tensorflow-cpu tensorflow-model-optimization numpy

# PC - GPU 사용 (CUDA, cuDNN 필요)
pip install tensorflow tensorflow-model-optimization numpy

# Raspberry Pi 4 - 64-bit OS
pip install tensorflow==2.11.0 numpy

# Raspberry Pi 4 - 메모리 제약시 (벤치마크만)
pip install tensorflow-lite-runtime numpy
```

### 선택 사항 (메모리/CPU 모니터링)

```bash
pip install psutil
```

### 단계별 실행 가이드

#### 1단계: PC에서 모델 생성

먼저 PC 또는 PowerPC 환경에서 모든 양자화 모델을 생성합니다.

```bash
# 모든 양자화 모델 생성 및 저장
python create_models.py
```

이 명령은:

- MNIST 데이터셋 다운로드 및 훈련
- 5가지 양자화 방식의 모델 생성
- 모든 모델을 `mnist_tflite_models/` 디렉토리에 저장
- 생성 완료 후 Raspberry Pi 4로 전달할 수 있게 함

**생성되는 파일** (약 200-300 KB):

- `mnist_model_float32.tflite` (원본)
- `mnist_model_quant_f16.tflite` (Float16)
- `mnist_model_quant_dynamic.tflite` (동적 범위)
- `mnist_model_quant_int8.tflite` (정수)
- `mnist_model_quant_qat.tflite` (QAT)
- `mnist_model_quant_16x8.tflite` (16x8)

#### 2단계: 개별 양자화 예제 실행 (선택사항)

각 양자화 방식에 대해 자세히 학습하려면 개별 스크립트를 실행하세요.

```bash
# Float16 양자화 예제
python 01-float16-quantization.py

# 동적 범위 양자화 예제
python 02-dynamic-range-quantization.py

# 정수 양자화 예제
python 03-integer-quantization.py

# QAT 예제
python 04-quantization-aware-training.py

# 16x8 양자화 예제
python 05-int16-int8-quantization.py
```

#### 3단계: Raspberry Pi 4로 전달

생성된 `mnist_tflite_models/` 디렉토리를 Raspberry Pi 4로 복사합니다.

**방법 1: scp를 사용하여 복사**

```bash
scp -r mnist_tflite_models pi@raspberrypi.local:/home/pi/
```

**방법 2: 직접 저장소 복제**

```bash
# Pi에서 전체 저장소 복제
git clone https://github.com/newracom/litert-example.git
cd litert-example/examples/02-quantization

# PC에서 모델 생성
python create_models.py  # (PC에서 먼저 실행)
```

#### 4단계: Raspberry Pi 4에서 벤치마크 실행

Raspberry Pi 4에서 다음을 실행합니다.

```bash
cd examples/02-quantization

# 벤치마크 수행 (미리 생성된 모델들 사용)
python benchmark_rpi4.py
```

이 명령은:

- 시스템 정보 표시 (ARM 아키텍처, CPU, 메모리)
- 각 모델의 정확도 평가
- 각 모델의 추론 속도 측정 (50회 반복)
- 결과를 `benchmark_results_rpi4.json`에 저장

**생성되는 결과 파일**:

```json
{
  "timestamp": "2026-02-27 10:30:45",
  "system_info": {
    "device": "Raspberry Pi",
    "arch": "ARMv8 (64-bit)",
    "total_memory_gb": 4.0,
    ...
  },
  "baseline_accuracy": "96.52%",
  "models": {
    "Float32": {
      "accuracy": "96.52%",
      "fps": "4.2",
      ...
    }
  }
}
```

---

## � PC vs Raspberry Pi 실행 환경

### 각 스크립트의 역할

| 스크립트            | 실행 환경        | 목적      | 설명                                                                                                         |
| ------------------- | ---------------- | --------- | ------------------------------------------------------------------------------------------------------------ |
| `01-05-*.py`        | PC               | 학습      | 각 양자화 방식의 개념 학습<br>모델 생성부터 평가까지 완전한 과정                                             |
| `create_models.py`  | **PC**           | 모델 생성 | 모든 양자화 방식의 모델을 한 번에 생성<br>Raspberry Pi용 모델 파일 생성                                      |
| `benchmark_rpi4.py` | **Raspberry Pi** | 성능 측정 | 실제 Pi 환경에서 모델 성능 측정<br>ARM CPU의 실제 성능 데이터 수집<br>메모리, CPU 사용량 등 시스템 정보 표기 |

### 리소스 요구사항

```
PC (create_models.py):
  - 메모리: 8 GB 이상 추천
  - 저장소: 500 MB (모델 + 데이터)
  - 실행 시간: 5-10분 (GPU 있으면 더 빠름)

Raspberry Pi 4 (benchmark_rpi4.py):
  - 메모리: 2 GB 이상 (충분함)
  - 저장소: 300 MB (모델들만)
  - 실행 시간: 30초-1분
```

---

```
시작: 양자화할 모델이 있는가?
  ├─ 아니오 → 예제 실행 후 자신의 모델에 적용
  └─ 예
     ↓
EdgeTPU를 사용하는가?
  ├─ 예 → 정수 양자화 또는 QAT 추천
  └─ 아니오
     ↓
추가 훈련 데이터가 있는가?
  ├─ 예 → QAT 추천 (최고 정확도)
  └─ 아니오
     ↓
정확도가 매우 중요한가?
  ├─ 예 → Float16 또는 16x8 추천
  └─ 아니오 → 동적 범위 양자화 추천
```

---

## 📈 결과 분석

### 모델 크기 최적화

정확도를 유지하면서 모델 크기를 최적화하는 순서:

1. **Float16**: 최소한의 정확도 손실로 2x 감소
2. **동적 범위**: 추가 데이터 없이 4x 감소
3. **정수 양자화**: 대표 데이터로 4x 감소, 가장 빠른 추론
4. **16x8**: 정수보다 더 정확, 3-4x 감소
5. **QAT**: 훈련을 통해 정확도 보존

### 추론 속도 최적화

최대 성능을 원할 때:

1. 정수 양자화 (가장 빠름)
2. 16x8 양자화
3. 동적 범위
4. QAT
5. Float16

### 정확도 보존

정확도가 중요할 때:

1. **QAT** (최고 정확도)
2. **16x8** (높은 정확도)
3. **Float16** (거의 손실 없음)
4. 동적 범위
5. 정수 양자화

---

## 🔧 문제 해결

### PC에서의 문제

#### "create_models.py 실행 중 메모리 부족"

```bash
# 해결방법 1: TensorFlow 메모리 제한
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=-1  # GPU 미사용

# 해결방법 2: 스크립트 수정 (일부 모델만 생성)
# create_models.py에서 불필요한 모델 생성 부분 주석 처리
```

#### "TensorFlow 설치 오류"

```bash
# CPU 버전 설치 (GPU 없을 경우)
pip install tensorflow-cpu

# GPU 버전 설치 (NVIDIA CUDA 필요)
pip install tensorflow-gpu
```

#### "정확도 손실이 너무 큼"

- QAT 시도 (훈련을 통한 양자화) - 가장 높은 정확도
- 16x8 양자화 시도
- 대표 데이터셋 개선 (더 많은 샘플: 1000으로 증가)
- 모델 아키텍처 검토 (더 깊은 네트워크 사용)

### Raspberry Pi 4에서의 문제

#### "benchmark_rpi4.py: 모델을 찾을 수 없음"

```bash
# 확인사항
1. mnist_tflite_models/ 디렉토리 확인
   ls -la mnist_tflite_models/

2. 모든 모델 파일이 있는지 확인
   - mnist_model_float32.tflite
   - mnist_model_quant_f16.tflite
   - mnist_model_quant_dynamic.tflite
   - mnist_model_quant_int8.tflite
   - mnist_model_quant_qat.tflite
   - mnist_model_quant_16x8.tflite

3. PC에서 create_models.py 재실행 후 복사
```

#### "Pi에서 실행 중 메모리 부족"

```bash
# 해결방법 1: 스왑 메모리 확인
free -h

# 해결방법 2: 백그라운드 프로세스 종료
sudo systemctl stop bluetooth
sudo systemctl stop avahi-daemon

# 해결방법 3: 더 많은 스왑 메모리 설정
sudo dphys-swapfile swapoff
# /etc/dphys-swapfile에서 CONF_SWAPSIZE=2048로 변경
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### "Pi에서 실행이 기대보다 느림"

주의: Raspberry Pi 4의 성능은 원래 제한적입니다.

```bash
# 성능 최적화
1. CPU 클록 스케일 확인
   /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

2. 전원 모드 설정
   performance 모드로 변경하면 약간 빨라짐

3. 컴파일 플래그 확인
   - ARMv8 (64-bit OS) vs ARMv7 (32-bit OS)
   - 64-bit가 ~10% 더 빠름

4. 배경 프로세스 종료
   - X11 (GUI) 종료
   - SSH를 통한 실행 권장
```

#### "Pi에서 TensorFlow 설치 오류"

```bash
# Raspberry Pi 4용 특정 버전 설치
pip install tensorflow==2.11.0

# 또는 최적화된 Pi용 빌드
pip install \
  https://github.com/PINTO0309/Tensorflow-bin/releases/download/v2.11.0/tensorflow-2.11.0-cp39-none-linux_aarch64.whl

# ARMv7 (32-bit) 경우
pip install tensorflow-lite-runtime
```

#### "벤치마크 결과가 예상과 다름"

```bash
# 원인 확인
1. 시스템 온도 확인 (과열 시 성능 저하)
   vcgencmd measure_temp

2. CPU 클록 스로틀링 확인
   grep throttled /proc/device-tree/thermal_zones/cpu-thermal/trip_point_0_hyst

3. 메모리 사용량 확인
   free -h

4. 다른 실행 중인 프로세스 확인
   ps aux | grep -v grep | grep -v benchmark
```

---

## 📚 참고 자료

- [LiteRT 공식 문서](https://ai.google.dev/edge/litert)
- [LiteRT 양자화 가이드](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/model_optimization)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [LiteRT Conversion 가이드](https://ai.google.dev/edge/litert/conversion/tensorflow/convert_tf)

---

## 🤝 피드백

이 예제에 대한 피드백과 개선 제안을 환영합니다!

---

## 📄 라이선스

Apache 2.0
