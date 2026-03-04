# LiteRT 활용 예제 (LiteRT Example)

LiteRT를 활용한 다양한 ML/GenAI 모델 배포 예제 모음입니다. Google의 엣지 AI 런타임인 LiteRT를 통해 빠르고 효율적인 온디바이스 머신러닝 추론을 구현하는 방법을 보여줍니다.

## 📌 LiteRT란?

**LiteRT**는 엣지 플랫폼에서 고성능 ML 및 GenAI 모델을 배포하기 위한 Google의 업계 표준 머신러닝 런타임입니다.

### 주요 특징

- **TensorFlow Lite 기반**: 전 세계 수십억 개의 기기에서 검증된 기술
- **크로스 플랫폼**: Android, iOS, 웹, 임베디드 등 다양한 플랫폼 지원
- **GenAI 지원**: LLM 및 생성형 AI 모델 배포 가능
- **하드웨어 가속화**: GPU, NPU 등 다양한 가속기 지원
- **다중 프레임워크**: PyTorch, JAX, TensorFlow 모델 지원
- **최적화**: 양자화 등을 통한 모델 최적화 가능

## 🚀 시작하기

### 필수 요구사항

- Python 3.8 이상
- TensorFlow (모델 변환용)
- LiteRT 런타임

## 📂 저장소 구조

```
litert-example/
├── README.md
├── requirements.txt                         # 공통 의존성
├── requirements-rpi.txt                     # Raspberry Pi 의존성
│
└── examples/
    ├── 01-quantization/                    # 양자화 예제
    │   ├── 01-float16-quantization.py
    │   ├── 02-dynamic-range-quantization.py
    │   ├── 03-integer-quantization.py
    │   ├── 04-quantization-aware-training.py
    │   ├── 05-int16-int8-quantization.py
    │   ├── create_models.py
    │   ├── benchmark_rpi4.py
    │   └── README.md
    │
    ├── 02-pruning/                         # 프루닝 예제
    │   ├── 01-basic-pruning.py
    │   ├── 02-pruning-with-quantization.py
    │   ├── create_models.py
    │   ├── benchmark_rpi4.py
    │   └── README.md
    │
    ├── 03-clustering/                      # 클러스터링 예제
    │   ├── 01-basic-clustering.py
    │   ├── 02-clustering-with-quantization.py
    │   ├── create_models.py
    │   ├── benchmark_rpi4.py
    │   └── README.md
    │
    └── 04-collaborative-optimization/      # 협업 최적화 예제 ⭐
        ├── 01-cqat.py                     # Clustering + QAT
        ├── 02-pqat.py                     # Pruning + QAT
        ├── 03-pcqat.py                    # Pruning + Clustering + QAT (최고 압축)
        ├── create_models.py
        ├── benchmark_rpi4.py
        └── README.md
```

## 💡 예제 목록

### 1️⃣ 양자화 (Quantization) - `01-quantization/`

**Float32 모델을 Int8로 변환하여 모델 크기와 추론 속도를 개선합니다.**

5가지 양자화 기법을 다룹니다:

- **01-float16-quantization.py**: Float16 양자화
- **02-dynamic-range-quantization.py**: 동적 범위 양자화
- **03-integer-quantization.py**: 정수 양자화
- **04-quantization-aware-training.py**: 양자화 인식 훈련 (QAT)
- **05-int16-int8-quantization.py**: Int16/Int8 양자화

**기대 결과:**

- 모델 크기: 75-80% 감소
- 추론 속도: 4-6배 향상
- 정확도 손실: <1%

---

### 2️⃣ 프루닝 (Pruning) - `02-pruning/`

**가중치의 일부(~50%)를 0으로 설정하여 모델을 희소화합니다.**

- **01-basic-pruning.py**: 기본 프루닝 개념 학습
- **02-pruning-with-quantization.py**: 프루닝 + 양자화 조합
- **create_models.py**: 배치 모델 생성
- **benchmark_rpi4.py**: Raspberry Pi 4 벤치마킹

**기대 결과:**

- 스파시티(희소성): 50%
- 모델 크기: 20-30% 감소 (gzip 압축 시 더 효율적)
- 정확도 손실: <1%

---

### 3️⃣ 클러스터링 (Clustering) - `03-clustering/`

**가중치들을 N개의 그룹으로 묶어 고유 값을 줄입니다.**

- **01-basic-clustering.py**: 기본 클러스터링 (16개 클러스터)
- **02-clustering-with-quantization.py**: 클러스터링 + 양자화 조합
- **create_models.py**: 다양한 클러스터 수(8, 16, 32)로 모델 생성
- **benchmark_rpi4.py**: Raspberry Pi 4 벤치마킹

**기대 결과:**

- 고유 값: 1000→16 (16배 감소)
- 모델 크기: 20-25% 감소
- 정확도 손실: <1%

---

### 4️⃣ 협업 최적화 (Collaborative Optimization) - ⭐ `04-collaborative-optimization/`

**여러 최적화 기법을 순차적으로 조합하여 최대 압축을 달성합니다.**

3가지 협업 최적화 경로:

#### 🔸 CQAT (Clustering + Quantization Aware Training)

- **01-cqat.py**: 클러스터링 → QAT
- 압축율: **30-35%**
- 정확도: 높음 ✅
- 추천: 균형잡힌 선택

#### 🔹 PQAT (Pruning + Quantization Aware Training)

- **02-pqat.py**: 프루닝 → QAT
- 압축율: **35-40%**
- 정확도: 중상 ✅
- 추천: 희소성 활용 필요시

#### 🔺 PCQAT (Pruning + Clustering + QAT) ⭐⭐⭐

- **03-pcqat.py**: 프루닝 → 클러스터링 → QAT
- 압축율: **45-50%** 🏆
- 정확도: 중 ✅
- 추천: **최대 압축 필요시** (권장)

**추가 파일:**

- **create_models.py**: 3가지 협업 최적화 모델 배치 생성
- **benchmark_rpi4.py**: Raspberry Pi 4 성능 벤치마킹

---

## 📊 최적화 기법 비교

| 기법       | 압축율     | 추론속도   | 구현 난이도 | 추천대상      |
| ---------- | ---------- | ---------- | ----------- | ------------- |
| 양자화     | 75-80%     | ⬆️⬆️⬆️     | 낮음        | 시작점        |
| 프루닝     | 20-30%     | ➡️         | 중간        | 희소성 활용   |
| 클러스터링 | 20-25%     | ➡️         | 중간        | 고유값 감소   |
| CQAT       | 30-35%     | ⬆️⬆️       | 중간        | 균형          |
| PQAT       | 35-40%     | ⬆️⬆️       | 중간        | 희소성 보존   |
| **PCQAT**  | **45-50%** | **⬆️⬆️⬆️** | **높음**    | **최대 압축** |

---

## � 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/newracom/litert-example.git
cd litert-example
```

### 2. 가상환경 설정

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\Activate  # Windows
```

### 3. 의존성 설치

#### PC에서 (TensorFlow 필요)

```bash
pip install -r requirements.txt
```

#### Raspberry Pi에서 (가벼운 버전)

```bash
pip install -r requirements-rpi.txt
```

### 4. 예제 실행

#### PC에서 모델 생성 (권장)

```bash
# 양자화 모델 생성
cd examples/01-quantization
python create_models.py

# 또는 협업 최적화 (최고 압축)
cd ../04-collaborative-optimization
python create_models.py
```

#### Raspberry Pi 4에서 벤치마킹

```bash
# 벤치마크 실행
python benchmark_rpi4.py
```

---

## 📱 타겟 플랫폼

이 리파지토리는 **임베디드 Linux** 환경에서 LiteRT를 활용한 예제를 제공합니다.

### 지원되는 하드웨어

- Raspberry Pi (4)

## � 각 폴더별 실행 방법

### 01-quantization 실행

```bash
cd examples/01-quantization
pip install -r requirements.txt

# 개별 예제 실행
python 01-float16-quantization.py
python 02-dynamic-range-quantization.py
python 03-integer-quantization.py

# 모든 모델 한 번에 생성
python create_models.py

# Raspberry Pi에서 벤치마킹
pip install -r requirements-rpi.txt
python benchmark_rpi4.py
```

### 02-pruning 실행

```bash
cd examples/02-pruning
pip install -r requirements.txt

# 개별 예제 실행
python 01-basic-pruning.py
python 02-pruning-with-quantization.py

# 모든 모델 한 번에 생성
python create_models.py

# Raspberry Pi에서 벤치마킹
pip install -r requirements-rpi.txt
python benchmark_rpi4.py
```

### 03-clustering 실행

```bash
cd examples/03-clustering
pip install -r requirements.txt

# 개별 예제 실행
python 01-basic-clustering.py
python 02-clustering-with-quantization.py

# 모든 모델 한 번에 생성
python create_models.py

# Raspberry Pi에서 벤치마킹
pip install -r requirements-rpi.txt
python benchmark_rpi4.py
```

### 04-collaborative-optimization 실행 (권장)

```bash
cd examples/04-collaborative-optimization
pip install -r requirements.txt

# 3가지 협업 최적화 기법 학습
python 01-cqat.py      # Clustering + QAT
python 02-pqat.py      # Pruning + QAT
python 03-pcqat.py     # Pruning + Clustering + QAT (최고 압축)

# 모든 모델 한 번에 생성
python create_models.py

# Raspberry Pi에서 벤치마킹
pip install -r requirements-rpi.txt
python benchmark_rpi4.py
```

---

## 💡 팁

### 최적화 선택 기준

| 상황                        | 추천 기법              | 이유                 |
| --------------------------- | ---------------------- | -------------------- |
| 엣지 디바이스 (매우 제한적) | **PCQAT**              | 최대 압축 50%        |
| 모바일 앱                   | **CQAT** 또는 **PQAT** | 균형잡힌 압축 35-40% |
| 정확도 우선                 | **양자화**             | 정확도 손실 최소     |
| 시간이 부족할 때            | **01-quantization**    | 가장 빨리 배우기     |

더 많은 정보와 최신 예제는 [공식 LiteRT 문서](https://ai.google.dev/edge/litert)를 참고하세요.
