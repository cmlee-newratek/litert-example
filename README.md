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

### 설치

```bash
# LiteRT 런타임 설치
pip install litert

# 필수 의존성
pip install tensorflow
```

## 📂 저장소 구조

```
litert-example/
├── README.md
├── examples/
│   ├── 01-basic-inference/          # 기본 추론 예제
│   ├── 02-quantization/              # 양자화 예제
│   │   ├── 01-float16-quantization.py
│   │   ├── 02-dynamic-range-quantization.py
│   │   ├── 03-integer-quantization.py
│   │   ├── 04-quantization-aware-training.py
│   │   ├── 05-int16-int8-quantization.py
│   │   ├── benchmark_rpi4.py        # Raspberry Pi 4 성능 비교
│   │   └── README.md
│   ├── 03-image-classification/      # 이미지 분류 예제
│   ├── 04-image-segmentation/        # 이미지 세그멘테이션 예제
│   ├── 05-object-detection/          # 객체 탐지 예제
│   └── 06-nlp-inference/             # NLP 추론 예제
├── models/                            # 사전 변환된 .tflite 모델
├── datasets/                          # 양자화용 샘플 데이터
├── tools/
│   ├── convert-tensorflow.py         # TensorFlow → TFLite 변환 스크립트
│   └── benchmark-utils.py            # 벤치마크 유틸리티
├── requirements.txt                   # Python 의존성
└── requirements-rpi4.txt              # Raspberry Pi 4용 의존성
```

## 💡 예제 목록

### 1. 이미지 분류 (Image Classification)

```python
import litert.runtime as rt
import numpy as np

# 모델 로드
interpreter = rt.Interpreter(model_file='model.tflite')
interpreter.allocate_tensors()

# 입력 데이터
input_image = np.random.rand(1, 224, 224, 3).astype(np.float32)

# 추론
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

# 결과
predictions = interpreter.get_tensor(output_details[0]['index'])
print(predictions)
```

### 2. 이미지 세그멘테이션 (Image Segmentation)

엣지 디바이스에서 실시간 이미지 세그멘테이션을 수행합니다.

### 3. 객체 탐지 (Object Detection)

카메라 입력으로부터 실시간 객체 탐지를 구현합니다.

### 4. NLP 추론

텍스트 분류, 감정 분석 등의 NLP 작업을 수행합니다.

### 5. GenAI 모델 배포

LLM 및 생성형 AI 모델을 엣지 디바이스에 배포합니다.

## 🔄 모델 변환 과정

LiteRT는 다양한 프레임워크의 모델을 `.tflite` 형식으로 변환합니다.

### 1단계: 모델 획득

- 사전 학습된 `.tflite` 모델 사용
- TensorFlow 및 Keras 모델 변환
- HuggingFace에서 LiteRT 커뮤니티 모델 다운로드

### 2단계: 모델 최적화

```bash
# 양자화를 통한 모델 최적화
python tools/quantize-model.py --input model.tflite --output model_quant.tflite
```

### 3단계: 배포

최적화된 모델을 선택한 플랫폼에 배포합니다.

## 📱 타겟 플랫폼

이 리파지토리는 **임베디드 Linux** 환경에서 LiteRT를 활용한 예제를 제공합니다.

### 지원되는 하드웨어

- Raspberry Pi (3B+, 4, 5)
- NVIDIA Jetson (Orin, Xavier, Nano)
- MediaTek 칩셋
- Qualcomm Snapdragon용 임베디드 Linux

## 🎯 성능 특성

- **저지연성**: 밀리초 단위 추론
- **높은 개인정보보호**: 온디바이스 처리
- **효율성**: 낮은 메모리 및 전력 소비
- **확장성**: 경량 모델부터 대규모 GenAI까지 지원

## 📚 학습 자료

### 공식 문서

- [LiteRT 공식 문서](https://ai.google.dev/edge/litert)
- [LiteRT Overview](https://ai.google.dev/edge/litert/overview)
- [마이그레이션 가이드](https://ai.google.dev/edge/litert/migration)

### 튜토리얼

- [TensorFlow 모델 변환](https://ai.google.dev/edge/litert/conversion/tensorflow)
- [임베디드 Linux 배포 가이드](https://ai.google.dev/edge/litert/inference)
- [GPU 가속화](https://ai.google.dev/edge/litert/next/gpu)

### 모델 자료

- [HuggingFace LiteRT 커뮤니티](https://huggingface.co/litert-community)

## 🤝 커뮤니티

- [GitHub LiteRT 리포지토리](https://github.com/google-ai-edge/LiteRT)
- [HuggingFace LiteRT 커뮤니티](https://huggingface.co/litert-community)
- [Issue 및 토론](https://github.com/google-ai-edge/LiteRT/discussions)

## 💻 개발 환경 설정

### 가상 환경 생성

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 예제 실행

```bash
cd examples/01-image-classification
python main.py
```

## 🛠️ 유용한 도구

### 모델 변환 스크립트

```bash
# TensorFlow 모델 변환
python tools/convert-tensorflow.py --input model.pb --output model.tflite

# 모델 양자화
python tools/quantize-model.py --input model.tflite --output model_quant.tflite
```

## ⚙️ 하드웨어 가속화 (임베디드 Linux)

LiteRT는 다음의 임베디드 Linux 가속기를 지원합니다:

- **GPU**: Vulkan, OpenGL (Mali GPU, Adreno GPU)
- **NPU**: MediaTek Neuron VPU, Qualcomm Hexagon
- **CUDA**: NVIDIA Jetson GPU (CUDA 지원)
- **CPU**: ARM NEON, x86 SSE/AVX

## 📊 모델 최적화 팁

1. **양자화**: 모델 크기 및 지연시간 감소
2. **프루닝**: 불필요한 파라미터 제거
3. **Knowledge Distillation**: 작은 모델로 지식 전이
4. **적절한 입력 크기 선택**: 성능과 정확도 균형

## 🐛 문제 해결

### 모델 로드 실패

- `.tflite` 파일 경로 확인
- 모델 버전 호환성 확인

### 추론 성능 저하

- 하드웨어 가속화 활성화 확인
- 모델 양자화 고려

### 메모리 부족

- 배치 크기 감소
- 모델 양자화 또는 프루닝

## 📄 라이선스

이 저장소는 Apache 2.0 라이선스 하에 공개됩니다.

## 🙋 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 📞 연락처

- Issue 제출: GitHub Issues
- 토론: GitHub Discussions
- 문의: [Google AI Edge Support](https://ai.google.dev/edge)

---

**마지막 업데이트**: 2026년 2월 27일

더 많은 정보와 최신 예제는 [공식 LiteRT 문서](https://ai.google.dev/edge/litert)를 참고하세요.
