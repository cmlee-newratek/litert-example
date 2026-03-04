"""
동적 범위 양자화 예제 (Post-training Dynamic Range Quantization)

동적 범위 양자화는 모델의 가중치를 int8로 변환하고 활성화값은 동적으로 양자화합니다.

특징:
- 추가 데이터 필요 없음
- 모델 크기를 약 75% 감소 (4x 축소)
- CPU/GPU 지원
- 모든 연산에 양자화 커널 사용

참고: https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quant
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os
import time


def main():
    print("=" * 70)
    print("동적 범위 양자화 예제 (MNIST)")
    print("=" * 70)

    # 1. MNIST 데이터셋 로드
    print("\n[1] MNIST 데이터셋 로드 중...")
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 정규화: 0~1 범위로
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print(f"    훈련 이미지: {train_images.shape}, 테스트 이미지: {test_images.shape}")

    # 2. 모델 아키텍처 정의
    print("\n[2] 모델 아키텍처 정의 중...")
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(28, 28)),
            keras.layers.Reshape(target_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10),
        ]
    )
    model.summary()

    # 3. 모델 컴파일
    print("\n[3] 모델 컴파일 중...")
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # 4. 모델 훈련
    print("\n[4] 모델 훈련 중... (1 epoch)")
    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_data=(test_images, test_labels),
        verbose=1,
    )

    # 5. 원본 모델 평가
    print("\n[5] 원본 모델 평가 중...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"    원본 모델 정확도: {test_accuracy * 100:.2f}%")

    # 6. 디렉토리 생성
    print("\n[6] 모델 저장 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_tflite_models/")
    models_dir.mkdir(exist_ok=True, parents=True)

    # 7. Float32 모델 변환
    print("\n[7] Float32 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    float32_path = models_dir / "mnist_model.tflite"
    float32_path.write_bytes(tflite_model)
    float32_size = os.path.getsize(float32_path)
    print(f"    ✅ Float32 모델 저장: {float32_size / 1024:.2f} KB")

    interpreter_float32 = tf.lite.Interpreter(model_path=str(float32_path))
    interpreter_float32.allocate_tensors()

    # 8. 동적 범위 양자화 변환
    print("\n[8] 동적 범위 양자화 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_quant_model = converter.convert()

    quant_path = models_dir / "mnist_model_quant_dynamic.tflite"
    quant_path.write_bytes(tflite_quant_model)
    quant_size = os.path.getsize(quant_path)
    print(f"    ✅ 동적 범위 양자화 모델 저장: {quant_size / 1024:.2f} KB")

    # 9. 크기 비교
    print("\n[9] 모델 크기 비교")
    compression_ratio = (1 - quant_size / float32_size) * 100
    print(f"    Float32 모델:   {float32_size / 1024:8.2f} KB")
    print(f"    동적 범위 모델: {quant_size / 1024:8.2f} KB")
    print(f"    압축률:         {compression_ratio:8.1f}%")

    # 10. 양자화 모델 평가
    print("\n[10] 양자화 모델 정확도 평가 중...")

    def evaluate_model(interpreter, test_data, test_labels):
        """LiteRT 모델 평가"""
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_index = input_details["index"]
        output_index = output_details["index"]

        correct_count = 0
        for i, test_image in enumerate(test_data):
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)[0]
            prediction = np.argmax(output)
            if prediction == test_labels[i]:
                correct_count += 1

        accuracy = correct_count / len(test_data)
        return accuracy

    interpreter = tf.lite.Interpreter(model_path=str(quant_path))
    interpreter.allocate_tensors()

    quant_accuracy = evaluate_model(interpreter, test_images, test_labels)
    print(f"    양자화 모델 정확도: {quant_accuracy * 100:.2f}%")

    # 11. 추론 속도 비교
    print("\n[11] 추론 속도 측정 중...")

    def measure_inference_time(interpreter, test_data, num_runs=100):
        """추론 시간 측정"""
        input_index = interpreter.get_input_details()[0]["index"]

        times = []
        for _ in range(num_runs):
            test_image = np.expand_dims(test_data[0], axis=0).astype(np.float32)
            start = time.time()
            interpreter.set_tensor(input_index, test_image)
            interpreter.invoke()
            times.append(time.time() - start)

        return np.mean(times) * 1000

    float32_inference_time = measure_inference_time(interpreter_float32, test_images)
    quant_inference_time = measure_inference_time(interpreter, test_images)
    print(f"    Float32 평균 추론 시간: {float32_inference_time:.2f} ms")
    print(f"    양자화 모델 평균 추론 시간: {quant_inference_time:.2f} ms")

    # 12. 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"원본 모델 정확도:          {test_accuracy * 100:.2f}%")
    print(f"양자화 모델 정확도:        {quant_accuracy * 100:.2f}%")
    print(
        f"정확도 차이:               {abs(test_accuracy - quant_accuracy) * 100:.2f}%p"
    )
    print("\n모델 크기")
    print(f"  Float32:                {float32_size / 1024:.2f} KB")
    print(f"  동적 범위:              {quant_size / 1024:.2f} KB")
    print(f"  감소율:                 {compression_ratio:.1f}%")
    print("\n추론 시간")
    print(f"  Float32:                {float32_inference_time:.2f} ms")
    print(f"  양자화:                 {quant_inference_time:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
