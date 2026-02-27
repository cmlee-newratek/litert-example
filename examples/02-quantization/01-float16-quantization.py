"""
Float16 양자화 예제 (Post-training Float16 Quantization)

Float16 양자화는 모델의 가중치를 32비트 부동소수점에서 16비트 부동소수점으로 변환합니다.

특징:
- 추가 데이터 필요 없음
- 모델 크기를 약 50% 감소 (2x 축소)
- 정확도 손실 최소화
- GPU에서 최적화

참고: https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_float16_quant
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os
import time


def main():
    print("=" * 70)
    print("Float16 양자화 예제 (MNIST)")
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

    # 8. Float16 양자화 변환
    print("\n[8] Float16 양자화 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_fp16_model = converter.convert()

    float16_path = models_dir / "mnist_model_quant_f16.tflite"
    float16_path.write_bytes(tflite_fp16_model)
    float16_size = os.path.getsize(float16_path)
    print(f"    ✅ Float16 양자화 모델 저장: {float16_size / 1024:.2f} KB")

    # 9. 크기 비교
    print("\n[9] 모델 크기 비교")
    compression_ratio = (1 - float16_size / float32_size) * 100
    print(f"    Float32 모델:   {float32_size / 1024:8.2f} KB")
    print(f"    Float16 모델:   {float16_size / 1024:8.2f} KB")
    print(f"    압축률:         {compression_ratio:8.1f}%")

    # 10. Float16 모델 평가
    print("\n[10] Float16 모델 정확도 평가 중...")

    def evaluate_model(interpreter, test_data, test_labels):
        """LiteRT 모델 평가"""
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

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

    interpreter_fp16 = tf.lite.Interpreter(model_path=str(float16_path))
    interpreter_fp16.allocate_tensors()

    fp16_accuracy = evaluate_model(interpreter_fp16, test_images, test_labels)
    print(f"    Float16 모델 정확도: {fp16_accuracy * 100:.2f}%")

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
    fp16_inference_time = measure_inference_time(interpreter_fp16, test_images)
    print(f"    Float32 평균 추론 시간: {float32_inference_time:.2f} ms")
    print(f"    Float16 평균 추론 시간: {fp16_inference_time:.2f} ms")

    # 12. 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"원본 모델 정확도:          {test_accuracy * 100:.2f}%")
    print(f"Float16 모델 정확도:       {fp16_accuracy * 100:.2f}%")
    print(
        f"정확도 차이:               {abs(test_accuracy - fp16_accuracy) * 100:.2f}%p"
    )
    print("\n모델 크기")
    print(f"  Float32:                {float32_size / 1024:.2f} KB")
    print(f"  Float16:                {float16_size / 1024:.2f} KB")
    print(f"  감소율:                 {compression_ratio:.1f}%")
    print("\n추론 시간")
    print(f"  Float32:                {float32_inference_time:.2f} ms")
    print(f"  Float16:                {fp16_inference_time:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
