"""
양자화 인식 훈련 예제 (Quantization Aware Training, QAT)

QAT는 훈련 중에 양자화를 시뮬레이션하여 가중치를 조정합니다.

특징:
- 훈련 데이터 필요
- 정수 양자화보다 높은 정확도
- 훈련 시간 소요
- EdgeTPU 호환 가능

참고: https://www.tensorflow.org/model_optimization/guide/quantization/training_example
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib
import os
import time


def main():
    print("=" * 70)
    print("양자화 인식 훈련 (QAT) 예제 (MNIST)")
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

    # 3. 모델 컴파일 및 훈련
    print("\n[3] 모델 컴파일 중...")
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("[4] 기본 모델 훈련 중... (1 epoch)")
    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_data=(test_images, test_labels),
        verbose=1,
    )

    # 5. 원본 모델 평가
    print("\n[5] 원본 모델 평가 중...")
    test_loss, baseline_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"    원본 모델 정확도: {baseline_accuracy * 100:.2f}%")

    # 6. 양자화 인식 모델로 변환
    print("\n[6] 양자화 인식 모델로 변환 중...")
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)

    q_aware_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("    ✅ 양자화 인식 모델 생성 완료")

    # 7. 양자화 인식 모델 미세조정 훈련
    print("\n[7] 양자화 인식 모델 미세조정 중... (1 epoch)")
    q_aware_model.fit(
        train_images,
        train_labels,
        batch_size=500,
        epochs=1,
        validation_data=(test_images, test_labels),
        verbose=1,
    )

    # 8. 양자화 인식 모델 평가
    print("\n[8] 양자화 인식 모델 평가 중...")
    _, q_aware_accuracy = q_aware_model.evaluate(test_images, test_labels, verbose=0)
    print(f"    양자화 인식 모델 정확도: {q_aware_accuracy * 100:.2f}%")

    # 9. 디렉토리 생성
    print("\n[9] 모델 저장 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_tflite_models/")
    models_dir.mkdir(exist_ok=True, parents=True)

    # 10. 원본 Float32 모델 변환
    print("\n[10] 원본 Float32 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    float32_path = models_dir / "mnist_model.tflite"
    float32_path.write_bytes(tflite_model)
    float32_size = os.path.getsize(float32_path)
    print(f"     ✅ Float32 모델 저장: {float32_size / 1024:.2f} KB")

    interpreter_float32 = tf.lite.Interpreter(model_path=str(float32_path))
    interpreter_float32.allocate_tensors()

    # 11. 양자화 인식 모델을 Int8로 변환
    print("\n[11] 양자화 인식 모델을 Int8로 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_qat_model = converter.convert()

    qat_path = models_dir / "mnist_model_quant_qat.tflite"
    qat_path.write_bytes(tflite_qat_model)
    qat_size = os.path.getsize(qat_path)
    print(f"     ✅ QAT 양자화 모델 저장: {qat_size / 1024:.2f} KB")

    # 12. 크기 비교
    print("\n[12] 모델 크기 비교")
    compression_ratio = (1 - qat_size / float32_size) * 100
    print(f"     Float32 모델:   {float32_size / 1024:8.2f} KB")
    print(f"     QAT 모델:       {qat_size / 1024:8.2f} KB")
    print(f"     압축률:         {compression_ratio:8.1f}%")

    # 13. QAT 모델 평가
    print("\n[13] QAT 모델 정확도 평가 중...")

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

    interpreter = tf.lite.Interpreter(model_path=str(qat_path))
    interpreter.allocate_tensors()

    qat_accuracy = evaluate_model(interpreter, test_images, test_labels)
    print(f"     QAT 모델 정확도: {qat_accuracy * 100:.2f}%")

    # 14. 추론 속도 측정
    print("\n[14] 추론 속도 측정 중...")

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
    qat_inference_time = measure_inference_time(interpreter, test_images)
    print(f"     Float32 평균 추론 시간: {float32_inference_time:.2f} ms")
    print(f"     QAT 평균 추론 시간: {qat_inference_time:.2f} ms")

    # 15. 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"원본 모델 정확도:          {baseline_accuracy * 100:.2f}%")
    print(f"QAT 모델 정확도:           {qat_accuracy * 100:.2f}%")
    print(
        f"정확도 차이:               {abs(baseline_accuracy - qat_accuracy) * 100:.2f}%p"
    )
    print("\n모델 크기")
    print(f"  Float32:                {float32_size / 1024:.2f} KB")
    print(f"  QAT (양자화됨):         {qat_size / 1024:.2f} KB")
    print(f"  감소율:                 {compression_ratio:.1f}%")
    print("\n추론 시간")
    print(f"  Float32:                {float32_inference_time:.2f} ms")
    print(f"  QAT:                    {qat_inference_time:.2f} ms")
    print("\n설명")
    print("  QAT는 훈련 중에 양자화를 시뮬레이션합니다.")
    print("  따라서 정수 양자화보다 정확도가 높습니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
