"""
정수 양자화 예제 (Post-training Integer Quantization)

정수 양자화는 가중치와 활성화값을 모두 int8로 변환합니다. EdgeTPU와 호환됩니다.

특징:
- 대표 데이터셋 필요
- 모델 크기를 약 75% 감소 (4x 축소)
- 가장 빠른 CPU 추론
- EdgeTPU 완벽 호환

참고: https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os
import time


def main():
    print("=" * 70)
    print("정수 양자화 예제 (MNIST)")
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
    float32_input_details = interpreter_float32.get_input_details()[0]

    # 8. 대표 데이터셋 생성 (representative dataset generator)
    print("\n[8] 대표 데이터셋 생성 중...")

    def representative_data_gen():
        """양자화를 위한 대표 데이터셋 생성기"""
        for input_value in (
            tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100)
        ):
            yield [tf.cast(input_value, tf.float32)]

    print("    ✅ 대표 데이터셋 준비 완료 (100 샘플)")

    # 9. 정수 양자화 변환
    print("\n[9] 정수 양자화 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_int8_model = converter.convert()

    int8_path = models_dir / "mnist_model_quant_int8.tflite"
    int8_path.write_bytes(tflite_int8_model)
    int8_size = os.path.getsize(int8_path)
    print(f"    ✅ 정수 양자화 모델 저장: {int8_size / 1024:.2f} KB")

    # 10. 크기 비교
    print("\n[10] 모델 크기 비교")
    compression_ratio = (1 - int8_size / float32_size) * 100
    print(f"    Float32 모델:   {float32_size / 1024:8.2f} KB")
    print(f"     정수 양자화:    {int8_size / 1024:8.2f} KB")
    print(f"    압축률:         {compression_ratio:8.1f}%")

    # 11. 정수 양자화 모델 평가
    print("\n[11] 정수 양자화 모델 정확도 평가 중...")

    def evaluate_model(interpreter, test_data, test_labels, input_dtype):
        """LiteRT 모델 평가"""
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_index = input_details["index"]
        output_index = output_details["index"]

        correct_count = 0
        for i, test_image in enumerate(test_data):
            # 정수 입력인 경우 스케일링 필요
            if input_dtype == np.uint8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image_scaled = test_image / input_scale + input_zero_point
            else:
                test_image_scaled = test_image

            test_image_expanded = np.expand_dims(test_image_scaled, axis=0).astype(
                input_dtype
            )
            interpreter.set_tensor(input_index, test_image_expanded)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)[0]
            prediction = np.argmax(output)
            if prediction == test_labels[i]:
                correct_count += 1

        accuracy = correct_count / len(test_data)
        return accuracy

    interpreter = tf.lite.Interpreter(model_path=str(int8_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]

    int8_accuracy = evaluate_model(
        interpreter, test_images, test_labels, input_details["dtype"]
    )
    print(f"    정수 양자화 모델 정확도: {int8_accuracy * 100:.2f}%")

    # 12. 추론 속도 측정
    print("\n[12] 추론 속도 측정 중...")

    def measure_inference_time(interpreter, test_data, input_dtype, num_runs=100):
        """추론 시간 측정"""
        input_details = interpreter.get_input_details()[0]
        input_index = input_details["index"]

        times = []
        for _ in range(num_runs):
            test_image = test_data[0]
            if input_dtype == np.uint8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image = test_image / input_scale + input_zero_point

            test_image_expanded = np.expand_dims(test_image, axis=0).astype(input_dtype)
            start = time.time()
            interpreter.set_tensor(input_index, test_image_expanded)
            interpreter.invoke()
            times.append(time.time() - start)

        return np.mean(times) * 1000

    float32_inference_time = measure_inference_time(
        interpreter_float32, test_images, float32_input_details["dtype"]
    )
    int8_inference_time = measure_inference_time(
        interpreter, test_images, input_details["dtype"]
    )
    print(f"    Float32 평균 추론 시간: {float32_inference_time:.2f} ms")
    print(f"    정수 양자화 평균 추론 시간: {int8_inference_time:.2f} ms")

    # 13. 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"원본 모델 정확도:          {test_accuracy * 100:.2f}%")
    print(f"정수 양자화 모델 정확도:   {int8_accuracy * 100:.2f}%")
    print(
        f"정확도 차이:               {abs(test_accuracy - int8_accuracy) * 100:.2f}%p"
    )
    print("\n모델 크기")
    print(f"  Float32:                {float32_size / 1024:.2f} KB")
    print(f"  정수 양자화 (Int8):      {int8_size / 1024:.2f} KB")
    print(f"  감소율:                 {compression_ratio:.1f}%")
    print("\n추론 시간")
    print(f"  Float32:                {float32_inference_time:.2f} ms")
    print(f"  정수 양자화:             {int8_inference_time:.2f} ms")
    print(f"  입력 타입:               {input_details['dtype']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
