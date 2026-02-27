"""
Int16 활성화 + Int8 가중치 양자화 예제 (16x8 Quantization)

16x8 양자화는 활성화값을 int16으로, 가중치를 int8로 양자화합니다.

특징:
- 대표 데이터셋 필요
- 정수 양자화보다 높은 정확도
- 약 3-4x 모델 크기 감소
- 활성화에 민감한 모델에 적합

참고: https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant_16x8
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os
import time


def main():
    print("=" * 70)
    print("Int16/Int8 양자화 예제 (MNIST)")
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

    # 8. 대표 데이터셋 생성 (representative dataset generator)
    print("\n[8] 대표 데이터셋 생성 중...")

    def representative_data_gen():
        """양자화를 위한 대표 데이터셋 생성기"""
        for input_value in (
            tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100)
        ):
            yield [tf.cast(input_value, tf.float32)]

    print("    ✅ 대표 데이터셋 준비 완료 (100 샘플)")

    # 9. 16x8 양자화 변환
    print("\n[9] 16x8 양자화 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.representative_dataset = representative_data_gen

    tflite_16x8_model = converter.convert()

    int16x8_path = models_dir / "mnist_model_quant_16x8.tflite"
    int16x8_path.write_bytes(tflite_16x8_model)
    int16x8_size = os.path.getsize(int16x8_path)
    print(f"    ✅ 16x8 양자화 모델 저장: {int16x8_size / 1024:.2f} KB")

    # 10. 크기 비교
    print("\n[10] 모델 크기 비교")
    compression_ratio = (1 - int16x8_size / float32_size) * 100
    print(f"     Float32 모델:   {float32_size / 1024:8.2f} KB")
    print(f"     16x8 모델:      {int16x8_size / 1024:8.2f} KB")
    print(f"     압축률:         {compression_ratio:8.1f}%")

    # 11. 16x8 모델 평가
    print("\n[11] 16x8 모델 정확도 평가 중...")

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

    interpreter = tf.lite.Interpreter(model_path=str(int16x8_path))
    interpreter.allocate_tensors()

    int16x8_accuracy = evaluate_model(interpreter, test_images, test_labels)
    print(f"     16x8 모델 정확도: {int16x8_accuracy * 100:.2f}%")

    # 12. 입출력 타입 확인
    print("\n[12] 모델 입출력 타입 확인")
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"     입력 타입:  {input_details['dtype']}")
    print(f"     출력 타입:  {output_details['dtype']}")

    # 13. 추론 속도 측정
    print("\n[13] 추론 속도 측정 중...")

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
    int16x8_inference_time = measure_inference_time(interpreter, test_images)
    print(f"     Float32 평균 추론 시간: {float32_inference_time:.2f} ms")
    print(f"     16x8 평균 추론 시간: {int16x8_inference_time:.2f} ms")

    # 14. 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"원본 모델 정확도:          {test_accuracy * 100:.2f}%")
    print(f"16x8 모델 정확도:          {int16x8_accuracy * 100:.2f}%")
    print(
        f"정확도 차이:               {abs(test_accuracy - int16x8_accuracy) * 100:.2f}%p"
    )
    print("\n모델 크기")
    print(f"  Float32:                {float32_size / 1024:.2f} KB")
    print(f"  16x8 (Int16/Int8):      {int16x8_size / 1024:.2f} KB")
    print(f"  감소율:                 {compression_ratio:.1f}%")
    print("\n추론 시간")
    print(f"  Float32:                {float32_inference_time:.2f} ms")
    print(f"  16x8:                   {int16x8_inference_time:.2f} ms")
    print("\n입출력 타입")
    print(f"  입력:                   {input_details['dtype']}")
    print(f"  출력:                   {output_details['dtype']}")
    print("\n설명")
    print("  16x8 양자화는 활성화값에 민감한 모델에 좋습니다.")
    print("  정수 양자화보다 정확도가 높으며 더 강건합니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
