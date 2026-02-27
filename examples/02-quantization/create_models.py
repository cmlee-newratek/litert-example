"""
양자화 모델 생성 스크립트

PC 환경에서 5가지 양자화 방식의 MNIST 모델들을 생성합니다.
생성된 모델들은 Raspberry Pi 4에서 벤치마크하는데 사용됩니다.
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
    print("MNIST 양자화 모델 생성 (PC 환경)")
    print("=" * 70)

    # 1. MNIST 데이터셋 로드
    print("\n[1] MNIST 데이터셋 로드 중...")
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print(f"    훈련 이미지: {train_images.shape}")
    print(f"    테스트 이미지: {test_images.shape}")

    # 2. 모델 생성
    print("\n[2] 모델 생성 및 훈련 중...")
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

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_data=(test_images, test_labels),
        verbose=1,
    )

    _, baseline_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"    ✅ 원본 모델 정확도: {baseline_accuracy * 100:.2f}%")

    # 3. 모델 저장 디렉토리 생성
    print("\n[3] 모델 저장 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_tflite_models/")
    models_dir.mkdir(exist_ok=True, parents=True)
    print(f"    경로: {models_dir.absolute()}")

    # 4. 대표 데이터셋 생성
    print("\n[4] 대표 데이터셋 생성 중...")

    def representative_data_gen():
        for input_value in (
            tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100)
        ):
            yield [tf.cast(input_value, tf.float32)]

    # 5. 모든 양자화 방식 모델 생성
    print("\n[5] 양자화 모델들 생성 중...")
    models_created = {}

    # 5.1 Float32 (기본 모델)
    print("    - Float32 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    float32_path = models_dir / "mnist_model.tflite"
    float32_path.write_bytes(tflite_model)
    float32_size = os.path.getsize(float32_path) / 1024
    models_created["Float32"] = float32_size
    print(f"      ✅ 저장: {float32_path.name} ({float32_size:.2f} KB)")

    # 5.2 Float16
    print("    - Float16 양자화 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    float16_path = models_dir / "mnist_model_quant_f16.tflite"
    float16_path.write_bytes(tflite_model)
    float16_size = os.path.getsize(float16_path) / 1024
    models_created["Float16"] = float16_size
    print(f"      ✅ 저장: {float16_path.name} ({float16_size:.2f} KB)")

    # 5.3 Dynamic Range
    print("    - 동적 범위 양자화 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    dynamic_path = models_dir / "mnist_model_quant_dynamic.tflite"
    dynamic_path.write_bytes(tflite_model)
    dynamic_size = os.path.getsize(dynamic_path) / 1024
    models_created["DynamicRange"] = dynamic_size
    print(f"      ✅ 저장: {dynamic_path.name} ({dynamic_size:.2f} KB)")

    # 5.4 Integer Quantization
    print("    - 정수 양자화 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    int8_path = models_dir / "mnist_model_quant_int8.tflite"
    int8_path.write_bytes(tflite_model)
    int8_size = os.path.getsize(int8_path) / 1024
    models_created["Integer"] = int8_size
    print(f"      ✅ 저장: {int8_path.name} ({int8_size:.2f} KB)")

    # 5.5 QAT (Quantization Aware Training)
    print("    - QAT 모델 생성 중...")
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)
    q_aware_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    q_aware_model.fit(
        train_images,
        train_labels,
        batch_size=500,
        epochs=1,
        validation_data=(test_images, test_labels),
        verbose=0,
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    qat_path = models_dir / "mnist_model_quant_qat.tflite"
    qat_path.write_bytes(tflite_model)
    qat_size = os.path.getsize(qat_path) / 1024
    models_created["QAT"] = qat_size
    print(f"      ✅ 저장: {qat_path.name} ({qat_size:.2f} KB)")

    # 5.6 16x8 Quantization
    print("    - 16x8 양자화 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()
    int16x8_path = models_dir / "mnist_model_quant_16x8.tflite"
    int16x8_path.write_bytes(tflite_model)
    int16x8_size = os.path.getsize(int16x8_path) / 1024
    models_created["Int16x8"] = int16x8_size
    print(f"      ✅ 저장: {int16x8_path.name} ({int16x8_size:.2f} KB)")

    # 6. MNIST 테스트 데이터 저장 (Raspberry Pi용)
    print("\n[6] MNIST 테스트 데이터 저장 중 (Raspberry Pi용)...")
    test_images_path = models_dir / "mnist_test_images.npy"
    test_labels_path = models_dir / "mnist_test_labels.npy"

    # 정규화되지 않은 원본 데이터 저장 (0-255 범위)
    np.save(test_images_path, test_images * 255.0)
    np.save(test_labels_path, test_labels)

    test_images_size = os.path.getsize(test_images_path) / 1024
    test_labels_size = os.path.getsize(test_labels_path) / 1024
    print(f"    ✅ 테스트 이미지: {test_images_path.name} ({test_images_size:.2f} KB)")
    print(f"    ✅ 테스트 레이블: {test_labels_path.name} ({test_labels_size:.2f} KB)")
    print("    → Raspberry Pi에서 TensorFlow 없이 데이터 로드 가능")

    # 7. 추론 시간 측정
    print("\n[7] 추론 시간 측정 중...")

    def measure_tflite_inference_time(model_path, test_data, num_runs=100):
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        input_index = input_details["index"]
        input_dtype = input_details["dtype"]
        input_scale, input_zero_point = input_details["quantization"]

        test_image = test_data[0]
        if input_dtype == np.uint8:
            if input_scale > 0:
                test_image = test_image / input_scale + input_zero_point
        test_image = np.expand_dims(test_image, axis=0).astype(input_dtype)

        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()

        times = []
        for _ in range(num_runs):
            start = time.time()
            interpreter.set_tensor(input_index, test_image)
            interpreter.invoke()
            times.append(time.time() - start)

        return np.mean(times) * 1000

    inference_times = {}
    model_paths = {
        "Float32": float32_path,
        "Float16": float16_path,
        "DynamicRange": dynamic_path,
        "Integer": int8_path,
        "QAT": qat_path,
        "Int16x8": int16x8_path,
    }

    for model_name, model_path in model_paths.items():
        try:
            inference_times[model_name] = measure_tflite_inference_time(
                model_path, test_images, num_runs=50
            )
            print(f"    ✅ {model_name}: {inference_times[model_name]:.2f} ms")
        except Exception as e:
            inference_times[model_name] = None
            print(f"    ❌ {model_name}: {str(e)}")

    # 8. 결과 요약
    print("\n" + "=" * 70)
    print("모델 생성 완료")
    print("=" * 70)
    print(f"\n저장 경로: {models_dir.absolute()}")
    print("\n생성된 모델들:")
    print(f"{'모델':<16} {'크기(KB)':>10} {'압축율(%)':>10} {'추론(ms)':>10}")
    print("-" * 52)

    for model_name, size in models_created.items():
        compression = (1 - size / float32_size) * 100 if model_name != "Float32" else 0
        inference_ms = inference_times.get(model_name)
        inference_text = f"{inference_ms:.2f}" if inference_ms is not None else "N/A"
        print(
            f"{model_name:<16} {size:>10.2f} {compression:>9.1f} % {inference_text:>10}"
        )

    print("\n" + "=" * 70)
    print("다음 단계: Raspberry Pi 4에서 benchmark_rpi4.py를 실행하세요!")
    print(f"모든 모델과 테스트 데이터가 {models_dir.absolute()}에 저장되었습니다.")
    print(
        "Raspberry Pi에서는 TensorFlow 없이 numpy 파일로 데이터를 로드할 수 있습니다."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
