"""
모든 프루닝 모델 생성 스크립트

이 스크립트는 PC에서 실행하여:
1. 기본 프루닝 모델 (Float32)
2. 프루닝 + 양자화 모델 (Dynamic Range, Int8)

모든 모델을 한 번에 생성합니다.
생성된 모델은 Raspberry Pi 4로 전달하여 벤치마크할 수 있습니다.
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib
import os
import time


def create_and_train_model(train_images, train_labels, val_images, val_labels):
    """기본 모델 생성 및 훈련"""
    print("\n[모델 생성 및 훈련]")
    print("-" * 70)

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

    print("  훈련 중... (2 epochs)")
    model.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=1,
    )

    return model


def apply_pruning(model, train_images, train_labels, val_images, val_labels):
    """프루닝 적용"""
    print("\n[프루닝 적용]")
    print("-" * 70)

    batch_size = 128
    epochs = 2
    num_images = len(train_images)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=end_step,
        )
    }

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model, **pruning_params
    )

    model_for_pruning.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    print("  프루닝 모델 훈련 중... (2 epochs)")
    model_for_pruning.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=1,
    )

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    # 프루닝 래퍼 제거 후 재컴파일
    model_for_export.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("  ✅ 프루닝 완료")

    return model_for_export


def save_tflite_model(
    model, filepath, quantization_type=None, representative_data=None
):
    """TFLite 모델 변환 및 저장"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization_type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization_type == "int8" and representative_data is not None:

        def representative_dataset():
            for i in range(100):
                yield [representative_data[i : i + 1].astype(np.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(filepath, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(filepath) / 1024
    return size_kb


def benchmark_inference(model_path, test_images, num_runs=100):
    """TFLite 모델의 추론 속도 벤치마크"""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]

    # 워밍업 (첫 1회 실행은 느릴 수 있음)
    test_image = test_images[0]
    if input_details["dtype"] == np.int8:
        input_scale, input_zero_point = input_details["quantization"]
        test_image = test_image / input_scale + input_zero_point
        test_image = np.expand_dims(test_image, axis=0).astype(np.int8)
    else:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()

    # 실제 벤치마크
    times = []
    for _ in range(num_runs):
        test_image = test_images[0]
        if input_details["dtype"] == np.int8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point
            test_image = np.expand_dims(test_image, axis=0).astype(np.int8)
        else:
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)

        start = time.time()
        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()
        times.append((time.time() - start) * 1000)

    times = np.array(times)
    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times),
    }


def main():
    print("=" * 70)
    print("프루닝 모델 생성 스크립트")
    print("=" * 70)

    # 1. MNIST 데이터셋 로드
    print("\n[1] MNIST 데이터셋 로드 중...")
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    validation_split_index = int(0.9 * len(train_images))
    train_images_split = train_images[:validation_split_index]
    train_labels_split = train_labels[:validation_split_index]
    val_images = train_images[validation_split_index:]
    val_labels = train_labels[validation_split_index:]

    print(f"    훈련 이미지: {train_images_split.shape}")
    print(f"    검증 이미지: {val_images.shape}")
    print(f"    테스트 이미지: {test_images.shape}")

    # 2. 디렉토리 생성
    print("\n[2] 모델 저장 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_pruned_models/")
    models_dir.mkdir(exist_ok=True, parents=True)
    print(f"    ✅ {models_dir}")

    # 3. 기본 모델 생성 및 훈련
    print("\n[3] 기본 모델 생성 및 훈련")
    baseline_model = create_and_train_model(
        train_images_split, train_labels_split, val_images, val_labels
    )

    baseline_loss, baseline_accuracy = baseline_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"  ✅ 기본 모델 정확도: {baseline_accuracy * 100:.2f}%")

    # 4. 프루닝 적용
    print("\n[4] 프루닝 적용")
    pruned_model = apply_pruning(
        baseline_model, train_images_split, train_labels_split, val_images, val_labels
    )

    pruned_loss, pruned_accuracy = pruned_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"  ✅ 프루닝 모델 정확도: {pruned_accuracy * 100:.2f}%")

    # 5. 모델 변환 및 저장
    print("\n[5] TFLite 모델 변환 및 저장")
    print("-" * 70)

    models_created = {}

    # 5-1. Baseline Float32
    print("  (1/5) Baseline Float32 변환 중...")
    size = save_tflite_model(baseline_model, models_dir / "mnist_model_baseline.tflite")
    models_created["Baseline"] = size
    print(f"        ✅ 크기: {size:.2f} KB")

    # 5-2. Pruned Float32
    print("  (2/5) Pruned Float32 변환 중...")
    size = save_tflite_model(pruned_model, models_dir / "mnist_model_pruned.tflite")
    models_created["Pruned"] = size
    print(f"        ✅ 크기: {size:.2f} KB")

    # 5-3. Baseline + Quantization (Dynamic Range)
    print("  (3/5) Baseline + Quantization 변환 중...")
    size = save_tflite_model(
        baseline_model,
        models_dir / "mnist_model_baseline_quant.tflite",
        quantization_type="dynamic",
    )
    models_created["Baseline+Quant"] = size
    print(f"        ✅ 크기: {size:.2f} KB")

    # 5-4. Pruned + Quantization (Dynamic Range)
    print("  (4/5) Pruned + Quantization 변환 중...")
    size = save_tflite_model(
        pruned_model,
        models_dir / "mnist_model_pruned_quant.tflite",
        quantization_type="dynamic",
    )
    models_created["Pruned+Quant"] = size
    print(f"        ✅ 크기: {size:.2f} KB")

    # 5-5. Pruned + Int8 Quantization
    print("  (5/5) Pruned + Int8 Quantization 변환 중...")
    size = save_tflite_model(
        pruned_model,
        models_dir / "mnist_model_pruned_int8.tflite",
        quantization_type="int8",
        representative_data=train_images_split,
    )
    models_created["Pruned+Int8"] = size
    print(f"        ✅ 크기: {size:.2f} KB")

    # 6. 테스트 데이터 저장
    print("\n[6] 테스트 데이터 저장 중 (벤치마크용)...")
    test_images_path = models_dir / "mnist_test_images.npy"
    test_labels_path = models_dir / "mnist_test_labels.npy"

    np.save(test_images_path, (test_images * 255).astype(np.uint8))
    np.save(test_labels_path, test_labels)
    print(f"    ✅ {test_images_path}")
    print(f"    ✅ {test_labels_path}")

    # 7. 요약
    print("\n" + "=" * 70)
    print("✅ 모든 모델 생성 완료!")
    print("=" * 70)

    # 8. 추론 속도 벤치마크
    print("\n[8] 추론 속도 벤치마크 중...")
    print("-" * 70)

    inference_results = {}
    for model_name, filename in [
        ("Baseline", "mnist_model_baseline.tflite"),
        ("Pruned", "mnist_model_pruned.tflite"),
        ("Baseline+Quant", "mnist_model_baseline_quant.tflite"),
        ("Pruned+Quant", "mnist_model_pruned_quant.tflite"),
        ("Pruned+Int8", "mnist_model_pruned_int8.tflite"),
    ]:
        model_path = models_dir / filename
        if model_path.exists():
            print(f"  {model_name} 벤치마크 중...")
            metrics = benchmark_inference(model_path, test_images, num_runs=50)
            inference_results[model_name] = metrics
            print(f"    ✅ 평균 추론 시간: {metrics['mean_ms']:.2f} ms")

    print("\n생성된 모델:")
    print(
        f"{'모델':<20} {'정확도(%)':>12} {'크기(KB)':>12} {'압축률(%)':>12} {'추론(ms)':>12} {'FPS':>10}"
    )
    print("-" * 94)

    baseline_size = models_created["Baseline"]
    baseline_accuracy_pct = baseline_accuracy * 100
    for model_name, size in models_created.items():
        compression = (1 - size / baseline_size) * 100 if model_name != "Baseline" else 0
        compression_text = f"{compression:.1f}%"
        inference = inference_results.get(model_name, {})
        accuracy_text = f"{baseline_accuracy_pct:.2f}%"
        inference_ms = f"{inference.get('mean_ms', 0):.2f}" if inference else "N/A"
        fps = f"{1000 / float(inference_ms):.1f}" if inference_ms != "N/A" else "N/A"
        print(
            f"{model_name:<22} {accuracy_text:>14} {size:>14.2f} {compression_text:>14} {inference_ms:>14} {fps:>12}"
        )

    print("\n📁 저장 위치:")
    print(f"   {models_dir.absolute()}")

    print("\n📝 다음 단계:")
    print("   1. 개별 예제 실행:")
    print("      - python 01-basic-pruning.py")
    print("      - python 02-pruning-with-quantization.py")
    print("   2. Raspberry Pi 4 벤치마크:")
    print("      - 이 디렉토리를 Pi로 복사")
    print("      - python benchmark_rpi4.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
