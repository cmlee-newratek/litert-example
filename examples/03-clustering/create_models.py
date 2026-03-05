"""
클러스터된 MNIST 모델 배치 생성 (Batch Model Generation)

다양한 클러스터링 옵션으로 MNIST 모델을 만들고 TFLite 변환합니다.

생성 모델:
1. mnist_model_baseline.tflite - Float32 원본 모델
2. mnist_model_clustered_8.tflite - 8개 클러스터
3. mnist_model_clustered_16.tflite - 16개 클러스터
4. mnist_model_clustered_32.tflite - 32개 클러스터
5. mnist_model_clustered_16_quant.tflite - 16 클러스터 + 양자화
6. mnist_model_clustered_16_int8.tflite - 16 클러스터 + Int8 양자화

테스트 데이터도 함께 저장됩니다.
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib
import os
import time


def create_model():
    """MNIST 모델 생성"""
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
    return model


def load_and_prepare_data():
    """MNIST 데이터 로드 및 정규화"""
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 정규화
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 검증 데이터 분리
    validation_split_index = int(0.9 * len(train_images))
    train_images_split = train_images[:validation_split_index]
    train_labels_split = train_labels[:validation_split_index]
    val_images = train_images[validation_split_index:]
    val_labels = train_labels[validation_split_index:]

    return (
        train_images_split,
        train_labels_split,
        val_images,
        val_labels,
        test_images,
        test_labels,
    )


def train_model(model, train_images, train_labels, val_images, val_labels):
    """모델 훈련"""
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )


def apply_clustering(model, num_clusters):
    """클러스터링 적용"""
    clustering_params = {
        "number_of_clusters": num_clusters,
        "cluster_centroids_init": tfmot.clustering.keras.CentroidsInitializer.LINEAR,
    }

    clustered_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)

    clustered_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return clustered_model


def convert_to_tflite(
    model, quantize=False, int8_quantize=False, representative_data=None
):
    """TFLite로 변환"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if int8_quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter.convert()


def representative_data_generator(train_images):
    """양자화용 대표 데이터셋 생성"""

    def generator():
        for i in range(100):
            yield [train_images[i : i + 1].astype(np.float32)]

    return generator


def benchmark_inference(model_path, test_images, num_runs=50):
    """TFLite 모델의 추론 속도 벤치마크"""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]

    # 워밍업
    test_image = test_images[0]
    if input_details["dtype"] == np.int8:
        input_scale, input_zero_point = input_details["quantization"]
        test_image = test_image / input_scale + input_zero_point
        test_image = np.expand_dims(test_image, axis=0).astype(np.int8)
    else:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()

    # 벤치마크
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
    print("클러스터된 MNIST 모델 배치 생성")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1] MNIST 데이터 로드 중...")
    train_images, train_labels, val_images, val_labels, test_images, test_labels = (
        load_and_prepare_data()
    )
    print(
        f"     훈련: {train_images.shape}, 검증: {val_images.shape}, 테스트: {test_images.shape}"
    )

    # 2. 디렉토리 생성
    print("\n[2] 출력 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_clustered_models/")
    models_dir.mkdir(exist_ok=True, parents=True)
    print(f"     ✅ {models_dir}")

    # 3. 기본 모델 훈련
    print("\n[3] 기본 모델 훈련 중...")
    model = create_model()
    train_model(model, train_images, train_labels, val_images, val_labels)
    print("     ✅ 기본 모델 훈련 완료")

    # 3-1. 기본 모델의 정확도 측정
    baseline_loss, baseline_accuracy = model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"     ✅ 기본 모델 정확도: {baseline_accuracy * 100:.2f}%")

    # 4. Float32 기본 모델 변환
    print("\n[4] Float32 기본 모델 변환 중...")
    baseline_tflite = convert_to_tflite(model)
    baseline_path = models_dir / "mnist_model_baseline.tflite"
    baseline_path.write_bytes(baseline_tflite)
    baseline_size = os.path.getsize(baseline_path)
    print(f"     ✅ {baseline_path.name}: {baseline_size / 1024:.2f} KB")

    # 5. 클러스터링 + 변환 (8, 16, 32 클러스터)
    for num_clusters in [8, 16, 32]:
        print(f"\n[5.{num_clusters // 8}] {num_clusters}개 클러스터 적용 중...")

        # 원본 모델 복사
        model = create_model()
        train_model(model, train_images, train_labels, val_images, val_labels)

        # 클러스터링 적용
        clustered_model = apply_clustering(model, num_clusters)
        train_model(clustered_model, train_images, train_labels, val_images, val_labels)

        # 클러스터링 완료
        model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)

        # TFLite 변환
        clustered_tflite = convert_to_tflite(model_for_export)
        clustered_path = models_dir / f"mnist_model_clustered_{num_clusters}.tflite"
        clustered_path.write_bytes(clustered_tflite)
        clustered_size = os.path.getsize(clustered_path)
        compression_ratio = baseline_size / clustered_size

        print(
            f"     ✅ {clustered_path.name}: {clustered_size / 1024:.2f} KB (압축율: {compression_ratio:.2f}x)"
        )

    # 6. 16 클러스터 + 동적 범위 양자화
    print("\n[6] 16 클러스터 + 동적 범위 양자화 변환 중...")
    model = create_model()
    train_model(model, train_images, train_labels, val_images, val_labels)

    clustered_model = apply_clustering(model, 16)
    train_model(clustered_model, train_images, train_labels, val_images, val_labels)

    model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)

    clustered_quant_tflite = convert_to_tflite(model_for_export, quantize=True)
    clustered_quant_path = models_dir / "mnist_model_clustered_16_quant.tflite"
    clustered_quant_path.write_bytes(clustered_quant_tflite)
    clustered_quant_size = os.path.getsize(clustered_quant_path)
    compression_ratio = baseline_size / clustered_quant_size
    print(
        f"     ✅ {clustered_quant_path.name}: {clustered_quant_size / 1024:.2f} KB (압축율: {compression_ratio:.2f}x)"
    )

    # 7. 16 클러스터 + Int8 양자화
    print("\n[7] 16 클러스터 + Int8 양자화 변환 중...")
    model = create_model()
    train_model(model, train_images, train_labels, val_images, val_labels)

    clustered_model = apply_clustering(model, 16)
    train_model(clustered_model, train_images, train_labels, val_images, val_labels)

    model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)

    rep_data_gen = representative_data_generator(train_images)
    clustered_int8_tflite = convert_to_tflite(
        model_for_export, int8_quantize=True, representative_data=rep_data_gen()
    )
    clustered_int8_path = models_dir / "mnist_model_clustered_16_int8.tflite"
    clustered_int8_path.write_bytes(clustered_int8_tflite)
    clustered_int8_size = os.path.getsize(clustered_int8_path)
    compression_ratio = baseline_size / clustered_int8_size
    print(
        f"     ✅ {clustered_int8_path.name}: {clustered_int8_size / 1024:.2f} KB (압축율: {compression_ratio:.2f}x)"
    )

    # 8. 테스트 데이터 저장 (벤치마크용)
    print("\n[8] 테스트 데이터 저장 중...")
    test_images_path = models_dir / "mnist_test_images.npy"
    test_labels_path = models_dir / "mnist_test_labels.npy"

    np.save(test_images_path, (test_images * 255).astype(np.uint8))
    np.save(test_labels_path, test_labels)

    print(f"     ✅ {test_images_path.name}")
    print(f"     ✅ {test_labels_path.name}")

    # 9. 추론 속도 벤치마크
    print("\n[9] 추론 속도 벤치마크 중...")
    print("-" * 70)

    inference_results = {}
    model_files = {
        "Baseline": models_dir / "mnist_model_baseline.tflite",
        "Clustered-8": models_dir / "mnist_model_clustered_8.tflite",
        "Clustered-16": models_dir / "mnist_model_clustered_16.tflite",
        "Clustered-32": models_dir / "mnist_model_clustered_32.tflite",
        "Clustered-16+Quant": models_dir / "mnist_model_clustered_16_quant.tflite",
        "Clustered-16+Int8": models_dir / "mnist_model_clustered_16_int8.tflite",
    }

    for model_name, model_path in model_files.items():
        if model_path.exists():
            print(f"  {model_name} 벤치마크 중...")
            metrics = benchmark_inference(model_path, test_images, num_runs=50)
            inference_results[model_name] = metrics
            print(f"    ✅ 평균 추론 시간: {metrics['mean_ms']:.2f} ms")

    # 10. 요약
    print("\n" + "=" * 70)
    print("✅ 배치 생성 완료!")
    print("=" * 70)

    print("\n생성된 모델:")
    print(
        f"{'모델':<20} {'정확도(%)':>12} {'크기(KB)':>12} {'압축률(%)':>12} {'추론(ms)':>12} {'FPS':>10}"
    )
    print("-" * 94)

    baseline_size_kb = baseline_size / 1024
    baseline_accuracy_pct = baseline_accuracy * 100

    # Baseline
    inference = inference_results.get("Baseline", {})
    accuracy_text = f"{baseline_accuracy_pct:.2f}%"
    inference_ms = f"{inference.get('mean_ms', 0):.2f}" if inference else "N/A"
    fps = f"{1000 / float(inference_ms):.1f}" if inference_ms != "N/A" else "N/A"
    print(
        f"{'Baseline':<22} {accuracy_text:>14} {baseline_size_kb:>14.2f} {'100.0%':>14} {inference_ms:>14} {fps:>12}"
    )

    # Clustered 모델들
    for num_clusters in [8, 16, 32]:
        model_path = models_dir / f"mnist_model_clustered_{num_clusters}.tflite"
        if model_path.exists():
            clustered_size_kb = os.path.getsize(model_path) / 1024
            compression = (1 - clustered_size_kb / baseline_size_kb) * 100

            model_name = f"Clustered-{num_clusters}"
            inference = inference_results.get(model_name, {})
            inference_ms = f"{inference.get('mean_ms', 0):.2f}" if inference else "N/A"
            fps = (
                f"{1000 / float(inference_ms):.1f}" if inference_ms != "N/A" else "N/A"
            )
            ratio = f"{100 - compression:.1f}%"
            print(
                f"{model_name:<22} {accuracy_text:>14} {clustered_size_kb:>14.2f} {ratio:>14} {inference_ms:>14} {fps:>12}"
            )

    # Clustered + Quant
    quant_path = models_dir / "mnist_model_clustered_16_quant.tflite"
    if quant_path.exists():
        quant_size_kb = clustered_quant_size / 1024
        compression = (1 - quant_size_kb / baseline_size_kb) * 100

        inference = inference_results.get("Clustered-16+Quant", {})
        inference_ms = f"{inference.get('mean_ms', 0):.2f}" if inference else "N/A"
        fps = f"{1000 / float(inference_ms):.1f}" if inference_ms != "N/A" else "N/A"
        ratio = f"{100 - compression:.1f}%"
        print(
            f"{'Clustered-16+Quant':<22} {accuracy_text:>14} {quant_size_kb:>14.2f} {ratio:>14} {inference_ms:>14} {fps:>12}"
        )

    # Clustered + Int8
    int8_path = models_dir / "mnist_model_clustered_16_int8.tflite"
    if int8_path.exists():
        int8_size_kb = clustered_int8_size / 1024
        compression = (1 - int8_size_kb / baseline_size_kb) * 100

        inference = inference_results.get("Clustered-16+Int8", {})
        inference_ms = f"{inference.get('mean_ms', 0):.2f}" if inference else "N/A"
        fps = f"{1000 / float(inference_ms):.1f}" if inference_ms != "N/A" else "N/A"
        ratio = f"{100 - compression:.1f}%"
        print(
            f"{'Clustered-16+Int8':<22} {accuracy_text:>14} {int8_size_kb:>14.2f} {ratio:>14} {inference_ms:>14} {fps:>12}"
        )

    print("\n💾 데이터 파일:")
    print("   • mnist_test_images.npy")
    print("   • mnist_test_labels.npy")

    print("\n📁 출력 디렉토리: ./mnist_clustered_models/")
    print("=" * 70)


if __name__ == "__main__":
    main()
