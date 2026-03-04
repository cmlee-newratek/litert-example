"""
협업 최적화 모델 배치 생성 (Collaborative Optimization Batch Generation)

3가지 협업 최적화 경로의 모델을 모두 생성합니다:
1. CQAT: Clustering → Quantization-Aware Training
2. PQAT: Pruning → Quantization-Aware Training
3. PCQAT: Pruning → Clustering → Quantization-Aware Training
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib
import os


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


def load_data():
    """MNIST 데이터 로드"""
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    validation_split = int(0.9 * len(train_images))
    train_images_split = train_images[:validation_split]
    train_labels_split = train_labels[:validation_split]
    val_images = train_images[validation_split:]
    val_labels = train_labels[validation_split:]

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


def get_representative_data(train_images):
    """양자화용 대표 데이터"""

    def gen():
        for i in range(100):
            yield [train_images[i : i + 1].astype(np.float32)]

    return gen


def convert_to_tflite_int8(keras_model, train_images):
    """Int8 양자화로 TFLite 변환"""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = get_representative_data(train_images)
    return converter.convert()


def main():
    print("=" * 70)
    print("협업 최적화 모델 배치 생성")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1] MNIST 데이터 로드 중...")
    train_images, train_labels, val_images, val_labels, test_images, test_labels = (
        load_data()
    )
    print(f"    ✅ 훈련: {train_images.shape}, 테스트: {test_images.shape}")

    # 2. 디렉토리 생성
    print("\n[2] 출력 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_collaborative_models/")
    models_dir.mkdir(exist_ok=True, parents=True)
    print(f"    ✅ {models_dir}")

    # 3. 기본 모델 훈련
    print("\n[3] 기본 모델 훈련 중...")
    baseline_model = create_model()
    train_model(baseline_model, train_images, train_labels, val_images, val_labels)
    print("    ✅ 기본 모델 훈련 완료")

    # 4. 기본 모델 TFLite 변환
    print("\n[4] 기본 모델 TFLite 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
    baseline_tflite = converter.convert()
    baseline_path = models_dir / "mnist_model_baseline.tflite"
    baseline_path.write_bytes(baseline_tflite)
    baseline_size = os.path.getsize(baseline_path)
    print(f"    ✅ {baseline_path.name}: {baseline_size / 1024:.2f} KB")

    # ===== CQAT 모델 생성 =====
    print("\n[5] CQAT 모델 생성 중 (Clustering → QAT)...")

    # 5.1 클러스터링 모델 생성
    model_for_cqat = create_model()
    train_model(model_for_cqat, train_images, train_labels, val_images, val_labels)

    clustering_params = {
        "number_of_clusters": 16,
        "cluster_centroids_init": tfmot.clustering.keras.CentroidsInitializer.LINEAR,
    }

    clustered = tfmot.clustering.keras.cluster_weights(
        model_for_cqat, **clustering_params
    )
    clustered.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    clustered.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )

    # 5.2 클러스터링 래퍼 제거 + QAT
    model_for_cqat_qat = tfmot.clustering.keras.strip_clustering(clustered)
    cqat_model = tfmot.quantization.keras.quantize_model(model_for_cqat_qat)
    cqat_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    cqat_model.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )

    # 5.3 CQAT TFLite 변환
    cqat_tflite = convert_to_tflite_int8(cqat_model, train_images)
    cqat_path = models_dir / "mnist_model_cqat.tflite"
    cqat_path.write_bytes(cqat_tflite)
    cqat_size = os.path.getsize(cqat_path)
    print(
        f"    ✅ CQAT: {cqat_path.name} ({cqat_size / 1024:.2f} KB, {cqat_size / baseline_size * 100:.1f}%)"
    )

    # ===== PQAT 모델 생성 =====
    print("\n[6] PQAT 모델 생성 중 (Pruning → QAT)...")

    # 6.1 프루닝 모델 생성
    model_for_pqat = create_model()
    train_model(model_for_pqat, train_images, train_labels, val_images, val_labels)

    steps_per_epoch = len(train_images) // 128
    total_steps = steps_per_epoch * 2

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=total_steps,
        )
    }

    pruned = tfmot.sparsity.keras.prune_low_magnitude(model_for_pqat, **pruning_params)
    pruned.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pruned.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )

    # 6.2 프루닝 래퍼 제거 + QAT
    model_for_pqat_qat = tfmot.sparsity.keras.strip_pruning(pruned)
    pqat_model = tfmot.quantization.keras.quantize_model(model_for_pqat_qat)
    pqat_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pqat_model.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )

    # 6.3 PQAT TFLite 변환
    pqat_tflite = convert_to_tflite_int8(pqat_model, train_images)
    pqat_path = models_dir / "mnist_model_pqat.tflite"
    pqat_path.write_bytes(pqat_tflite)
    pqat_size = os.path.getsize(pqat_path)
    print(
        f"    ✅ PQAT: {pqat_path.name} ({pqat_size / 1024:.2f} KB, {pqat_size / baseline_size * 100:.1f}%)"
    )

    # ===== PCQAT 모델 생성 (완전 최적화) =====
    print("\n[7] PCQAT 모델 생성 중 (Pruning → Clustering → QAT)...")

    # 7.1 프루닝 + 클러스터링
    model_for_pcqat = create_model()
    train_model(model_for_pcqat, train_images, train_labels, val_images, val_labels)

    # 프루닝
    pruned_pc = tfmot.sparsity.keras.prune_low_magnitude(
        model_for_pcqat, **pruning_params
    )
    pruned_pc.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pruned_pc.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )

    # 프루닝 제거 후 클러스터링
    model_after_prune = tfmot.sparsity.keras.strip_pruning(pruned_pc)

    clustered_pc = tfmot.clustering.keras.cluster_weights(
        model_after_prune, **clustering_params
    )
    clustered_pc.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    clustered_pc.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )

    # 7.2 클러스터링 래퍼 제거 + QAT
    model_for_pcqat_qat = tfmot.clustering.keras.strip_clustering(clustered_pc)
    pcqat_model = tfmot.quantization.keras.quantize_model(model_for_pcqat_qat)
    pcqat_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pcqat_model.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )

    # 7.3 PCQAT TFLite 변환
    pcqat_tflite = convert_to_tflite_int8(pcqat_model, train_images)
    pcqat_path = models_dir / "mnist_model_pcqat.tflite"
    pcqat_path.write_bytes(pcqat_tflite)
    pcqat_size = os.path.getsize(pcqat_path)
    print(
        f"    ✅ PCQAT: {pcqat_path.name} ({pcqat_size / 1024:.2f} KB, {pcqat_size / baseline_size * 100:.1f}%)"
    )

    # 8. 테스트 데이터 저장
    print("\n[8] 테스트 데이터 저장 중...")
    np.save(models_dir / "mnist_test_images.npy", (test_images * 255).astype(np.uint8))
    np.save(models_dir / "mnist_test_labels.npy", test_labels)
    print("    ✅ 테스트 데이터 저장 완료")

    # 9. 요약
    print("\n" + "=" * 70)
    print("✅ 협업 최적화 모델 배치 생성 완료!")
    print("=" * 70)

    print("\n📦 생성된 모델:")
    print(f"   • {baseline_path.name:40} {baseline_size / 1024:>8.2f} KB (100.0%)")
    print(
        f"   • {cqat_path.name:40} {cqat_size / 1024:>8.2f} KB ({cqat_size / baseline_size * 100:>6.1f}%)"
    )
    print(
        f"   • {pqat_path.name:40} {pqat_size / 1024:>8.2f} KB ({pqat_size / baseline_size * 100:>6.1f}%)"
    )
    print(
        f"   • {pcqat_path.name:40} {pcqat_size / 1024:>8.2f} KB ({pcqat_size / baseline_size * 100:>6.1f}%)"
    )

    print("\n🎯 협업 최적화 경로:")
    print("   1. CQAT:  클러스터링 → 양자화 인식 훈련")
    print("   2. PQAT:  프루닝 → 양자화 인식 훈련")
    print("   3. PCQAT: 프루닝 → 클러스터링 → 양자화 인식 훈련 (최고 압축)")

    print("\n💾 데이터 파일:")
    print("   • mnist_test_images.npy")
    print("   • mnist_test_labels.npy")

    print("\n📁 출력 디렉토리: ./mnist_collaborative_models/")
    print("=" * 70)


if __name__ == "__main__":
    main()
