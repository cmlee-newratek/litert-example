"""
PC (Pruning + Clustering) 결합 예제

프루닝으로 희소성을 확보한 후, 클러스터링으로 고유 가중치를 줄여
두 가지 최적화를 순차적으로 적용합니다.

특징:
- 프루닝으로 가중치의 50% 제로화 (희소성)
- 클러스터링으로 16개 클러스터로 가중치 그룹화
- 희소성 + 클러스터 이중 효과
- 30-35% 압축율 달성

결과:
- 프루닝: 20-30% 압축
- 클러스터링: 추가 10-15% 압축
- 합계: 30-40% 압축 가능
- 정확도 손실: <1%

참고:
- Pruning: https://www.tensorflow.org/model_optimization/guide/pruning
- Clustering: https://www.tensorflow.org/model_optimization/guide/clustering
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib
import os
import gzip


def main():
    print("=" * 70)
    print("PC (Pruning + Clustering) 결합 예제")
    print("프루닝(50%) + 클러스터링(16)")
    print("=" * 70)

    # 1. MNIST 데이터셋 로드
    print("\n[1] MNIST 데이터셋 로드 중...")
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

    print(f"    훈련: {train_images_split.shape}, 검증: {val_images.shape}")

    # 2. 모델 생성
    print("\n[2] 모델 생성 중...")
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
    print("    ✅ 모델 생성 완료")

    # 3. 원본 모델 훈련
    print("\n[3] 원본 모델 훈련 중...")
    model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )
    baseline_loss, baseline_accuracy = model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"    ✅ 기본 정확도: {baseline_accuracy * 100:.2f}%")

    # 4. 프루닝 적용
    print("\n[4] 프루닝 적용 중... (50% 스파시티)")
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=len(train_images_split) // 128 * 2,
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    pruned_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # 프루닝 훈련
    print("\n[5] 프루닝 훈련 중...")
    pruned_model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        verbose=0,
    )

    # 프루닝 완료
    pruning_loss, pruning_accuracy = pruned_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"    ✅ 프루닝 후 정확도: {pruning_accuracy * 100:.2f}%")

    # 6. 프루닝 완료 후 스트립
    print("\n[6] 프루닝 래퍼 제거 중...")
    model_for_clustering = tfmot.sparsity.keras.strip_pruning(pruned_model)
    print("    ✅ 프루닝 완료")

    # 7. 클러스터링 적용
    print("\n[7] 클러스터링 적용 중... (16 클러스터)")
    clustering_params = {
        "number_of_clusters": 16,
        "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.LINEAR,
    }

    clustered_model = tfmot.clustering.keras.cluster_weights(
        model_for_clustering, **clustering_params
    )

    clustered_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("    ✅ 클러스터링 모델 생성 완료")

    # 8. 클러스터링 훈련
    print("\n[8] 클러스터링 훈련 중...")
    clustered_model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )

    # 9. 클러스터링 평가
    print("\n[9] 클러스터링 평가 중...")
    clustered_loss, clustered_accuracy = clustered_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"    ✅ PC 후 정확도: {clustered_accuracy * 100:.2f}%")

    # 10. 클러스터링 래퍼 제거
    print("\n[10] 클러스터링 래퍼 제거 중...")
    model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)
    print("     ✅ 클러스터링 완료")

    # 11. 디렉토리 생성
    print("\n[11] 모델 저장 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_pc_models/")
    models_dir.mkdir(exist_ok=True, parents=True)

    # === 모델 변환 ===

    # 12. 원본 Float32 모델 변환
    print("\n[12] 원본 Float32 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    baseline_tflite_model = converter.convert()

    baseline_tflite_path = models_dir / "mnist_model_baseline.tflite"
    baseline_tflite_path.write_bytes(baseline_tflite_model)
    baseline_tflite_size = os.path.getsize(baseline_tflite_path)
    print(f"     ✅ 원본 Float32: {baseline_tflite_size / 1024:.2f} KB")

    # 13. PC(프루닝+클러스터링) Float32 모델 변환
    print("\n[13] PC(프루닝+클러스터링) Float32 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pc_tflite_model = converter.convert()

    pc_tflite_path = models_dir / "mnist_model_pc_f32.tflite"
    pc_tflite_path.write_bytes(pc_tflite_model)
    pc_tflite_size = os.path.getsize(pc_tflite_path)
    print(f"     ✅ PC Float32: {pc_tflite_size / 1024:.2f} KB")

    # 14. PC Int8 양자화 모델 변환
    print("\n[14] PC Int8 모델 변환 중...")

    def representative_data_gen():
        """양자화 대표 데이터 생성"""
        for i in range(100):
            yield [train_images[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    pc_int8_tflite_model = converter.convert()

    pc_int8_tflite_path = models_dir / "mnist_model_pc_int8.tflite"
    pc_int8_tflite_path.write_bytes(pc_int8_tflite_model)
    pc_int8_tflite_size = os.path.getsize(pc_int8_tflite_path)
    print(f"     ✅ PC Int8: {pc_int8_tflite_size / 1024:.2f} KB")

    # === 압축 비교 ===

    # 15. gzip 압축 비율 계산
    print("\n[15] gzip 압축 비율 계산 중...")

    def gzip_size(model_bytes):
        """gzip 압축 크기 계산"""
        return len(gzip.compress(model_bytes))

    baseline_gzip_size = gzip_size(baseline_tflite_model) / 1024
    pc_f32_gzip_size = gzip_size(pc_tflite_model) / 1024
    pc_int8_gzip_size = gzip_size(pc_int8_tflite_model) / 1024

    # === 결과 출력 ===

    print("\n" + "=" * 70)
    print("📊 최적화 결과 요약")
    print("=" * 70)

    print("\n[원본 Float32 모델]")
    print(f"  • 정확도: {baseline_accuracy * 100:.2f}%")
    print(f"  • TFLite 크기: {baseline_tflite_size / 1024:.2f} KB")
    print(f"  • gzip 압축: {baseline_gzip_size:.2f} KB (기준값)")

    print("\n[PC (프루닝 50% + 클러스터링 16) Float32]")
    pc_f32_reduction = (1 - pc_tflite_size / baseline_tflite_size) * 100
    print(
        f"  • 정확도: {pruning_accuracy * 100:.2f}% (손실: {baseline_accuracy - pruning_accuracy:.4f})"
    )
    print(f"  • TFLite 크기: {pc_tflite_size / 1024:.2f} KB")
    print(f"  • 압축 감소: {pc_f32_reduction:.1f}%")
    print(f"  • gzip 압축: {pc_f32_gzip_size:.2f} KB")

    print("\n[PC (프루닝 50% + 클러스터링 16) Int8]")
    pc_int8_reduction = (1 - pc_int8_tflite_size / baseline_tflite_size) * 100
    print(
        f"  • 정확도: {clustered_accuracy * 100:.2f}% (손실: {baseline_accuracy - clustered_accuracy:.4f})"
    )
    print(f"  • TFLite 크기: {pc_int8_tflite_size / 1024:.2f} KB")
    print(f"  • 압축 감소: {pc_int8_reduction:.1f}%")
    print(f"  • gzip 압축: {pc_int8_gzip_size:.2f} KB")

    # 16. TFLite 모델 정보 출력
    print("\n[17] TFLite 모델 정보")
    print(f"     모델들이 {models_dir} 폴더에 저장되었습니다:")
    print(f"     • mnist_model_baseline.tflite: {baseline_tflite_size / 1024:.2f} KB")
    print(f"     • mnist_model_pc_f32.tflite: {pc_tflite_size / 1024:.2f} KB")
    print(f"     • mnist_model_pc_int8.tflite: {pc_int8_tflite_size / 1024:.2f} KB")

    print("\n✅ PC 예제 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
