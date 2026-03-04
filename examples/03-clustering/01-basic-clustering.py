"""
클러스터링 예제 (Weight Clustering)

클러스터링은 모델의 가중치를 N개의 클러스터로 그룹화하여
고유한 가중치 값의 개수를 줄입니다.

특징:
- 훈련 데이터 필요
- 모델 압축률 향상 (최대 5x)
- 양자화와 결합 가능 (CQAT)
- 정확도 손실 최소화

참고: https://www.tensorflow.org/model_optimization/guide/clustering
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib
import os
import tempfile
import gzip


def main():
    print("=" * 70)
    print("클러스터링 (Weight Clustering) 예제 (MNIST)")
    print("=" * 70)

    # 1. MNIST 데이터셋 로드
    print("\n[1] MNIST 데이터셋 로드 중...")
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 정규화: 0~1 범위로
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 검증 데이터 분리
    validation_split_index = int(0.9 * len(train_images))
    train_images_split = train_images[:validation_split_index]
    train_labels_split = train_labels[:validation_split_index]
    val_images = train_images[validation_split_index:]
    val_labels = train_labels[validation_split_index:]

    print(f"    훈련 이미지: {train_images_split.shape}")
    print(f"    검증 이미지: {val_images.shape}")
    print(f"    테스트 이미지: {test_images.shape}")

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

    # 4. 기본 모델 훈련
    print("\n[4] 기본 모델 훈련 중... (2 epochs)")
    model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=1,
    )

    # 5. 원본 모델 평가
    print("\n[5] 원본 모델 평가 중...")
    baseline_loss, baseline_accuracy = model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"    원본 모델 정확도: {baseline_accuracy * 100:.2f}%")

    # 6. 클러스터링 설정
    print("\n[6] 클러스터링 설정 중...")
    # 16개 클러스터로 가중치를 그룹화
    clustering_params = {
        "number_of_clusters": 16,
        "cluster_centroids_init": tfmot.clustering.keras.CentroidsInitializer.LINEAR,
    }

    print("    클러스터 수: 16")
    print("    초기화 방식: LINEAR")

    # 7. 클러스터링 모델로 변환
    print("\n[7] 클러스터링 모델로 변환 중...")
    cluster_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)

    cluster_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("    ✅ 클러스터링 모델 생성 완료")

    # 8. 클러스터링 모델 훈련
    print("\n[8] 클러스터링 모델 훈련 중... (2 epochs)")
    cluster_model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=1,
    )

    # 9. 클러스터링 모델 평가
    print("\n[9] 클러스터링 모델 평가 중...")
    clustered_loss, clustered_accuracy = cluster_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"     클러스터링 모델 정확도: {clustered_accuracy * 100:.2f}%")
    print(f"     정확도 차이: {(clustered_accuracy - baseline_accuracy) * 100:+.2f}%p")

    # 10. 클러스터링 완료 후 최종 모델로 변환
    print("\n[10] 클러스터링 완료 후 최종 모델로 변환 중...")
    model_for_export = tfmot.clustering.keras.strip_clustering(cluster_model)
    print("     ✅ 클러스터링 래퍼 제거 완료")

    # 11. 디렉토리 생성
    print("\n[11] 모델 저장 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_clustered_models/")
    models_dir.mkdir(exist_ok=True, parents=True)

    # 12. Float32 원본 모델 변환
    print("\n[12] Float32 원본 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    baseline_tflite_path = models_dir / "mnist_model_baseline.tflite"
    baseline_tflite_path.write_bytes(tflite_model)
    baseline_tflite_size = os.path.getsize(baseline_tflite_path)
    print(f"     ✅ Float32 모델 저장: {baseline_tflite_size / 1024:.2f} KB")

    # 13. 클러스터링된 Float32 모델 변환
    print("\n[13] 클러스터링된 Float32 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    clustered_tflite_model = converter.convert()

    clustered_tflite_path = models_dir / "mnist_model_clustered.tflite"
    clustered_tflite_path.write_bytes(clustered_tflite_model)
    clustered_tflite_size = os.path.getsize(clustered_tflite_path)
    print(
        f"     ✅ 클러스터링 Float32 모델 저장: {clustered_tflite_size / 1024:.2f} KB"
    )

    # 14. gzip 압축 효과 확인
    print("\n[14] gzip 압축 효과 확인 중...")

    # 원본 모델 압축
    baseline_compressed = gzip.compress(tflite_model)
    _, baseline_zip_path = tempfile.mkstemp(".gz")
    with open(baseline_zip_path, "wb") as f:
        f.write(baseline_compressed)
    baseline_compressed_size = os.path.getsize(baseline_zip_path)

    # 클러스터링 모델 압축
    clustered_compressed = gzip.compress(clustered_tflite_model)
    _, clustered_zip_path = tempfile.mkstemp(".gz")
    with open(clustered_zip_path, "wb") as f:
        f.write(clustered_compressed)
    clustered_compressed_size = os.path.getsize(clustered_zip_path)

    compression_benefit = (
        1 - clustered_compressed_size / baseline_compressed_size
    ) * 100

    print(f"     원본 모델 (gzip):        {baseline_compressed_size / 1024:.2f} KB")
    print(f"     클러스터링 모델 (gzip):  {clustered_compressed_size / 1024:.2f} KB")
    print(f"     압축 개선율:             {compression_benefit:.1f}%")

    # 15. TFLite 모델 정확도 평가
    print("\n[15] TFLite 모델 정확도 평가 중...")

    def evaluate_tflite_model(tflite_model_content, test_images, test_labels):
        """TFLite 모델 정확도 평가"""
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        correct_count = 0
        for i in range(len(test_images)):
            test_image = np.expand_dims(test_images[i], axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            predicted_label = np.argmax(output[0])
            if predicted_label == test_labels[i]:
                correct_count += 1

        return correct_count / len(test_labels)

    baseline_tflite_accuracy = evaluate_tflite_model(
        tflite_model, test_images, test_labels
    )
    print(f"     원본 Float32 정확도:     {baseline_tflite_accuracy * 100:.2f}%")

    clustered_tflite_accuracy = evaluate_tflite_model(
        clustered_tflite_model, test_images, test_labels
    )
    print(f"     클러스터링 Float32 정확도: {clustered_tflite_accuracy * 100:.2f}%")

    # 16. 클러스터 분석
    print("\n[16] 클러스터 분석")
    for i, layer in enumerate(model_for_export.layers):
        if len(layer.weights) > 0:
            for weight in layer.weights:
                weight_array = weight.numpy()
                unique_values = len(np.unique(weight_array))
                total_values = weight_array.size
                compression_ratio = total_values / unique_values
                print(
                    f"     Layer {i} ({layer.name}): {unique_values} 고유값 (압축율: {compression_ratio:.1f}x)"
                )

    # 17. 테스트 데이터 저장 (벤치마크용)
    print("\n[17] 테스트 데이터 저장 중 (벤치마크용)...")
    test_images_path = models_dir / "mnist_test_images.npy"
    test_labels_path = models_dir / "mnist_test_labels.npy"

    np.save(test_images_path, (test_images * 255).astype(np.uint8))
    np.save(test_labels_path, test_labels)
    print(f"     ✅ {test_images_path}")
    print(f"     ✅ {test_labels_path}")

    # 18. 요약
    print("\n" + "=" * 70)
    print("✅ 클러스터링 예제 완료!")
    print("=" * 70)

    print("\n📊 주요 결과:")
    print(f"   • 원본 정확도:           {baseline_accuracy * 100:.2f}%")
    print(f"   • 클러스터링 후 정확도:  {clustered_tflite_accuracy * 100:.2f}%")
    print(
        f"   • 정확도 변화:           {(clustered_tflite_accuracy - baseline_accuracy) * 100:+.2f}%p"
    )

    print(f"\n   • 원본 크기:             {baseline_tflite_size / 1024:.2f} KB")
    print(f"   • 클러스터링 후:         {clustered_tflite_size / 1024:.2f} KB")
    print(f"   • 원본 (gzip):           {baseline_compressed_size / 1024:.2f} KB")
    print(
        f"   • 클러스터링 (gzip):     {clustered_compressed_size / 1024:.2f} KB ({compression_benefit:.1f}% 감소)"
    )

    print("\n💡 주요 개념:")
    print("   • 클러스터링은 고유한 가중치 값의 개수를 줄입니다")
    print("   • 16개 클러스터로 그룹화하면 더 효율적인 압축 가능")
    print("   • 클러스터 수를 조절하여 정확도와 크기의 균형 조정")
    print("   • 양자화와 결합하면 추가 압축 효과")

    print("\n📁 생성된 파일:")
    print(f"   • {baseline_tflite_path}")
    print(f"   • {clustered_tflite_path}")
    print(f"   • {test_images_path}")
    print(f"   • {test_labels_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
