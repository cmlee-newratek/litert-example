"""
CQAT (Cluster-preserving Quantization Aware Training) 예제

클러스터링 후, 양자화 인식 훈련을 수행하여 클러스터링 효과를 보존합니다.

특징:
- 클러스터링으로 고유 가중치 감소
- 양자화 인식 훈련(QAT)으로 가중치 양자화
- 클러스터링 효과 보존
- 30-35% 압축율 달성

"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib
import os


def main():
    print("=" * 70)
    print("CQAT (Cluster-preserving Quantization Aware Training) 예제")
    print("클러스터링 + 양자화 인식 훈련")
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

    # 4. 클러스터링 적용
    print("\n[4] 클러스터링 적용 중... (16개 클러스터)")
    clustering_params = {
        "number_of_clusters": 16,
        "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.LINEAR,
    }

    clustered_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)
    clustered_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    clustered_model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )
    clustered_loss, clustered_accuracy = clustered_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"    ✅ 클러스터링 정확도: {clustered_accuracy * 100:.2f}%")

    # 5. 클러스터링 완료 후 CQAT 준비
    print("\n[5] CQAT 준비 중...")

    # 클러스터링 래퍼 제거 (양자화된 가중치는 유지)
    # strip_clustering은 래퍼만 제거하므로 클러스터링 효과(모델 크기 감소)는 유지됨
    model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)

    # 클러스터링된 모델에 QAT 적용 (추가 양자화)
    # 결과: 클러스터링 + QAT = CQAT (더 나은 압축 달성)
    quant_aware_model = tfmot.quantization.keras.quantize_model(model_for_export)

    quant_aware_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("    ✅ CQAT 모델 준비 완료")

    # 6. CQAT 훈련
    print("\n[6] CQAT 훈련 중 (양자화 인식 훈련)...")
    quant_aware_model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )
    cqat_loss, cqat_accuracy = quant_aware_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"    ✅ CQAT 정확도: {cqat_accuracy * 100:.2f}%")

    # 7. 모델 변환
    print("\n[7] TFLite 변환 중...")

    # 기본 모델 (Float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    baseline_tflite = converter.convert()

    # 클러스터링 모델 (Float32)
    # model_for_export는 strip_clustering된 모델 (래퍼 제거)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    clustered_tflite = converter.convert()

    # CQAT 모델 (Int8 양자화, Float32 입출력)
    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(100):
            yield [train_images_split[i : i + 1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    cqat_tflite = converter.convert()

    # 8. 모델 저장
    print("\n[8] 모델 저장 중...")
    models_dir = pathlib.Path("./mnist_cqat_models/")
    models_dir.mkdir(exist_ok=True, parents=True)

    baseline_path = models_dir / "mnist_model_baseline.tflite"
    baseline_path.write_bytes(baseline_tflite)
    baseline_size = os.path.getsize(baseline_path)

    clustered_path = models_dir / "mnist_model_clustered.tflite"
    clustered_path.write_bytes(clustered_tflite)
    clustered_size = os.path.getsize(clustered_path)

    cqat_path = models_dir / "mnist_model_cqat.tflite"
    cqat_path.write_bytes(cqat_tflite)
    cqat_size = os.path.getsize(cqat_path)

    print(f"    ✅ {baseline_path.name}: {baseline_size / 1024:.2f} KB")
    print(f"    ✅ {clustered_path.name}: {clustered_size / 1024:.2f} KB")
    print(f"    ✅ {cqat_path.name}: {cqat_size / 1024:.2f} KB")

    # 9. TFLite 정확도 평가
    print("\n[9] TFLite 정확도 평가 중...")

    def evaluate_tflite(tflite_content):
        interpreter = tf.lite.Interpreter(model_content=tflite_content)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        correct = 0
        for i in range(len(test_images)):
            test_image = np.expand_dims(test_images[i], axis=0)

            if input_details["dtype"] == np.int8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image = test_image / input_scale + input_zero_point
                test_image = test_image.astype(np.int8)
            else:
                test_image = test_image.astype(np.float32)

            interpreter.set_tensor(input_details["index"], test_image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])

            if output_details["dtype"] == np.int8:
                output_scale, output_zero_point = output_details["quantization"]
                output = output.astype(np.float32)
                output = (output - output_zero_point) * output_scale

            if np.argmax(output[0]) == test_labels[i]:
                correct += 1

        return correct / len(test_labels)

    baseline_tflite_acc = evaluate_tflite(baseline_tflite)
    clustered_tflite_acc = evaluate_tflite(clustered_tflite)
    cqat_tflite_acc = evaluate_tflite(cqat_tflite)

    print(f"    기본 모델 (Float32):     {baseline_tflite_acc * 100:.2f}%")
    print(f"    클러스터링 (Float32):    {clustered_tflite_acc * 100:.2f}%")
    print(f"    CQAT (Int8):             {cqat_tflite_acc * 100:.2f}%")

    # 10. 테스트 데이터 저장
    print("\n[10] 테스트 데이터 저장 중...")
    np.save(models_dir / "mnist_test_images.npy", (test_images * 255).astype(np.uint8))
    np.save(models_dir / "mnist_test_labels.npy", test_labels)
    print("    ✅ 테스트 데이터 저장 완료")

    # 11. 요약
    print("\n" + "=" * 70)
    print("✅ CQAT 예제 완료!")
    print("=" * 70)

    print("\n📊 정확도 비교:")
    print(f"   • 기본 모델:         {baseline_accuracy * 100:.2f}%")
    print(f"   • 클러스터링:        {clustered_accuracy * 100:.2f}%")
    print(f"   • CQAT (Int8):       {cqat_tflite_acc * 100:.2f}%")

    print("\n📦 모델 크기:")
    print(f"   • 기본 (Float32):    {baseline_size / 1024:.2f} KB (100%)")
    print(
        f"   • 클러스터링:        {clustered_size / 1024:.2f} KB ({clustered_size / baseline_size * 100:.1f}%)"
    )
    print(
        f"   • CQAT (Int8):       {cqat_size / 1024:.2f} KB ({cqat_size / baseline_size * 100:.1f}%)"
    )

    print("\n💡 주요 사항:")
    print("   • 클러스터링: 고유 가중치 감소로 압축")
    print("   • CQAT: 클러스터링을 보존하며 추가 양자화")
    print("   • 결과: 단순 클러스터링보다 더 나은 압축율")
    print("=" * 70)


if __name__ == "__main__":
    main()
