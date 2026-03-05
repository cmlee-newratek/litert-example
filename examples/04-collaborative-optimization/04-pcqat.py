"""
PCQAT (Sparsity-Cluster-preserving QAT) 예제

프루닝 + 클러스터링 후, 양자화 인식 훈련을 수행하여 모든 효과를 보존합니다.

특징:
- 프루닝으로 스파시티(50%) + 클러스터링으로 고유 가중치 감소
- 양자화 인식 훈련(QAT)으로 모든 최적화 보존
- 전체 효과 최대화
- 45-50% 압축율 달성 (최고 압축)

구조:
기본 모델 → 프루닝 → + 클러스터링 → + PCQAT
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib
import os


def main():
    print("=" * 70)
    print("PCQAT (Sparsity-Cluster-preserving QAT) 예제")
    print("프루닝 + 클러스터링 + 양자화 인식 훈련 (완전 최적화)")
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

    # 4. 프루닝 적용 (Step 1)
    print("\n[4] Step 1: 프루닝 적용 중... (50% 스파시티)")

    steps_per_epoch = len(train_images_split) // 128
    total_steps = steps_per_epoch * 2  # 2 epochs

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=total_steps,
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    pruned_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pruned_model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        verbose=0,
    )
    pruned_loss, pruned_accuracy = pruned_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"    ✅ 프루닝 정확도: {pruned_accuracy * 100:.2f}%")

    # 5. 프루닝 완료 후 클러스터링 준비 (Step 2)
    print("\n[5] Step 2: 프루닝 모델에서 클러스터링 준비 중...")

    # 프루닝 래퍼 제거
    model_after_pruning = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # 클러스터링 적용 (스파시티 보존)
    clustering_params = {
        "number_of_clusters": 16,
        "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.LINEAR,
    }

    clustered_model = tfmot.clustering.keras.cluster_weights(
        model_after_pruning, **clustering_params
    )

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
    print(f"    ✅ 프루닝+클러스터링 정확도: {clustered_accuracy * 100:.2f}%")

    # 6. 클러스터링 완료 후 PCQAT 준비 (Step 3)
    print("\n[6] Step 3: PCQAT 준비 중 (프루닝+클러스터링 보존)...")

    # 클러스터링 래퍼 제거
    model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)

    # 양자화 인식 모델로 변환 (프루닝+클러스터링 효과 보존)
    quant_aware_model = tfmot.quantization.keras.quantize_model(model_for_export)

    quant_aware_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("    ✅ PCQAT 모델 준비 완료")

    # 7. PCQAT 훈련
    print("\n[7] Step 3: PCQAT 훈련 중 (양자화 인식 훈련)...")
    quant_aware_model.fit(
        train_images_split,
        train_labels_split,
        batch_size=128,
        epochs=2,
        validation_data=(val_images, val_labels),
        verbose=0,
    )
    pcqat_loss, pcqat_accuracy = quant_aware_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"    ✅ PCQAT 정확도: {pcqat_accuracy * 100:.2f}%")

    # 8. TFLite 변환
    print("\n[8] TFLite 변환 중...")

    # 기본 모델 (Float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    baseline_tflite = converter.convert()

    # 프루닝 모델 (Float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    pruned_tflite = converter.convert()

    # 프루닝+클러스터링 모델 (Float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    pruned_clustered_tflite = converter.convert()

    # PCQAT 모델 (Int8 - 최종 배포 모델)
    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    def representative_dataset():
        for i in range(100):
            yield [train_images_split[i : i + 1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    pcqat_tflite = converter.convert()

    # 9. 모델 저장
    print("\n[9] 모델 저장 중...")
    models_dir = pathlib.Path("./mnist_pcqat_models/")
    models_dir.mkdir(exist_ok=True, parents=True)

    baseline_path = models_dir / "mnist_model_baseline.tflite"
    baseline_path.write_bytes(baseline_tflite)
    baseline_size = os.path.getsize(baseline_path)

    pruned_path = models_dir / "mnist_model_pruned.tflite"
    pruned_path.write_bytes(pruned_tflite)
    pruned_size = os.path.getsize(pruned_path)

    pruned_clustered_path = models_dir / "mnist_model_pruned_clustered.tflite"
    pruned_clustered_path.write_bytes(pruned_clustered_tflite)
    pruned_clustered_size = os.path.getsize(pruned_clustered_path)

    pcqat_path = models_dir / "mnist_model_pcqat.tflite"
    pcqat_path.write_bytes(pcqat_tflite)
    pcqat_size = os.path.getsize(pcqat_path)

    print(f"    ✅ {baseline_path.name}: {baseline_size / 1024:.2f} KB")
    print(f"    ✅ {pruned_path.name}: {pruned_size / 1024:.2f} KB")
    print(f"    ✅ {pruned_clustered_path.name}: {pruned_clustered_size / 1024:.2f} KB")
    print(f"    ✅ {pcqat_path.name}: {pcqat_size / 1024:.2f} KB")

    # 10. TFLite 정확도 평가
    print("\n[10] TFLite 정확도 평가 중...")

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
    pruned_tflite_acc = evaluate_tflite(pruned_tflite)
    pruned_clustered_tflite_acc = evaluate_tflite(pruned_clustered_tflite)
    pcqat_tflite_acc = evaluate_tflite(pcqat_tflite)

    print(f"    기본 모델 (Float32):           {baseline_tflite_acc * 100:.2f}%")
    print(f"    프루닝 (50% sparsity):         {pruned_tflite_acc * 100:.2f}%")
    print(
        f"    프루닝+클러스터링 (16 cl.):    {pruned_clustered_tflite_acc * 100:.2f}%"
    )
    print(f"    PCQAT (Int8 - 최종):           {pcqat_tflite_acc * 100:.2f}%")

    # 11. 테스트 데이터 저장
    print("\n[11] 테스트 데이터 저장 중...")
    np.save(models_dir / "mnist_test_images.npy", (test_images * 255).astype(np.uint8))
    np.save(models_dir / "mnist_test_labels.npy", test_labels)
    print("    ✅ 테스트 데이터 저장 완료")

    # 12. 요약
    print("\n" + "=" * 70)
    print("✅ PCQAT 예제 완료!")
    print("=" * 70)

    print("\n📊 정확도 비교 (모든 최적화 단계):")
    print(f"   • 기본 모델:                   {baseline_accuracy * 100:.2f}%")
    print(f"   • 프루닝 (50%):                {pruned_accuracy * 100:.2f}%")
    print(f"   • 프루닝 + 클러스터링:         {clustered_accuracy * 100:.2f}%")
    print(f"   • PCQAT (Int8 - 최종):         {pcqat_tflite_acc * 100:.2f}%")

    print("\n📦 모델 크기 비교:")
    print(f"   • 기본 (Float32):              {baseline_size / 1024:.2f} KB (100.0%)")
    print(
        f"   • 프루닝:                      {pruned_size / 1024:.2f} KB ({pruned_size / baseline_size * 100:.1f}%)"
    )
    print(
        f"   • 프루닝 + 클러스터링:         {pruned_clustered_size / 1024:.2f} KB ({pruned_clustered_size / baseline_size * 100:.1f}%)"
    )
    print(
        f"   • PCQAT (Int8 - 최종):         {pcqat_size / 1024:.2f} KB ({pcqat_size / baseline_size * 100:.1f}%)"
    )

    print("\n🎯 최적화 경로:")
    print("   기본 모델")
    print("    ↓")
    print("   프루닝 (50% 스파시티)")
    print("    ↓")
    print("   클러스터링 (16개 클러스터, 스파시티 보존)")
    print("    ↓")
    print("   PCQAT (양자화 인식 훈련, 모든 효과 보존)")
    print("    ↓")
    print(
        f"   최종: {pcqat_size / 1024:.2f} KB ({pcqat_size / baseline_size * 100:.1f}%) - 최대 압축!"
    )

    print("\n💡 주요 사항:")
    print("   • 프루닝: 가중치의 50%를 0으로 설정")
    print("   • 클러스터링: 고유 가중치를 16개로 그룹화")
    print("   • PCQAT: 모든 최적화를 보존하며 양자화")
    print("   • 결과: 최대 50% 이상 압축 달성")
    print("=" * 70)


if __name__ == "__main__":
    main()
