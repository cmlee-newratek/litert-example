"""
프루닝 예제 (Magnitude-based Weight Pruning)

프루닝은 훈련 중에 중요하지 않은 가중치를 점진적으로 0으로 만들어
희소(sparse) 모델을 생성합니다.

특징:
- 훈련 데이터 필요
- 모델 압축률 향상 (희소 모델)
- gzip 압축 시 효과적
- 추론 속도 개선 가능 (프레임워크 지원 시)

참고: https://www.tensorflow.org/model_optimization/guide/pruning
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
    print("프루닝 (Pruning) 예제 (MNIST)")
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

    # 6. 프루닝 스케줄 설정
    print("\n[6] 프루닝 스케줄 설정 중...")
    batch_size = 128
    epochs = 2
    num_images = len(train_images_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # 프루닝 파라미터 설정
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,  # 초기 희소성 (0%)
            final_sparsity=0.5,  # 최종 희소성 (50%)
            begin_step=0,
            end_step=end_step,
        )
    }

    print("    초기 희소성: 0%")
    print("    최종 희소성: 50%")
    print(f"    프루닝 단계: 0 ~ {end_step}")

    # 7. 프루닝 모델로 변환
    print("\n[7] 프루닝 모델로 변환 중...")
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model, **pruning_params
    )

    model_for_pruning.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print("    ✅ 프루닝 모델 생성 완료")

    # 8. 프루닝 콜백 설정
    print("\n[8] 프루닝 콜백 설정 중...")
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]

    # 9. 프루닝 모델 훈련
    print("\n[9] 프루닝 모델 훈련 중... (2 epochs)")
    model_for_pruning.fit(
        train_images_split,
        train_labels_split,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=1,
    )

    # 10. 프루닝 모델 평가
    print("\n[10] 프루닝 모델 평가 중...")
    pruned_loss, pruned_accuracy = model_for_pruning.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"     프루닝 모델 정확도: {pruned_accuracy * 100:.2f}%")
    print(f"     정확도 차이: {(pruned_accuracy - baseline_accuracy) * 100:+.2f}%p")

    # 11. 프루닝 완료 후 sparse 모델로 변환
    print("\n[11] 프루닝 완료 후 최종 모델로 변환 중...")
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    print("     ✅ 프루닝 래퍼 제거 완료")

    # 12. 디렉토리 생성
    print("\n[12] 모델 저장 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_pruned_models/")
    models_dir.mkdir(exist_ok=True, parents=True)

    # 13. Keras 모델 저장
    print("\n[13] Keras 모델 저장 중...")
    baseline_h5_path = models_dir / "mnist_model_baseline.h5"
    model.save(baseline_h5_path, include_optimizer=False)
    baseline_h5_size = os.path.getsize(baseline_h5_path)
    print(f"     ✅ 원본 Keras 모델: {baseline_h5_size / 1024:.2f} KB")

    pruned_h5_path = models_dir / "mnist_model_pruned.h5"
    model_for_export.save(pruned_h5_path, include_optimizer=False)
    pruned_h5_size = os.path.getsize(pruned_h5_path)
    print(f"     ✅ 프루닝 Keras 모델: {pruned_h5_size / 1024:.2f} KB")

    # 14. TFLite 모델 변환
    print("\n[14] TFLite 모델 변환 중...")

    # 원본 모델
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    baseline_tflite_path = models_dir / "mnist_model_baseline.tflite"
    baseline_tflite_path.write_bytes(tflite_model)
    baseline_tflite_size = os.path.getsize(baseline_tflite_path)
    print(f"     ✅ 원본 TFLite 모델: {baseline_tflite_size / 1024:.2f} KB")

    # 프루닝 모델
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()
    pruned_tflite_path = models_dir / "mnist_model_pruned.tflite"
    pruned_tflite_path.write_bytes(pruned_tflite_model)
    pruned_tflite_size = os.path.getsize(pruned_tflite_path)
    print(f"     ✅ 프루닝 TFLite 모델: {pruned_tflite_size / 1024:.2f} KB")

    # 15. gzip 압축 효과 확인
    print("\n[15] gzip 압축 효과 확인 중...")

    # 원본 모델 압축
    baseline_compressed = gzip.compress(tflite_model)
    _, baseline_zip_path = tempfile.mkstemp(".gz")
    with open(baseline_zip_path, "wb") as f:
        f.write(baseline_compressed)
    baseline_compressed_size = os.path.getsize(baseline_zip_path)

    # 프루닝 모델 압축
    pruned_compressed = gzip.compress(pruned_tflite_model)
    _, pruned_zip_path = tempfile.mkstemp(".gz")
    with open(pruned_zip_path, "wb") as f:
        f.write(pruned_compressed)
    pruned_compressed_size = os.path.getsize(pruned_zip_path)

    compression_benefit = (1 - pruned_compressed_size / baseline_compressed_size) * 100

    print(f"     원본 모델 (gzip):    {baseline_compressed_size / 1024:.2f} KB")
    print(f"     프루닝 모델 (gzip):  {pruned_compressed_size / 1024:.2f} KB")
    print(f"     압축 개선율:         {compression_benefit:.1f}%")

    # 16. TFLite 모델 정확도 평가
    print("\n[16] TFLite 모델 정확도 평가 중...")

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
    print(f"     원본 TFLite 정확도:     {baseline_tflite_accuracy * 100:.2f}%")

    pruned_tflite_accuracy = evaluate_tflite_model(
        pruned_tflite_model, test_images, test_labels
    )
    print(f"     프루닝 TFLite 정확도:   {pruned_tflite_accuracy * 100:.2f}%")

    # 17. 희소성 확인
    print("\n[17] 모델 희소성(sparsity) 확인")

    # 가중치 분석
    total_params = 0
    zero_params = 0
    for layer in model_for_export.layers:
        for weight in layer.weights:
            weight_array = weight.numpy()
            total_params += weight_array.size
            zero_params += np.sum(weight_array == 0)

    sparsity = zero_params / total_params * 100
    print(f"     전체 파라미터: {total_params:,}")
    print(f"     0인 파라미터: {zero_params:,}")
    print(f"     희소성: {sparsity:.2f}%")

    # 18. 레이어별 희소성 분석
    print("\n[18] 레이어별 희소성 분석")
    for i, layer in enumerate(model_for_export.layers):
        if len(layer.weights) > 0:
            layer_total = 0
            layer_zeros = 0
            for weight in layer.weights:
                weight_array = weight.numpy()
                layer_total += weight_array.size
                layer_zeros += np.sum(weight_array == 0)
            if layer_total > 0:
                layer_sparsity = layer_zeros / layer_total * 100
                print(
                    f"     Layer {i} ({layer.name}): {layer_sparsity:.2f}% (0개수: {layer_zeros:,}/{layer_total:,})"
                )

    # 19. 요약
    print("\n" + "=" * 70)
    print("✅ 프루닝 예제 완료!")
    print("=" * 70)
    print("\n📊 주요 결과:")
    print(f"   • 원본 정확도:           {baseline_accuracy * 100:.2f}%")
    print(f"   • 프루닝 후 정확도:      {pruned_tflite_accuracy * 100:.2f}%")
    print(
        f"   • 정확도 변화:           {(pruned_tflite_accuracy - baseline_accuracy) * 100:+.2f}%p"
    )
    print(f"\n   • 원본 크기:             {baseline_tflite_size / 1024:.2f} KB")
    print(f"   • 프루닝 후:             {pruned_tflite_size / 1024:.2f} KB")
    print(f"   • 원본 (gzip):           {baseline_compressed_size / 1024:.2f} KB")
    print(
        f"   • 프루닝 후 (gzip):      {pruned_compressed_size / 1024:.2f} KB ({compression_benefit:.1f}% 감소)"
    )
    print(f"\n   • 모델 희소성:           {sparsity:.1f}%")

    print("\n💡 주요 개념:")
    print("   • 프루닝은 중요하지 않은 가중치를 0으로 만듭니다")
    print("   • gzip 압축 시 0이 많을수록 압축률이 높아집니다")
    print("   • 희소성 비율을 조절하여 정확도와 크기의 균형을 맞출 수 있습니다")
    print("   • 양자화와 결합하면 더 큰 압축 효과를 얻을 수 있습니다")

    print("\n📁 생성된 파일:")
    print(f"   • {baseline_h5_path}")
    print(f"   • {pruned_h5_path}")
    print(f"   • {baseline_tflite_path}")
    print(f"   • {pruned_tflite_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
