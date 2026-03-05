"""
프루닝 + 양자화 결합 예제 (Pruning + Quantization)

프루닝으로 희소 모델을 만든 후 양자화를 적용하여 최대 압축률을 달성합니다.

특징:
- 프루닝으로 가중치를 희소하게 만듦
- 양자화로 Float32 → Int8 변환
- 최대 압축 효과 (10-15x)
- 정확도 손실 최소화

참고:
- Pruning: https://www.tensorflow.org/model_optimization/guide/pruning
- Quantization: https://www.tensorflow.org/model_optimization/guide/quantization
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
    print("프루닝 + 양자화 결합 예제 (MNIST)")
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

    # 11. 프루닝 완료 후 sparse 모델로 변환
    print("\n[11] 프루닝 완료 후 최종 모델로 변환 중...")
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    print("     ✅ 프루닝 래퍼 제거 완료")

    # 12. 디렉토리 생성
    print("\n[12] 모델 저장 디렉토리 생성 중...")
    models_dir = pathlib.Path("./mnist_pruned_models/")
    models_dir.mkdir(exist_ok=True, parents=True)

    # === 원본 모델 변환 ===

    # 13. Float32 원본 모델 변환
    print("\n[13] Float32 원본 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    baseline_tflite_model = converter.convert()

    baseline_tflite_path = models_dir / "mnist_model_baseline.tflite"
    baseline_tflite_path.write_bytes(baseline_tflite_model)
    baseline_tflite_size = os.path.getsize(baseline_tflite_path)
    print(f"     ✅ 원본 Float32 모델: {baseline_tflite_size / 1024:.2f} KB")

    # === 프루닝만 적용 ===

    # 14. 프루닝된 Float32 모델 변환
    print("\n[14] 프루닝 Float32 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()

    pruned_tflite_path = models_dir / "mnist_model_pruned.tflite"
    pruned_tflite_path.write_bytes(pruned_tflite_model)
    pruned_tflite_size = os.path.getsize(pruned_tflite_path)
    print(f"     ✅ 프루닝 Float32 모델: {pruned_tflite_size / 1024:.2f} KB")

    # === 양자화만 적용 ===

    # 15. 원본 모델 양자화 (Dynamic Range)
    print("\n[15] 원본 모델 양자화 중 (Dynamic Range)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    baseline_quant_tflite_model = converter.convert()

    baseline_quant_path = models_dir / "mnist_model_baseline_quant.tflite"
    baseline_quant_path.write_bytes(baseline_quant_tflite_model)
    baseline_quant_size = os.path.getsize(baseline_quant_path)
    print(f"     ✅ 원본 양자화 모델: {baseline_quant_size / 1024:.2f} KB")

    # === 프루닝 + 양자화 결합 ===

    # 16. 프루닝 + 동적 범위 양자화
    print("\n[16] 프루닝 + 양자화 (Dynamic Range) 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    pruned_quant_tflite_model = converter.convert()

    pruned_quant_path = models_dir / "mnist_model_pruned_quant.tflite"
    pruned_quant_path.write_bytes(pruned_quant_tflite_model)
    pruned_quant_size = os.path.getsize(pruned_quant_path)
    print(f"     ✅ 프루닝+양자화 모델: {pruned_quant_size / 1024:.2f} KB")

    # 17. 프루닝 + 정수 양자화 (Int8)
    print("\n[17] 프루닝 + 정수 양자화 (Int8) 변환 중...")

    # 대표 데이터셋 생성
    def representative_dataset():
        for i in range(100):
            yield [train_images_split[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # 모든 연산을 Int8로 강제
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    pruned_int8_tflite_model = converter.convert()

    pruned_int8_path = models_dir / "mnist_model_pruned_int8.tflite"
    pruned_int8_path.write_bytes(pruned_int8_tflite_model)
    pruned_int8_size = os.path.getsize(pruned_int8_path)
    print(f"     ✅ 프루닝+Int8 모델: {pruned_int8_size / 1024:.2f} KB")

    # 18. gzip 압축 효과 확인
    print("\n[18] gzip 압축 효과 확인 중...")

    def get_gzip_size(model_bytes):
        compressed = gzip.compress(model_bytes)
        return len(compressed)

    baseline_gz_size = get_gzip_size(baseline_tflite_model)
    pruned_gz_size = get_gzip_size(pruned_tflite_model)
    baseline_quant_gz_size = get_gzip_size(baseline_quant_tflite_model)
    pruned_quant_gz_size = get_gzip_size(pruned_quant_tflite_model)
    pruned_int8_gz_size = get_gzip_size(pruned_int8_tflite_model)

    print(f"     원본 (gzip):             {baseline_gz_size / 1024:.2f} KB")
    print(f"     프루닝 (gzip):           {pruned_gz_size / 1024:.2f} KB")
    print(f"     원본+양자화 (gzip):      {baseline_quant_gz_size / 1024:.2f} KB")
    print(f"     프루닝+양자화 (gzip):    {pruned_quant_gz_size / 1024:.2f} KB")
    print(f"     프루닝+Int8 (gzip):      {pruned_int8_gz_size / 1024:.2f} KB")

    # 19. TFLite 모델 정확도 평가
    print("\n[19] TFLite 모델 정확도 평가 중...")

    def evaluate_tflite_model(tflite_model_content, test_images, test_labels):
        """TFLite 모델 정확도 평가"""
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        input_index = input_details["index"]
        output_index = output_details["index"]
        input_dtype = input_details["dtype"]

        correct_count = 0
        for i in range(len(test_images)):
            test_image = np.expand_dims(test_images[i], axis=0)

            # Int8 모델의 경우 양자화 수행
            if input_dtype == np.int8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image = test_image / input_scale + input_zero_point
                test_image = test_image.astype(np.int8)
            else:
                test_image = test_image.astype(np.float32)

            interpreter.set_tensor(input_index, test_image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)

            # Int8 모델의 경우 역양자화
            if output_details["dtype"] == np.int8:
                output_scale, output_zero_point = output_details["quantization"]
                output = output.astype(np.float32)
                output = (output - output_zero_point) * output_scale

            predicted_label = np.argmax(output[0])
            if predicted_label == test_labels[i]:
                correct_count += 1

        return correct_count / len(test_labels)

    baseline_tflite_accuracy = evaluate_tflite_model(
        baseline_tflite_model, test_images, test_labels
    )
    print(f"     원본 Float32:            {baseline_tflite_accuracy * 100:.2f}%")

    pruned_tflite_accuracy = evaluate_tflite_model(
        pruned_tflite_model, test_images, test_labels
    )
    print(f"     프루닝 Float32:          {pruned_tflite_accuracy * 100:.2f}%")

    baseline_quant_accuracy = evaluate_tflite_model(
        baseline_quant_tflite_model, test_images, test_labels
    )
    print(f"     원본 양자화:             {baseline_quant_accuracy * 100:.2f}%")

    pruned_quant_accuracy = evaluate_tflite_model(
        pruned_quant_tflite_model, test_images, test_labels
    )
    print(f"     프루닝+양자화:           {pruned_quant_accuracy * 100:.2f}%")

    pruned_int8_accuracy = evaluate_tflite_model(
        pruned_int8_tflite_model, test_images, test_labels
    )
    print(f"     프루닝+Int8:             {pruned_int8_accuracy * 100:.2f}%")

    # 20. 희소성 확인
    print("\n[20] 모델 희소성(sparsity) 확인")

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

    # 21. 테스트 데이터 저장 (벤치마크용)
    print("\n[21] 테스트 데이터 저장 중 (벤치마크용)...")
    test_images_path = models_dir / "mnist_test_images.npy"
    test_labels_path = models_dir / "mnist_test_labels.npy"

    np.save(test_images_path, (test_images * 255).astype(np.uint8))
    np.save(test_labels_path, test_labels)
    print(f"     ✅ {test_images_path}")
    print(f"     ✅ {test_labels_path}")

    # 22. 요약
    print("\n" + "=" * 70)
    print("✅ 프루닝 + 양자화 결합 예제 완료!")
    print("=" * 70)

    print("\n📊 정확도 비교:")
    print(f"   • 원본 Float32:          {baseline_tflite_accuracy * 100:.2f}%")
    print(f"   • 프루닝:                {pruned_tflite_accuracy * 100:.2f}%")
    print(f"   • 원본+양자화:           {baseline_quant_accuracy * 100:.2f}%")
    print(f"   • 프루닝+양자화:         {pruned_quant_accuracy * 100:.2f}%")
    print(f"   • 프루닝+Int8:           {pruned_int8_accuracy * 100:.2f}%")

    print("\n📦 모델 크기 비교 (압축 전):")
    print(f"   • 원본:                  {baseline_tflite_size / 1024:.2f} KB (100.0%)")
    print(
        f"   • 프루닝:                {pruned_tflite_size / 1024:.2f} KB ({pruned_tflite_size / baseline_tflite_size * 100:.1f}%)"
    )
    print(
        f"   • 원본+양자화:           {baseline_quant_size / 1024:.2f} KB ({baseline_quant_size / baseline_tflite_size * 100:.1f}%)"
    )
    print(
        f"   • 프루닝+양자화:         {pruned_quant_size / 1024:.2f} KB ({pruned_quant_size / baseline_tflite_size * 100:.1f}%)"
    )
    print(
        f"   • 프루닝+Int8:           {pruned_int8_size / 1024:.2f} KB ({pruned_int8_size / baseline_tflite_size * 100:.1f}%)"
    )

    print("\n📦 모델 크기 비교 (gzip 압축 후):")
    print(f"   • 원본:                  {baseline_gz_size / 1024:.2f} KB (100.0%)")
    print(
        f"   • 프루닝:                {pruned_gz_size / 1024:.2f} KB ({pruned_gz_size / baseline_gz_size * 100:.1f}%)"
    )
    print(
        f"   • 원본+양자화:           {baseline_quant_gz_size / 1024:.2f} KB ({baseline_quant_gz_size / baseline_gz_size * 100:.1f}%)"
    )
    print(
        f"   • 프루닝+양자화:         {pruned_quant_gz_size / 1024:.2f} KB ({pruned_quant_gz_size / baseline_gz_size * 100:.1f}%)"
    )
    print(
        f"   • 프루닝+Int8:           {pruned_int8_gz_size / 1024:.2f} KB ({pruned_int8_gz_size / baseline_gz_size * 100:.1f}%)"
    )

    print(f"\n   • 모델 희소성:           {sparsity:.1f}%")

    print("\n💡 주요 결과:")
    print("   • 프루닝만으로는 크기가 줄지 않지만 gzip 압축 시 효과 큼")
    print("   • 양자화는 4배 압축 효과")
    print("   • 프루닝+양자화는 두 기법의 장점 결합")
    print("   • 프루닝+Int8은 가장 작지만 약간의 정확도 손실")

    print("\n📁 생성된 파일:")
    print(f"   • {baseline_tflite_path}")
    print(f"   • {pruned_tflite_path}")
    print(f"   • {baseline_quant_path}")
    print(f"   • {pruned_quant_path}")
    print(f"   • {pruned_int8_path}")
    print(f"   • {test_images_path}")
    print(f"   • {test_labels_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
