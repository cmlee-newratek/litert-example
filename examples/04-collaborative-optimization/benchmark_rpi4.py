"""
라즈베리파이 4 벤치마크 스크립트 (Collaborative Optimization 모델)

협업 최적화된 모델들의 성능을 라즈베리파이 4에서 측정합니다.

측정:
- 4가지 협업 최적화 모델 (CQAT, PQAT, PC, PCQAT)
- 정확도, 추론 속도, FPS, 모델 크기
"""

import sys
import json
import time
import platform
import numpy as np
from pathlib import Path


def get_system_info():
    """시스템 정보 수집"""
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "Model" in line:
                    system_info["rpi_model"] = line.strip()
                elif "Hardware" in line:
                    system_info["hardware"] = line.strip()
    except:
        pass

    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    system_info["total_memory"] = line.strip()
                elif "MemAvailable" in line:
                    system_info["available_memory"] = line.strip()
    except:
        pass

    return system_info


def load_mnist_data():
    """MNIST 테스트 데이터 로드"""
    data_dir = Path(__file__).parent / "mnist_collaborative_models"

    test_images_path = data_dir / "mnist_test_images.npy"
    test_labels_path = data_dir / "mnist_test_labels.npy"

    if test_images_path.exists() and test_labels_path.exists():
        print(f"   ✅ 로컬 데이터 로드: {test_images_path}")
        test_images = np.load(test_images_path) / 255.0
        test_labels = np.load(test_labels_path)
        return test_images, test_labels

    print("   ℹ️  로컬 데이터 없음. HTTP에서 다운로드 시도 중...")
    try:
        import urllib.request

        url_images = (
            "https://github.com/newracom/litert-example/raw/main/examples/"
            "02-quantization/mnist_tflite_models/mnist_test_images.npy"
        )
        url_labels = (
            "https://github.com/newracom/litert-example/raw/main/examples/"
            "02-quantization/mnist_tflite_models/mnist_test_labels.npy"
        )

        test_images_path = data_dir / "mnist_test_images.npy"
        test_labels_path = data_dir / "mnist_test_labels.npy"

        urllib.request.urlretrieve(url_images, test_images_path)
        urllib.request.urlretrieve(url_labels, test_labels_path)

        print(f"   ✅ HTTP 다운로드: {test_images_path}")
        test_images = np.load(test_images_path) / 255.0
        test_labels = np.load(test_labels_path)
        return test_images, test_labels
    except Exception as e:
        print(f"   ⚠️  HTTP 다운로드 실패: {e}")

    print("   ℹ️  TensorFlow에서 MNIST 데이터 로드 중...")
    try:
        import tensorflow as tf

        _, (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        test_images = test_images / 255.0
        return test_images, test_labels
    except Exception as e:
        print(f"   ❌ 데이터 로드 실패: {e}")
        sys.exit(1)


class ModelBenchmark:
    """모델 벤치마크 클래스"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = self._load_model()

    def _load_model(self):
        """모델 로드"""
        try:
            try:
                import tflite_runtime.interpreter as tflite

                return tflite.Interpreter(model_path=str(self.model_path))
            except ImportError:
                pass

            import tensorflow as tf

            return tf.lite.Interpreter(model_path=str(self.model_path))

        except Exception as e:
            print(f"   ❌ 모델 로드 실패: {e}")
            return None

    def evaluate_accuracy(self, test_images, test_labels):
        """정확도 평가"""
        if self.interpreter is None:
            return 0.0

        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        input_index = input_details["index"]
        output_index = output_details["index"]
        input_dtype = input_details["dtype"]

        correct_count = 0
        for i in range(len(test_images)):
            test_image = np.expand_dims(test_images[i], axis=0)

            if input_dtype == np.int8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image = test_image / input_scale + input_zero_point
                test_image = test_image.astype(np.int8)
            else:
                test_image = test_image.astype(np.float32)

            self.interpreter.set_tensor(input_index, test_image)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(output_index)

            if output_details["dtype"] == np.int8:
                output_scale, output_zero_point = output_details["quantization"]
                output = output.astype(np.float32)
                output = (output - output_zero_point) * output_scale

            predicted_label = np.argmax(output[0])
            if predicted_label == test_labels[i]:
                correct_count += 1

        return correct_count / len(test_labels)

    def benchmark_inference(self, test_images, num_runs=10):
        """추론 성능 측정"""
        if self.interpreter is None:
            return None

        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()[0]
        input_index = input_details["index"]
        input_dtype = input_details["dtype"]

        times = []

        # 워밍업
        test_image = np.expand_dims(test_images[0], axis=0)
        if input_dtype == np.int8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point
            test_image = test_image.astype(np.int8)
        else:
            test_image = test_image.astype(np.float32)
        self.interpreter.set_tensor(input_index, test_image)
        self.interpreter.invoke()

        # 벤치마크
        for i in range(min(num_runs, len(test_images))):
            test_image = np.expand_dims(test_images[i], axis=0)

            if input_dtype == np.int8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image = test_image / input_scale + input_zero_point
                test_image = test_image.astype(np.int8)
            else:
                test_image = test_image.astype(np.float32)

            start_time = time.perf_counter()
            self.interpreter.set_tensor(input_index, test_image)
            self.interpreter.invoke()
            end_time = time.perf_counter()

            times.append((end_time - start_time) * 1000)  # ms

        return {
            "avg_ms": np.mean(times),
            "median_ms": np.median(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000 / np.mean(times),
        }


def main():
    print("=" * 70)
    print("Raspberry Pi 4 벤치마크 - 협업 최적화 모델")
    print("=" * 70)

    # 1. 시스템 정보
    print("\n[1] 시스템 정보 수집 중...")
    system_info = get_system_info()
    print(f"    Platform: {system_info.get('platform', 'Unknown')}")
    print(f"    Python: {system_info.get('python_version', 'Unknown')}")

    # 2. MNIST 데이터 로드
    print("\n[2] MNIST 테스트 데이터 로드 중...")
    test_images, test_labels = load_mnist_data()
    print(f"    ✅ 로드됨: {test_images.shape[0]} 이미지")

    # 3. 모델 경로
    print("\n[3] 벤치마크할 모델 검색 중...")
    models_dir = Path(__file__).parent / "mnist_collaborative_models"

    models = {
        "baseline": models_dir / "mnist_model_baseline.tflite",
        "cqat": models_dir / "mnist_model_cqat.tflite",
        "pqat": models_dir / "mnist_model_pqat.tflite",
        "pc_int8": models_dir / "mnist_model_pc_int8.tflite",
        "pcqat": models_dir / "mnist_model_pcqat.tflite",
    }

    results = {
        "system_info": system_info,
        "models": {},
    }

    # 4. 각 모델 벤치마크
    print("\n[4] 모델 벤치마크 중...")
    for model_name, model_path in models.items():
        if not model_path.exists():
            print(f"    ⚠️  {model_name}: 파일 없음")
            continue

        print(f"\n    📊 {model_name.upper()}...")

        # 모델 크기
        model_size_kb = model_path.stat().st_size / 1024

        # 벤치마크
        benchmark = ModelBenchmark(model_path)

        # 정확도
        accuracy = benchmark.evaluate_accuracy(test_images, test_labels)
        print(f"       정확도: {accuracy * 100:.2f}%")

        # 추론 속도
        inference_results = benchmark.benchmark_inference(test_images, num_runs=20)
        if inference_results:
            print(f"       평균 추론: {inference_results['avg_ms']:.2f} ms")
            print(f"       중앙값 추론: {inference_results['median_ms']:.2f} ms")
            print(f"       FPS: {inference_results['fps']:.1f}")
            print(f"       모델 크기: {model_size_kb:.2f} KB")

            results["models"][model_name] = {
                "accuracy": accuracy,
                "model_size_kb": model_size_kb,
                "inference_ms": {
                    "avg": inference_results["avg_ms"],
                    "median": inference_results["median_ms"],
                    "min": inference_results["min_ms"],
                    "max": inference_results["max_ms"],
                },
                "fps": inference_results["fps"],
            }
        else:
            print("       ❌ 벤치마크 실패")

    # 5. JSON 결과 저장
    print("\n[5] 결과 저장 중...")
    results_path = models_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    ✅ {results_path}")

    # 6. 요약
    print("\n" + "=" * 70)
    print("✅ 벤치마크 완료!")
    print("=" * 70)

    if results["models"]:
        print("\n📊 벤치마크 결과 요약:")
        print(
            f"    {'모델':<20} {'크기':<12} {'정확도':<12} {'평균 추론':<12} {'FPS':<10}"
        )
        print("    " + "-" * 65)
        for model_name, result in results["models"].items():
            print(
                f"    {model_name:<20} "
                f"{result['model_size_kb']:<12.2f} "
                f"{result['accuracy'] * 100:<12.2f} "
                f"{result['inference_ms']['avg']:<12.2f} "
                f"{result['fps']:<10.1f}"
            )

    print("\n🎯 협업 최적화 효과:")
    print("    • CQAT (클러스터링+QAT): 적당한 압축 + 정확도 유지")
    print("    • PQAT (프루닝+QAT): 희소성 보존 + 추론 이득")
    print("    • PC (프루닝+클러스터링): 두 기법 조합 + 정확도 유지")
    print("    • PCQAT (프루닝+클러스터링+QAT): 최대 압축 달성")

    print("\n💾 상세 결과: ./mnist_collaborative_models/benchmark_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
