"""
Raspberry Pi 4 성능 벤치마크

주의: create_models.py로 PC에서 미리 생성한 모델들을 벤치마크합니다.
Raspberry Pi 4에서 실행하세요.

MNIST 데이터 로드:
1. PC에서 생성한 numpy 파일 우선 로드 (TensorFlow 불필요)
2. 없으면 HTTP로 직접 다운로드
3. 최후에 tf.keras.datasets 사용
"""

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    # tensorflow-lite-runtime만 있는 경우
    try:
        import tflite_runtime.interpreter as tflite

        TF_AVAILABLE = False
    except ImportError:
        raise ImportError(
            "TensorFlow 또는 tflite_runtime이 필요합니다.\n"
            "설치: pip install tensorflow 또는 pip install tflite-runtime"
        )

import numpy as np
import pathlib
import os
import time
import json
import platform
import urllib.request
import gzip


def load_mnist_data(models_dir):
    """
    MNIST 테스트 데이터 로드

    우선순위:
    1. PC에서 생성한 numpy 파일 (.npy)
    2. HTTP로 직접 다운로드
    3. tf.keras.datasets 사용 (TensorFlow 설치된 경우)
    """
    test_images = None
    test_labels = None

    # 방법 1: PC에서 생성한 numpy 파일 로드
    test_images_path = models_dir / "mnist_test_images.npy"
    test_labels_path = models_dir / "mnist_test_labels.npy"

    if test_images_path.exists() and test_labels_path.exists():
        print("    방법: PC에서 생성한 numpy 파일 로드")
        test_images = np.load(test_images_path) / 255.0
        test_labels = np.load(test_labels_path)
        print(f"    ✅ 테스트 이미지: {test_images.shape}")
        return test_images, test_labels

    # 방법 2: HTTP로 직접 다운로드
    print("    방법: HTTP로 직접 다운로드 시도...")
    try:
        base_url = "http://yann.lecun.com/exdb/mnist/"

        # 테스트 이미지 다운로드
        images_url = base_url + "t10k-images-idx3-ubyte.gz"
        labels_url = base_url + "t10k-labels-idx1-ubyte.gz"

        print("    다운로드 중... (최초 1회만, 이후 캐시 사용)")

        # 임시 디렉토리에 캐시
        cache_dir = models_dir / ".mnist_cache"
        cache_dir.mkdir(exist_ok=True)

        images_cache = cache_dir / "t10k-images-idx3-ubyte.gz"
        labels_cache = cache_dir / "t10k-labels-idx1-ubyte.gz"

        if not images_cache.exists():
            urllib.request.urlretrieve(images_url, images_cache)
        if not labels_cache.exists():
            urllib.request.urlretrieve(labels_url, labels_cache)

        # 파일 파싱
        with gzip.open(images_cache, "rb") as f:
            f.read(16)  # 헤더 스킵
            buf = f.read()
            test_images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            test_images = test_images.reshape(10000, 28, 28) / 255.0

        with gzip.open(labels_cache, "rb") as f:
            f.read(8)  # 헤더 스킵
            buf = f.read()
            test_labels = np.frombuffer(buf, dtype=np.uint8)

        print(f"    ✅ 테스트 이미지: {test_images.shape}")

        # numpy 파일로 저장 (다음 실행 시 빠르게 로드)
        np.save(test_images_path, test_images * 255.0)
        np.save(test_labels_path, test_labels)
        print("    ✅ numpy 파일로 저장 완료 (다음 실행 시 더 빠름)")

        return test_images, test_labels

    except Exception as e:
        print(f"    ⚠️  HTTP 다운로드 실패: {str(e)}")

    # 방법 3: tf.keras.datasets 사용 (fallback)
    if TF_AVAILABLE:
        print("    방법: tf.keras.datasets 사용 (fallback)")
        try:
            mnist = tf.keras.datasets.mnist
            _, (test_images, test_labels) = mnist.load_data()
            test_images = test_images / 255.0
            print(f"    ✅ 테스트 이미지: {test_images.shape}")
            return test_images, test_labels
        except Exception as e:
            print(f"    ❌ tf.keras.datasets 로드 실패: {str(e)}")

    raise RuntimeError(
        "MNIST 데이터를 로드할 수 없습니다.\n"
        "해결 방법:\n"
        "1. PC에서 create_models.py를 실행하여 numpy 파일 생성\n"
        "2. 인터넷 연결 확인 (HTTP 다운로드용)\n"
        "3. TensorFlow 설치: pip install tensorflow"
    )


def get_system_info():
    """시스템 정보 조회"""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__
        if TF_AVAILABLE
        else "N/A (tflite-runtime)",
    }

    # Raspberry Pi 정보 시도
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            if "BCM" in cpuinfo:
                info["device"] = "Raspberry Pi"
                if "ARMv7" in cpuinfo:
                    info["arch"] = "ARMv7 (32-bit)"
                elif "ARMv8" in cpuinfo:
                    info["arch"] = "ARMv8 (64-bit)"
    except:
        info["device"] = "Unknown"

    # 메모리 정보 시도
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_kb = int(line.split()[1])
                    info["total_memory_gb"] = mem_kb / 1024 / 1024
                elif line.startswith("MemAvailable"):
                    avail_kb = int(line.split()[1])
                    info["available_memory_gb"] = avail_kb / 1024 / 1024
    except:
        pass

    return info


class ModelBenchmark:
    """모델 벤치마크 클래스"""

    def __init__(self, model_path):
        self.model_path = model_path
        if TF_AVAILABLE:
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        else:
            self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

    def get_model_details(self):
        """모델 상세 정보"""
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        return {
            "size_kb": os.path.getsize(self.model_path) / 1024,
            "input_dtype": str(input_details["dtype"]),
            "output_dtype": str(output_details["dtype"]),
        }

    def benchmark_inference(self, test_data, num_runs=100):
        """추론 성능 벤치마크"""
        input_details = self.interpreter.get_input_details()[0]
        input_index = input_details["index"]

        # 워밍업 (첫 1회 실행은 느릴 수 있음)
        test_image = test_data[0]
        if input_details["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        self.interpreter.set_tensor(input_index, test_image)
        self.interpreter.invoke()

        # 실제 벤치마크
        times = []
        for _ in range(num_runs):
            test_image = test_data[0]
            if input_details["dtype"] == np.uint8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image = test_image / input_scale + input_zero_point

            test_image = np.expand_dims(test_image, axis=0).astype(
                input_details["dtype"]
            )
            start = time.time()
            self.interpreter.set_tensor(input_index, test_image)
            self.interpreter.invoke()
            times.append((time.time() - start) * 1000)

        times = np.array(times)

        return {
            "mean_ms": np.mean(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "std_ms": np.std(times),
            "median_ms": np.median(times),
            "fps": 1000.0 / np.mean(times),
        }

    def evaluate_accuracy(self, test_images, test_labels):
        """정확도 평가"""
        input_details = self.interpreter.get_input_details()[0]
        output_index = self.interpreter.get_output_details()[0]["index"]

        correct_count = 0
        total_count = 0

        for i, test_image in enumerate(test_images):
            # 정수 입력인 경우 처리
            if input_details["dtype"] == np.uint8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image_scaled = test_image / input_scale + input_zero_point
            else:
                test_image_scaled = test_image

            test_image_expanded = np.expand_dims(test_image_scaled, axis=0).astype(
                input_details["dtype"]
            )
            self.interpreter.set_tensor(input_details["index"], test_image_expanded)
            self.interpreter.invoke()

            output = self.interpreter.get_tensor(output_index)[0]
            prediction = np.argmax(output)

            if prediction == test_labels[i]:
                correct_count += 1
            total_count += 1

        return correct_count / total_count


def main():
    print("=" * 80)
    print("Raspberry Pi 4 성능 벤치마크 (LiteRT MNIST 양자화 모델)")
    print("=" * 80)

    # 1. 시스템 정보 출력
    print("\n[1] 시스템 정보")
    print("-" * 80)
    sys_info = get_system_info()
    for key, value in sys_info.items():
        if isinstance(value, float):
            print(f"    {key:<25}: {value:.2f}")
        else:
            print(f"    {key:<25}: {value}")

    # 2. MNIST 데이터셋 로드
    print("\n[2] MNIST 테스트 데이터셋 로드 중...")
    print("-" * 80)

    # 모델 디렉토리 먼저 확인
    models_dir = pathlib.Path("./mnist_tflite_models/")

    test_images, test_labels = load_mnist_data(models_dir)
    print("-" * 80)

    # 3. 모델 경로 확인
    print("\n[3] 모델 파일 확인 중...")

    if not models_dir.exists():
        print(f"    ❌ 오류: {models_dir} 디렉토리를 찾을 수 없습니다.")
        print("    PC에서 create_models.py를 먼저 실행하세요.")
        return

    model_files = {
        "Float32": "mnist_model.tflite",
        "Float16": "mnist_model_quant_f16.tflite",
        "DynamicRange": "mnist_model_quant_dynamic.tflite",
        "Integer": "mnist_model_quant_int8.tflite",
        "QAT": "mnist_model_quant_qat.tflite",
        "Int16x8": "mnist_model_quant_16x8.tflite",
    }

    models_to_test = {}
    missing_models = []

    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            models_to_test[model_name] = str(model_path)
            print(f"    ✅ {model_name}: {filename}")
        else:
            missing_models.append(filename)
            print(f"    ❌ {model_name}: {filename} (찾을 수 없음)")

    if not models_to_test:
        print("\n    ❌ 사용 가능한 모델이 없습니다.")
        print("    PC에서 create_models.py를 실행하여 모델들을 생성하세요.")
        return

    # 4. 벤치마크 수행
    print("\n[4] 벤치마크 수행 중...")
    print("-" * 80)

    results = {}
    baseline_accuracy = None

    for model_name, model_path in models_to_test.items():
        print(f"\n   {model_name} 벤치마크 중...")

        try:
            benchmark = ModelBenchmark(model_path)

            # 모델 정보
            model_details = benchmark.get_model_details()

            # 정확도 평가
            accuracy = benchmark.evaluate_accuracy(test_images, test_labels)

            if model_name == "Float32":
                baseline_accuracy = accuracy

            # 성능 측정 (50회 실행)
            perf_metrics = benchmark_inference(benchmark, test_images, num_runs=50)

            results[model_name] = {
                "accuracy": f"{accuracy * 100:.2f}%",
                "baseline_accuracy_diff": (
                    f"{(accuracy - baseline_accuracy) * 100:+.2f}%p"
                    if baseline_accuracy
                    else "N/A"
                ),
                "model_size_kb": f"{model_details['size_kb']:.2f}",
                "input_dtype": model_details["input_dtype"],
                "output_dtype": model_details["output_dtype"],
                "inference_mean_ms": f"{perf_metrics['mean_ms']:.2f}",
                "inference_median_ms": f"{perf_metrics['median_ms']:.2f}",
                "inference_min_ms": f"{perf_metrics['min_ms']:.2f}",
                "inference_max_ms": f"{perf_metrics['max_ms']:.2f}",
                "inference_std_ms": f"{perf_metrics['std_ms']:.2f}",
                "fps": f"{perf_metrics['fps']:.1f}",
            }

            print(f"      ✅ 정확도: {accuracy * 100:.2f}%")
            print(f"      ✅ 모델 크기: {model_details['size_kb']:.2f} KB")
            print(f"      ✅ 평균 추론 시간: {perf_metrics['mean_ms']:.2f} ms")
            print(f"      ✅ FPS: {perf_metrics['fps']:.1f}")

        except Exception as e:
            print(f"      ❌ 오류: {str(e)}")
            results[model_name] = {"error": str(e)}

    # 5. 결과 저장
    print("\n[5] 결과 저장 중...")
    results_file = models_dir / "benchmark_results_rpi4.json"

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": sys_info,
        "baseline_accuracy": f"{baseline_accuracy * 100:.2f}%"
        if baseline_accuracy
        else "N/A",
        "models": results,
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"    ✅ 결과 저장: {results_file}")

    # 6. 결과 요약 출력
    print("\n" + "=" * 80)
    print("벤치마크 결과 요약")
    print("=" * 80)

    if baseline_accuracy:
        print(f"\n기준 모델 (Float32) 정확도: {baseline_accuracy * 100:.2f}%")

    print(f"\n{'모델':<15} {'정확도':<12} {'크기(KB)':<12} {'추론(ms)':<12} {'FPS':<8}")
    print("-" * 70)

    for model_name, metrics in results.items():
        if "error" not in metrics:
            accuracy = metrics["accuracy"]
            size = metrics["model_size_kb"]
            inference = metrics["inference_mean_ms"]
            fps = metrics["fps"]
            print(
                f"{model_name:<15} {accuracy:<12} {size:<12} {inference:<12} {fps:<8}"
            )

    print("\n" + "=" * 80)
    print(f"✅ 벤치마크 완료! 결과: {results_file}")
    print("=" * 80)


def benchmark_inference(benchmark, test_data, num_runs=50):
    """추론 벤치마크 수행"""
    return benchmark.benchmark_inference(test_data, num_runs=num_runs)


if __name__ == "__main__":
    main()
