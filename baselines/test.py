from pathlib import Path
from divr_benchmark import Benchmark

storage_path = Path("/home/divr_benchmark/storage")
storage_path.mkdir(parents=True, exist_ok=True)
benchmark = Benchmark(storage_path=storage_path, version="v1")
print(benchmark)
