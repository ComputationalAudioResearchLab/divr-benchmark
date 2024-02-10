from tqdm import tqdm
from pathlib import Path
from divr_benchmark import Benchmark

storage_path = Path("/home/divr_benchmark/storage")
storage_path.mkdir(parents=True, exist_ok=True)
benchmark = Benchmark(storage_path=storage_path, version="v1")

tasks = [4, 16, 4, 5]
for stream, num_tasks in tqdm(enumerate(tasks), total=len(tasks), position=0):
    for task_idx in tqdm(range(num_tasks), position=1):
        task = benchmark.task(stream=stream, task=task_idx + 1)
        tqdm.write(
            f"{stream, task_idx}: {len(task.train), len(task.val), len(task.test)}"
        )
