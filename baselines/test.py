import torch
from tqdm import tqdm
from pathlib import Path
from divr_benchmark import Benchmark
from data_loader import MeanMfcc

storage_path = Path("/home/divr_benchmark/storage")
storage_path.mkdir(parents=True, exist_ok=True)
benchmark = Benchmark(storage_path=storage_path, version="v1")

# tasks = [4, 16, 4, 5]
# for stream, num_tasks in tqdm(enumerate(tasks), total=len(tasks), position=0):
#     for task_idx in tqdm(range(num_tasks), position=1):
#         task = benchmark.task(stream=stream, task=task_idx + 1)
#         tqdm.write(
#             f"{stream, task_idx}: {len(task.train), len(task.val), len(task.test)}"
#         )


data_loader = MeanMfcc(
    task=benchmark.task(stream=1, task=1),
    device=torch.device("cpu"),
    batch_size=32,
    random_seed=42,
    shuffle_train=True,
)
data_loader.train()
for (inputs, input_lens), labels in data_loader:
    print(inputs.shape, input_lens.shape, labels.shape)
