from divr_diagnosis import diagnosis_maps
from divr_benchmark import Benchmark, Dataset

# Define diagnostic framework
diag_map = diagnosis_maps.CaRLab_2025()

# Instantiate benchmarking toolkit
benchmark = Benchmark(
    storage_path="/home/user/storage",
    version="v1",
    sample_rate=16000,
)

# Select vowel /a/ samples for SVD
def select_a(tasks):
    filtered_tasks = []
    for task in tasks:
        new_audios = []
        for audio_path in task.audio_keys:
            if audio_path.endswith("-a_n.wav"):
                new_audios += [audio_path]
        if len(new_audios) > 0:
            task.audio_keys = new_audios
            filtered_tasks += [task]
    return filtered_tasks

# Filter incompletely classified samples
def valid(tasks):
    return [
        t for t in tasks
        if not t.label.incompletely_classified
    ]

# Define filter function for cross-database task
def filter_func(db_func):
    svd = await db_func(name="svd")
    meei = await db_func(name="meei")
    svd_train = valid(select_a(svd.all_train()))
    svd_test = valid(select_a(svd.all_test()))
    svd_val = valid(select_a(svd.all_val()))
    meei_all = valid(meei.all())
    return Dataset(
        train=svd_train + svd_val,
        val=svd_test,
        test=meei_all,
    )

# Generate the cross-database benchmark task
benchmark.generate_task(
    filter_func=lambda db_func: filter_func(db_func),
    task_path="/home/user/svd_a_to_meei_test",
    diagnosis_map=diag_map,
    allow_incomplete_classification=False,
)
