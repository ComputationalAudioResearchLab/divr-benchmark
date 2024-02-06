from typing import List
from divr_benchmark.prepare_dataset.processed import ProcessedSession, ProcessedDataset


def assert_all_sessions_allocated(
    sessions: List[ProcessedSession], dataset: ProcessedDataset
):
    sessions_in_db = (
        dataset.train_sessions + dataset.test_sessions + dataset.val_sessions
    )
    ids_in_db = set([s.id for s in sessions_in_db])
    ids_in_input = set([s.id for s in sessions])
    assert ids_in_db == ids_in_input
