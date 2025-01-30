import statistics
import yaml
from pathlib import Path
from typing import List, Callable, Awaitable
from dataclasses import dataclass

from .task import Task
from .databases import Base as Database

DatabaseFunc = Callable[[str], Awaitable[Database]]


@dataclass
class Dataset:
    train: List[Task]
    val: List[Task]
    test: List[Task]


class Generator:
    def to_task_file(self, tasks: List[Task], output_path: Path) -> None:
        tasks_dict = {}
        for task in tasks:
            task_data = task.__dict__.copy()
            del task_data["id"]
            task_data["label"] = task.label.name
            tasks_dict[task.id] = task_data
        with open(f"{output_path}.yml", "w") as output_file:
            yaml.dump(tasks_dict, output_file)
        self.generate_demographics(tasks=tasks, output_path=output_path)

    def generate_demographics(self, tasks: List[Task], output_path: Path) -> None:
        demographics = {}
        for task in tasks:
            diagnosis_name = task.label.name
            if diagnosis_name not in demographics:
                demographics[diagnosis_name] = {}
            diagnosis = demographics[diagnosis_name]
            gender = task.gender
            if gender not in diagnosis:
                diagnosis[gender] = {"ages": [], "total": 0}
            diagnosis[gender]["total"] += 1
            if task.age is not None:
                diagnosis[gender]["ages"] += [task.age]
        for diagnosis in demographics:
            for gender in demographics[diagnosis]:
                ages = demographics[diagnosis][gender]["ages"]
                total = demographics[diagnosis][gender]["total"]
                total_ages = len(ages)
                age_stats = None
                if total_ages > 0:
                    age_stats = {
                        "mean": statistics.mean(ages),
                        "std": statistics.stdev(ages) if total_ages > 1 else None,
                        "min": min(ages),
                        "max": max(ages),
                    }

                demographics[diagnosis][gender] = {
                    "total": total,
                    "age_stats": age_stats,
                }
        with open(f"{output_path}.demographics.yml", "w") as output_file:
            yaml.dump(demographics, output_file)

    def truncate_low_resource_classes(
        self, task_list: List[List[Task]], min_examples: int
    ) -> List[List[Task]]:
        to_remove = []
        for tasks in task_list:
            counts = {}
            for task in tasks:
                diagnosis_name = task.label.name
                if diagnosis_name not in counts:
                    counts[diagnosis_name] = 0
                counts[diagnosis_name] += 1
            for label, count in counts.items():
                if count < min_examples:
                    to_remove += [label]
        new_list = []
        for tasks in task_list:
            new_tasks = []
            for task in tasks:
                diagnosis_name = task.label.name
                if diagnosis_name not in to_remove:
                    new_tasks += [task]
            new_list += [new_tasks]
        return new_list
