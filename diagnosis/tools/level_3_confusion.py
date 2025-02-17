import yaml
import pandas as pd
from pathlib import Path


def level_3_confusion():
    curdir = Path(__file__).parent.resolve()
    with open(f"{curdir}/../divr_diagnosis/diagnosis_maps/USVAC_2025.yml", "r") as df:
        data = yaml.load(df, Loader=yaml.FullLoader)

    confusion = {}
    for row in data.values():
        if row["level"] == 3:
            parents = list(set(row["parents"].keys()))
            for parent1 in parents:
                if parent1 not in confusion:
                    confusion[parent1] = {"full_consensus": 0}
                parent1 = confusion[parent1]
                for parent2 in parents:
                    if parent2 not in parent1:
                        parent1[parent2] = 1
                    else:
                        parent1[parent2] += 1
            if len(parents) == 1:
                confusion[parents[0]]["full_consensus"] += 1

    df = pd.DataFrame.from_records(confusion)
    print(df)
