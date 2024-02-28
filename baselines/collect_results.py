import pandas as pd
from pathlib import Path


def main():
    start_path = Path("/home/workspace/baselines/data/divr_benchmark/results")
    result_files = list(start_path.rglob("result.log"))
    df = []
    for file in result_files:
        model = file.parent.name
        batch = model.removeprefix("Simple")
        model = model.removesuffix(batch)
        if batch == "":
            batch = 1
        feature = file.parent.parent.name
        task = int(file.parent.parent.parent.name.removeprefix("T"))
        stream = int(file.parent.parent.parent.parent.name.removeprefix("S"))
        with open(file, "r") as result_file:
            result_data = result_file.readlines()
            accuracy = float(result_data[0].strip().split(":")[1]) * 100
            df += [
                {
                    "Stream": stream,
                    "Task": task,
                    "Feature": feature,
                    "Batch": batch,
                    "Model": model,
                    "Weighted Accuracy": accuracy,
                }
            ]
    df = pd.DataFrame.from_records(df)
    df["Weighted Accuracy"] = df["Weighted Accuracy"].apply(lambda x: f"{x:.02f}")
    df = df.sort_values(by=["Stream", "Task", "Feature"])
    df.to_csv("results.csv", index=False)

    def best_model(group):
        return group.loc[group["Weighted Accuracy"].idxmax()]

    best_models = (
        df.groupby(by=["Stream", "Task"])
        .apply(best_model, include_groups=False)
        .reset_index()
    )
    best_models.to_csv("best_results.csv", index=False)


if __name__ == "__main__":
    main()
