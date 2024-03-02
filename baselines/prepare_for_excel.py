import pandas as pd


def combiner(groups, key):
    new_df = pd.DataFrame()
    for (col_name,), val in groups:
        val = val[["task_key", key]]
        val.columns = ["task_key", col_name]
        if new_df.empty:
            new_df = val
        else:
            new_df = pd.merge(left=new_df, right=val, on="task_key")
    return new_df


def process(name: str, df: pd.DataFrame):
    groups = df.groupby(by=["feature_name"])
    for key in ["acc", "confidence_high", "confidence_low"]:
        combiner(groups=groups, key=key).to_csv(
            f"/home/workspace/baselines/data/df_{name}_{key}.csv", index=False
        )


def main():
    df = pd.read_csv("/home/workspace/baselines/data/collector_results.csv")
    level_1 = df[df["task_key"].isin([1, 2, 3, 4, 5, 6, 7])]
    level_2 = df[~df["task_key"].isin([1, 2, 3, 4, 5, 6, 7])]
    process(name="level_1", df=level_1)
    process(name="level_2", df=level_2)


if __name__ == "__main__":
    main()
