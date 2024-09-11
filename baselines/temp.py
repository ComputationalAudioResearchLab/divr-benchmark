import pandas as pd

df = pd.read_csv("/home/workspace/baselines/data/results.csv")
df = df[["Stream", "Task", "RecallPerClass"]]
df.to_csv("./class_recall.csv", index=False)
